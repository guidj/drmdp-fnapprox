"""
Bias-Variance Decomposition Experiment for Reward Estimators.

This module runs experiments to empirically measure the bias and variance
of reward estimators (least-lfa and bayes-least-lfa) compared to true rewards.

Methodology
1. Train reward estimator models with different seeds
2. Collect one fixed dataset of (state, action, true_reward) samples
3. Apply all models to predict rewards for each sample
4. Compute bias/variance across the predictions for the same sample

This ensures all predictions target the same true reward value, making the
bias-variance decomposition mathematically valid.

Mathematical Background:
For predictions {ŷ₁, ŷ₂, ..., ŷₙ} for the same sample with true value y:
- Bias = mean(ŷ) - y
- Variance = var(ŷ)
- MSE = mean((ŷ - y)²) = Bias² + Variance
"""

import argparse
import copy
import dataclasses
import itertools
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import ray
import ray.data
from ray.data import aggregate

from drmdp import core, dataproc, envs, ioutils, metrics, rewdelay, task, transform

MAX_STEPS_PER_EPISODE_GEM = 10_000
SAMPLES_PER_ENV = 100_000
DATASET_SEED = 42


class BiasVarianceAggregator(aggregate.AggregateFn):
    """
    Aggregates predictions to compute bias-variance decomposition.
    """

    def __init__(self, name: str = "BiasVarianceAggregator"):
        super().__init__(
            init=self._init,
            accumulate_row=self._accumulate_row,
            merge=self._merge,
            finalize=self._finalize,
            name=name,
        )

    def _init(self, key: Any) -> Dict[str, Any]:
        """Initialize empty accumulator."""
        del key
        return {
            "predictions": [],
            "true_reward": None,
            "metadata": {},
            "count": 0,
            "true_reward_mismatch": False,
        }

    def _accumulate_row(
        self, acc: Dict[str, Any], row: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Accumulate a single row."""
        new_acc = copy.deepcopy(acc)

        # Append prediction
        new_acc["predictions"].append(row["pred_reward"])

        # Set or verify true_reward
        if new_acc["true_reward"] is None:
            new_acc["true_reward"] = row["true_reward"]
        elif not np.isclose(new_acc["true_reward"], row["true_reward"]):
            new_acc["true_reward_mismatch"] = True

        # Extract metadata from first row
        if new_acc["count"] == 0:
            new_acc["metadata"] = {
                "sample_id": int(row["sample_id"]),
                "env_name": row["env_name"],
                "rewest_method": row["rewest_method"],
                "estimation_episode": int(row["estimation_episode"]),
                "delay": int(row["delay"]),
                "gamma": float(row["gamma"]),
            }

        new_acc["count"] += 1
        return new_acc

    def _merge(
        self, acc_left: Dict[str, Any], acc_right: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two accumulators."""
        acc = copy.deepcopy(acc_left)

        # Combine predictions
        acc["predictions"].extend(acc_right["predictions"])

        # Verify true_reward consistency
        if acc["true_reward"] is None:
            acc["true_reward"] = acc_right["true_reward"]
        elif acc_right["true_reward"] is not None:
            if not np.isclose(acc["true_reward"], acc_right["true_reward"]):
                acc["true_reward_mismatch"] = True

        # Merge metadata (prefer non-empty)
        if not acc["metadata"] and acc_right["metadata"]:
            acc["metadata"] = acc_right["metadata"]

        # Combine counts and mismatch flags
        acc["count"] += acc_right["count"]
        acc["true_reward_mismatch"] = (
            acc["true_reward_mismatch"] or acc_right["true_reward_mismatch"]
        )

        return acc

    def _finalize(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        """Compute bias-variance statistics and return flat dict."""
        # Validate
        if acc["count"] == 0:
            raise ValueError("Empty group - no predictions to aggregate")

        # Log warning for true_reward mismatch
        if acc["true_reward_mismatch"]:
            logging.warning(
                "True reward mismatch detected for sample %s",
                acc["metadata"].get("sample_id", "unknown"),
            )

        # Convert predictions to numpy array
        predictions = np.array(acc["predictions"])
        true_reward = acc["true_reward"]

        # Compute bias-variance decomposition
        bv_stats = metrics.bias_variance_decomposition(predictions, true_reward)

        # Return flat dict with metadata + stats
        return {
            **acc["metadata"],
            "true_reward": float(true_reward),
            "num_models": acc["count"],
            **bv_stats,
        }


@dataclasses.dataclass(frozen=True)
class Args:
    """Task arguments."""

    num_runs: int
    num_episodes: int
    output_path: str


@dataclasses.dataclass(frozen=True)
class JobSpec:
    """Specification for a single experiment job."""

    env_name: str
    env_args: Mapping[str, Any]
    control_feats_spec: Sequence[Mapping[str, Any]]
    rewest_method: str
    rewest_args: Mapping[str, Any]
    reward_delay: int
    gamma: float
    estimation_episode: int
    num_episodes: int
    use_bias: bool
    epochs: int
    run_id: int


@dataclasses.dataclass(frozen=True)
class RewardModelArtifact:
    """Trained reward model artifact stored in memory."""

    model_id: str
    run_id: int
    env_name: str
    rewest_method: str
    estimation_episode: int
    reward_delay: int
    gamma: float
    weights: np.ndarray
    use_bias: bool
    ft_op_spec: List[Mapping[str, Any]]
    # Bayesian-specific fields
    mv_normal_mean: Optional[np.ndarray] = None
    mv_normal_cov: Optional[np.ndarray] = None
    estimation_meta: Optional[Dict[str, Any]] = None


def least_lfa_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    """Generate specs for least-lfa estimator."""
    specs: List[Mapping[str, Any]] = []
    for aee, feat_spec in itertools.product(attempt_estimation_episodes, feat_specs):
        specs.append(
            {
                "name": "least-lfa",
                "args": {
                    "attempt_estimation_episode": aee,
                    "feats_spec": feat_spec,
                    "estimation_buffer_mult": 25,
                },
            }
        )
    return specs


def bayes_least_lfa_specs(
    init_attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    """Generate specs for bayes-least-lfa estimator."""
    specs: List[Mapping[str, Any]] = []
    for iaee, feat_spec in itertools.product(
        init_attempt_estimation_episodes, feat_specs
    ):
        specs.append(
            {
                "name": "bayes-least-lfa",
                "args": {
                    "init_attempt_estimation_episode": iaee,
                    "mode": "double",
                    "feats_spec": feat_spec,
                    "estimation_buffer_mult": 25,
                },
            }
        )
    return specs


def experiment_specs() -> Sequence[Mapping[str, Any]]:
    """
    Returns experiment configurations.
    """
    return (
        {
            "name": "Finite-CC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [
                    {
                        "name": "splice-tile-observation-action-ft",
                        "args": {"tiling_dim": 4},
                    }
                ]
            ],
            "rewest": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-CC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "rewest": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [
                    {
                        "name": "splice-tile-observation-action-ft",
                        "args": {"tiling_dim": 3},
                    }
                ]
            ],
            "rewest": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ],
            ],
            "rewest": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "rewest": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ]
            ],
            "rewest": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            ),
            "epochs": 1,
        },
    )


def create_all_job_specs(
    specs: Sequence[Mapping[str, Any]],
    num_runs: int,
    num_episodes: int,
    delays: Sequence[int] = (2,),  # 4, 6, 8),
    gammas: Sequence[float] = (1.0,),  # 0.99),
) -> Sequence[JobSpec]:
    """
    Create all job specifications for the bias-variance experiment.

    Generates jobs for all combinations of:
    - environments
    - estimators (least-lfa, bayes-least-lfa)
    - estimation episodes e.g. (10, 50, 100)
    - delays e.g. (2, 4, 6, 8)
    - gammas e.g. (1.0, 0.99)
    - runs (0 to num_runs-1)
    """
    jobs = []
    for spec in specs:
        for feats_spec, rewest_spec in itertools.product(
            spec["feats_specs"], spec["rewest"]
        ):
            # Extract estimation episode parameter
            if rewest_spec["name"] == "least-lfa":
                est_episode = rewest_spec["args"]["attempt_estimation_episode"]
            elif rewest_spec["name"] == "bayes-least-lfa":
                est_episode = rewest_spec["args"]["init_attempt_estimation_episode"]
            else:
                continue

            for delay, gamma, run_id in itertools.product(
                delays, gammas, range(num_runs)
            ):
                job_spec = JobSpec(
                    env_name=spec["name"],
                    env_args=spec["args"],
                    control_feats_spec=feats_spec,
                    num_episodes=num_episodes,
                    reward_delay=delay,
                    gamma=gamma,
                    estimation_episode=est_episode,
                    rewest_method=rewest_spec["name"],
                    rewest_args=rewest_spec["args"],
                    use_bias=False,
                    epochs=spec["epochs"],
                    run_id=run_id,
                )
                jobs.append(job_spec)
    return jobs


def fixed_delay_config(delay: int) -> Dict[str, Any]:
    """Create a fixed delay configuration."""
    return {"name": "fixed", "args": {"delay": delay}}


def create_exp_instance(job_spec: JobSpec) -> core.ExperimentInstance:
    """Create an experiment instance from a job spec."""
    env_spec = core.EnvSpec(
        job_spec.env_name,
        args=job_spec.env_args,
        feats_spec=job_spec.control_feats_spec,
    )
    problem_spec = core.ProblemSpec(
        policy_type="markovian",
        reward_mapper={
            "name": job_spec.rewest_method,
            "args": job_spec.rewest_args,
        },
        delay_config=fixed_delay_config(job_spec.reward_delay),
        epsilon=0.1,
        gamma=job_spec.gamma,
        learning_rate_config={
            "name": "constant",
            "args": {"initial_lr": 0.01},
        },
    )
    run_config = core.RunConfig(
        num_runs=1,
        episodes_per_run=job_spec.num_episodes * job_spec.epochs,
        log_episode_frequency=100,
        use_seed=True,
        output_dir=tempfile.gettempdir(),
    )
    return core.ExperimentInstance(
        exp_id=str(uuid.uuid4()),
        instance_id=job_spec.run_id,
        experiment=core.Experiment(
            env_spec=env_spec, problem_spec=problem_spec, epochs=job_spec.epochs
        ),
        run_config=run_config,
        context={},
        export_model=False,
    )


def extract_reward_model_from_wrapper(
    wrapper: gym.Wrapper, job_spec: JobSpec
) -> Optional[RewardModelArtifact]:
    """
    Extract reward model components from trained wrapper.

    Args:
        wrapper: Trained reward wrapper (LeastLfa or BayesLeastLfa)
        job_spec: Job specification containing ft_op_spec

    Returns:
        RewardModelArtifact if estimation succeeded, None otherwise
    """
    # Get ft_op_spec from job_spec
    ft_op_spec = list(job_spec.rewest_args.get("feats_spec", []))

    model_id = f"{job_spec.env_name}_{job_spec.rewest_method}_d{job_spec.reward_delay}_g{job_spec.gamma}_e{job_spec.estimation_episode}_r{job_spec.run_id}"

    if isinstance(wrapper, rewdelay.LeastLfaGenerativeRewardWrapper):
        if wrapper.weights is None:
            logging.warning("Model %s has no weights - estimation failed", model_id)
            return None

        return RewardModelArtifact(
            model_id=model_id,
            run_id=job_spec.run_id,
            env_name=job_spec.env_name,
            rewest_method=job_spec.rewest_method,
            estimation_episode=job_spec.estimation_episode,
            reward_delay=job_spec.reward_delay,
            gamma=job_spec.gamma,
            weights=wrapper.weights.copy(),
            use_bias=wrapper.use_bias,
            ft_op_spec=ft_op_spec,
            mv_normal_mean=None,
            mv_normal_cov=None,
            estimation_meta=dict(wrapper.estimation_meta)
            if wrapper.estimation_meta
            else None,
        )

    elif isinstance(wrapper, rewdelay.BayesLeastLfaGenerativeRewardWrapper):
        if wrapper.mv_normal_rewards is None:
            logging.warning("Model %s has no posterior - estimation failed", model_id)
            return None

        return RewardModelArtifact(
            model_id=model_id,
            run_id=job_spec.run_id,
            env_name=job_spec.env_name,
            rewest_method=job_spec.rewest_method,
            estimation_episode=job_spec.estimation_episode,
            reward_delay=job_spec.reward_delay,
            gamma=job_spec.gamma,
            weights=wrapper.mv_normal_rewards.mean.copy(),
            use_bias=wrapper.use_bias,
            ft_op_spec=ft_op_spec,
            mv_normal_mean=wrapper.mv_normal_rewards.mean.copy(),
            mv_normal_cov=wrapper.mv_normal_rewards.cov.copy(),
            estimation_meta=dict(wrapper.estimation_meta)
            if wrapper.estimation_meta
            else None,
        )

    else:
        raise ValueError(f"Unknown wrapper type: {type(wrapper)}")


@ray.remote
def train_reward_model_run(job_spec: JobSpec) -> Optional[RewardModelArtifact]:
    """
    Phase 1: Train single reward estimator and extract model components.

    Returns:
        RewardModelArtifact if training succeeded, None otherwise
    """
    task_id = str(uuid.uuid4())
    logging.info(
        "Training model for task %s: %s",
        task_id,
        job_spec.model_id if hasattr(job_spec, "model_id") else job_spec.run_id,
    )

    try:
        # Create experiment instance
        exp_instance = create_exp_instance(job_spec)

        # Setup environment
        env_spec = exp_instance.experiment.env_spec
        problem_spec = exp_instance.experiment.problem_spec

        # Create base environment
        env = envs.make(
            env_name=env_spec.name,
            **env_spec.args if env_spec.args else {},
        )
        proxy_env = envs.make(
            env_name=env_spec.name,
            **env_spec.args if env_spec.args else {},
        )

        # Add monitor
        env, monitor = task.monitor_wrapper(env)

        # Apply delay
        rew_delay = task.reward_delay_distribution(problem_spec.delay_config)
        env_delayed = task.delay_wrapper(env, rew_delay)

        # Apply reward mapper (this creates the wrapper we need to extract)
        env_mapped = task.reward_mapper(
            env_delayed,
            proxy_env=proxy_env,
            mapping_spec=problem_spec.reward_mapper,
        )

        # Create algorithm
        feats_op = transform.transform_pipeline(
            env=env_mapped, specs=env_spec.feats_spec
        )
        lr = task.learning_rate(**problem_spec.learning_rate_config)
        algorithm = task.create_algorithm(
            env=env_mapped,
            ft_op=feats_op,
            delay_reward=rew_delay,
            lr=lr,
            gamma=problem_spec.gamma,
            epsilon=problem_spec.epsilon,
            policy_type=problem_spec.policy_type,
            base_seed=exp_instance.instance_id,
        )

        # Train
        snapshots = algorithm.train(
            env=env_mapped,
            num_episodes=exp_instance.run_config.episodes_per_run,
            monitor=monitor,
        )

        # Trigger training (`train` returns a generator)
        for _ in snapshots:
            pass

        # Extract model artifact from the reward wrapper
        # The reward wrapper is env_mapped
        model_artifact = extract_reward_model_from_wrapper(env_mapped, job_spec)

        env.close()
        proxy_env.close()

        if model_artifact is not None:
            logging.info("Extracted model %s", model_artifact.model_id)
        else:
            logging.warning("Failed to extract model for task %s", task_id)

        return model_artifact

    except Exception as err:
        logging.error("Error training model for task %s: %s", task_id, err)
        return None


@ray.remote
def collect_sample_dataset(
    env_name: str, env_args: Mapping[str, Any], num_samples: int, seed: int
) -> Dict[str, Any]:
    """
    Phase 2: Collect fixed dataset using random exploration.

    Returns:
        Dictionary with observations, actions, rewards, sample_ids
    """
    logging.info("Collecting %d samples for %s", num_samples, env_name)

    env = envs.make(env_name, **env_args)
    buffer = dataproc.collection_traj_data(env, steps=num_samples, seed=seed)

    # Convert to arrays
    observations = np.array([obs for obs, _, _, _ in buffer])
    actions = np.array([action for _, action, _, _ in buffer])
    rewards = np.array([rew for _, _, _, rew in buffer])
    sample_ids = np.arange(len(buffer))

    env.close()

    dataset = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "sample_ids": sample_ids,
        "env_name": env_name,
    }

    logging.info("Collected %d samples for %s", len(sample_ids), env_name)
    return dataset


def create_predict_fn(artifact: RewardModelArtifact, env: gym.Env):
    """
    Create prediction function from in-memory artifact.

    Args:
        artifact: Trained model artifact
        env: Environment (for creating feature transformation pipeline)

    Returns:
        Function that predicts rewards: predict(obs, action) -> reward
    """
    # Reconstruct feature transformation pipeline from spec
    ft_op = transform.transform_pipeline(env=env, specs=artifact.ft_op_spec)

    def predict(obs: np.ndarray, action: int) -> float:
        """Predict reward for (observation, action) pair."""
        # Transform observation-action pair to features
        example = ft_op.apply(transform.Example(observation=obs, action=action))
        features = example.observation

        # Apply linear model
        reward = float(np.dot(features, artifact.weights))
        return reward

    return predict


@ray.remote
def predict_on_dataset(
    model_artifact: RewardModelArtifact,
    dataset: Dict[str, Any],
    env_args: Mapping[str, Any],
    output_path: str,
) -> str:
    """
    Phase 3: Apply in-memory model to all samples in dataset.

    Args:
        model_artifact: Trained model
        dataset: Fixed sample dataset
        env_args: Environment arguments for creating prediction function

    Returns:
        Path of predictions.
    """
    logging.info(
        "Predicting with model %s on %d samples",
        model_artifact.model_id,
        len(dataset["sample_ids"]),
    )

    # Create temporary environment for feature transformation
    env = envs.make(model_artifact.env_name, **env_args)
    predict_fn = create_predict_fn(model_artifact, env)

    # Vectorized prediction
    predictions = []
    for obs, action, true_reward, sample_id in zip(
        dataset["observations"],
        dataset["actions"],
        dataset["rewards"],
        dataset["sample_ids"],
    ):
        pred_reward = predict_fn(obs, action)
        predictions.append(
            {
                "sample_id": int(sample_id),
                "model_id": model_artifact.model_id,
                "run_id": model_artifact.run_id,
                "pred_reward": pred_reward,
                "true_reward": true_reward,
                "env_name": model_artifact.env_name,
                "rewest_method": model_artifact.rewest_method,
                "estimation_episode": model_artifact.estimation_episode,
                "delay": model_artifact.reward_delay,
                "gamma": model_artifact.gamma,
            }
        )

    env.close()
    ioutils.write_records_json(output_path, predictions, gzip_compression=False)
    logging.info(
        "Generated %d predictions for model %s. Exported to: %s", len(predictions), model_artifact.model_id, output_path
    )
    return output_path


def compute_bias_variance_from_predictions(
    ds_predictions: ray.data.Dataset,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 4: Compute bias-variance statistics from predictions.

    Args:
        predictions_dir: Directory containing prediction parquet files

    Returns:
        Tuple of (sample_level_df, summary_df)
    """
    logging.info("Computing bias-variance from predictions")

    # Read all predictions
    logging.info("Loaded predictions dataset")

    # Group by sample_id and configuration
    group_cols = [
        "sample_id",
        "env_name",
        "rewest_method",
        "delay",
        "gamma",
        "estimation_episode",
    ]

    # Apply groupby and compute stats
    logging.info("Columns: %s", ds_predictions.columns())
    logging.info("Grouping by %s and computing bias-variance...", group_cols)
    ds_result = ds_predictions.groupby(group_cols).aggregate(
        BiasVarianceAggregator(name="bias_variance_agg")
    )

    # Convert to pandas
    df_sample = ds_result.to_pandas()
    logging.info("Computed bias-variance for %d samples", len(df_sample))

    # Aggregate summary statistics
    logging.info("Computing summary statistics...")
    summary_group_cols = [
        "env_name",
        "rewest_method",
        "delay",
        "gamma",
        "estimation_episode",
    ]

    summary_stats = []
    for group_key, df_group in df_sample.groupby(summary_group_cols):
        config = dict(zip(summary_group_cols, group_key))
        summary_stats.append(
            {
                **config,
                "num_samples": len(df_group),
                "mean_bias": df_group["bias"].mean(),
                "std_bias": df_group["bias"].std(),
                "mean_variance": df_group["variance"].mean(),
                "std_variance": df_group["variance"].std(),
                "mean_mse": df_group["mse"].mean(),
                "std_mse": df_group["mse"].std(),
                "mean_bias_squared": df_group["bias_squared"].mean(),
                "mean_verification_error": df_group["verification_error"].mean(),
                "max_verification_error": df_group["verification_error"].max(),
            }
        )

    df_summary = pd.DataFrame(summary_stats)
    logging.info("Computed summary for %d configurations", len(df_summary))

    return df_sample, df_summary


def export_results(df_sample: pd.DataFrame, df_summary: pd.DataFrame, output_dir: str):
    """Export results to parquet."""
    # Export sample-level results
    sample_parquet_path = os.path.join(output_dir, "bias_var_sample.parquet")
    df_sample.to_parquet(sample_parquet_path, index=False)
    logging.info("Exported sample results to %s", sample_parquet_path)

    # Export summary results
    summary_parquet_path = os.path.join(output_dir, "bias_var_summary.parquet")
    df_summary.to_parquet(summary_parquet_path, index=False)
    logging.info("Exported summary results to %s", summary_parquet_path)


def main():
    """
    Main entry point for bias-variance experiment.

    4-Phase Workflow:
    1. Train reward models (in-memory artifacts)
    2. Collect fixed sample datasets
    3. Generate predictions (models applied to fixed datasets)
    4. Compute bias-variance and export results
    """
    args = parse_args()

    # Create all job specifications
    specs = experiment_specs()
    job_specs = create_all_job_specs(
        specs=specs,
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
    )
    logging.info("Created %d job specifications", len(job_specs))

    # Shuffle jobs to balance workload
    np.random.shuffle(job_specs)  # type: ignore

    # Initialize Ray
    with ray.init({"env_vars": {"RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": 0.5}}):
        logging.info("Ray initialized")
        # PHASE 1: Train models and extract artifacts (in-memory)
        logging.info("PHASE 1: Training %d reward models...", len(job_specs))
        model_refs = [train_reward_model_run.remote(spec) for spec in job_specs]
        model_artifacts: Sequence[Optional[RewardModelArtifact]] = wait_till_completion(
            model_refs, fetch=True
        )
        model_artifacts: Sequence[RewardModelArtifact] = [
            model_artifact
            for model_artifact in model_artifacts
            if model_artifact is not None
        ]
        logging.info(
            "Phase 1 complete: %d/%d models trained successfully",
            len(model_artifacts),
            len(job_specs),
        )

        # Envs with omodels
        envs_with_model = set([artifact.env_name for artifact in model_artifacts])

        # PHASE 2: Collect fixed sample datasets
        logging.info("PHASE 2: Collecting sample datasets...")
        env_configs = {spec["name"]: spec["args"] for spec in specs}
        dataset_refs = {
            env_name: collect_sample_dataset.remote(
                env_name, env_configs[env_name], SAMPLES_PER_ENV, DATASET_SEED
            )
            for env_name in sorted(envs_with_model)
        }
        logging.info(
            "Phase 2 complete: Collected datasets for %d environments",
            len(dataset_refs),
        )

        # PHASE 3: Generate predictions
        logging.info(
            "PHASE 3: Generating predictions for %d models...", len(model_artifacts)
        )
        pred_refs = []
        for artifact in model_artifacts:
            dataset_ref = dataset_refs[artifact.env_name]
            env_args = env_configs[artifact.env_name]
            pred_refs.append(predict_on_dataset.remote(artifact, dataset_ref, env_args))

        # Wait till all predictions are complete
        prediction_file_paths = wait_till_completion(pred_refs, fetch=True)

        ds_predictions = ray.data.read_json(prediction_file_paths, lines=True)

        # # PHASE 4: Compute bias-variance
        logging.info("PHASE 4: Computing bias-variance statistics...")
        df_sample, df_summary = compute_bias_variance_from_predictions(ds_predictions)

        # Export final results
        export_results(df_sample, df_summary, args.output_path)

        logging.info("Bias-variance analysis complete!")
        logging.info("Results: %s", args.output_path)
        logging.info(
            "Summary: %d samples, %d configurations", len(df_sample), len(df_summary)
        )


def wait_till_completion(
    tasks_refs, fetch: bool = False
) -> Sequence[ray.ObjectRef | Any]:
    """
    Waits for every ray task to complete.
    """
    results = []
    unfinished_tasks = tasks_refs
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        if fetch:
            results.extend(
                [ray.get(task) if fetch else task for task in finished_tasks]
            )
        logging.info(
            "Completed %d task(s). %d left out of %d.",
            len(finished_tasks),
            len(unfinished_tasks),
            len(tasks_refs),
        )

        if len(unfinished_tasks) == 0:
            break
    return results


def parse_args() -> Args:
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description="Run bias-variance decomposition experiments"
    )
    arg_parser.add_argument(
        "--num-runs",
        type=int,
        required=True,
        help="Number of runs per configuration",
    )
    arg_parser.add_argument(
        "--num-episodes",
        type=int,
        required=True,
        help="Number of episodes per run",
    )
    arg_parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory path",
    )
    known_args, _ = arg_parser.parse_known_args()
    return Args(**vars(known_args))


if __name__ == "__main__":
    main()
