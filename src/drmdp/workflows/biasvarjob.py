"""
Bias-Variance Decomposition Experiment for Reward Estimators.

This module runs experiments to empirically measure the bias and variance
of reward estimators (least-lfa and bayes-least-lfa) compared to true rewards.

The experiment workflow:
1. Runs multiple trials with identical configurations but different seeds
2. Tracks both true and predicted rewards at each timestep
3. Aligns rewards across runs by timestep
4. Computes bias-variance decomposition at each timestep
5. Aggregates statistics over time windows
6. Exports results to parquet and CSV files

Mathematical Background:
For predictions {ŷ₁, ŷ₂, ..., ŷₙ} at timestep t with true value y:
- Bias = mean(ŷ) - y
- Variance = var(ŷ)
- MSE = mean((ŷ - y)²) = Bias² + Variance
"""

import argparse
import dataclasses
import gzip
import itertools
import json
import logging
import math
import os
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import ray
import tensorflow as tf

from drmdp import core, envs, metrics, task, transform

MAX_STEPS_PER_EPISODE_GEM = 10_000
REWARD_EVAL_SAMPLES = 25_000
NUM_TASKS_PER_WRITER = 100


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


class BiasVarRewardStoreWrapper(gym.Wrapper):
    """
    Stores rewards with timestep metadata for cross-run alignment.

    This wrapper tracks rewards along with global timestep and episode counters,
    enabling alignment across multiple experimental runs.
    """

    def __init__(self, env, buffer_size: int):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, Any]] = []
        self.global_step = 0
        self.episode = 0

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        # Store reward with metadata
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(
                {
                    "reward": float(reward),
                    "global_step": self.global_step,
                    "episode": self.episode,
                }
            )
        self.global_step += 1
        if term or trunc:
            self.episode += 1
        return obs, reward, term, trunc, info


@ray.remote
class ResultWriter:
    """Remote task to export results to disk."""

    def __init__(self, prefix: str, output_path: str, partition_size: int = 100):
        self.prefix = prefix
        self.output_path = output_path
        self.partition_size = partition_size
        self.results: List[Any] = []
        self.partition = 0

    def write(self, result):
        """Add data to buffer, sync if partition size reached."""
        self.results.append(result)
        if len(self.results) >= self.partition_size:
            self.sync()

    def sync(self):
        """Write buffered data to file and clear buffer."""
        if self.results:
            write_records(
                os.path.join(
                    self.output_path, f"result-{self.prefix}-{self.partition}.jsonl"
                ),
                records=self.results,
            )
            self.results = []
            self.partition += 1


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
    delays: Sequence[int] = (2, 4, 6, 8),
    gammas: Sequence[float] = (1.0, 0.99),
) -> Sequence[JobSpec]:
    """
    Create all job specifications for the bias-variance experiment.

    Generates jobs for all combinations of:
    - environments
    - estimators (least-lfa, bayes-least-lfa)
    - estimation episodes (10, 50, 100)
    - delays (2, 4, 6, 8)
    - gammas (1.0, 0.99)
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


@ray.remote
def run_bias_var_experiment(
    job_spec: JobSpec, result_writer: ResultWriter
) -> Dict[str, Any]:
    """
    Execute single bias-variance experiment run with dual reward tracking.

    This function:
    1. Creates the environment
    2. Wraps it BEFORE delay to track true rewards
    3. Applies delay wrapper
    4. Applies reward mapper (least-lfa or bayes-least-lfa)
    5. Wraps it AFTER mapper to track predicted rewards
    6. Trains the algorithm
    7. Returns results with both reward buffers
    """
    task_id = str(uuid.uuid4())
    logging.info("Starting task %s: %s", task_id, job_spec)

    try:
        # Create experiment instance
        exp_instance = create_exp_instance(job_spec)

        # Setup environment with dual reward tracking
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

        # Wrap BEFORE delay to capture true rewards
        true_wrapper = BiasVarRewardStoreWrapper(env, buffer_size=REWARD_EVAL_SAMPLES)

        # Apply delay
        rew_delay = task.reward_delay_distribution(problem_spec.delay_config)
        env_delayed = task.delay_wrapper(true_wrapper, rew_delay)

        # Apply reward mapper
        env_mapped = task.reward_mapper(
            env_delayed,
            proxy_env=proxy_env,
            mapping_spec=problem_spec.reward_mapper,
        )

        # Wrap AFTER mapper to capture predicted rewards
        pred_wrapper = BiasVarRewardStoreWrapper(
            env_mapped, buffer_size=REWARD_EVAL_SAMPLES
        )

        # Create algorithm
        feats_op = transform.transform_pipeline(
            env=pred_wrapper, specs=env_spec.feats_spec
        )
        lr = task.learning_rate(**problem_spec.learning_rate_config)
        algorithm = task.create_algorithm(
            env=pred_wrapper,
            ft_op=feats_op,
            delay_reward=rew_delay,
            lr=lr,
            gamma=problem_spec.gamma,
            epsilon=problem_spec.epsilon,
            policy_type=problem_spec.policy_type,
            base_seed=exp_instance.instance_id,
        )

        # Train
        results = algorithm.train(
            env=pred_wrapper,
            num_episodes=exp_instance.run_config.episodes_per_run,
            monitor=monitor,
        )

        # Collect returns
        returns = []
        for _, snapshot in enumerate(results):
            returns.append(snapshot.returns)

        # Prepare result
        result = {
            "task_id": task_id,
            **dataclasses.asdict(job_spec),
            "true_rewards": true_wrapper.buffer,
            "pred_rewards": pred_wrapper.buffer,
            "returns": returns,
        }

        env.close()
        proxy_env.close()

        logging.info("Completed task %s: %s", task_id, job_spec)
        return result_writer.write.remote(result)  # type: ignore

    except Exception as err:
        logging.error("Error in experiment %s: %s", task_id, err)
        raise RuntimeError(f"Task {task_id} `{job_spec}` failed") from err


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
        epsilon=0.2,
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


def fixed_delay_config(delay: int) -> Dict[str, Any]:
    """Create a fixed delay configuration."""
    return {"name": "fixed", "args": {"delay": delay}}


def write_records(
    output_path: str,
    records: Sequence[Mapping[str, Any]],
    gzip_compression: bool = True,
) -> None:
    """Export records to JSON files."""
    bytes_size = sys.getsizeof(records)
    if gzip_compression and not output_path.endswith(".gzip"):
        output_path = ".".join([output_path, "gzip"])

    logging.debug(
        "Writing partition of %fMB to %s",
        bytes_size / 1024 / 1024,
        output_path,
    )

    if gzip_compression:
        with tf.io.gfile.GFile(output_path, "wb") as writable:
            with gzip.GzipFile(fileobj=writable, mode="wb") as writer:
                for record in records:
                    content = "".join([json.dumps(record), "\n"])
                    writer.write(content.encode("UTF-8"))
    else:
        with tf.io.gfile.GFile(output_path, "w") as writable:
            for record in records:
                json.dump(record, fp=writable)
                writable.write("\n")


def read_records(
    input_path: str, gzip_compression: bool = True
) -> Sequence[Mapping[str, Any]]:
    """Read records from JSON."""
    logging.debug("Reading file %s", input_path)

    records = []
    if gzip_compression:
        with tf.io.gfile.GFile(input_path, "rb") as readable:
            with gzip.GzipFile(fileobj=readable, mode="rb") as reader:
                for line in reader:
                    records.append(json.loads(line.decode("UTF-8")))
    else:
        with tf.io.gfile.GFile(input_path, "r") as readable:
            for line in readable:
                records.append(json.loads(line))
    return records


def load_and_align_results(results_dir: str) -> pd.DataFrame:
    """
    Load all experiment results and align rewards by timestep.

    Reads result-*.jsonl.gzip files, expands reward buffers into rows,
    and groups by configuration + timestep to align predictions across runs.
    """
    logging.info("Loading results from %s", results_dir)

    # Find all result files
    result_files = tf.io.gfile.glob(os.path.join(results_dir, "result-*.jsonl.gzip"))
    logging.info("Found %d result files", len(result_files))

    # Load and flatten all results
    rows = []
    for file_path in result_files:
        records = read_records(file_path, gzip_compression=True)
        for record in records:
            # Extract metadata
            env_name = record["env_name"]
            rewest_method = record["rewest_method"]
            delay = record["reward_delay"]
            gamma = record["gamma"]
            estimation_episode = record["estimation_episode"]
            run_id = record["run_id"]

            # Expand reward buffers
            true_rewards = record["true_rewards"]
            pred_rewards = record["pred_rewards"]

            # Align by using minimum length
            min_len = min(len(true_rewards), len(pred_rewards))

            for i in range(min_len):
                true_entry = true_rewards[i]
                pred_entry = pred_rewards[i]

                # Verify timestep alignment
                if true_entry["global_step"] != pred_entry["global_step"]:
                    logging.warning(
                        "Timestep mismatch for run %s: true=%d, pred=%d",
                        run_id,
                        true_entry["global_step"],
                        pred_entry["global_step"],
                    )
                    continue

                rows.append(
                    {
                        "env_name": env_name,
                        "rewest_method": rewest_method,
                        "delay": delay,
                        "gamma": gamma,
                        "estimation_episode": estimation_episode,
                        "run_id": run_id,
                        "global_step": true_entry["global_step"],
                        "episode": true_entry["episode"],
                        "true_reward": true_entry["reward"],
                        "pred_reward": pred_entry["reward"],
                    }
                )

    df = pd.DataFrame(rows)
    logging.info("Loaded %d reward observations", len(df))
    return df


def compute_bias_variance(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bias, variance, and MSE at each timestep.

    Groups by configuration + timestep and computes bias-variance decomposition
    for all predictions at that timestep.
    """
    logging.info("Computing bias-variance decomposition")

    def decompose(group):
        """Compute bias-variance statistics for a group."""
        # Verify all true rewards are identical
        true_rewards = group["true_reward"].values
        if not np.allclose(true_rewards, true_rewards[0], rtol=1e-9):
            logging.warning(
                "True rewards not identical at timestep %d: %s",
                group["global_step"].iloc[0],
                true_rewards,
            )

        true_reward = float(true_rewards[0])
        pred_rewards = group["pred_reward"].values

        # Compute decomposition
        decomp = metrics.bias_variance_decomposition(pred_rewards, true_reward)

        return pd.Series(
            {
                "true_reward": true_reward,
                "mean_pred_reward": float(np.mean(pred_rewards)),
                "bias": decomp["bias"],
                "variance": decomp["variance"],
                "bias_squared": decomp["bias_squared"],
                "mse": decomp["mse"],
                "num_runs": len(pred_rewards),
                "verification_error": decomp["verification_error"],
            }
        )

    result = (
        aligned_df.groupby(
            [
                "env_name",
                "rewest_method",
                "delay",
                "gamma",
                "estimation_episode",
                "global_step",
            ]
        )
        .apply(decompose)
        .reset_index()
    )

    # Add episode information (take from first run)
    episode_info = (
        aligned_df.groupby(
            [
                "env_name",
                "rewest_method",
                "delay",
                "gamma",
                "estimation_episode",
                "global_step",
            ]
        )["episode"]
        .first()
        .reset_index()
    )

    result = result.merge(
        episode_info,
        on=[
            "env_name",
            "rewest_method",
            "delay",
            "gamma",
            "estimation_episode",
            "global_step",
        ],
    )

    logging.info("Computed bias-variance for %d timesteps", len(result))
    return result


def aggregate_over_windows(bias_var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bias/variance statistics over different time windows.

    Creates aggregations for:
    - overall: All timesteps
    - pre_estimation: Before estimation episode
    - post_estimation: After estimation episode
    - episode ranges: Grouped by episode bins
    """
    logging.info("Aggregating over time windows")

    aggregations = []

    # Group by configuration (excluding timestep)
    group_cols = [
        "env_name",
        "rewest_method",
        "delay",
        "gamma",
        "estimation_episode",
    ]

    for group_key, group_df in bias_var_df.groupby(group_cols):
        config = dict(zip(group_cols, group_key))
        est_episode = config["estimation_episode"]

        # Overall statistics
        aggregations.append(
            {
                **config,
                "window": "overall",
                "window_start": group_df["global_step"].min(),
                "window_end": group_df["global_step"].max(),
                "num_timesteps": len(group_df),
                "mean_bias": group_df["bias"].mean(),
                "mean_variance": group_df["variance"].mean(),
                "mean_mse": group_df["mse"].mean(),
                "mean_bias_squared": group_df["bias_squared"].mean(),
                "mean_verification_error": group_df["verification_error"].mean(),
            }
        )

        # Pre-estimation statistics
        pre_est = group_df[group_df["episode"] < est_episode]
        if len(pre_est) > 0:
            aggregations.append(
                {
                    **config,
                    "window": "pre_estimation",
                    "window_start": pre_est["global_step"].min(),
                    "window_end": pre_est["global_step"].max(),
                    "num_timesteps": len(pre_est),
                    "mean_bias": pre_est["bias"].mean(),
                    "mean_variance": pre_est["variance"].mean(),
                    "mean_mse": pre_est["mse"].mean(),
                    "mean_bias_squared": pre_est["bias_squared"].mean(),
                    "mean_verification_error": pre_est["verification_error"].mean(),
                }
            )

        # Post-estimation statistics
        post_est = group_df[group_df["episode"] >= est_episode]
        if len(post_est) > 0:
            aggregations.append(
                {
                    **config,
                    "window": "post_estimation",
                    "window_start": post_est["global_step"].min(),
                    "window_end": post_est["global_step"].max(),
                    "num_timesteps": len(post_est),
                    "mean_bias": post_est["bias"].mean(),
                    "mean_variance": post_est["variance"].mean(),
                    "mean_mse": post_est["mse"].mean(),
                    "mean_bias_squared": post_est["bias_squared"].mean(),
                    "mean_verification_error": post_est["verification_error"].mean(),
                }
            )

    summary_df = pd.DataFrame(aggregations)
    logging.info("Created %d window aggregations", len(summary_df))
    return summary_df


def export_results(
    timestep_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str
):
    """Export results to parquet and CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Export timestep-level results
    timestep_parquet = os.path.join(output_dir, "bias_var_timestep.parquet")
    timestep_csv = os.path.join(output_dir, "bias_var_timestep.csv")
    timestep_df.to_parquet(timestep_parquet, index=False)
    timestep_df.to_csv(timestep_csv, index=False)
    logging.info("Exported timestep results to %s", timestep_parquet)

    # Export summary results
    summary_parquet = os.path.join(output_dir, "bias_var_summary.parquet")
    summary_csv = os.path.join(output_dir, "bias_var_summary.csv")
    summary_df.to_parquet(summary_parquet, index=False)
    summary_df.to_csv(summary_csv, index=False)
    logging.info("Exported summary results to %s", summary_parquet)


def wait_till_completion(tasks_refs, name: Optional[str] = None):
    """Wait for all Ray tasks to complete."""
    unfinished_tasks = tasks_refs
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        logging.info(
            "Completed %d %s task(s). %d left out of %d.",
            len(finished_tasks),
            name,
            len(unfinished_tasks),
            len(tasks_refs),
        )

        if len(unfinished_tasks) == 0:
            break


def main():
    """
    Main entry point for bias-variance experiment.

    Workflow:
    1. Parse arguments
    2. Initialize Ray
    3. Create all job specifications
    4. Submit jobs to Ray cluster
    5. Wait for completion
    6. Load and aggregate results
    7. Export final statistics
    """
    args = parse_args()

    # Initialize Ray
    with ray.init(address=os.environ.get("RAY_ADDRESS", "auto")):
        logging.info("Ray initialized")

        # Create output directories
        results_dir = os.path.join(args.output_path, "raw_results")
        os.makedirs(results_dir, exist_ok=True)

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

        # Create result writers
        num_writers = max(math.floor(len(job_specs) / NUM_TASKS_PER_WRITER), 1)
        result_writers = [
            ResultWriter.remote(prefix=idx, output_path=results_dir)  # type: ignore
            for idx in range(num_writers)
        ]
        logging.info("Created %d result writers", num_writers)

        # Submit all jobs
        logging.info("Submitting %d experiment runs...", len(job_specs))
        results_refs = [
            run_bias_var_experiment.remote(job, result_writers[idx % num_writers])
            for idx, job in enumerate(job_specs)
        ]

        # Wait for experiments to complete
        wait_till_completion(results_refs, name="Bias-Var-Experiment")

        # Flush writer buffers
        wait_till_completion(
            [writer.sync.remote() for writer in result_writers],  # type: ignore
            name="Flush-Buffer",
        )

        logging.info("All experiments complete. Aggregating results...")

        # Load and process results
        aligned_df = load_and_align_results(results_dir)
        bias_var_df = compute_bias_variance(aligned_df)
        summary_df = aggregate_over_windows(bias_var_df)

        # Export final results
        output_dir = os.path.join(args.output_path, "aggregated")
        export_results(bias_var_df, summary_df, output_dir)

        logging.info("Bias-variance analysis complete!")
        logging.info("Raw results: %s", results_dir)
        logging.info("Aggregated results: %s", output_dir)


def parse_args() -> Args:
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description="Run bias-variance decomposition experiments"
    )
    arg_parser.add_argument(
        "--num-runs", type=int, required=True, help="Number of runs per configuration"
    )
    arg_parser.add_argument(
        "--num-episodes", type=int, required=True, help="Number of episodes per run"
    )
    arg_parser.add_argument(
        "--output-path", type=str, required=True, help="Output directory path"
    )
    known_args, _ = arg_parser.parse_known_args()
    return Args(**vars(known_args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
