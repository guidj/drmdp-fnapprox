import argparse
import dataclasses
import itertools
import logging
import random
from collections.abc import Mapping, Sequence
from typing import Any

import ray

from drmdp import constants, core, task
from drmdp.workflows import raytasks, specutils

logger = logging.getLogger(__name__)

MAX_STEPS = 200


def bayes_least_specs(feats_spec: Mapping[str, Any]):
    def specs(mode: str, init_attempt_estimation_episodes: int):
        return tuple(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_attempt_estimation_episode": init_attempt_estimation_episodes,
                        "mode": mode,
                        "transition_feats_spec": {
                            "name": "current-features",
                            "args": {"current_spec": feats_spec},
                        },
                        "intercept": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": delay}},
                "epsilon": 0.2,
                "gamma": gamma,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            }
            for gamma, delay in itertools.product((1.0, 0.99, 0.995), (2, 4, 6, 8))
        )

    return (
        specs(mode="exponential", init_attempt_estimation_episodes=10)
        + specs(mode="exponential", init_attempt_estimation_episodes=50)
        + specs(mode="exponential", init_attempt_estimation_episodes=100)
    )


COMMON_PROBLEM_SPECS = (
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": None,
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": None,
        "epsilon": 0.2,
        "gamma": 0.99,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": None,
        "epsilon": 0.2,
        "gamma": 0.995,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
)

SPECS: Sequence[Mapping[str, Any]] = (
    {
        "name": "Finite-CC-PermExDc-v0",
        "args": {
            "reward_fn": "pos-enf",
            "penalty_gamma": 1.0,
            "constraint_violation_reward": 0.0,
            "max_episode_steps": 200,
        },
        "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 4}}],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
        "epochs": 1,
    },
    {
        "name": "Finite-CC-ShuntDc-v0",
        "args": {
            "reward_fn": "pos-enf",
            "penalty_gamma": 1.0,
            "constraint_violation_reward": 0.0,
            "max_episode_steps": 200,
        },
        "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
        "epochs": 1,
    },
    {
        "name": "Finite-SC-PermExDc-v0",
        "args": {
            "reward_fn": "pos-enf",
            "penalty_gamma": 1.0,
            "constraint_violation_reward": 0.0,
            "max_episode_steps": 200,
        },
        "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 3}}],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
        "epochs": 1,
    },
    {
        "name": "Finite-SC-ShuntDc-v0",
        "args": {
            "reward_fn": "pos-enf",
            "penalty_gamma": 1.0,
            "constraint_violation_reward": 0.0,
            "max_episode_steps": 200,
        },
        "feats_specs": [{"name": "scale", "args": None}],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
        "epochs": 1,
    },
    {
        "name": "Finite-TC-PermExDc-v0",
        "args": {
            "reward_fn": "pos-enf",
            "penalty_gamma": 1.0,
            "constraint_violation_reward": 0.0,
            "max_episode_steps": 200,
        },
        "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
        "epochs": 1,
    },
    {
        "name": "RedGreen-v0",
        "args": None,
        "feats_specs": [
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 6}}),
        "epochs": 100,
    },
    {
        "name": "IceWorld-v0",
        "args": None,
        "feats_specs": [
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 6}}),
        "epochs": 100,
    },
    {
        "name": "MountainCar-v0",
        "args": {
            "max_episode_steps": 2500,
        },
        "feats_specs": [
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 6}}),
        "epochs": 10,
    },
    {
        "name": "GridWorld-v0",
        "args": {"max_episode_steps": 200},
        "feats_specs": [
            {"name": "tiles", "args": {"tiling_dim": 8}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 8}}),
        "epochs": 10,
    },
)


@dataclasses.dataclass(frozen=True)
class ControlPipelineArgs:
    """
    Program arguments.
    """

    # problem args
    num_runs: int
    num_episodes: int
    output_dir: str
    log_episode_frequency: int
    task_prefix: str
    bundle_size: int
    use_seed: bool
    # ray args
    cluster_uri: str | None


def main(args: ControlPipelineArgs):
    """
    Program entry point.
    """

    ray_env: dict[str, Any] = {}
    logger.info("Ray environment: %s", ray_env)
    with ray.init(args.cluster_uri, runtime_env=ray_env) as context:
        logger.info("Ray Context: %s", context)
        logger.info("Ray Nodes: %s", ray.nodes())

        tasks_results_refs = create_tasks(
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            output_dir=args.output_dir,
            task_prefix=args.task_prefix,
            bundle_size=args.bundle_size,
            log_episode_frequency=args.log_episode_frequency,
            use_seed=args.use_seed,
        )

        # since ray tracks objectref items
        # we swap the key:value
        results_refs = [result_ref for _, result_ref in tasks_results_refs]
        raytasks.wait_till_completion(results_refs, name="experiment")


def create_tasks(
    num_runs: int,
    num_episodes: int,
    output_dir: str,
    task_prefix: str,
    bundle_size: int,
    log_episode_frequency: int,
    use_seed: bool,
) -> Sequence[tuple[ray.ObjectRef, core.ExperimentInstance]]:
    """
    Runs numerical experiments on policy evaluation.
    """
    experiments = tuple(specutils.parse_experiment_specs(specs=SPECS))
    experiment_instances = tuple(
        task.generate_experiments_instances(
            experiments=experiments,
            num_runs=num_runs,
            num_episodes_per_epoch=num_episodes,
            log_episode_frequency=log_episode_frequency,
            use_seed=use_seed,
            output_dir=output_dir,
            task_prefix=task_prefix,
            export_model=False,
        )
    )
    # shuffle tasks to balance workload
    experiment_instances = random.sample(
        experiment_instances,
        len(experiment_instances),  # type: ignore
    )
    experiment_batches = task.bundle(experiment_instances, bundle_size=bundle_size)
    logger.info(
        "Parsed %d experiments into %d instances and %d ray tasks",
        len(experiments),
        len(experiment_instances),
        len(experiment_batches),
    )
    results_refs = []
    for batch in experiment_batches:
        result_ref = run_experiments.remote(batch)
        results_refs.append((batch, result_ref))
    return results_refs


@ray.remote
def run_experiments(
    experiments_batch: Sequence[core.ExperimentInstance],
) -> Sequence[str]:
    """
    Run experiments.
    """
    ids: list[str] = []
    for experiment_task in experiments_batch:
        task_id = f"{experiment_task.exp_id}/{experiment_task.instance_id}"
        logger.info(
            "Experiment %s starting: %s",
            task_id,
            experiment_task,
        )
        try:
            task.policy_control(experiment_task)
        except Exception:
            logger.error("Experiment %s failed", experiment_task)
            raise
        ids.append(task_id)
        logger.info("Experiment %s finished", task_id)
    return ids


def parse_args() -> ControlPipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-runs", type=int, required=True)
    arg_parser.add_argument("--num-episodes", type=int, required=True)
    arg_parser.add_argument("--output-dir", type=str, required=True)
    arg_parser.add_argument("--log-episode-frequency", type=int, required=True)
    arg_parser.add_argument("--task-prefix", type=str, required=True)
    arg_parser.add_argument(
        "--bundle-size", type=int, default=constants.DEFAULT_BATCH_SIZE
    )
    arg_parser.add_argument("--use-seed", action="store_true")
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    known_args, unknown_args = arg_parser.parse_known_args()
    logger.info("Unknown args: %s", unknown_args)
    return ControlPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
