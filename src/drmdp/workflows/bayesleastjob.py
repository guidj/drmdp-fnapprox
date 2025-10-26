import argparse
import dataclasses
import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ray

from drmdp import constants, core, task

MAX_STEPS = 200


def bayes_least_specs(feats_spec: Mapping[str, Any]):
    def specs(mode: str, init_update_episode: int):
        return (
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 2}},
                "epsilon": 0.2,
                "gamma": 1.0,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 4}},
                "epsilon": 0.2,
                "gamma": 1.0,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 6}},
                "epsilon": 0.2,
                "gamma": 1.0,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 8}},
                "epsilon": 0.2,
                "gamma": 1.0,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 2}},
                "epsilon": 0.2,
                "gamma": 0.99,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 4}},
                "epsilon": 0.2,
                "gamma": 0.99,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 6}},
                "epsilon": 0.2,
                "gamma": 0.99,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_estimation_sample_size": init_update_episode,
                        "mode": mode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": 8}},
                "epsilon": 0.2,
                "gamma": 0.99,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
        )

    return (
        specs(mode="double", init_update_episode=10)
        + specs(mode="double", init_update_episode=50)
        + specs(mode="double", init_update_episode=100)
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
    cluster_uri: Optional[str]


def main(args: ControlPipelineArgs):
    """
    Program entry point.
    """

    ray_env: Dict[str, Any] = {}
    logging.info("Ray environment: %s", ray_env)
    with ray.init(args.cluster_uri, runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

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
        unfinished_tasks = results_refs
        while True:
            finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
            for finished_task in finished_tasks:
                logging.info(
                    "Completed task %s, %d left out of %d.",
                    ray.get(finished_task),
                    len(unfinished_tasks),
                    len(results_refs),
                )

            if len(unfinished_tasks) == 0:
                break


def create_tasks(
    num_runs: int,
    num_episodes: int,
    output_dir: str,
    task_prefix: str,
    bundle_size: int,
    log_episode_frequency: int,
    use_seed: bool,
) -> Sequence[Tuple[ray.ObjectRef, core.ExperimentInstance]]:
    """
    Runs numerical experiments on policy evaluation.
    """
    experiments = tuple(parse_experiments(specs=SPECS))
    experiment_instances = tuple(
        task.generate_experiments_instances(
            experiments=experiments,
            num_runs=num_runs,
            num_episodes_per_epoch=num_episodes,
            log_episode_frequency=log_episode_frequency,
            use_seed=use_seed,
            output_dir=output_dir,
            task_prefix=task_prefix,
        )
    )
    # shuffle tasks to balance workload
    experiment_instances = random.sample(
        experiment_instances,
        len(experiment_instances),  # type: ignore
    )
    experiment_batches = task.bundle(experiment_instances, bundle_size=bundle_size)
    logging.info(
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


def parse_experiments(
    specs: Sequence[Mapping[str, Any]],
) -> Sequence[core.Experiment]:
    """
    Convert experiments from Dict into typed datastructures.
    """
    experiment_specs: List[core.Experiment] = []
    for spec in specs:
        for feat_tfx_spec in spec["feats_specs"]:
            for problem_spec in spec["problem_specs"]:
                experiment_specs.append(
                    core.Experiment(
                        env_spec=core.EnvSpec(
                            name=spec["name"],
                            args=spec["args"],
                            feats_spec=feat_tfx_spec,
                        ),
                        problem_spec=core.ProblemSpec(**problem_spec),
                        epochs=spec["epochs"],
                    )
                )
    return experiment_specs


@ray.remote
def run_experiments(
    experiments_batch: Sequence[core.ExperimentInstance],
) -> Sequence[str]:
    """
    Run experiments.
    """
    ids: List[str] = []
    for experiment_task in experiments_batch:
        task_id = f"{experiment_task.exp_id}/{experiment_task.instance_id}"
        logging.info(
            "Experiment %s starting: %s",
            task_id,
            experiment_task,
        )
        try:
            task.policy_control(experiment_task)
        except Exception as err:
            logging.error("Experiment %s failed", experiment_task)
            raise err
        ids.append(task_id)
        logging.info("Experiment %s finished", task_id)
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
    logging.info("Unknown args: %s", unknown_args)
    return ControlPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
