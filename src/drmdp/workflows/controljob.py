"""
Boostrap and run distributed jobs.
"""

import argparse
import dataclasses
import itertools
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import ray

from drmdp import core, task
from drmdp.workflows import controlexps


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
    use_seed: bool
    export_model: bool
    # ray args
    cluster_uri: Optional[str]


def wait_till_completion(tasks_refs):
    """
    Waits for every ray task to complete.
    """
    unfinished_tasks = tasks_refs
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        logging.info(
            "Completed %d task(s). %d left out of %d.",
            len(finished_tasks),
            len(unfinished_tasks),
            len(tasks_refs),
        )

        if len(unfinished_tasks) == 0:
            break


def create_tasks(
    num_runs: int,
    num_episodes: int,
    output_dir: str,
    task_prefix: str,
    log_episode_frequency: int,
    use_seed: bool,
    export_model: bool,
) -> Sequence[core.ExperimentInstance]:
    """
    Runs numerical experiments on policy evaluation.
    """
    experiments = parse_experiments(specs=controlexps.experiment_specs())
    experiment_instances = list(
        task.generate_experiments_instances(
            experiments=experiments,
            num_runs=num_runs,
            num_episodes_per_epoch=num_episodes,
            log_episode_frequency=log_episode_frequency,
            use_seed=use_seed,
            output_dir=output_dir,
            task_prefix=task_prefix,
            export_model=export_model,
        )
    )
    # shuffle tasks to balance workload
    np.random.shuffle(experiment_instances)  # type: ignore

    logging.info(
        "Parsed %d experiments into %d instances.",
        len(experiments),
        len(experiment_instances),
    )
    return experiment_instances


def parse_experiments(
    specs: Sequence[Mapping[str, Any]],
) -> Sequence[core.Experiment]:
    """
    Convert experiments from Dict into typed datastructures.
    """
    experiment_specs: List[core.Experiment] = []
    for spec in specs:
        for feat_tfx_spec, problem_spec in itertools.product(
            spec["feats_specs"], spec["problem_specs"]
        ):
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
def run_experiment(
    experiment_instance: core.ExperimentInstance,
) -> str:
    """
    Run experiments.
    """
    task_id = f"{experiment_instance.exp_id}/{experiment_instance.instance_id}"
    logging.info(
        "Experiment %s starting: %s",
        task_id,
        experiment_instance,
    )
    try:
        task.policy_control(experiment_instance)
    except Exception as err:
        logging.error("Error in experiment %s: %s", task_id, err)
        raise RuntimeError(f"Experiment {experiment_instance} failed") from err
    logging.info("Experiment %s finished", task_id)
    return task_id


def main(args: ControlPipelineArgs):
    """
    Program entry point.
    """

    ray_env: Dict[str, Any] = {}
    logging.info("Ray environment: %s", ray_env)
    experiment_instances = create_tasks(
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        task_prefix=args.task_prefix,
        log_episode_frequency=args.log_episode_frequency,
        use_seed=args.use_seed,
        export_model=args.export_model,
    )

    with ray.init(args.cluster_uri, runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

        logging.info("Submitting %d tasks", len(experiment_instances))
        results_refs = []
        for experiment_instance in experiment_instances:
            result_ref = run_experiment.remote(experiment_instance)
            results_refs.append(result_ref)

        wait_till_completion(results_refs)


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
    arg_parser.add_argument("--use-seed", action="store_true")
    arg_parser.add_argument(
        "--export-model", action=argparse.BooleanOptionalAction, default=False
    )
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return ControlPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
