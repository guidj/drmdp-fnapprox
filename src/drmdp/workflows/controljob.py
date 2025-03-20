"""
Boostrap and run distributed jobs.
"""

import argparse
import dataclasses
import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ray

from drmdp import constants, core, task
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
    bundle_size: int
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
) -> Sequence[Tuple[ray.ObjectRef, core.ExperimentInstance]]:
    """
    Runs numerical experiments on policy evaluation.
    """
    experiments = tuple(parse_experiments(specs=controlexps.SPECS))
    experiment_instances = tuple(
        task.generate_experiments_instances(
            run_config=core.RunConfig(
                num_runs=num_runs,
                episodes_per_run=num_episodes,
                log_episode_frequency=log_episode_frequency,
                output_dir=output_dir,
            ),
            experiments=experiments,
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
        logging.debug(
            "Experiment %s starting: %s",
            task_id,
            experiment_task,
        )
        try:
            task.policy_control_run_fn(experiment_task)
        except Exception as err:
            logging.error("Experiment %s failed", experiment_task)
            raise err
        ids.append(task_id)
        logging.debug("Experiment %s finished", task_id)
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
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return ControlPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
