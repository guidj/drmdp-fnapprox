"""
Bootstrap and run estimation-only distributed jobs.

The estimation-only campaign measures reward-estimation encodings without
control: every experiment runs `task.estimation_only`, the random-policy
loop that exercises the production estimator stack and records
true-reward errors per estimation event.
"""

import argparse
import dataclasses
import logging
from collections.abc import Sequence

import numpy as np
import ray

from drmdp import core, task
from drmdp.workflows import estimationexps, raytasks, specutils

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class EstimationPipelineArgs:
    """
    Program arguments.
    """

    # problem args
    num_runs: int
    output_dir: str
    log_episode_frequency: int
    task_prefix: str
    use_seed: bool
    # ray args
    cluster_uri: str | None


def create_tasks(
    num_runs: int,
    output_dir: str,
    task_prefix: str,
    log_episode_frequency: int,
    use_seed: bool,
) -> Sequence[core.ExperimentInstance]:
    """
    Experiment instances of the estimation-only campaign.

    Instances are generated per campaign cell — (env, encoding, method,
    gate, delay, noise) — with the cell in the task prefix, so each cell
    receives a distinct `exp_id`. The aggregation pipeline identifies
    runs by (exp_id, instance_id); without the per-cell prefix, the
    cells of one environment would collapse onto each other in the
    estimation-event dedup.
    """
    specs = estimationexps.estimation_experiment_specs()
    experiments = specutils.parse_experiment_specs(specs=specs)

    cells: dict[str, list[core.Experiment]] = {}
    for experiment in experiments:
        cells.setdefault(cell_key(experiment), []).append(experiment)

    # per-env episode caps: instances are generated per cell so the
    # cap rides the run config
    instances: list[core.ExperimentInstance] = []
    for key, cell_experiments in cells.items():
        env_name = cell_experiments[0].env_spec.name
        instances.extend(
            task.generate_experiments_instances(
                experiments=cell_experiments,
                num_runs=num_runs,
                num_episodes_per_epoch=estimationexps.episode_cap(env_name),
                log_episode_frequency=log_episode_frequency,
                use_seed=use_seed,
                output_dir=output_dir,
                task_prefix=f"{task_prefix}-{key}",
                export_model=False,
            )
        )
    # shuffle tasks to balance workload
    np.random.shuffle(instances)  # type: ignore

    logger.info(
        "Parsed %d experiments into %d instances.",
        len(experiments),
        len(instances),
    )
    return instances


def cell_key(experiment: core.Experiment) -> str:
    """Identity of a campaign cell within one environment."""
    env_name = experiment.env_spec.name
    encoding = (experiment.env_spec.metadata or {}).get("encoding", "none")
    mapper = experiment.problem_spec.reward_mapper
    mapper_args = mapper["args"] or {}
    check_mode = mapper_args.get("estimation_check", "none")
    delay_args = (experiment.problem_spec.delay_config or {})["args"]
    delay_label = f"d{delay_args['min_delay']}x{delay_args['max_delay']}"
    env_args = experiment.env_spec.args or {}
    noise_args = (env_args.get("reward_noise") or {}).get("args") or {}
    noise_label = f"n{noise_args.get('std_multiplier', 'none')}"
    return f"{env_name}-{encoding}-{mapper['name']}-{check_mode}-{delay_label}-{noise_label}"


@ray.remote
def run_estimation_instance(experiment_instance: core.ExperimentInstance) -> str:
    """
    Run one estimation-only experiment instance.
    """
    task_id = f"{experiment_instance.exp_id}/{experiment_instance.instance_id}"
    logger.info("Estimation experiment %s starting: %s", task_id, experiment_instance)
    try:
        task.estimation_only(experiment_instance)
    except Exception as err:
        logger.error("Error in estimation experiment %s: %s", task_id, err)
        raise RuntimeError(
            f"Estimation experiment {experiment_instance} failed"
        ) from err
    logger.info("Estimation experiment %s finished", task_id)
    return task_id


def main(args: EstimationPipelineArgs) -> None:
    """
    Program entry point.
    """
    experiment_instances = create_tasks(
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        task_prefix=args.task_prefix,
        log_episode_frequency=args.log_episode_frequency,
        use_seed=args.use_seed,
    )

    with ray.init(args.cluster_uri) as context:
        logger.info("Ray Context: %s", context)
        logger.info("Ray Nodes: %s", ray.nodes())

        logger.info("Submitting %d tasks", len(experiment_instances))
        results_refs = []
        for experiment_instance in experiment_instances:
            result_ref = run_estimation_instance.remote(experiment_instance)
            results_refs.append(result_ref)

        raytasks.wait_till_completion(results_refs, name="estimation-experiment")


def parse_args() -> EstimationPipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-runs", type=int, required=True)
    arg_parser.add_argument("--output-dir", type=str, required=True)
    arg_parser.add_argument("--log-episode-frequency", type=int, required=True)
    arg_parser.add_argument("--task-prefix", type=str, required=True)
    arg_parser.add_argument("--use-seed", action="store_true")
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    known_args, unknown_args = arg_parser.parse_known_args()
    logger.info("Unknown args: %s", unknown_args)
    return EstimationPipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
