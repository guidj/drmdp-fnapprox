import argparse
import dataclasses
import itertools
import json
import logging
import os.path
import tempfile
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import ray
import ray.data
import tensorflow as tf

from drmdp import core, envs, feats, logger, task
from drmdp.workflows import controlexps

MAX_STEPS = 2500
EST_SAMPLE_SIZES = (10_000, 25_000, 50_000, 100_000, 150_000)
REWARD_DELAYS = (2, 4, 6)


@dataclasses.dataclass(frozen=True)
class Args:
    num_runs: int
    num_episodes: int
    output_path: str


@dataclasses.dataclass(frozen=True)
class JobSpec:
    env_name: str
    env_args: Mapping[str, Any]
    feats_spec: Mapping[str, Any]
    rwest_sample_size: int
    num_episodes: int
    reward_delay: int
    use_bias: bool
    turn: int


def run_reward_estimation_study(
    specs, sample_sizes: Sequence[int], turns: int, num_episodes: int, output_path: str
):
    configs = itertools.product(specs, sample_sizes, range(turns))
    jobs = []
    for spec, sample_size, turn in configs:
        for feats_spec in spec["feats_specs"]:
            for reward_delay in REWARD_DELAYS:
                job_spec = JobSpec(
                    env_name=spec["name"],
                    env_args=spec["args"],
                    feats_spec=feats_spec,
                    rwest_sample_size=sample_size,
                    num_episodes=num_episodes,
                    reward_delay=reward_delay,
                    use_bias=True,
                    turn=turn,
                )
                jobs.extend([job_spec, dataclasses.replace(job_spec, use_bias=False)])
    np.random.shuffle(jobs)

    with ray.init() as context:
        logging.info("Starting ray task: %s", context)
        results_refs = [run_fn.remote(job) for job in jobs]
        results = wait_till_completion(results_refs)
        write_records(os.path.join(output_path, "result.jsonl"), results)


@ray.remote
def run_fn(job_spec: JobSpec):
    task_id = str(uuid.uuid4())
    logging.info("Starting task %s: %s", task_id, job_spec)
    try:
        result = reward_estimation(job_spec)
    except Exception as err:
        raise RuntimeError(f"Task {task_id} `{job_spec}` failed") from err
    logging.info("Completed task %s: %s", task_id, job_spec)
    return {"task_id": task_id, **dataclasses.asdict(job_spec), **result}


def wait_till_completion(tasks_refs):
    """
    Waits for every ray task to complete.
    """
    results = []
    unfinished_tasks = tasks_refs
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        for finished_task in finished_tasks:
            result = ray.get(finished_task)
            results.append(result)
            logging.info(
                "Completed task %s, %d left out of %d.",
                result["task_id"],
                len(unfinished_tasks),
                len(tasks_refs),
            )

        if len(unfinished_tasks) == 0:
            break
    return results


def reward_estimation(job_spec: JobSpec):
    exp_instance = create_exp_instance(job_spec)
    env, algorithm, monitor = setup_experiment(exp_instance)

    logging.debug("Starting DRMDP Control Experiments: %s", exp_instance)
    results = algorithm.train(
        env=env, num_episodes=exp_instance.run_config.episodes_per_run, monitor=monitor
    )

    with logger.ExperimentLogger(
        log_dir=exp_instance.run_config.output_dir, experiment_instance=exp_instance
    ) as exp_logger:
        returns = []
        try:
            for episode, snapshot in enumerate(results):
                returns.append(snapshot.returns)
                if episode % exp_instance.run_config.log_episode_frequency == 0:
                    exp_logger.log(
                        episode=episode,
                        steps=snapshot.steps,
                        returns=np.mean(returns).item(),
                        info={},
                    )

            logging.debug(
                "\nReturns for run %d of %s:\n%s",
                exp_instance.instance_id,
                exp_instance.exp_id,
                np.mean(returns),
            )
        except Exception as err:
            raise RuntimeError(
                f"Task {exp_instance.exp_id}, run {exp_instance.instance_id} failed"
            ) from err

    meta = env.get_wrapper_attr("estimation_meta")
    env.close()
    return meta


def create_exp_instance(job_spec: JobSpec):
    env_spec = core.EnvSpec(
        job_spec.env_name,
        args=job_spec.env_args,
        feats_spec=job_spec.feats_spec,
    )
    problem_spec = core.ProblemSpec(
        policy_type="markovian",
        reward_mapper={
            "name": "least-lfa",
            "args": {
                "estimation_sample_size": job_spec.rwest_sample_size,
                "use_bias": job_spec.use_bias,
                "feats_spec": {"name": "scale", "args": None},
            },
        },
        delay_config={"name": "fixed", "args": {"delay": job_spec.reward_delay}},
        epsilon=0.2,
        gamma=1.0,
        learning_rate_config={
            "name": "constant",
            "args": {"initial_lr": 0.01},
        },
    )
    run_config = core.RunConfig(
        num_runs=1,
        episodes_per_run=job_spec.num_episodes,
        log_episode_frequency=100,
        use_seed=True,
        output_dir=tempfile.gettempdir(),
    )
    return core.ExperimentInstance(
        exp_id=str(uuid.uuid4()),
        instance_id=job_spec.turn,
        experiment=core.Experiment(
            env_spec=env_spec,
            problem_spec=problem_spec,
        ),
        run_config=run_config,
        context={},
    )


def setup_experiment(exp_instance: core.ExperimentInstance):
    env_spec = exp_instance.experiment.env_spec
    problem_spec = exp_instance.experiment.problem_spec
    env = envs.make(
        env_name=env_spec.name,
        **env_spec.args if env_spec.args else {},
    )
    env, monitor = task.monitor_wrapper(env)
    rew_delay = task.reward_delay_distribution(problem_spec.delay_config)
    env = task.delay_wrapper(env, rew_delay)
    env = task.reward_mapper(
        env,
        mapping_spec=problem_spec.reward_mapper,
    )

    # measure samples required to estimate rewards
    # random policy vs control
    feats_tfx = feats.create_feat_transformer(env=env, **env_spec.feats_spec)
    lr = task.learning_rate(**problem_spec.learning_rate_config)
    # Create spec using provided name and args for feature spec
    algorithm = task.create_algorithm(
        env=env,
        feats_transform=feats_tfx,
        delay_reward=rew_delay,
        lr=lr,
        gamma=problem_spec.gamma,
        epsilon=problem_spec.epsilon,
        policy_type=problem_spec.policy_type,
        base_seed=exp_instance.instance_id,
    )
    return env, algorithm, monitor


def write_records(output_path: str, records):
    with tf.io.gfile.GFile(output_path, "w") as writable:
        for record in records:
            json.dump(record, writable)
            writable.write("\n")


def main():
    args = parse_args()
    os.path.join(args.output_path)
    specs = [
        {"name": spec["name"], "args": spec["args"], "feats_specs": spec["feats_specs"]}
        for spec in controlexps.SPECS
    ]

    run_reward_estimation_study(
        specs,
        sample_sizes=EST_SAMPLE_SIZES,
        turns=args.num_runs,
        num_episodes=args.num_episodes,
        output_path=args.output_path,
    )
    logging.info("Output dir: %s", args.output_path)
    logging.info("Done")


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-runs", type=int, required=True)
    arg_parser.add_argument("--num-episodes", type=int, required=True)
    arg_parser.add_argument("--output-path", required=True)
    known_args, _ = arg_parser.parse_known_args()
    return Args(**vars(known_args))


if __name__ == "__main__":
    main()
