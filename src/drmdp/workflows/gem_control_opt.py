import argparse
import dataclasses
import itertools
import logging
import os.path
import uuid
from typing import Any, Mapping

import numpy as np
import ray
import ray.data

from drmdp import feats, task
from drmdp.envs import gem

MAX_STEPS = 2500


@dataclasses.dataclass(frozen=True)
class Args:
    num_runs: int
    num_episodes: int
    output_path: str


@dataclasses.dataclass(frozen=True)
class JobSpec:
    env_name: str
    feats_spec: Mapping[str, Any]
    policy_type: str
    num_episodes: int
    turn: int
    output_path: str


def run_feats_spec_control_study(
    envs, feats_specs, turns: int, num_episodes: int, output_path: str
):
    baseline_spec = {"name": "identity", "args": None}
    # pair speccs with policies
    configs = list(
        itertools.product(envs, feats_specs, ["markovian"], range(turns))
    ) + list(itertools.product(envs, [baseline_spec], ["uniform-random"], range(turns)))

    jobs = []
    for env_name, feats_spec, policy_type, turn in configs:
        args = JobSpec(
            env_name, feats_spec, policy_type, num_episodes, turn, output_path
        )
        jobs.append(args)

    with ray.init() as context:
        logging.info("Starting ray task: %s", context)
        results_refs = [run_fn.remote(args) for args in jobs]
        wait_till_completion(results_refs)


def wait_till_completion(tasks_refs):
    """
    Waits for every ray task to complete.
    """
    unfinished_tasks = tasks_refs
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        for finished_task in finished_tasks:
            logging.info(
                "Completed task %s, %d left out of %d.",
                ray.get(finished_task),
                len(unfinished_tasks),
                len(tasks_refs),
            )

        if len(unfinished_tasks) == 0:
            break


@ray.remote
def run_fn(job_spec: JobSpec):
    task_id = str(uuid.uuid4())
    logging.info("Starting task %s: %s", task_id, job_spec)
    try:
        feats_spec_control(job_spec, task_id)
    except Exception as err:
        raise RuntimeError(f"Task {task_id} `{job_spec}` failed") from err
    logging.info("Completed task %s: %s", task_id, job_spec)
    return task_id


def feats_spec_control(job_spec: JobSpec, task_id: str):
    env = gem.make(job_spec.env_name, reward_fn="pos-enf", max_episode_steps=MAX_STEPS)
    env, monitor = task.monitor_wrapper(env)
    rew_delay = task.reward_delay_distribution(None)
    env = task.delay_wrapper(env, rew_delay)
    env = task.reward_mapper(
        env,
        mapping_spec={"name": "identity", "args": None},
        feats_spec={},
    )
    feats_tfx = feats.create_feat_transformer(env=env, **job_spec.feats_spec)
    lr = task.learning_rate(**{"name": "constant", "args": {"initial_lr": 0.01}})
    # Create spec using provided name and args for feature spec
    algorithm = task.create_algorithm(
        env=env,
        feats_transform=feats_tfx,
        delay_reward=rew_delay,
        lr=lr,
        gamma=1.0,
        epsilon=0.2,
        policy_type=job_spec.policy_type,
        base_seed=0,
    )

    results = task.policy_control(
        env=env,
        algorithm=algorithm,
        num_episodes=job_spec.num_episodes,
        monitor=monitor,
    )

    records = []
    for episode, snapshot in enumerate(results):
        if episode % max((job_spec.num_episodes // 5), 1) == 0:
            logging.info(
                "Episode: %d; Steps: %d, Mean returns: %f; Task: %s",
                episode,
                snapshot.steps,
                np.mean(monitor.returns + [monitor.rewards]).item(),
                job_spec,
            )
        records.append(
            {
                "env_name": job_spec.env_name,
                "feats_spec": job_spec.feats_spec,
                "policy": job_spec.policy_type,
                "turn": job_spec.turn,
                "episode": episode,
                "returns": np.mean(monitor.returns + [monitor.rewards]).item(),
                "steps": monitor.step,
            }
        )

    ds_result = ray.data.from_items(records)
    # write to a single file
    ds_result.repartition(1).write_json(
        os.path.join(job_spec.output_path, task_id),
    )


def main():
    args = parse_args()
    os.path.join(args.output_path)
    run_feats_spec_control_study(
        [
            "Finite-CC-PMSM-v0",
            "Finite-TC-PermExDc-v0",
            "Finite-CC-SeriesDc-v0",
            "Finite-TC-ShuntDc-v0",
            "Finite-SC-SynRM-v0",
            "Finite-CC-SCIM-v0",
        ],
        [
            {"name": "tiles", "args": {"tiling_dim": 4}},
            {"name": "tiles", "args": {"tiling_dim": 3}},
            {"name": "tiles", "args": {"tiling_dim": 2}},
            {"name": "scale", "args": None},
        ],
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
