import argparse
import dataclasses
import itertools
import json
import logging
import os.path
import tempfile
import uuid
from typing import Any, List, Mapping, Sequence

import gymnasium as gym
import numpy as np
import ray
import tensorflow as tf

from drmdp import core, envs, feats, logger, rewdelay, task

MAX_STEPS = 200
REWARD_DELAYS = (2, 4, 6, 8)
REWARD_EVAL_SAMPLES = 25_000


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Task arguments.
    """

    num_runs: int
    num_episodes: int
    output_path: str


@dataclasses.dataclass(frozen=True)
class JobSpec:
    """
    Spec for a job config.
    """

    env_name: str
    env_args: Mapping[str, Any]
    control_feats_spec: Mapping[str, Any]
    rewest_method: str
    rewest_args: Mapping[str, Any]
    reward_delay: int
    num_episodes: int
    use_bias: bool
    epochs: int
    turn: int


@ray.remote
class ResultWriter:
    """
    Remote task to export results.
    """

    def __init__(self, output_path: str, partition_size: int = 100):
        self.output_path = output_path
        self.partition_size = partition_size
        self.results: List[Any] = []
        self.partition = 0

    def write(self, result):
        """
        Adds data to the write buffer.
        Can sync output file if `partition_size`
        is reached.
        """
        self.results.append(result)
        if len(self.results) >= self.partition_size:
            self.sync()

    def sync(self):
        """
        Writes data to a file and cleans
        the buffer.
        """
        if self.results:
            write_records(
                os.path.join(self.output_path, f"result-{self.partition}.jsonl"),
                records=self.results,
            )
            self.results = []
            self.partition += 1


class RewardStoreWrapper(gym.Wrapper):
    """
    Keeps track of rewards.
    """

    def __init__(self, env, buffer_size: int):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.buffer = []
        self.solver_state = {"solution_found_step": None}

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(reward)
            if (
                self.solver_state["solution_found_step"] is None
                and "estimator" in info
                and info["estimator"]["state"] == rewdelay.OptState.SOLVED
            ):
                self.solver_state["solution_found_step"] = len(self.buffer) - 1
        return obs, reward, term, trunc, info


def discrete_least_specs(
    attempt_estimation_episodes: Sequence[int], feat_spec: Mapping[str, Any]
):
    return [
        {
            "name": "discrete-least-lfa",
            "args": {
                "attempt_estimation_episode": attempt_estimation_episode,
                "feats_spec": feat_spec,
            },
        }
        for attempt_estimation_episode in attempt_estimation_episodes
    ]


def least_specs(
    attempt_estimation_episodes: Sequence[int], feat_spec: Mapping[str, Any]
):
    return [
        {
            "name": "least-lfa",
            "args": {
                "attempt_estimation_episode": attempt_estimation_episode,
                "feats_spec": feat_spec,
            },
        }
        for attempt_estimation_episode in attempt_estimation_episodes
    ]


def bayes_least_specs(
    init_attempt_estimation_episodes: Sequence[int], feat_spec: Mapping[str, Any]
):
    return [
        {
            "name": "bayes-least-lfa",
            "args": {
                "init_attempt_estimation_episode": init_attempt_estimation_episode,
                "mode": "double",
                "feats_spec": feat_spec,
            },
        }
        for init_attempt_estimation_episode in init_attempt_estimation_episodes
    ]


def cvlps_specs(
    attempt_estimation_episodes: Sequence[int], feat_spec: Mapping[str, Any]
):
    return [
        {
            "name": "cvlps",
            "args": {
                "attempt_estimation_episode": attempt_estimation_episode,
                "feats_spec": feat_spec,
            },
        }
        for attempt_estimation_episode in attempt_estimation_episodes
    ]


def bayes_cvlps_specs(
    init_attempt_estimation_episodes: Sequence[int], feat_spec: Mapping[str, Any]
):
    return [
        {
            "name": "bayes-cvlps",
            "args": {
                "init_attempt_estimation_episode": init_attempt_estimation_episode,
                "feats_spec": feat_spec,
            },
        }
        for init_attempt_estimation_episode in init_attempt_estimation_episodes
    ]


def experiment_specs() -> Sequence[Mapping[str, Any]]:
    """
    Returns experiments configurations.
    """
    return (
        {
            "name": "Finite-CC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 4}}],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-CC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 4}}],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 3}}],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "scale", "args": None},
            ),
            "epochs": 1,
        },
        {
            "name": "RedGreen-v0",
            "args": None,
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 6}},
            ],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            ),
            "epochs": 100,
        },
        {
            "name": "IceWorld-v0",
            "args": None,
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 6}},
            ],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            ),
            "epochs": 100,
        },
        {
            "name": "MountainCar-v0",
            "args": {
                "max_episode_steps": MAX_STEPS,
            },
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 6}},
            ],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            ),
            "epochs": 10,
        },
        {
            "name": "GridWorld-v0",
            "args": {"max_episode_steps": 2500},
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 8}},
            ],
            "rewest": discrete_least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "cluster-c", "args": {"num_clusters": 100}},
            )
            + least_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 8}},
            )
            + cvlps_specs(
                attempt_estimation_episodes=(10, 50, 100),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 8}},
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 8}},
            )
            + bayes_cvlps_specs(
                init_attempt_estimation_episodes=(10, 20, 30),
                feat_spec={"name": "tiles", "args": {"tiling_dim": 8}},
            ),
            "epochs": 10,
        },
    )


def run_reward_estimation_study(specs, turns: int, num_episodes: int, output_path: str):
    """
    Runs a reward estimation study.
    """
    configs = itertools.product(specs, REWARD_DELAYS, range(turns))
    jobs = []
    for spec, reward_delay, turn in configs:
        for feats_spec, rewest_spec in itertools.product(
            spec["feats_specs"], spec["rewest"]
        ):
            job_spec = JobSpec(
                env_name=spec["name"],
                env_args=spec["args"],
                control_feats_spec=feats_spec,
                num_episodes=num_episodes,
                reward_delay=reward_delay,
                rewest_method=rewest_spec["name"],
                rewest_args=rewest_spec["args"],
                use_bias=False,
                epochs=spec["epochs"],
                turn=turn,
            )
            jobs.append(job_spec)
    np.random.shuffle(jobs)  # type: ignore

    with ray.init() as context:
        logging.info("Starting ray task: %s", context)
        result_writer = ResultWriter.remote(output_path)  # pylint: disable=no-member
        results_refs = [run_fn.remote(job) for job in jobs]
        wait_till_completion(results_refs, result_writer)


@ray.remote
def run_fn(job_spec: JobSpec):
    """
    Remote task to execute reward estimation
    experiment.
    """
    task_id = str(uuid.uuid4())
    logging.info("Starting task %s, %s", task_id, job_spec)
    try:
        result = reward_estimation(job_spec)
    except Exception as err:
        raise RuntimeError(f"Task {task_id} `{job_spec}` failed") from err
    logging.info("Completed task %s: %s", task_id, job_spec)
    result = {"task_id": task_id, **dataclasses.asdict(job_spec), "meta": result}
    return result


def wait_till_completion(tasks_refs, result_writer: ResultWriter):
    """
    Waits for every ray task to complete.
    """
    unfinished_tasks = tasks_refs
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        for finished_task in finished_tasks:
            ray.get(result_writer.write.remote(finished_task))  # type: ignore
            logging.info(
                "Completed task. %d left out of %d.",
                len(unfinished_tasks),
                len(tasks_refs),
            )

        if len(unfinished_tasks) == 0:
            break

    # Flush remaining files
    ray.get(result_writer.sync.remote())  # type: ignore


def reward_estimation(job_spec: JobSpec):
    """
    Runs a reward estimation experiment.
    """
    exp_instance = create_exp_instance(job_spec)
    env, algorithm, monitor, opt_logs = setup_experiment(exp_instance)

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

    meta: Mapping[str, Any] = env.get_wrapper_attr("estimation_meta")
    env.close()

    return {
        "returns": np.mean(returns).item(),
        **opt_logs,
        **meta,
    }


def create_exp_instance(job_spec: JobSpec):
    """
    Creates an experiment instance given a spec.
    """
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
        episodes_per_run=job_spec.num_episodes * job_spec.epochs,
        log_episode_frequency=100,
        use_seed=True,
        output_dir=tempfile.gettempdir(),
    )
    return core.ExperimentInstance(
        exp_id=str(uuid.uuid4()),
        instance_id=job_spec.turn,
        experiment=core.Experiment(
            env_spec=env_spec, problem_spec=problem_spec, epochs=job_spec.epochs
        ),
        run_config=run_config,
        context={},
    )


def setup_experiment(exp_instance: core.ExperimentInstance):
    """
    Sets up an experiment run given an instance.
    """
    opt_logs = {}
    env_spec = exp_instance.experiment.env_spec
    problem_spec = exp_instance.experiment.problem_spec
    env = envs.make(
        env_name=env_spec.name,
        **env_spec.args if env_spec.args else {},
    )
    env, monitor = task.monitor_wrapper(env)
    # Save true reward, prior to any change.
    env = RewardStoreWrapper(env, buffer_size=REWARD_EVAL_SAMPLES)
    opt_logs["true_reward_buffer"] = env.buffer
    rew_delay = task.reward_delay_distribution(problem_spec.delay_config)
    env = task.delay_wrapper(env, rew_delay)
    env = task.reward_mapper(
        env,
        mapping_spec=problem_spec.reward_mapper,
    )
    # Save rewards post transformation
    env = RewardStoreWrapper(env, buffer_size=REWARD_EVAL_SAMPLES)
    opt_logs["pred_reward_buffer"] = env.buffer
    opt_logs["solver_state"] = env.solver_state

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
    return env, algorithm, monitor, opt_logs


def write_records(output_path: str, records):
    """
    Exports records into JSON files.
    """
    with tf.io.gfile.GFile(output_path, "w") as writable:
        for record in records:
            json.dump(record, writable)
            writable.write("\n")


def main():
    """
    Entry point for running experiments.
    """
    args = parse_args()
    os.path.join(args.output_path)

    run_reward_estimation_study(
        experiment_specs(),
        turns=args.num_runs,
        num_episodes=args.num_episodes,
        output_path=args.output_path,
    )
    logging.info("Output dir: %s", args.output_path)
    logging.info("Done")


def parse_args() -> Args:
    """
    Parse task arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-runs", type=int, required=True)
    arg_parser.add_argument("--num-episodes", type=int, required=True)
    arg_parser.add_argument("--output-path", required=True)
    known_args, _ = arg_parser.parse_known_args()
    return Args(**vars(known_args))


if __name__ == "__main__":
    main()
