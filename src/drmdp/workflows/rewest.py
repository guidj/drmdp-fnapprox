import argparse
import copy
import dataclasses
import gzip
import itertools
import json
import logging
import math
import os.path
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import ray
import tensorflow as tf

from drmdp import core, envs, feats, logger, mathutils, metrics, rewdelay, task

MAX_STEPS = 10_000
REWARD_DELAYS = (2, 4, 6)
REWARD_EVAL_SAMPLES = 25_000
MINES_GW_GRID = [
    "ooooxooooooo",
    "oooooooooxoo",
    "oxoooooxoooo",
    "oooooxoooooo",
    "ooxooooooxoo",
    "sxxxxxxxxxxg",
]
NUM_TASKS_PER_WRITER = 100


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

    def __init__(self, prefix: str, output_path: str, partition_size: int = 100):
        self.prefix = prefix
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
                os.path.join(
                    self.output_path, f"result-{self.prefix}-{self.partition}.jsonl"
                ),
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
        self.buffer: List[float] = []
        self.solver_state = {"solution_found_step": None}
        self.steps_counter = 0

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        # Add to buffer
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(reward)
        # Check if solution exists and log step
        if (
            self.solver_state["solution_found_step"] is None
            and "estimator" in info
            and info["estimator"]["state"] == rewdelay.OptState.SOLVED
        ):
            self.solver_state["solution_found_step"] = self.steps_counter
        # Increment step counter
        self.steps_counter += 1
        return obs, reward, term, trunc, info


def discrete_least_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Mapping[str, Any]],
    estimation_buffer_multiples: Sequence[Optional[int]] = (25,),
):
    """
    Discretised Least Squares specs.
    """
    specs = []
    for aee, feat_spec, est_buffer_mult in itertools.product(
        attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "discrete-least-lfa",
                "args": {
                    "attempt_estimation_episode": aee,
                    "feats_spec": feat_spec,
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def least_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Mapping[str, Any]],
    estimation_buffer_multiples: Sequence[Optional[int]] = (25,),
):
    """
    Least Squares specs.
    """
    specs = []
    for aee, feat_spec, est_buffer_mult in itertools.product(
        attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "least-lfa",
                "args": {
                    "attempt_estimation_episode": aee,
                    "feats_spec": feat_spec,
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def bayes_least_specs(
    init_attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Mapping[str, Any]],
    estimation_buffer_multiples: Sequence[Optional[int]] = (25,),
):
    """
    Bayesian linear regression specs.
    """
    specs = []
    for iaee, feat_spec, est_buffer_mult in itertools.product(
        init_attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "bayes-least-lfa",
                "args": {
                    "init_attempt_estimation_episode": iaee,
                    "mode": "double",
                    "feats_spec": feat_spec,
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def cvlps_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Mapping[str, Any]],
    estimation_buffer_multiples: Sequence[Optional[int]] = (25,),
    constraints_buffer_limit: Optional[int] = 100,
):
    """
    Constrained optimisation specs.
    """
    specs = []
    for aee, feat_spec, est_buffer_mult in itertools.product(
        attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "cvlps",
                "args": {
                    "attempt_estimation_episode": aee,
                    "feats_spec": feat_spec,
                    "constraints_buffer_limit": constraints_buffer_limit,
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def recurring_cvlps(
    init_attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Mapping[str, Any]],
    estimation_buffer_multiples: Sequence[Optional[int]] = (25,),
    constraints_buffer_limit: Optional[int] = 100,
):
    """
    Recurring convex linear estimation specs.
    """
    specs = []
    for iaee, feat_spec, est_buffer_mult in itertools.product(
        init_attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "recurring-cvlps",
                "args": {
                    "init_attempt_estimation_episode": iaee,
                    "feats_spec": feat_spec,
                    "constraints_buffer_limit": constraints_buffer_limit,
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


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
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS,
                "emit_state": False,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 4}}],
            "rewest": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-CC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS,
                "emit_state": False,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
            "rewest": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS,
                "emit_state": False,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 3}}],
            "rewest": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS,
                "emit_state": True,
            },
            "feats_specs": [
                {"name": "scale", "args": None},
            ],
            "rewest": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS,
                "emit_state": False,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
            "rewest": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS,
                "emit_state": True,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
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
        logging.info("Parsed %d jobs in total", len(jobs))
        num_writers = max(math.floor(len(jobs) / NUM_TASKS_PER_WRITER), 1)
        result_writers = [
            ResultWriter.remote(prefix=idx, output_path=output_path)  # type: ignore
            for idx in range(num_writers)
        ]

        results_refs = [
            run_fn.remote(job, result_writers[idx % num_writers])
            for idx, job in enumerate(jobs)
        ]
        # Finish all estimation tasks.
        wait_till_completion(results_refs, name="Reward-Estimation")
        # Flush buffers
        wait_till_completion(
            [
                result_writer.sync.remote()
                for result_writer in result_writers  # type: ignore
            ],
            name="Flush-Buffer",
        )


@ray.remote
def run_fn(job_spec: JobSpec, result_writer: ResultWriter):
    """
    Remote task to execute reward estimation
    experiment.
    """
    task_id = str(uuid.uuid4())
    logging.info("Starting task %s, %s", task_id, job_spec)
    try:
        output = reward_estimation(job_spec)
        result = {"task_id": task_id, **dataclasses.asdict(job_spec), "meta": output}
    except Exception as err:
        logging.error("Error in experiment %s: %s", task_id, err)
        raise RuntimeError(f"Task {task_id} `{job_spec}` failed") from err
    logging.info("Completed task %s: %s", task_id, job_spec)
    return result_writer.write.remote(proc_result(result))  # type: ignore


def proc_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Calculates final results.
    """
    output = copy.deepcopy(result)
    meta = output["meta"]
    # all-steps error
    r_true = np.array(meta["true_reward_buffer"], dtype=np.float64)
    r_pred = np.array(meta["pred_reward_buffer"], dtype=np.float64)
    # use the smallest, working backwards
    # (workaround for setup order)
    min_size = np.minimum(np.size(r_true), np.size(r_pred))
    r_true, r_pred = r_true[-min_size:], r_pred[-min_size:]
    all_steps_error = metrics.rmse(v_true=r_true, v_pred=r_pred, axis=0)

    # post estimation error
    solution_step: Optional[int] = meta["solver_state"]["solution_found_step"]
    post_est_error = (
        metrics.rmse(
            v_true=r_true[solution_step:], v_pred=r_pred[solution_step:], axis=0
        )
        if solution_step
        else np.nan
    )

    del meta["true_reward_buffer"]
    del meta["pred_reward_buffer"]
    meta["all_steps_error"] = all_steps_error
    meta["post_est_error"] = post_est_error
    return output


def wait_till_completion(tasks_refs, name: Optional[str] = None):
    """
    Waits for every ray task to complete.
    """
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


def yield_as_completed(tasks_refs, name: Optional[str] = None):
    """
    Waits for every ray task to complete.
    """
    unfinished_tasks = tasks_refs
    finished_tasks = []
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        logging.info(
            "Yielding %d %s task(s). %d left out of %d.",
            len(finished_tasks),
            name,
            len(unfinished_tasks),
            len(tasks_refs),
        )
        for finished_task in finished_tasks:
            yield finished_task

        if len(unfinished_tasks) == 0:
            break


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
        delay_config=poisson_delay_config(job_spec.reward_delay),
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


def poisson_delay_config(lam: int):
    """
    Natural Poisson bounds:
    low, lambda, high
    0 2 5
    0 3 7
    1 4 8
    1 5 10
    2 6 11
    2 7 13
    3 8 14
    """
    lb, _ = mathutils.poisson_exact_confidence_interval(lam)
    return {"name": "clipped-poisson", "args": {"lam": lam, "min_delay": max(2, lb)}}


def setup_experiment(exp_instance: core.ExperimentInstance):
    """
    Sets up an experiment run given an instance.
    """
    opt_logs: Dict[str, Any] = {}
    env_spec = exp_instance.experiment.env_spec
    problem_spec = exp_instance.experiment.problem_spec
    env = envs.make(
        env_name=env_spec.name,
        **env_spec.args if env_spec.args else {},
    )
    # For non-control related ops, e.g. data collection
    # or sampling for state projection.
    proxy_env = envs.make(
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
        proxy_env=proxy_env,
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


def write_records(
    output_path: str,
    records: Sequence[Mapping[str, Any]],
    gzip_compression: bool = True,
) -> None:
    """
    Exports records into JSON files.
    """
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
    """
    Read records from JSON.
    """
    logging.debug(
        "Reading file %s",
        input_path,
    )

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
