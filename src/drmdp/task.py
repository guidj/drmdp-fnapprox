import dataclasses
import logging
import os
import os.path
import uuid
from typing import Any, Iterator, List, Mapping, Sequence

import gymnasium as gym
import numpy as np

from drmdp import algorithms, core, envs, feats, logger, optsol


def policy_control_run_fn(exp_instance: core.ExperimentInstance):
    """
    Entry point running on-policy evaluation for DAAF.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env = envs.make(
        env_name=exp_instance.experiment.env_spec.name,
        **exp_instance.experiment.env_spec.args
        if exp_instance.experiment.env_spec.args
        else {},
    )
    feats_tfx = feats.create_feat_transformer(
        env=env, **exp_instance.experiment.env_spec.feats_spec
    )
    policy = algorithms.LinearFnApproxPolicy(
        feat_transform=feats_tfx, action_space=env.action_space
    )

    logging.debug("Starting DAAF Control Experiments: %s", exp_instance)

    results = policy_control(
        env=env,
        policy=policy,
        problem_spec=exp_instance.experiment.problem_spec,
        num_episodes=exp_instance.run_config.episodes_per_run,
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
                        # Action values can be large tables
                        # especially for options policies
                        # so we log state values and best actions
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
    env.close()


def policy_control(
    env: gym.Env,
    policy: core.PyValueFnPolicy,
    problem_spec: core.ProblemSpec,
    num_episodes: int,
) -> Iterator[algorithms.PolicyControlSnapshot]:
    """
    Runs policy control with given algorithm, env, and policy spec.
    """
    # create lrs
    # init values
    # traj-mapper/wrapper
    # run algorithms

    # Create spec using provided name and args for feature spec
    lr = learning_rate(**problem_spec.learning_rate_config)
    sarsa = algorithms.SemigradietSARSAFnApprox(
        env=env,
        lr=lr,
        gamma=problem_spec.gamma,
        epsilon=problem_spec.epsilon,
        policy=policy,
    )
    return sarsa.train(
        num_episodes=num_episodes,
    )


def create_task_id(task_prefix: str) -> str:
    """
    Creates a task id using a given prefix
    and a generated partial uuid.
    """
    return f"{task_prefix}-{uid()}"


def uid() -> str:
    """
    Generate a uuid from partial uuid
    """
    return next(iter(str(uuid.uuid4()).split("-")))


def generate_experiments_instances(
    experiments: Sequence[core.Experiment],
    run_config: core.RunConfig,
    task_prefix: str,
) -> Iterator[core.ExperimentInstance]:
    for experiment in experiments:
        exp_id = "-".join([create_task_id(task_prefix), experiment.env_spec.name])
        for idx in range(run_config.num_runs):
            yield core.ExperimentInstance(
                exp_id=exp_id,
                instance_id=idx,
                experiment=experiment,
                run_config=dataclasses.replace(
                    run_config,
                    # replace run output with run specific values
                    output_dir=os.path.join(
                        run_config.output_dir,
                        exp_id,
                        f"run_{idx}",
                        experiment.problem_spec.traj_mapping_method,
                        uid(),
                    ),
                ),
                context={"dummy": 0},
            )


def bundle(items: Sequence[Any], bundle_size: int) -> Sequence[Sequence[Any]]:
    """
    Bundles items into groups of size `bundle_size`, if possible.
    The last bundle may have fewer items.
    """
    if bundle_size < 1:
        raise ValueError("`bundle_size` must be positive.")

    bundles: List[List[Any]] = []
    bundle_: List[Any] = []
    for idx, item in enumerate(items):
        if idx > 0 and (idx % bundle_size) == 0:
            if bundle_:
                bundles.append(bundle_)
            bundle_ = []
        bundle_.append(item)
    if bundle_:
        bundles.append(bundle_)
    return bundles


def learning_rate(name: str, args: Mapping[str, Any]) -> optsol.LearningRateSchedule:
    """
    Returns a learning rate scheduler.
    """
    if "initial_lr" not in args:
        raise ValueError(f"Missing `initial_lr` from lr config: {args}")
    if name == "fixed":
        return optsol.FixedLRSchedule(args["initial_lr"])
    else:
        raise ValueError(f"Unknown lr {name}")
