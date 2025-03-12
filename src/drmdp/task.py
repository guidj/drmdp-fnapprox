import copy
import dataclasses
import logging
import os
import os.path
import random
import uuid
from typing import Any, Iterator, List, Mapping, Sequence

import gymnasium as gym
import numpy as np

from drmdp import core, envs, feats, logger, optsol


@dataclasses.dataclass(frozen=True)
class PolicyControlSnapshot:
    steps: int
    returns: float
    weights: np.ndarray


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
    logging.debug("Starting DAAF Control Experiments")
    results = policy_control(
        env=env,
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
    problem_spec: core.ProblemSpec,
    num_episodes: int,
) -> Iterator[PolicyControlSnapshot]:
    """
    Runs policy control with given algorithm, env, and policy spec.
    """
    # create lrs
    # init values
    # traj-mapper/wrapper
    # run algorithms

    # Create spec using provided name and args for feature spec
    feats_tfx = feats.create_feat_transformer(env=env, **problem_spec.feats_spec)
    lr = learning_rate(**problem_spec.learning_rate_config)
    return semi_gradient_sarsa(
        env=env,
        lr=lr,
        gamma=problem_spec.gamma,
        epsilon=problem_spec.epsilon,
        num_episodes=num_episodes,
        feat_transform=feats_tfx,
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


def semi_gradient_sarsa(
    env: gym.Env,
    lr: optsol.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    num_episodes: int,
    feat_transform: feats.FeatTransform,
    verbose: bool = True,
) -> Iterator[PolicyControlSnapshot]:
    actions = tuple(range(env.action_space.n))
    weights = np.zeros(feat_transform.output_shape, dtype=np.float64)
    returns = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state_qvalues, gradients = action_values(obs, actions, weights, feat_transform)
        rewards = 0
        # choose action
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.random.choice(
                np.flatnonzero(state_qvalues == state_qvalues.max())
            )
        steps = 0
        while True:
            # greedy
            (
                next_obs,
                reward,
                term,
                trunc,
                _,
            ) = env.step(action)
            rewards += reward

            if term or trunc:
                weights = (
                    weights
                    + lr(episode, steps)
                    * (reward - state_qvalues[action])
                    * gradients[action]
                )
                break

            next_state_qvalues, next_gradients = action_values(
                next_obs, actions, weights, feat_transform
            )

            if random.random() <= epsilon:
                next_action = env.action_space.sample()
            else:
                # greedy
                next_action = np.random.choice(
                    np.flatnonzero(next_state_qvalues == next_state_qvalues.max())
                )

            weights = (
                weights
                + lr(episode, steps)
                * (
                    reward
                    + gamma * next_state_qvalues[next_action]
                    - state_qvalues[action]
                )
                * gradients[action]
            )
            obs = next_obs
            action = next_action
            state_qvalues = next_state_qvalues
            gradients = next_gradients
            steps += 1
        returns.append(rewards)
        if verbose and (episode + 1) % (num_episodes // 5) == 0:
            logging.info("Episode %d mean returns: %f", episode + 1, np.mean(returns))
        yield PolicyControlSnapshot(
            steps=steps, returns=rewards, weights=copy.copy(weights)
        )


def action_values(
    observation, actions: Sequence[int], weights, feat_transform: feats.FeatTransform
):
    observations = [observation] * len(actions)
    state_action_m = feat_transform.batch_transform(observations, actions)
    return np.dot(state_action_m, weights), state_action_m


def learning_rate(name: str, args: Mapping[str, Any]):
    if "initial_lr" not in args:
        raise ValueError(f"Missing `initial_lr` from lr config: {args}")
    if name == "fixed":
        return optsol.FixedLRSchedule(args["initial_lr"])
    else:
        raise ValueError(f"Unknown lr {name}")
