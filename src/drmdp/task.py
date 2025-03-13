import copy
import dataclasses
import logging
import os
import os.path
import random
import uuid
from typing import Any, Iterator, List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from drmdp import core, envs, feats, logger, optsol


@dataclasses.dataclass(frozen=True)
class PolicyControlSnapshot:
    steps: int
    returns: float
    weights: np.ndarray


class PyEGreedyValueFnPolicy(core.PyValueFnPolicy):
    """
    A e-greedy, which randomly chooses actions with e probability,
    and the chooses teh best action otherwise.
    """

    def __init__(
        self,
        exploit_policy: core.PyValueFnPolicy,
        epsilon: float,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"Epsilon must be between [0, 1]: {epsilon}")
        super().__init__(
            action_space=exploit_policy.action_space,
            emit_log_probability=emit_log_probability,
            seed=seed,
        )

        self.exploit_policy = exploit_policy
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        policy_step = self.exploit_policy.action(observation, policy_state)
        # greedy move, find out the greedy arm
        if self._rng.random() <= self.epsilon:
            action = self.action_space.sample()
            return dataclasses.replace(policy_step, action=action)
        return policy_step

    def action_values_gradients(self, observation, actions):
        return self.exploit_policy.action_values_gradients(observation, actions)

    def update(self, scaled_gradients):
        return self.exploit_policy.update(scaled_gradients)

    @property
    def model(self):
        return self.exploit_policy.model


class LinearFnApproxPolicy(core.PyValueFnPolicy):
    def __init__(
        self,
        feat_transform: feats.FeatTransform,
        action_space: gym.Space,
        emit_log_probability=False,
        seed=None,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.feat_transform = feat_transform
        self.weights = np.zeros(feat_transform.output_shape, dtype=np.float64)
        self.actions = tuple(range(action_space.n))
        self.rng = np.random.default_rng()

    def get_initial_state(self, batch_size=None):
        del batch_size
        return ()

    def action(self, observation, policy_state: Any = (), seed=None):
        del seed
        state_qvalues, gradients = self.action_values_gradients(
            observation, self.actions
        )
        # Choose highest value action
        # breaking ties are random
        action = self.rng.choice(np.flatnonzero(state_qvalues == state_qvalues.max()))
        return core.PolicyStep(
            action,
            state=policy_state,
            info={"values": state_qvalues, "gradients": gradients},
        )

    def action_values_gradients(self, observation, actions):
        observations = [observation] * len(actions)
        state_action_m = self.feat_transform.batch_transform(observations, actions)
        return np.dot(state_action_m, self.weights), state_action_m

    def update(self, scaled_gradients):
        self.weights += scaled_gradients

    @property
    def model(self):
        return self.weights


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
    policy = LinearFnApproxPolicy(
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
) -> Iterator[PolicyControlSnapshot]:
    """
    Runs policy control with given algorithm, env, and policy spec.
    """
    # create lrs
    # init values
    # traj-mapper/wrapper
    # run algorithms

    # Create spec using provided name and args for feature spec
    lr = learning_rate(**problem_spec.learning_rate_config)
    return semi_gradient_sarsa(
        env=env,
        lr=lr,
        gamma=problem_spec.gamma,
        epsilon=problem_spec.epsilon,
        num_episodes=num_episodes,
        policy=policy,
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
    policy: core.PyValueFnPolicy,
    verbose: bool = True,
) -> Iterator[PolicyControlSnapshot]:
    egreedy_policy = PyEGreedyValueFnPolicy(exploit_policy=policy, epsilon=epsilon)
    returns = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        policy_step = egreedy_policy.action(obs)
        state_qvalues, gradients = (
            policy_step.info["values"],
            policy_step.info["gradients"],
        )
        steps = 0
        rewards = 0
        while True:
            (
                next_obs,
                reward,
                term,
                trunc,
                _,
            ) = env.step(policy_step.action)
            rewards += reward

            if term or trunc:
                scaled_gradients = (
                    lr(episode, steps)
                    * (reward - state_qvalues[policy_step.action])
                    * gradients[policy_step.action]
                )
                egreedy_policy.update(scaled_gradients)
                break

            next_policy_step = egreedy_policy.action(next_obs)
            next_state_qvalues, next_gradients = (
                next_policy_step.info["values"],
                next_policy_step.info["gradients"],
            )
            scaled_gradients = (
                lr(episode, steps)
                * (
                    reward
                    + gamma * next_state_qvalues[next_policy_step.action]
                    - state_qvalues[policy_step.action]
                )
                * gradients[policy_step.action]
            )
            egreedy_policy.update(scaled_gradients)
            obs = next_obs
            policy_step = next_policy_step
            state_qvalues = next_state_qvalues
            gradients = next_gradients
            steps += 1
        returns.append(rewards)
        if verbose and (episode + 1) % (num_episodes // 5) == 0:
            logging.info("Episode %d mean returns: %f", episode + 1, np.mean(returns))
        yield PolicyControlSnapshot(
            steps=steps, returns=rewards, weights=copy.copy(egreedy_policy.model)
        )


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
