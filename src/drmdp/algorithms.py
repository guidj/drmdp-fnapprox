import abc
import copy
import dataclasses
import logging
import random
from typing import Any, Iterator, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from drmdp import core, feats, optsol


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


class FnApproxAlgorithm(abc.ABC):
    @abc.abstractmethod
    def train(self, num_episodes: int) -> Iterator[PolicyControlSnapshot]: ...


class SemigradietSARSAFnApprox(FnApproxAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        lr: optsol.LearningRateSchedule,
        gamma: float,
        epsilon: float,
        policy: core.PyValueFnPolicy,
        verbose: bool = True,
    ):
        super().__init__()

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.verbose = verbose
        self.egreedy_policy = PyEGreedyValueFnPolicy(
            exploit_policy=policy, epsilon=epsilon
        )

    def train(self, num_episodes: int) -> Iterator[PolicyControlSnapshot]:
        returns = []
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            policy_step = self.egreedy_policy.action(obs)
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
                ) = self.env.step(policy_step.action)
                rewards += reward

                if term or trunc:
                    scaled_gradients = (
                        self.lr(episode, steps)
                        * (reward - state_qvalues[policy_step.action])
                        * gradients[policy_step.action]
                    )
                    self.egreedy_policy.update(scaled_gradients)
                    break

                next_policy_step = self.egreedy_policy.action(next_obs)
                next_state_qvalues, next_gradients = (
                    next_policy_step.info["values"],
                    next_policy_step.info["gradients"],
                )
                scaled_gradients = (
                    self.lr(episode, steps)
                    * (
                        reward
                        + self.gamma * next_state_qvalues[next_policy_step.action]
                        - state_qvalues[policy_step.action]
                    )
                    * gradients[policy_step.action]
                )
                self.egreedy_policy.update(scaled_gradients)
                obs = next_obs
                policy_step = next_policy_step
                state_qvalues = next_state_qvalues
                gradients = next_gradients
                steps += 1
            returns.append(rewards)
            if self.verbose and (episode + 1) % (num_episodes // 5) == 0:
                logging.info(
                    "Episode %d mean returns: %f", episode + 1, np.mean(returns)
                )
            yield PolicyControlSnapshot(
                steps=steps,
                returns=rewards,
                weights=copy.copy(self.egreedy_policy.model),
            )
