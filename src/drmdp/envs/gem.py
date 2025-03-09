import copy
import functools
from typing import Optional

import gym_electric_motor
import gymnasium as gym
import numpy as np
from gym_electric_motor import reward_functions

from drmdp.envs import wrappers


class StrictWeightedSumOfErrors(reward_functions.WeightedSumOfErrors):
    def __init__(
        self, reward_weights=None, normed_reward_weights=False, violation_reward=None
    ):
        super().__init__(
            reward_weights,
            normed_reward_weights,
            violation_reward,
            gamma=1.0,
            reward_power=1,
            bias=0,
        )

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        del k
        del action
        return (
            self._wse_reward(state, reference)
            + violation_degree * self._violation_reward
        )


class GemObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = getattr(env.reference_generator, "referenced_states")
        state_obs_space, ref_state_obs_space = env.observation_space

        self._weights = getattr(env.reward_function, "_reward_weights")[self._mask]
        self._expo = getattr(env.reward_function, "_n")[self._mask]
        self._bias = getattr(env.reward_function, "_bias")
        self._denom = (state_obs_space.high - state_obs_space.low)[self._mask]
        self._prev_ref_state = None  # np.zeros_like(state_obs_space.high[self._mask])

        bounds = [
            np.abs(state_obs_space.high[self._mask] - ref_state_obs_space.low),
            np.abs(state_obs_space.high[self._mask] - ref_state_obs_space.high),
            np.abs(state_obs_space.low[self._mask] - ref_state_obs_space.high),
            np.abs(state_obs_space.low[self._mask] - ref_state_obs_space.low),
        ]
        obs_space_low = np.concatenate(
            [
                np.zeros_like(state_obs_space.low[self._mask]) + self._bias,
                # constraint violation
                np.array([0.0]),
            ]
        )
        obs_space_high = np.concatenate(
            [
                (functools.reduce(np.maximum, bounds) / self._denom) ** self._expo
                + self._bias,
                # constraint violation
                np.array([1.0]),
            ]
        )
        self.observation_space = gym.spaces.Box(
            low=obs_space_low, high=obs_space_high, dtype=state_obs_space.dtype
        )
        self._cvfn = getattr(self.env.constraint_monitor, "check_constraints")

    def observation(self, observation):
        prev_ref_state = copy.copy(self._prev_ref_state)
        next_state, ref_state = observation
        cv = self._cvfn(next_state)
        next_state = next_state[self._mask]

        if prev_ref_state is None:
            prev_ref_state = ref_state

        wrapped_next_state = np.concatenate(
            [
                (abs(next_state - prev_ref_state) / self._denom) ** self._expo
                + self._bias,
                np.array([cv]),
            ]
        )
        self._prev_ref_state = ref_state
        return wrapped_next_state


def make(
    env_name: str,
    constraint_violation_reward: Optional[float] = -10.0,
    wrapper: Optional[str] = None,
    **kwargs,
) -> gym.Env:
    rf = StrictWeightedSumOfErrors(violation_reward=constraint_violation_reward)
    env = GemObsAsVectorWrapper(gym_electric_motor.make(env_name, reward_function=rf))
    return wrappers.wrap(env, wrapper=wrapper, **kwargs)
