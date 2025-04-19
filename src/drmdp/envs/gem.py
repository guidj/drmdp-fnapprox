import copy
import functools
from typing import Optional

import gym_electric_motor
import gymnasium as gym
import numpy as np
from gym_electric_motor import reward_functions

from drmdp.envs import wrappers


class StrictWeightedSumOfErrors(reward_functions.WeightedSumOfErrors):
    """
    This reward function applies the violation penalty
    whilst keeping the reward linear w.r.t to the
    penalty team and state.
    """

    def __init__(
        self,
        penalty_gamma: Optional[float] = None,
        reward_weights=None,
        normed_reward_weights=False,
        violation_reward=None,
    ):
        super().__init__(
            reward_weights,
            normed_reward_weights,
            violation_reward,
            gamma=penalty_gamma,
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


class EarlyStopPenaltyWeightedSumOfErrors(reward_functions.WeightedSumOfErrors):
    """
    This reward function applies the violation penalty
    whilst keeping the reward linear w.r.t to the
    penalty team and state.

    The violation penalty is set to an extreme value
    to encourage continued exploration.
    """

    def __init__(
        self,
        reward_weights=None,
        normed_reward_weights=False,
    ):
        super().__init__(
            reward_weights,
            normed_reward_weights,
            violation_reward=-((2**31) - 1),
            gamma=None,
            reward_power=1,
            bias=0,
        )
        self.penalty_set = False

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        del k
        del action
        return (
            self._wse_reward(state, reference)
            + violation_degree * self._violation_reward
        )


class PositiveEnforcementWeightedSumOfErrors(reward_functions.WeightedSumOfErrors):
    """
    This function assumes the reward is bounded
    between [-x, 0].
    Shifts the rewards range for positive re-enforcement.
    """

    def __init__(
        self,
        penalty_gamma: Optional[float] = None,
        reward_weights=None,
        normed_reward_weights=False,
        violation_reward=None,
    ):
        super().__init__(
            reward_weights,
            normed_reward_weights,
            violation_reward,
            gamma=penalty_gamma,
            reward_power=1,
            bias=0,
        )
        self.checked = False

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        del k
        del action

        if not self.checked:
            rw_lb, rw_ub = self.reward_range
            if rw_lb >= 0:
                raise ValueError(f"Lower bound should negative: {self.reward_range}")
            if rw_ub != 0:
                raise ValueError(f"Upper bound should zero: {self.reward_range}")
            self.checked = True
        return (
            self._wse_reward(state, reference)
            + violation_degree * self._violation_reward
        ) + (2 * abs(self.reward_range[0]))


class GemObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
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
                # delta
                np.zeros_like(state_obs_space.low[self._mask]) + self._bias,
                # values
                state_obs_space.low[self._mask],
                # constraint violation
                np.array([0.0]),
            ]
        )
        obs_space_high = np.concatenate(
            [
                # delta
                (functools.reduce(np.maximum, bounds) / self._denom) ** self._expo
                + self._bias,
                # values
                state_obs_space.high[self._mask],
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
                next_state,
                np.array([cv]),
            ]
        )
        self._prev_ref_state = ref_state
        return wrapped_next_state


class DiscretiseActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = env.action_space
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            if not np.size(env.action_space.nvec) == 2:
                raise ValueError(
                    f"Only MultiDiscrete(2) is supported. Got {env.action_space}"
                )
            self.action_space = gym.spaces.Discrete(np.prod(env.action_space.nvec))
        else:
            raise ValueError(
                f"Action space must be Discrete or MultiDiscrete. Got: {env.action_space}"
            )

    def action(self, action):
        # covert a discrete action into a multidiscrete one
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return action
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            nvec = self.env.action_space.nvec
            c1 = action % nvec[1]
            c0 = (action - c1) // nvec[1]
            return [c0, c1]
        else:
            raise ValueError(f"Unsupported action {action}")


def make(
    env_name: str,
    reward_fn: str,
    constraint_violation_reward: Optional[float] = 0.0,
    penalty_gamma: Optional[float] = 1.0,
    wrapper: Optional[str] = None,
    **kwargs,
) -> gym.Env:
    if reward_fn == "default":
        rf = StrictWeightedSumOfErrors(
            violation_reward=constraint_violation_reward, penalty_gamma=penalty_gamma
        )
    elif reward_fn == "pos-enf":
        rf = PositiveEnforcementWeightedSumOfErrors(
            violation_reward=constraint_violation_reward, penalty_gamma=penalty_gamma
        )
    elif reward_fn == "esp-neg":
        rf = EarlyStopPenaltyWeightedSumOfErrors()
    else:
        raise ValueError(f"Unknown reward fn: {reward_fn}")
    env = GemObsAsVectorWrapper(gym_electric_motor.make(env_name, reward_function=rf))
    env = DiscretiseActionWrapper(env)
    max_episode_steps = kwargs.get("max_episode_steps", None)
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
    return wrappers.wrap(env, wrapper=wrapper, **kwargs)
