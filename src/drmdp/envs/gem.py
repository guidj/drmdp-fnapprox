import copy
import functools
from typing import Any, Optional

import gym_electric_motor
import gymnasium as gym
import numpy as np
from gym_electric_motor import reward_functions
from sklearn import mixture, model_selection

from drmdp import data, hashtutils, tiles, constants


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


class GemScaleObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, hash_dim: Optional[int] = None):
        super().__init__(env)
        self.hash_dim = hash_dim

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.hash_dim = hash_dim
        self.obs_dim = np.size(self.obs_space.high)

    def observation(self, obs: Any):
        obs_scaled_01 = (obs - self.obs_space.low) / (
            self.obs_space.high - self.obs_space.low
        )
        return obs_scaled_01


class GemGaussianMixObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, param_grid, steps: int):
        super().__init__(env)
        buffer = data.collection_traj_data(env, steps=steps)
        self.grid_search = self.gm_proj(buffer, param_grid)
        self.estimator = self.grid_search.best_estimator_
        print("Best estimator:", self.grid_search.best_estimator_)
        self.obs_dim = self.grid_search.best_estimator_.n_components + 1

        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape=self.obs_dim, dtype=np.float32),
            high=np.ones(shape=self.obs_dim, dtype=np.float32),
        )

    def gm_proj(self, buffer, param_grid):
        # exclude last state component
        obs = np.stack([example[2][:-1] for example in buffer])
        grid_search = model_selection.GridSearchCV(
            mixture.GaussianMixture(), param_grid=param_grid, scoring=self.gmm_bic_score
        )
        return grid_search.fit(obs)

    def gmm_bic_score(self, estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    def observation(self, obs):
        # Apply to first N dimensions
        xs = self.estimator.predict_proba(np.array([obs[:-1]]))
        return np.concatenate([xs[0], np.array([obs[-1]])])


class GemTileObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        tiling_dim: int,
        num_tilings: int = None,
        hash_dim: Optional[int] = None,
    ):
        super().__init__(env)
        self.hash_dim = hash_dim
        self.tiles = tiles.Tiles(
            dims_min=env.observation_space.low[:-1],
            dims_max=env.observation_space.high[:-1],
            tiling_dim=tiling_dim,
            num_tilings=num_tilings,
        )
        self.observation_space = gym.spaces.Box(
            low=np.zeros(
                shape=self.hash_dim or self.tiles.max_size + 1, dtype=np.float64
            ),
            high=np.ones(
                shape=self.hash_dim or self.tiles.max_size + 1, dtype=np.float64
            ),
        )

    def observation(self, obs):
        # Apply to first N dimensions
        xs = self.tiles(obs[:-1])
        if self.hash_dim:
            xs = hashtutils.hashtrick(xs, dim=self.hash_dim - 1)
        return np.concatenate([xs, np.array([obs[-1]])])


def make(
    env_name: str,
    constraint_violation_reward: Optional[float] = -10.0,
    wrapper: Optional[str] = None,
) -> gym.Env:
    rf = StrictWeightedSumOfErrors(violation_reward=constraint_violation_reward)
    env = GemObsAsVectorWrapper(gym_electric_motor.make(env_name, reward_function=rf))
    if wrapper is None:
        return env
    if wrapper == constants.SCALE:
        return GemScaleObsWrapper(env)
    elif wrapper == constants.GAUSSIAN_MIX:
        return GemGaussianMixObsWrapper(env)
    elif wrapper == constants.TILES:
        return GemTileObsWrapper(env)
    raise ValueError(f"Wrapper `{wrapper}` unknown")
