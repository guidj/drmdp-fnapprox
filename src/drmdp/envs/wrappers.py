from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from sklearn import mixture, model_selection

from drmdp import constants, data, hashtutils, tiles


class ScaleObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = np.size(self.obs_space.high)

    def observation(self, observation: ObsType):
        obs_scaled_01 = (observation - self.obs_space.low) / (
            self.obs_space.high - self.obs_space.low
        )
        return obs_scaled_01


class GaussianMixObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, param_grid, sample_steps: int):
        super().__init__(env)
        buffer = data.collection_traj_data(env, steps=sample_steps)
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
        obs = np.stack([example[2] for example in buffer])
        grid_search = model_selection.GridSearchCV(
            mixture.GaussianMixture(), param_grid=param_grid, scoring=self.gmm_bic_score
        )
        return grid_search.fit(obs)

    def gmm_bic_score(self, estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    def observation(self, observation: ObsType):
        # Apply to first N dimensions
        basis_probs_batch = self.estimator.predict_proba([observation])
        return basis_probs_batch[0]


class TilesObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        tiling_dim: int,
        num_tilings: int = None,
        hash_dim: Optional[int] = None,
    ):
        super().__init__(env)
        self.tiles = tiles.Tiles(
            dims_min=env.observation_space.low,
            dims_max=env.observation_space.high,
            tiling_dim=tiling_dim,
            num_tilings=num_tilings,
        )

        # Hashing applies only to reduce the obs space.
        self.hash_dim = hash_dim if self.tiles.max_size > hash_dim else None
        if self.hash_dim:
            print("Hashed dim:", self.hash_dim)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape=self.hash_dim or self.tiles.max_size),
            high=np.ones(shape=self.hash_dim or self.tiles.max_size),
        )

    def observation(self, observation: ObsType):
        # Apply to first N dimensions
        xs = self.tiles(observation)
        if self.hash_dim:
            xs = hashtutils.hashtrick(xs, dim=self.hash_dim)
        return xs


def wrap(env: gym.Env, wrapper: Optional[str] = None, **kwargs):
    if wrapper is None:
        return env
    if wrapper == constants.SCALE:
        return ScaleObsWrapper(env)
    if wrapper == constants.GAUSSIAN_MIX:
        param_grid = kwargs.get("param_grid", constants.DEFAULT_PARAMS_GRID)
        steps = kwargs.get("steps", constants.DEFAULT_GM_STEPS)
        return GaussianMixObsWrapper(env, param_grid=param_grid, sample_steps=steps)
    if wrapper == constants.TILES:
        tiling_dim = kwargs.get("tiling_dim", constants.DEFAULT_TILING_DIM)
        hash_dim = kwargs.get("hash_dim", constants.DEFAULT_HASH_DIM)
        return TilesObsWrapper(env, tiling_dim=tiling_dim, hash_dim=hash_dim)
    raise ValueError(f"Wrapper `{wrapper}` unknown")
