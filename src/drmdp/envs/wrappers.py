import copy
from typing import Any, Dict, Hashable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from sklearn import cluster, mixture, model_selection

from drmdp import constants, dataproc, mathutils, tiles


class RandomBinaryObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, enc_size: int):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(enc_size), high=np.ones(enc_size), dtype=np.int64
        )
        self.enc_size = enc_size
        self._representations: Dict[Hashable, Any] = {}

        if not isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Discrete)):
            raise ValueError(
                f"Environment space must be either Box or Discrete, not {type(env.observation_space)}"
            )

    def observation(self, observation: ObsType):
        array = np.array(observation, dtype=np.int64)
        key: Hashable = -1
        if np.shape(array) == ():
            key = array.item()
        else:
            key = tuple(array.tolist())
        if key not in self._representations:
            indices = np.random.randint(
                0,
                high=self.enc_size,
                size=self.enc_size // 2,
            )
            vec = np.zeros(shape=self.enc_size)
            vec[indices] = 1
            self._representations[key] = vec
        return self._representations[key]


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
    def __init__(self, env, param_grid, sample_steps: int, **kwargs):
        super().__init__(env)
        self.param_grid = param_grid
        self.sample_steps = sample_steps
        # Make a copy of the env
        # to preserve state.
        buffer = dataproc.collection_traj_data(copy.copy(env), steps=sample_steps)
        self.grid_search = self.gm_proj(buffer, param_grid, **kwargs)
        self.estimator = self.grid_search.best_estimator_
        self.obs_dim = self.grid_search.best_estimator_.n_components

        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape=self.obs_dim),
            high=np.ones(shape=self.obs_dim),
            dtype=np.float64,
        )

    def gm_proj(self, buffer, param_grid, **kwargs):
        grid_search = model_selection.GridSearchCV(
            mixture.GaussianMixture(init_params="k-means++", **kwargs),
            param_grid=param_grid,
            scoring=self.gmm_bic_score,
        )
        return grid_search.fit(np.stack([tup[0] for tup in buffer]))

    def gmm_bic_score(self, estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    def observation(self, observation: ObsType):
        # Apply to first N dimensions
        basis_probs_batch = self.estimator.predict_proba([observation])
        return basis_probs_batch[0]


class ClusterCentroidObsWrapper(gym.ObservationWrapper):
    """
    Clusters the input space.
    As such, maps each state into a centroid id.
    Centroids are in the range [0, `num_clusters`).
    """

    def __init__(
        self,
        env: gym.Env,
        num_clusters: int,
        sample_steps: int,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.num_clusters = num_clusters
        self.sample_steps = sample_steps
        # Make a copy of the env
        # to preserve state.
        buffer = dataproc.collection_traj_data(
            copy.copy(env), steps=self.sample_steps, seed=seed
        )
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Source `env` observation space must be of type `Box`. Got {type(env.observation_space)}"
            )
        # To make centroids float64.
        # Bypass KMeans issues with float32 centroids
        # when calling `predict`
        matrix = np.stack([tup[0] for tup in buffer], dtype=np.float64)
        self.estimator = self.fit_clusters(matrix)
        self.obs_dim = self.num_clusters
        self.observation_space = gym.spaces.Discrete(self.num_clusters)

    def fit_clusters(self, matrix, seed: Optional[int] = None):
        """
        Fit clusters.
        """
        clustering = cluster.KMeans(
            n_clusters=self.num_clusters, init="k-means++", random_state=seed
        )
        clustering.fit(matrix)
        return clustering

    def observation(self, observation: ObsType):
        """
        Returns cluster assignment.
        """
        clusters_batch = self.estimator.predict([observation])
        return clusters_batch[0]


class FlatGridCoordObsWrapper(gym.ObservationWrapper):
    """
    Maps n-dimension array grid positions to a
    single value.
    """

    def __init__(self, env: gym.Env, ohe: bool = False):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Source `env` observation space must be of type `Box`. Got {type(env.observation_space)}"
            )

        if np.size(env.observation_space.shape) != 1:
            raise ValueError(
                f"Env should be a 1D array. Got {env.observation_space.shape}"
            )

        self.ohe = ohe
        shape = env.observation_space.shape
        self.ndims = (
            shape[0] if isinstance(env.observation_space.shape, Sequence) else shape
        )
        value_ranges = env.observation_space.high - env.observation_space.low
        self.value_ranges = np.array(value_ranges, dtype=np.int64)
        if np.sum(value_ranges - self.value_ranges) != 0:
            raise ValueError(
                f"Bad value range: {env.observation_space}. Make sure all values are integers."
            )

        # The obs space doesn't map to states, i.e. some coordinates
        # don't actually exist.
        # This will overspecify matricies.
        self._state_ops = {}
        if self.has_wrapper_attr("transition") and self.has_wrapper_attr(
            "get_state_id"
        ):
            self._state_ops["transition"] = self.get_wrapper_attr("transition")
            self._state_ops["get_state_id"] = self.get_wrapper_attr("get_state_id")

        # num coordinates
        self.nstates = (
            len(self._state_ops["transition"])
            if self._state_ops
            else int(np.prod(self.value_ranges))
        )
        # Cache op
        self.value_range_prod = [
            np.prod(self.value_ranges[idx + 1 :]) for idx in range(self.ndims)
        ]
        self.output_size = self.nstates if self.ohe else 1
        self.observation_space = (
            gym.spaces.Box(low=np.zeros(self.nstates), high=np.ones(self.nstates))
            if self.ohe
            else gym.spaces.Discrete(self.nstates)
        )

    def observation(self, observation: ObsType):
        """
        Returns cluster assignment.
        """
        pos: int = 0
        if self._state_ops:
            pos = self._state_ops["get_state_id"](observation)
        else:
            xs = np.concatenate([observation, [1]])
            for idx in range(self.ndims):
                pos += xs[idx] * self.value_range_prod[idx]
            pos = int(pos)
        if self.ohe:
            output = np.zeros(self.nstates)
            output[pos] = 1
            return output
        return pos


class TilesObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
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
        self.hash_dim = (
            hash_dim if hash_dim and self.tiles.max_size > hash_dim else None
        )
        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape=self.hash_dim or self.tiles.max_size),
            high=np.ones(shape=self.hash_dim or self.tiles.max_size),
            dtype=np.int64,
        )

    def observation(self, observation: ObsType):
        # Apply to first N dimensions
        xs = self.tiles(observation)
        if self.hash_dim:
            xs = mathutils.hashtrick(xs, dim=self.hash_dim)
        return xs


def wrap(env: gym.Env, wrapper: Optional[str] = None, **kwargs):
    """
    Creates an environment observation wrappers.
    """
    if wrapper is None:
        return env
    if wrapper == constants.RANDOM_VEC:
        enc_size = kwargs["enc_size"]
        return RandomBinaryObsWrapper(env, enc_size=enc_size)
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
    if wrapper == constants.CLUSTER_CENTROID:
        num_clusters = kwargs["num_clusters"]
        steps = kwargs.get("steps", constants.DEFAULT_CLUSTER_STEPS)
        return ClusterCentroidObsWrapper(
            env, num_clusters=num_clusters, sample_steps=steps
        )
    if wrapper == constants.FLAT_GRID_COORD:
        ohe = kwargs.get("ohe", False)
        return FlatGridCoordObsWrapper(env, ohe=ohe)
    raise ValueError(f"Wrapper `{wrapper}` unknown")
