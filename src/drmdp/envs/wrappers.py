import copy
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from sklearn import cluster, mixture, model_selection

from drmdp import constants, dataproc, mathutils, tiles
from drmdp.envs import gridutils


class RandomBinaryObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, enc_size: int):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(enc_size,), dtype=np.int64
        )
        self.enc_size = enc_size
        self._representations: dict[Hashable, Any] = {}

        if not isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Discrete)):
            raise TypeError(
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
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                f"Expected Box observation_space. Got: {env.observation_space}"
            )

        self.num_actions = env.action_space.n
        self.obs_dim = np.size(env.observation_space.high)

        self.observation_space = gym.spaces.Box(
            high=np.ones_like(env.observation_space.high),
            low=np.zeros_like(env.observation_space.high),
            dtype=np.float64,
        )

    def observation(self, observation: ObsType):
        obs_scaled_01 = (observation - self.env.observation_space.low) / (
            self.env.observation_space.high - self.env.observation_space.low
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
            low=0,
            high=1,
            shape=(self.obs_dim,),
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
        seed: int | None = None,
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
            raise TypeError(
                f"Source `env` observation space must be of type `Box`. Got {type(env.observation_space)}"
            )
        # To make centroids float64.
        # Bypass KMeans issues with float32 centroids
        # when calling `predict`
        matrix = np.stack([tup[0] for tup in buffer], dtype=np.float64)
        self.estimator = self.fit_clusters(matrix)
        self.obs_dim = self.num_clusters
        self.observation_space = gym.spaces.Discrete(self.num_clusters)

    def fit_clusters(self, matrix, seed: int | None = None):
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
            raise TypeError(
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
            gym.spaces.Box(low=0, high=1, shape=(self.nstates,))
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
        num_tilings: int | None = None,
        hash_dim: int | None = None,
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
        obs_dim = self.hash_dim or self.tiles.max_size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(obs_dim,),
            dtype=np.int64,
        )

    def observation(self, observation: ObsType):
        # Apply to first N dimensions
        xs = self.tiles(observation)
        if self.hash_dim:
            xs = mathutils.hashtrick(xs, dim=self.hash_dim)
        return xs


class PotentialShapingWrapper(gym.Wrapper):
    """Potential-based reward shaping: r' = r + gamma * Phi(s') - Phi(s).

    Preserves the optimal policy for any potential (Ng, Harada &
    Russell 1999) and shifts the action values by the state-only term
    ``-Phi(s)``.  Subclasses implement ``_potential(obs)`` and set
    ``goal_potential`` to the potential at the goal, so that Phi
    vanishes on terminating transitions and the telescoping sum over
    an episode cancels exactly.

    ``scale`` multiplies the potential.  A scaled potential is still a
    potential, so the invariance result applies to every scale; the
    scale only sets how far the action values move and how much the
    per-step reward varies around the base reward.
    """

    goal_potential: float = 0.0

    def __init__(self, env: gym.Env, gamma: float = 1.0, scale: float = 1.0):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale
        self._prev_potential: float = 0.0

    def _potential(self, obs: np.ndarray) -> float:
        raise NotImplementedError

    def _shaped(self, potential: float) -> float:
        return self.scale * (potential - self.goal_potential)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_potential = self._shaped(self._potential(obs))
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        potential = 0.0 if term else self._shaped(self._potential(obs))
        reward = reward + self.gamma * potential - self._prev_potential
        self._prev_potential = potential
        return obs, reward, term, trunc, info


class MountainCarHeightShaping(PotentialShapingWrapper):
    """Phi(s) = sin(3 * position) - sin(1.5).

    Hill height of the car measured from the goal height, so the
    potential is 0 at the goal and negative below it.  The shaped
    reward rises when the car climbs, which is the direction the
    additive bonus rewards, but the optimum is unchanged because the
    potential shift is a function of position only.
    """

    goal_potential = float(np.sin(1.5))

    def _potential(self, obs: np.ndarray) -> float:
        position = float(obs[0])
        return float(np.sin(3.0 * position))


class AcrobotTipHeightShaping(PotentialShapingWrapper):
    """Phi(s) = -(cos(theta1) + cos(theta1 + theta2)) - 1.

    Tip height of the second link measured from the goal height of 1,
    so the potential is 0 at termination and negative below it.
    Bounded to [-3, 1].
    """

    goal_potential = 1.0

    def _potential(self, obs: np.ndarray) -> float:
        cos_theta1 = float(obs[0])
        sin_theta1 = float(obs[1])
        cos_theta2 = float(obs[2])
        sin_theta2 = float(obs[3])
        cos_sum = cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2
        return float(-(cos_theta1 + cos_sum))


class GridWorldDistanceShaping(PotentialShapingWrapper):
    """Phi(s) = -scale * D * (d(s)/D)**exponent, d = geodesic distance.

    ``d(s)`` is the number of moves from the agent's cell to the
    nearest exit, avoiding cliffs; ``D`` is the largest distance on the
    grid.  The potential is zero on exit cells, so the goal reference
    coincides with the terminal convention: a GridWorld exit is a state
    with ``d = 0``, and no separate terminal value is needed.

    ``exponent = 1`` makes the potential proportional to the
    undiscounted optimal cost-to-go ``-d``, so the shaped action values
    are the advantage function of the base task at ``gamma = 1``.  That
    potential pays a constant ``-1 + scale`` on every step of a shortest
    path, because each such step reduces ``d`` by exactly one.  Any
    other exponent keeps the potential monotone in distance while
    grading the step reward by how far from the exit the step happens.

    ``scale`` multiplies the potential and cannot move the argmax; it
    sets the amplitude of the per-step variation.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 1.0,
        scale: float = 1.0,
        exponent: float = 2.0,
    ):
        super().__init__(env=env, gamma=gamma, scale=scale)
        self.exponent = exponent
        self.distances = _grid_geodesic_distances(env)
        self.max_distance = float(max(self.distances.values()))
        if self.max_distance == 0:
            raise ValueError("grid has no cell at a positive distance from an exit")

    def _potential(self, obs: np.ndarray) -> float:
        cell = (int(obs[0]), int(obs[1]))
        relative = self.distances[cell] / self.max_distance
        return float(-self.max_distance * relative**self.exponent)


class AdditiveShapingWrapper(gym.Wrapper):
    """Additive reward bonus: r' = r + scale * bonus(obs).

    Unlike PBRS, this changes the optimal policy but creates
    substantial state-dependent reward variation for testing
    reward estimation quality.

    The bonus is NOT applied on terminal steps so that
    terminal-state zero-reward semantics are preserved.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def _bonus(self, obs: np.ndarray) -> float:
        raise NotImplementedError

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if not term:
            reward = reward + self.scale * self._bonus(obs)
        return obs, reward, term, trunc, info


class MountainCarHeightBonus(AdditiveShapingWrapper):
    """bonus(s) = sin(3 * position), scaled by constructor arg.

    Creates position-dependent rewards: the car is rewarded for
    being high on the hill.  Bounded to [-scale, +scale].
    """

    def _bonus(self, obs: np.ndarray) -> float:
        return float(np.sin(3.0 * float(obs[0])))


class AcrobotTipHeightBonus(AdditiveShapingWrapper):
    """bonus(s) = -(cos(θ1) + cos(θ1+θ2)), scaled by constructor arg.

    Rewards the tip being high.  Bounded to [-2·scale, +2·scale].
    """

    def _bonus(self, obs: np.ndarray) -> float:
        cos_theta1 = float(obs[0])
        sin_theta1 = float(obs[1])
        cos_theta2 = float(obs[2])
        sin_theta2 = float(obs[3])
        cos_sum = cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2
        return float(-(cos_theta1 + cos_sum))


class ActionCostShapingWrapper(gym.Wrapper):
    """Action-dependent reward bonus: different actions incur different costs.

    Creates non-constant, action-dependent rewards that any
    action-aware count-based encoding (tile coding, OHE) can
    represent exactly.

    The bonus is NOT applied on terminal steps so that
    terminal-state zero-reward semantics are preserved.
    """

    def __init__(self, env: gym.Env, action_costs: Sequence[float]):
        super().__init__(env)
        self.action_costs = np.array(action_costs, dtype=np.float64)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if not term:
            reward = reward + self.action_costs[action]
        return obs, reward, term, trunc, info


class GaussianRewardNoiseWrapper(gym.Wrapper):
    """Adds clipped Gaussian noise to rewards.

    The noise standard deviation is ``sigma = std_multiplier *
    _reference_std(obs, action)``, so the multiplier is expressed in units of
    the reference standard deviation and no square or square root sits between
    a configured value and the sampled distribution. Samples are clipped to
    the 95% energy interval [-clip_std * sigma, clip_std * sigma]. Subclasses
    implement ``_reference_std(obs, action)`` for state/action-dependent noise
    magnitude.

    Noise is NOT applied on terminal steps so that
    terminal-state zero-reward semantics are preserved.
    """

    def __init__(
        self,
        env: gym.Env,
        std_multiplier: float = 1.0,
        clip_std: float = 1.96,
        seed: int | None = None,
    ):
        super().__init__(env)
        if std_multiplier < 0:
            raise ValueError(
                f"std_multiplier must be non-negative, not {std_multiplier}"
            )
        self.std_multiplier = std_multiplier
        self.clip_std = clip_std
        self._rng = np.random.default_rng(seed)

    def _reference_std(self, obs: np.ndarray, action: int) -> float:
        raise NotImplementedError

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if not term:
            sigma = self.std_multiplier * self._reference_std(obs, action)
            noise = self._rng.normal(0.0, sigma)
            bound = self.clip_std * sigma
            noise = float(np.clip(noise, -bound, bound))
            reward = reward + noise
        return obs, reward, term, trunc, info


class ConstantGaussianNoiseReward(GaussianRewardNoiseWrapper):
    """Unit reference standard deviation: the noise std is std_multiplier."""

    def _reference_std(self, obs: np.ndarray, action: int) -> float:
        return 1.0


class RewardRangeGaussianNoiseReward(GaussianRewardNoiseWrapper):
    """Noise standard deviation proportional to a reference standard deviation.

    ``reference_std`` is the standard deviation of the post-shaping per-step
    reward distribution of the environment, so the multiplier acts as a
    relative perturbation: an equal ``std_multiplier`` produces an equal
    relative perturbation across environments with different reward
    magnitudes. The parameter is named for a standard deviation, not for the
    ``(low, high)`` reward range of the ``gym.Env`` interface.
    """

    def __init__(
        self,
        env: gym.Env,
        reference_std: float,
        std_multiplier: float = 1.0,
        clip_std: float = 1.96,
        seed: int | None = None,
    ):
        super().__init__(
            env, std_multiplier=std_multiplier, clip_std=clip_std, seed=seed
        )
        self.reference_std = reference_std

    def _reference_std(self, obs: np.ndarray, action: int) -> float:
        return float(self.reference_std)


def normalize_noise_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Maps legacy variance-multiplier reward-noise args to std multipliers.

    Records and in-flight configs written before the rename hold ``scale``
    (a multiplier on the reference variance) and ``reward_range`` (a
    reference standard deviation under a misleading name). Both are decoded
    to ``std_multiplier`` and ``reference_std``; the mapping is exact because
    ``sqrt(scale)`` with a variance-multiplier parameterization is the same
    number. Ambiguous or unknown keys raise instead of being ignored, since a
    config carrying both shapes would hide which one took effect.
    """
    legacy_names = {"scale": "std_multiplier", "reward_range": "reference_std"}
    known_names = {"std_multiplier", "reference_std", "clip_std", "seed"}
    normalized: dict[str, Any] = {}
    for name, value in args.items():
        if name in legacy_names:
            new_name = legacy_names[name]
            if new_name in args:
                raise TypeError(
                    f"reward noise args carry both `{name}` and `{new_name}`"
                )
            if name == "scale":
                if value < 0:
                    raise ValueError(
                        f"legacy `scale` must be non-negative, not {value}"
                    )
                normalized[new_name] = np.sqrt(value)
            else:
                normalized[new_name] = value
            continue
        if name not in known_names:
            raise TypeError(f"unknown reward noise arg `{name}`")
        normalized[name] = value
    return normalized


def wrap(env: gym.Env, wrapper: str | None = None, **kwargs):
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


def _grid_geodesic_distances(env: gym.Env) -> dict[tuple[int, int], int]:
    """Geodesic distance from every grid cell to its nearest exit.

    Cells no exit reaches (cliffs, and open cells walled off from every
    exit) take the largest finite distance, so the potential is defined
    on the whole observation space.
    """
    grid_env = env.unwrapped
    size = getattr(grid_env, "_size", None)
    cliffs = getattr(grid_env, "_cliffs", None)
    exits = getattr(grid_env, "_exits", None)
    if size is None or cliffs is None or exits is None:
        raise ValueError(
            f"{type(grid_env).__name__} exposes no grid geometry; "
            "distance shaping requires a gridworld environment"
        )
    grid = np.zeros(shape=size, dtype=np.int8)
    for position in cliffs:
        grid[position] = gridutils.CELL_CLIFF
    for position in exits:
        grid[position] = gridutils.CELL_GOAL
    num_cols = size[1]
    sources = [row * num_cols + col for row, col in exits]
    flat_distances = gridutils.grid_bfs_distances(grid=grid, sources=sources)
    finite = flat_distances[flat_distances >= 0]
    if finite.size == 0:
        raise ValueError("no grid cell reaches an exit")
    max_distance = int(finite.max())
    distances = {}
    for row in range(size[0]):
        for col in range(size[1]):
            distance = int(flat_distances[row * num_cols + col])
            distances[(row, col)] = max_distance if distance < 0 else distance
    return distances
