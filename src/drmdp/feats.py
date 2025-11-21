import abc
import math
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from sklearn import mixture

from drmdp import constants, dataproc, mathutils, tiles


class FeatTransform(abc.ABC):
    """
    Encodes state-actions into a single vector output.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def transform(self, observation: ObsType, action: ActType):
        pass

    @abc.abstractmethod
    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        pass

    @property
    @abc.abstractmethod
    def output_shape(self) -> int:
        pass


class IdentityFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `n` x num_actions.
    - The observation is placed in a position corresponding to the action taken.
    """

    def __init__(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"env.observation_space must be `spaces.Box`. Got {env.observation_space}",
                env,
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"env.action_space must be `spaces.Discrete`. Got {env.action_space}",
                env,
            )

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = np.size(self.obs_space.high)

    def transform(self, observation: ObsType, action: ActType):
        output = np.zeros(shape=self.obs_dim * self.num_actions)
        idx = self.obs_dim * action
        output[idx : idx + self.obs_dim] = observation
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        batch_size = len(observations)
        output = np.zeros((batch_size, self.obs_dim * self.num_actions))

        # make obs an array if necessary
        obs = np.asarray(observations)
        # Fill output array using loop
        for i, (obs, action) in enumerate(zip(obs, actions)):
            idx = self.obs_dim * action
            output[i, idx : idx + self.obs_dim] = obs

        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim * self.num_actions  # type: ignore


class FiniteIdentityFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `n`.
    """

    def __init__(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"env.observation_space must be `spaces.Box`. Got {env.observation_space}",
                env,
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"env.action_space must be `spaces.Discrete`. Got {env.action_space}",
                env,
            )

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = np.size(self.obs_space.high)

    def transform(self, observation: ObsType, action: ActType):
        del action
        return observation

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        batch_size = len(observations)
        output = np.zeros((batch_size, self.obs_dim))

        # make obs an array if necessary
        obs = np.asarray(observations)
        # Fill output array using loop
        for i, obs in enumerate(obs):
            output[i, :] = obs
        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim  # type: ignore


class RandomBinaryFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `n` x num_actions.
    - Each observation is a discrete value that gets mapped to a
    random binary vector.
    - The random vector is placed in the output array in a position
    corresponding to the action.
    """

    def __init__(self, env: gym.Env, enc_size: int):
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")
        if not isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Discrete)):
            raise ValueError(
                f"Environment space must be either Box or Discrete, not {type(env.observation_space)}"
            )
        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = enc_size
        self._representations: Dict[Hashable, Any] = {}

    def transform(self, observation: ObsType, action: ActType):
        array = np.asarray(observation, dtype=np.int64)
        key: Hashable = -1
        if np.shape(array) == ():
            key = array.item()
        else:
            key = tuple(array.tolist())
        if key not in self._representations:
            indices = np.random.randint(
                0,
                high=self.obs_dim,
                size=math.floor(self.obs_dim / 2),
            )
            vec = np.zeros(shape=self.obs_dim)
            vec[indices] = 1
            self._representations[key] = vec
        output = np.zeros(shape=self.obs_dim * self.num_actions)
        pos = self.obs_dim * action
        output[pos : pos + self.obs_dim] = self._representations[key]
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        batch_size = len(observations)
        output = np.zeros((batch_size, self.obs_dim * self.num_actions))

        # Convert observations to keys for lookup
        obs_arrays = [np.asarray(obs, dtype=np.int64) for obs in observations]
        keys = [
            arr.item() if np.shape(arr) == () else tuple(arr.tolist())
            for arr in obs_arrays
        ]

        # Generate representations for any new keys
        new_keys = set(keys) - self._representations.keys()
        for key in new_keys:
            indices = np.random.randint(
                0, high=self.obs_dim, size=math.floor(self.obs_dim / 2)
            )
            vec = np.zeros(shape=self.obs_dim)
            vec[indices] = 1
            self._representations[key] = vec

        # Fill output array
        for i, (key, action) in enumerate(zip(keys, actions)):
            pos = self.obs_dim * action
            output[i, pos : pos + self.obs_dim] = self._representations[key]

        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim * self.num_actions  # type: ignore


class FiniteRandomBinaryFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `n` x num_actions.
    - Each observation is a discrete value that gets mapped to a
    random binary vector.
    - The random vector is placed in the output array in a position
    corresponding to the action.
    """

    def __init__(self, env: gym.Env, enc_size: int):
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")
        if not isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Discrete)):
            raise ValueError(
                f"Environment space must be either Box or Discrete, not {type(env.observation_space)}"
            )
        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = enc_size
        self._representations: Dict[Hashable, Any] = {}

    def transform(self, observation: ObsType, action: ActType):
        del action
        array = np.asarray(observation, dtype=np.int64)
        key: Hashable = -1
        if np.shape(array) == ():
            key = array.item()
        else:
            key = tuple(array.tolist())
        if key not in self._representations:
            indices = np.random.randint(
                0,
                high=self.obs_dim,
                size=math.floor(self.obs_dim / 2),
            )
            vec = np.zeros(shape=self.obs_dim)
            vec[indices] = 1
            self._representations[key] = vec
        return self._representations[key]

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        del actions
        batch_size = len(observations)
        output = np.zeros((batch_size, self.obs_dim))

        # Convert observations to keys for lookup
        obs_arrays = [np.asarray(obs, dtype=np.int64) for obs in observations]
        keys = [
            arr.item() if np.shape(arr) == () else tuple(arr.tolist())
            for arr in obs_arrays
        ]

        # Generate representations for any new keys
        new_keys = set(keys) - self._representations.keys()
        for key in new_keys:
            indices = np.random.randint(
                0, high=self.obs_dim, size=math.floor(self.obs_dim / 2)
            )
            vec = np.zeros(shape=self.obs_dim)
            vec[indices] = 1
            self._representations[key] = vec

        # Fill output array
        for i, key in enumerate(keys):
            output[i, :] = self._representations[key]
        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim  # type: ignore


class ScaleFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `n` x num_actions.
    - Each observation scaled according to its bounds so each
    dimension has a value betwen [0, 1].
    - The scaled vector is placed in the output array in a position
    corresponding to the action.
    """

    def __init__(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"env.observation_space must be `spaces.Box`. Got {env.observation_space}",
                env,
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"env.action_space must be `spaces.Discrete`. Got {env.action_space}",
                env,
            )

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = np.size(self.obs_space.high)
        self.obs_range = self.obs_space.high - self.obs_space.low

    def transform(self, observation: ObsType, action: ActType):
        obs_scaled_01 = (observation - self.obs_space.low) / self.obs_range
        output = np.zeros(shape=self.obs_dim * self.num_actions)
        idx = self.obs_dim * action
        output[idx : idx + self.obs_dim] = obs_scaled_01
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        batch_size = len(observations)
        output = np.zeros((batch_size, self.obs_dim * self.num_actions))

        # Scale all observations at once
        obs_scaled_01 = (np.asarray(observations) - self.obs_space.low) / self.obs_range

        # Fill output array using loop
        for i, (obs, action) in enumerate(zip(obs_scaled_01, actions)):
            start_index = self.obs_dim * action
            output[i, start_index : start_index + self.obs_dim] = obs

        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim * self.num_actions  # type: ignore


class FiniteScaleFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `n`.
    - Each observation scaled according to its bounds so each
    dimension has a value betwen [0, 1].
    """

    def __init__(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"env.observation_space must be `spaces.Box`. Got {env.observation_space}",
                env,
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"env.action_space must be `spaces.Discrete`. Got {env.action_space}",
                env,
            )

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = np.size(self.obs_space.high)
        self.obs_range = self.obs_space.high - self.obs_space.low

    def transform(self, observation: ObsType, action: ActType):
        del action
        obs_scaled_01 = (observation - self.obs_space.low) / self.obs_range
        return obs_scaled_01

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        del actions
        # Scale all observations at once
        obs_scaled_01 = (np.asarray(observations) - self.obs_space.low) / self.obs_range
        return obs_scaled_01

    @property
    def output_shape(self) -> int:
        return self.obs_dim  # type: ignore


class GaussianMixFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation, returns an observation
     of size `num_mixtures` x num_actions.
    - `num_mixtures` is learned through grid-search,
    using Expectation-Maximisation.
    - Each observation projected into a basis vectors, where each component
     represents probability of membership to a distribution.
    - The basis membership vector is placed in the output array in a position
    corresponding to the action.
    """

    def __init__(
        self, env: gym.Env, sample_steps: int = constants.DEFAULT_GM_STEPS, **kwargs
    ):
        # params or hps_params can be provided
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self._gm = mixture.GaussianMixture(**kwargs, init_params="k-means++")
        self._gm.fit(
            [tup[0] for tup in dataproc.collection_traj_data(env, steps=sample_steps)]
        )
        self.obs_dim = self._gm.n_components

    def transform(self, observation: ObsType, action: ActType):
        output = np.zeros(shape=self.obs_dim * self.num_actions)
        idx = self.obs_dim * action
        output[idx : idx + self.obs_dim] = self._gm.predict_proba([observation])[0]
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        """Transform multiple observations and actions at once.

        Args:
            observations: Array of shape (batch_size, obs_dim) containing observations
            actions: Array of shape (batch_size,) containing discrete actions

        Returns:
            Array of shape (batch_size, obs_dim * num_actions) containing transformed features
        """
        batch_size = len(observations)
        # Pre-allocate output array
        output = np.zeros((batch_size, self.obs_dim * self.num_actions))

        # Get all probabilities at once
        all_probs = self._gm.predict_proba(observations)

        # Calculate indices for each action
        start_indices = actions * self.obs_dim

        # Assign probabilities for each observation-action pair
        for i in range(batch_size):
            start_idx = start_indices[i]
            output[i, start_idx : start_idx + self.obs_dim] = all_probs[i]
        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim * self.num_actions  # type: ignore


class FiniteGaussianMixFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation, returns an observation
     of size `num_mixtures` x num_actions.
    - `num_mixtures` is learned through grid-search,
    using Expectation-Maximisation.
    - Each observation projected into a basis vectors, where each component
     represents probability of membership to a distribution.
    - The basis membership vector is the representation.
    """

    def __init__(
        self, env: gym.Env, sample_steps: int = constants.DEFAULT_GM_STEPS, **kwargs
    ):
        # params or hps_params can be provided
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self._gm = mixture.GaussianMixture(**kwargs, init_params="k-means++")
        self._gm.fit(
            [tup[0] for tup in dataproc.collection_traj_data(env, steps=sample_steps)]
        )
        self.obs_dim = self._gm.n_components

    def transform(self, observation: ObsType, action: ActType):
        del action
        return self._gm.predict_proba([observation])[0]

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        """Transform multiple observations and actions at once.

        Args:
            observations: Array of shape (batch_size, obs_dim) containing observations
            actions: Array of shape (batch_size,) containing discrete actions

        Returns:
            Array of shape (batch_size, obs_dim * num_actions) containing transformed features
        """
        del actions
        # Get all probabilities at once
        return self._gm.predict_proba(observations)

    @property
    def output_shape(self) -> int:
        return self.obs_dim  # type: ignore


class TileFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `tiling_dim` x `num_tilings` x `num_actions`.
    - Each observation is tiled, which yields a vector of binary values. Tiles
    are coarse, overlapping representations.
    - The tile vector already encodes both the state and action.
    """

    def __init__(
        self,
        env: gym.Env,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
        hash_dim: Optional[int] = None,
    ):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.tiling_dim = tiling_dim
        self.wrapwidths = [tiling_dim] * np.size(self.obs_space.low)
        self.num_actions = env.action_space.n

        # For best results,
        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions.
        self.num_tilings = num_tilings or tiles.pow2geq(np.size(self.obs_space.low) * 4)
        self.max_size = (
            (tiling_dim ** np.size(self.obs_space.low))
            * self.num_tilings
            * self.num_actions
        )
        self.obs_dim = -1
        self.iht = tiles.IHT(self.max_size)
        self.hash_dim = hash_dim if hash_dim and self.max_size > hash_dim else None
        self.obs_range = self.obs_space.high - self.obs_space.low

    def transform(self, observation: ObsType, action: ActType):
        # Convert observations to numpy array if not already
        obs_scaled_01 = (np.asarray(observation) - self.obs_space.low) / self.obs_range
        output = np.zeros(shape=self.max_size)
        indices = self._tiles(obs_scaled_01, action)
        output[indices] = 1
        if self.hash_dim:
            return mathutils.hashtrick(output, self.hash_dim)
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        # Convert observations to numpy array if not already
        observations = np.asarray(observations)  # type: ignore

        # Scale observations to [0,1] range
        obs_scaled_01 = (observations - self.obs_space.low) / self.obs_range
        batch_size = len(observations)
        output = np.zeros((batch_size, self.max_size))
        hashed_output = np.zeros((batch_size, self.hash_dim)) if self.hash_dim else None

        # Get tile indices for each observation-action pair
        for i in range(batch_size):
            indices = self._tiles(obs_scaled_01[i], actions[i])
            output[i, indices] = 1
            if self.hash_dim:
                hashed_output[i] = mathutils.hashtrick(output[i], self.hash_dim)  # type:ignore
        return hashed_output if self.hash_dim else output

    @property
    def output_shape(self) -> int:
        return self.hash_dim or self.max_size

    def _tiles(self, scaled_obs: np.ndarray, action: ActType):
        return tiles.tileswrap(
            self.iht,
            numtilings=self.num_tilings,
            floats=scaled_obs * self.tiling_dim,  # type: ignore
            wrapwidths=self.wrapwidths,
            ints=[action] if action is not None else (),
        )


class FiniteTileFeatTransform(FeatTransform):
    """
    Transformation:
    - Given an input observation of size n, returns an observation
     of size `tiling_dim` x `num_tilings`.
    - Each observation is tiled, which yields a vector of binary values. Tiles
    are coarse, overlapping representations.
    - The tile vector encodes the state.
    """

    def __init__(
        self,
        env: gym.Env,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
        hash_dim: Optional[int] = None,
    ):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.tiling_dim = tiling_dim
        self.wrapwidths = [tiling_dim] * np.size(self.obs_space.low)
        self.num_actions = env.action_space.n

        # For best results,
        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions.
        self.num_tilings = num_tilings or tiles.pow2geq(np.size(self.obs_space.low) * 4)
        self.max_size = (tiling_dim ** np.size(self.obs_space.low)) * self.num_tilings
        self.obs_dim = -1
        self.iht = tiles.IHT(self.max_size)
        self.hash_dim = hash_dim if hash_dim and self.max_size > hash_dim else None
        self.obs_range = self.obs_space.high - self.obs_space.low

    def transform(self, observation: ObsType, action: ActType):
        # Convert observations to numpy array if not already
        obs_scaled_01 = (np.asarray(observation) - self.obs_space.low) / self.obs_range
        output = np.zeros(shape=self.max_size)
        indices = self._tiles(obs_scaled_01, action)
        output[indices] = 1
        if self.hash_dim:
            return mathutils.hashtrick(output, self.hash_dim)
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        # Convert observations to numpy array if not already
        observations = np.asarray(observations)  # type: ignore

        # Scale observations to [0,1] range
        obs_scaled_01 = (observations - self.obs_space.low) / self.obs_range
        batch_size = len(observations)
        output = np.zeros((batch_size, self.max_size))
        hashed_output = np.zeros((batch_size, self.hash_dim)) if self.hash_dim else None

        # Get tile indices for each observation-action pair
        for i in range(batch_size):
            indices = self._tiles(obs_scaled_01[i], actions[i])
            output[i, indices] = 1
            if self.hash_dim:
                hashed_output[i] = mathutils.hashtrick(output[i], self.hash_dim)  # type:ignore
        return hashed_output if self.hash_dim else output

    @property
    def output_shape(self) -> int:
        return self.hash_dim or self.max_size

    def _tiles(self, scaled_obs: np.ndarray, action: ActType):
        del action
        return tiles.tileswrap(
            self.iht,
            numtilings=self.num_tilings,
            floats=scaled_obs * self.tiling_dim,  # type: ignore
            wrapwidths=self.wrapwidths,
            ints=(),
        )


class ActionSplicedTileFeatTransform(FeatTransform):
    """
    - Given an input observation of size n, returns an observation
     of size `tiling_dim` x `num_tilings` x `num_actions`.
    - Each observation is tiled, which yields a vector of binary values. Tiles
    are coarse, overlapping representations.
    - Each action uses it's own separate tiling table to handle
    collisions.
    - The tile vector is placed in the output array in a position
    corresponding to the action.
    """

    def __init__(
        self,
        env: gym.Env,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
        hash_dim: Optional[int] = None,
    ):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.tiling_dim = tiling_dim
        self.wrapwidths = [tiling_dim] * np.size(self.obs_space.low)
        self.num_actions = env.action_space.n

        # For best results,
        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions.
        self.num_tilings = num_tilings or tiles.pow2geq(np.size(self.obs_space.low) * 4)
        self.max_size = (tiling_dim ** np.size(self.obs_space.low)) * self.num_tilings
        self.obs_dim = -1
        self.ihts = {
            action: tiles.IHT(self.max_size) for action in range(self.num_actions)
        }
        self.hash_dim = hash_dim if hash_dim and self.max_size > hash_dim else None
        self.obs_range = self.obs_space.high - self.obs_space.low

    def transform(self, observation: ObsType, action: ActType):
        # Convert observations to numpy array if not already
        obs_scaled_01 = (np.asarray(observation) - self.obs_space.low) / self.obs_range
        obs_tiled = np.zeros(shape=self.max_size)
        indices = self._tiles(obs_scaled_01, action)
        obs_tiled[indices] = 1
        if self.hash_dim:
            obs_hashed = mathutils.hashtrick(obs_tiled, self.hash_dim)
            output = np.zeros(shape=self.hash_dim * self.num_actions)
            start_index = self.hash_dim * action
            output[start_index : start_index + self.hash_dim] = obs_hashed
            return output

        output = np.zeros(shape=self.max_size * self.num_actions)
        start_index = self.max_size * action
        output[start_index : start_index + self.max_size] = obs_tiled
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        # Convert observations to numpy array if not already
        observations = np.asarray(observations)  # type: ignore

        # Scale observations to [0,1] range
        obs_scaled_01 = (observations - self.obs_space.low) / self.obs_range
        batch_size = len(observations)
        obs_tiled = np.zeros((batch_size, self.max_size))
        hashed_output = np.zeros((batch_size, self.hash_dim)) if self.hash_dim else None
        output_size = self.hash_dim or self.max_size
        output = np.zeros((batch_size, output_size * self.num_actions))
        # Get tile indices for each observation-action pair
        for i in range(batch_size):
            indices = self._tiles(obs_scaled_01[i], actions[i])
            obs_tiled[i, indices] = 1
            if self.hash_dim:
                hashed_output[i] = mathutils.hashtrick(obs_tiled[i], self.hash_dim)  # type:ignore
                start_index = self.hash_dim * actions[i]
                output[i, start_index : start_index + self.hash_dim] = hashed_output[i]  # type: ignore
            else:
                start_index = self.max_size * actions[i]
                output[i, start_index : start_index + self.max_size] = obs_tiled[i]
        return output

    @property
    def output_shape(self) -> int:
        return (self.hash_dim or self.max_size) * self.num_actions  # type: ignore

    def _tiles(self, scaled_obs: np.ndarray, action: ActType):
        return tiles.tileswrap(
            self.ihts[action],
            numtilings=self.num_tilings,
            floats=scaled_obs * self.tiling_dim,  # type: ignore
            wrapwidths=self.wrapwidths,
        )


class FiniteActionSplicedTileFeatTransform(FeatTransform):
    """
    - Given an input observation of size n, returns an observation
     of size `tiling_dim` x `num_tilings` + `num_actions`.
    - Each observation is tiled, which yields a vector of binary values. Tiles
    are coarse, overlapping representations.
    - Each action uses it's own separate tiling table to handle
    collisions.
    - The tile vector is placed in the output array in a position
    corresponding to the action.
    """

    def __init__(
        self,
        env: gym.Env,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
        hash_dim: Optional[int] = None,
    ):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.tiling_dim = tiling_dim
        self.wrapwidths = [tiling_dim] * np.size(self.obs_space.low)
        self.num_actions = env.action_space.n

        # For best results,
        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions.
        self.num_tilings = num_tilings or tiles.pow2geq(np.size(self.obs_space.low) * 4)
        self.max_size = (tiling_dim ** np.size(self.obs_space.low)) * self.num_tilings
        self.obs_dim = -1
        self.ihts = {
            action: tiles.IHT(self.max_size) for action in range(self.num_actions)
        }
        self.hash_dim = hash_dim if hash_dim and self.max_size > hash_dim else None
        self.obs_range = self.obs_space.high - self.obs_space.low

    def transform(self, observation: ObsType, action: ActType):
        # Convert observations to numpy array if not already
        obs_scaled_01 = (np.asarray(observation) - self.obs_space.low) / self.obs_range
        obs_tiled = np.zeros(shape=self.max_size)
        indices = self._tiles(obs_scaled_01, action)
        obs_tiled[indices] = 1
        action_ohe = np.zeros(self.num_actions)
        action_ohe[action] = 1
        if self.hash_dim:
            obs_hashed = mathutils.hashtrick(obs_tiled, self.hash_dim)
            return np.concatenate([obs_hashed, action_ohe])
        return np.concatenate([obs_tiled, action_ohe])

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        # Convert observations to numpy array if not already
        observations = np.asarray(observations)  # type: ignore

        # Scale observations to [0,1] range
        obs_scaled_01 = (observations - self.obs_space.low) / self.obs_range
        batch_size = len(observations)
        obs_tiled = np.zeros((batch_size, self.max_size))
        hashed_output = np.zeros((batch_size, self.hash_dim)) if self.hash_dim else None
        output_size = self.hash_dim or self.max_size
        output = np.zeros((batch_size, output_size + self.num_actions))
        # Get tile indices for each observation-action pair
        for i in range(batch_size):
            indices = self._tiles(obs_scaled_01[i], actions[i])
            obs_tiled[i, indices] = 1
            action_ohe = np.zeros(self.num_actions)
            action_ohe[actions[i]] = 1
            if self.hash_dim:
                hashed_output[i] = mathutils.hashtrick(obs_tiled[i], self.hash_dim)  # type:ignore
                output[i, :] = np.concatenate([hashed_output[i], action_ohe])  # type: ignore
            else:
                output[i, :] = np.concatenate([obs_tiled[i], action_ohe])
        return output

    @property
    def output_shape(self) -> int:
        return (self.hash_dim or self.max_size) + self.num_actions  # type: ignore

    def _tiles(self, scaled_obs: np.ndarray, action: ActType):
        return tiles.tileswrap(
            self.ihts[action],
            numtilings=self.num_tilings,
            floats=scaled_obs * self.tiling_dim,  # type: ignore
            wrapwidths=self.wrapwidths,
        )


class FlatGridCoorFeatTransform(FeatTransform):
    """
    - Given an input observation comprised of a tuple of integers,
    returns a integer.
    - This integer is placed in a vector of dim `num_actions`.
    - If OHE is true, the value is OHE into a vector of dim `num_states`,
    then placed into vector of dim `num_actions`x`num_states`
    """

    def __init__(self, env: gym.Env, ohe: bool = False):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Source `env` observation space must be of type `Box`. Got {type(env.observation_space)}"
            )
        if np.size(env.observation_space.shape) != 1:
            raise ValueError(
                f"Env should be a 1D array. Got {env.observation_space.shape}"
            )

        self.ohe = ohe
        self.obs_space = env.observation_space
        self.nactions: int = env.action_space.n
        self.obs_dims = (
            self.obs_space.shape[0]
            if isinstance(env.observation_space.shape, Sequence)
            else self.obs_space.shape
        )

        value_ranges = env.observation_space.high - env.observation_space.low
        self.value_ranges = np.array(value_ranges, dtype=np.int64)
        if np.sum(value_ranges - self.value_ranges) != 0:
            raise ValueError(
                f"Bad value range: {env.observation_space}. Make sure all values are integers."
            )
        # num coordinates
        self.nstates = int(np.prod(self.value_ranges))
        # Cache op
        self.value_range_prod = [
            np.prod(self.value_ranges[idx + 1 :]) for idx in range(self.obs_dims)
        ]

    def transform(self, observation: ObsType, action: ActType):
        xs = np.concatenate([observation, [1]])
        pos = 0
        for idx in range(self.obs_dims):
            pos += xs[idx] * self.value_range_prod[idx]
        pos = int(pos)
        output: np.ndarray = np.zeros(1)

        if self.ohe:
            output = np.zeros(shape=(self.nactions, self.nstates))
            output[action, pos] = 1
            return output.flatten()
        output = np.zeros(shape=self.nactions)
        output[action] = pos
        return output

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        batch_size = len(observations)
        output = np.zeros((batch_size, self.output_shape))

        # make obs an array if necessary
        obs = np.asarray(observations)
        # Fill output array using loop
        for i, (obs, action) in enumerate(zip(obs, actions)):
            # idx = self.obs_dim * action
            output[i] = self.transform(obs, action)
        return output

    @property
    def output_shape(self) -> int:
        return self.nstates * self.nactions if self.ohe else self.nactions


class FiniteFlatGridCoorFeatTransform(FeatTransform):
    """
    - Given an input observation comprised of a tuple of integers,
    returns a integer.
    - If OHE is true, the value is OHE.
    """

    def __init__(self, env: gym.Env, ohe: bool = False):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Source `env` observation space must be of type `Box`. Got {type(env.observation_space)}"
            )
        if np.size(env.observation_space.shape) != 1:
            raise ValueError(
                f"Env should be a 1D array. Got {env.observation_space.shape}"
            )

        self.ohe = ohe
        self.obs_space = env.observation_space
        self.nactions: int = env.action_space.n
        self.obs_dims = (
            self.obs_space.shape[0]
            if isinstance(env.observation_space.shape, Sequence)
            else self.obs_space.shape
        )

        value_ranges = env.observation_space.high - env.observation_space.low
        self.value_ranges = np.array(value_ranges, dtype=np.int64)
        if np.sum(value_ranges - self.value_ranges) != 0:
            raise ValueError(
                f"Bad value range: {env.observation_space}. Make sure all values are integers."
            )
        # num coordinates
        self.nstates = int(np.prod(self.value_ranges))
        # Cache op
        self.value_range_prod = [
            np.prod(self.value_ranges[idx + 1 :]) for idx in range(self.obs_dims)
        ]

    def transform(self, observation: ObsType, action: ActType):
        del action
        xs = np.concatenate([observation, [1]])
        pos = 0
        for idx in range(self.obs_dims):
            pos += xs[idx] * self.value_range_prod[idx]
        pos = int(pos)

        if self.ohe:
            output = np.zeros(shape=(self.nstates))
            output[pos] = 1
            return output
        return np.array(pos)

    def batch_transform(
        self, observations: Sequence[ObsType], actions: Sequence[ActType]
    ):
        batch_size = len(observations)
        output = np.zeros((batch_size, self.output_shape))

        # make obs an array if necessary
        obs = np.asarray(observations)
        # Fill output array using loop
        for i, (obs, action) in enumerate(zip(obs, actions)):
            # idx = self.obs_dim * action
            output[i] = self.transform(obs, action)
        return output

    @property
    def output_shape(self) -> int:
        return self.nstates if self.ohe else 1


def create_feat_transformer(env: gym.Env, name: str, args: Mapping[str, Any]):
    """
    Creates a FeatTransformer according to the spec.
    """
    constructors: Mapping[str, Callable[..., FeatTransform]] = {
        constants.IDENTITY: IdentityFeatTransform,
        constants.FINITE_IDENTITY: FiniteIdentityFeatTransform,
        constants.RANDOM_VEC: RandomBinaryFeatTransform,
        constants.FINITE_RANDOM_VEC: FiniteRandomBinaryFeatTransform,
        constants.SCALE: ScaleFeatTransform,
        constants.FINITE_SCALE: FiniteScaleFeatTransform,
        constants.GAUSSIAN_MIX: GaussianMixFeatTransform,
        constants.FINITE_GAUSSIAN_MIX: FiniteGaussianMixFeatTransform,
        constants.TILES: TileFeatTransform,
        constants.FINITE_TILES: FiniteTileFeatTransform,
        constants.SPLICED_TILES: ActionSplicedTileFeatTransform,
        constants.FINITE_SPLICED_TILES: FiniteActionSplicedTileFeatTransform,
        constants.FLAT_GRID_COORD: FlatGridCoorFeatTransform,
        constants.FINITE_FLAT_GRID_COORD: FiniteFlatGridCoorFeatTransform,
    }

    if name not in constructors:
        raise ValueError(f"FeatTransformer `{name}` unknown")
    constructor = constructors[name]
    return constructor(env, **(args or {}))
