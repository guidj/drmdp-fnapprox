import abc
import math
from typing import Any, Dict, Hashable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from sklearn import mixture

from drmdp import constants, dataproc, mathutils, tiles


class FeatTransform(abc.ABC):
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


class RandomBinaryFeatTransform(FeatTransform):
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
        array = np.array(observation, dtype=np.int64)
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
        obs_arrays = [np.array(obs, dtype=np.int64) for obs in observations]
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


class ScaleFeatTransform(FeatTransform):
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
            idx = self.obs_dim * action
            output[i, idx : idx + self.obs_dim] = obs

        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim * self.num_actions  # type: ignore


class GaussianMixFeatTransform(FeatTransform):
    def __init__(
        self, env: gym.Env, params, sample_steps: int = constants.DEFAULT_GM_STEPS
    ):
        # params or hps_params can be provided
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self._gm = mixture.GaussianMixture(**params)
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


class TileFeatTransform(FeatTransform):
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

        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions
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
        idx = self._tiles(obs_scaled_01, action)
        output[idx] = 1
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
            idx = self._tiles(obs_scaled_01[i], actions[i])
            output[i, idx] = 1
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
            ints=[action] if action is not None else [],
        )


def create_feat_transformer(env: gym.Env, name: str, **kwargs):
    if name == constants.RANDOM:
        return RandomBinaryFeatTransform(env, **kwargs)
    if name == constants.SCALE:
        return ScaleFeatTransform(env)
    if name == constants.GAUSSIAN_MIX:
        return GaussianMixFeatTransform(env, **kwargs)
    if name == constants.TILES:
        return TileFeatTransform(env, **kwargs)
    raise ValueError(f"FeatTransformer `{name}` unknown")
