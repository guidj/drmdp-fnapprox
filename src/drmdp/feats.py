import abc
import math
from typing import Any

import gymnasium as gym
import numpy as np
from sklearn import mixture

from drmdp import constants, data, tiles


class FeatTransform(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def transform(self, obs: Any, action: Any):
        pass

    @property
    @abc.abstractmethod
    def output_shape(self) -> int:
        pass


class TileTransform(FeatTransform):
    def __init__(
        self,
        env: gym.Env,
        tiling_dim: int,
        num_tilings: int = None,
        hash_dim: int = None,
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
        print("Num tilings", self.num_tilings, "\n", "Flat dim:", self.max_size)
        self.iht = tiles.IHT(self.max_size)
        self.hash_dim = hash_dim if self.max_size > hash_dim else None

    def transform(self, obs: Any, action: Any):
        obs_scaled_01 = (obs - self.obs_space.low) / (
            self.obs_space.high - self.obs_space.low
        )
        output = np.zeros(shape=self.max_size)
        idx = tiles.tileswrap(
            self.iht,
            numtilings=self.num_tilings,
            floats=obs_scaled_01 * self.tiling_dim,
            wrapwidths=self.wrapwidths,
            ints=[action] if action else [],
        )
        output[idx] = 1
        if self.hash_dim:
            return hashtrick(output, self.hash_dim)
        return output

    @property
    def output_shape(self) -> int:
        return self.hash_dim or self.max_size


class ScaleObsOheActTransform(FeatTransform):
    def __init__(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("env.observation_space must be `spaces.Box`")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")

        self.obs_space = env.observation_space
        self.num_actions = env.action_space.n
        self.obs_dim = np.size(self.obs_space.high)

    def transform(self, obs: Any, action: Any):
        obs_scaled_01 = (obs - self.obs_space.low) / (
            self.obs_space.high - self.obs_space.low
        )
        output = np.zeros(shape=self.obs_dim * self.num_actions)
        idx = np.size(self.obs_space.high) * action
        output[idx : idx + self.obs_dim] = obs_scaled_01
        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim * self.num_actions


class GaussianMixObsOheActTransform(FeatTransform):
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
            [obs for obs, _, _ in data.collection_traj_data(env, steps=sample_steps)]
        )
        self.obs_dim = self._gm.n_components

    def transform(self, obs: Any, action: Any):
        output = np.zeros(shape=self.obs_dim * self.num_actions)
        idx = self.obs_dim * action
        output[idx : idx + self.obs_dim] = self._gm.predict_proba([obs])[0]
        return output

    @property
    def output_shape(self) -> int:
        return self.obs_dim * self.num_actions


class RndBinaryTransform(FeatTransform):
    def __init__(self, env: gym.Env, enc_size: int):
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("env.action_space must be `spaces.Discrete`")
        self.obs_space = env.observation_space
        if isinstance(self.obs_space, gym.spaces.Discrete):
            self.num_states = self.obs_space.n
        elif isinstance(self.obs_space, gym.spaces.Box):
            (self.obs_space.high - self.obs_space.low).item()
        else:
            raise ValueError("Unsupport observation_space. Must be Box[1d] or Discrete")
        self.num_actions = env.action_space.n
        self.enc_size = enc_size
        self._idx = [
            np.array(xs)
            for xs in np.random.randint(
                0,
                high=self.enc_size,
                size=(self.num_states, math.floor(self.enc_size / 2)),
            )
        ]

    def transform(self, obs: Any, action: Any):
        output = np.zeros(shape=self.enc_size * self.num_actions)
        idx = self.enc_size * action
        output[self._idx[obs] + idx] = 1
        return output

    @property
    def output_shape(self) -> int:
        return self.enc_size * self.num_actions


def hashtrick(xs, dim: int):
    ys = np.zeros(dim, dtype=np.int32)
    (idx,) = np.where(xs == 1)
    for i in idx:
        ys[i % dim] += 1
    return ys
