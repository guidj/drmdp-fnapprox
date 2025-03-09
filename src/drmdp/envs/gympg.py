import copy
import functools
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gym_electric_motor import reward_functions
from sklearn import mixture, model_selection
from rlplg.environments import gridworld, redgreen, iceworld
from drmdp import constants, data, hashtutils, tiles


DEFAULT_GRID = ["oooooooooooo", "oooooooooooo", "oooooooooooo", "sxxxxxxxxxxg"]
DEFAULT_CURE = ["red", "green", "red", "green", "wait", "green"]
DEFAULT_MAX_EPISODE_STEPS = 10000

class GridWorldObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            high=np.array(
                [
                    env.observation_space["agent"][0].n,
                    env.observation_space["agent"][1].n,
                ]
            ),
            low=np.zeros(shape=2, dtype=np.int32),
        )

    def observation(self, obs):
        return np.array(obs["agent"])
    

class IceworldObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            high=np.array(
                [
                    env.observation_space["agent"][0].n,
                    env.observation_space["agent"][1].n,
                ]
            ),
            low=np.zeros(shape=2, dtype=np.int32),
        )

    def observation(self, obs):
        return np.array(obs["agent"])


class RedgreenObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            high=np.array([env.observation_space["pos"].n]), low=np.array([0])
        )

    def observation(self, obs):
        return obs["pos"]


class TilesObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, tiling_dim: int, num_tilings: int = None):
        super().__init__(env)
        self.tiles = tiles.Tiles(
                env.observation_space.low,
                env.observation_space.high,
                tiling_dim=tiling_dim,
                num_tilings=num_tilings
            )
        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape=self.tiles.max_size, dtype=np.int32), 
            high=np.ones(shape=self.tiles.max_size, dtype=np.int32)
        )

    def observation(self, obs):
        return self.tiles(obs)        


def make(env_name: str, wrapper: Optional[str] = None, **kwargs) -> gym.Env:
    if env_name == "GridWorld-v0":
        grid = kwargs.get("grid", DEFAULT_GRID)
        size, cliffs, exits, start = gridworld.parse_grid_from_text(
            grid
        )
        env = GridWorldObsAsVectorWrapper(
            gridworld.GridWorld(size, cliffs, exits, start)
        )        
    elif env_name == "RedGreen-v0":
        cure = kwargs.get("cure", DEFAULT_CURE)
        env = RedgreenObsAsVectorWrapper(redgreen.RedGreen(cure))
    elif env_name == "IceWorld-v0":
        map_name = kwargs.get("map_name", "4x4")
        map = iceworld.MAPS[map_name]
        size, lakes, goals, start = iceworld.parse_map_from_text(map)
        env = IceworldObsAsVectorWrapper(
            iceworld.IceWorld(size, lakes=lakes, goals=goals, start=start)
        )
    elif env_name == "MountainCar-v0":
        max_episode_steps = kwargs.get("max_episode_steps", DEFAULT_MAX_EPISODE_STEPS)
        env = gym.make("MountainCar-v0", max_episode_steps=max_episode_steps)
    else:
        raise ValueError(f"Environment `{env_name}` unknown")

    if wrapper is None:
        return env
    if wrapper == constants.SCALE:
        return (env)
    elif wrapper == constants.GAUSSIAN_MIX:
        return (env)
    elif wrapper == constants.TILES:
        return (env)
    raise ValueError(f"Wrapper `{wrapper}` unknown")
