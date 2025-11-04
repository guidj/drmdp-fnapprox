from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from rlplg.environments import gridworld, iceworld, redgreen

from drmdp.envs import wrappers

DEFAULT_GW_GRID = ["oooooooooooo", "oooooooooooo", "oooooooooooo", "sxxxxxxxxxxg"]
DEFAULT_RG_CURE = ["red", "green", "red", "green", "wait", "green"]
DEFAULT_ICE_MAP = "4x4"
DEFAULT_MC_MAX_EPISODE_STEPS = 10000


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
            low=np.zeros(shape=2),
            dtype=np.int64,
        )

    def observation(self, observation: ObsType):
        return np.array(observation["agent"], dtype=np.int64)


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
            low=np.zeros(shape=2),
            dtype=np.int64,
        )

    def observation(self, observation: ObsType):
        return np.array(observation["agent"], dtype=np.int64)


class RedgreenObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            high=np.array([env.observation_space["pos"].n]),
            low=np.zeros(shape=1),
            dtype=np.int64,
        )

    def observation(self, observation: ObsType):
        return np.array([observation["pos"]], dtype=np.int64)


def make(env_name: str, wrapper: Optional[str] = None, **kwargs) -> gym.Env:
    if env_name == "GridWorld-v0":
        grid = kwargs.get("grid", DEFAULT_GW_GRID)
        size, cliffs, exits, start = gridworld.parse_grid_from_text(grid)
        env = GridWorldObsAsVectorWrapper(
            gridworld.GridWorld(size, cliffs, exits, start)
        )
        env = episode_steps_limit(env, kwargs.get("max_episode_steps", None))
    elif env_name == "RedGreen-v0":
        cure = kwargs.get("cure", DEFAULT_RG_CURE)
        env = RedgreenObsAsVectorWrapper(redgreen.RedGreenSeq(cure))
        env = episode_steps_limit(env, kwargs.get("max_episode_steps", None))
    elif env_name == "IceWorld-v0":
        map_name = kwargs.get("map_name", DEFAULT_ICE_MAP)
        map_ = iceworld.MAPS[map_name]
        size, lakes, goals, start = iceworld.parse_map_from_text(map_)
        env = IceworldObsAsVectorWrapper(
            iceworld.IceWorld(size, lakes=lakes, goals=goals, start=start)
        )
        env = episode_steps_limit(env, kwargs.get("max_episode_steps", None))
    elif env_name == "MountainCar-v0":
        max_episode_steps = kwargs.get(
            "max_episode_steps", DEFAULT_MC_MAX_EPISODE_STEPS
        )
        env = gym.make("MountainCar-v0", max_episode_steps=max_episode_steps)
    else:
        raise ValueError(f"Environment `{env_name}` unknown")
    return wrappers.wrap(env, wrapper=wrapper, **kwargs)


def episode_steps_limit(env: gym.Env, max_episode_steps: Optional[int] = None):
    """
    Applies a `TimeLimit` wrapper, if `max_episode_steps` is defined.
    """
    if max_episode_steps:
        return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
