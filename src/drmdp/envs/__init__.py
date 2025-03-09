from typing import Optional
import gymnasium as gym

from drmdp.envs import gem, gympg


def make(env_name: str, wrapper: Optional[str] = None, **kwargs) -> gym.Env:
    try:
        return gympg.make(env_name, wrapper=wrapper, **kwargs)
    except ValueError:
        return gem.make(env_name, wrapper=wrapper, **kwargs)
        