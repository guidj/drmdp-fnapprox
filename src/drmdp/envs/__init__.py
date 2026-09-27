import gymnasium as gym

from drmdp.envs import gem, gympg


def make(env_name: str, wrapper: str | None = None, **kwargs) -> gym.Env:
    """
    Create a supported environments.
    """
    try:
        return gympg.make(env_name, wrapper=wrapper, **kwargs)
    except ValueError:
        return gem.make(env_name, wrapper=wrapper, **kwargs)
