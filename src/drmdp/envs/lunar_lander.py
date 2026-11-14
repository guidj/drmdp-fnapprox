"""
LunarLander-v3 with a reward-linear observation.

The environment's per-step reward is a difference of the potential

$$\\Phi(s) = -100\\lVert(s_0,s_1)\rVert_2 - 100\\lVert(s_2,s_3)\rVert_2
- 100\\lvert s_4\rvert + 10s_6 + 10s_7$$

plus a fuel cost, so it is not a function of the observation the environment
emits. `LunarLanderRewardObs` appends the potential's five terms, which puts
the reward on a linear map of the observation: the appended basis carries the
magnitude terms and the raw block carries the signs the basis discards.

The appended block precedes the raw observation so that the reward basis is
read as a fixed-width prefix.
"""

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

REWARD_BLOCK_SIZE = 5
RAW_OBSERVATION_SIZE = 8
RAW_OBSERVATION_OFFSET = REWARD_BLOCK_SIZE
OBSERVATION_SIZE = REWARD_BLOCK_SIZE + RAW_OBSERVATION_SIZE
ACTION_COUNT = 4
DEFAULT_MAX_EPISODE_STEPS = 1000


def reward_basis(observation: ObsType) -> np.ndarray:
    r"""The five potential terms $\psi(s)$ of a raw LunarLander observation."""
    state = np.asarray(observation, dtype=np.float64)
    if state.shape != (RAW_OBSERVATION_SIZE,):
        raise ValueError(
            f"expected a raw observation of shape {(RAW_OBSERVATION_SIZE,)}, "
            f"got {state.shape}"
        )
    return np.array(
        [
            np.linalg.norm(state[0:2]),
            np.linalg.norm(state[2:4]),
            abs(state[4]),
            state[6],
            state[7],
        ],
        dtype=np.float64,
    )


def observation_space(raw_space: gym.spaces.Box) -> gym.spaces.Box:
    """Bounds of the reward basis followed by the raw observation.

    Each norm is bounded by the norm of its coordinate bounds, the angle
    magnitude by the angle bound, and the leg indicators keep the raw leg
    bounds because the basis copies them.
    """
    low = np.asarray(raw_space.low, dtype=np.float32)
    high = np.asarray(raw_space.high, dtype=np.float32)
    if low.shape != (RAW_OBSERVATION_SIZE,):
        raise ValueError(
            f"expected an observation space of shape {(RAW_OBSERVATION_SIZE,)}, "
            f"got {low.shape}"
        )
    basis_low = np.zeros(REWARD_BLOCK_SIZE, dtype=np.float32)
    basis_low[3] = low[6]
    basis_low[4] = low[7]
    basis_high = np.array(
        [
            np.linalg.norm(np.maximum(np.abs(low[0:2]), np.abs(high[0:2]))),
            np.linalg.norm(np.maximum(np.abs(low[2:4]), np.abs(high[2:4]))),
            float(max(abs(low[4]), abs(high[4]))),
            float(high[6]),
            float(high[7]),
        ],
        dtype=np.float32,
    )
    basis_high = np.maximum(basis_high, basis_low)
    return gym.spaces.Box(
        low=np.concatenate([basis_low, low]),
        high=np.concatenate([basis_high, high]),
        dtype=np.float32,
    )


AWAKE_INFO_KEY = "awake"


class LunarLanderAwakeInfoWrapper(gym.Wrapper):
    r"""Adds the lander's `awake` bit to the step info.

    The terminal reward branch is `not lander.awake`, the sleep state Box2D
    assigns a body at rest, which the observation does not encode. The bit is
    read from the unwrapped box2d environment because the `gym.Env` interface
    carries no lander reference.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        lander = self.env.unwrapped.lander
        info = {**info, AWAKE_INFO_KEY: bool(lander.awake)}
        return observation, reward, terminated, truncated, info


class LunarLanderRewardObs(gym.ObservationWrapper):
    r"""Emits $(\psi(s) \Vert s)$: the reward basis followed by the raw state."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                f"Expected a Box observation space. Got: {env.observation_space}"
            )
        self.observation_space = observation_space(env.observation_space)

    def observation(self, observation: ObsType) -> np.ndarray:
        state = np.asarray(observation, dtype=np.float64)
        return np.concatenate([reward_basis(state), state]).astype(
            self.observation_space.dtype
        )
