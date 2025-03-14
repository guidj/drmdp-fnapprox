import abc
import random
from typing import Callable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from drmdp import mathutils


class RewardDelay(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> int: ...

    @abc.abstractmethod
    def range(self) -> Tuple[int, int]: ...

    @classmethod
    @abc.abstractmethod
    def id(cls) -> str: ...


class FixedDelay(RewardDelay):
    def __init__(self, delay: int):
        super().__init__()
        self.delay = delay

    def sample(self) -> int:
        return self.delay

    def range(self) -> Tuple[int, int]:
        return self.delay, self.delay

    @classmethod
    def id(cls):
        return "fixed"


class UniformDelay(RewardDelay):
    def __init__(self, min_delay: int, max_delay: int):
        super().__init__()
        self.min_delay = min_delay
        self.max_delay = max_delay

    def sample(self):
        return random.randint(self.min_delay, self.max_delay)

    def range(self) -> Tuple[int, int]:
        return self.min_delay, self.max_delay

    @classmethod
    def id(cls):
        return "uniform"


class ClippedPoissonDelay(RewardDelay):
    def __init__(
        self, lam: int, min_delay: Optional[int] = None, max_delay: Optional[int] = None
    ):
        """
        Calculate upper and lower bounds if not provided.
        """
        super().__init__()
        lower, upper = mathutils.poisson_exact_confidence_interval(lam)
        self.lam = lam
        self.min_delay = min_delay or lower
        self.max_delay = max_delay or upper
        self.rng = np.random.default_rng()

    def sample(self):
        return np.clip(self.rng.poisson(self.lam), self.min_delay, self.max_delay)

    def range(self) -> Tuple[int, int]:
        return self.min_delay, self.max_delay

    @classmethod
    def id(cls):
        return "clipped-poisson"


class DelayedRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_delay: RewardDelay,
        op: Callable[[Sequence[float]], float] = sum,
    ):
        super().__init__(env)
        self.reward_delay = reward_delay
        self.segment: Optional[int] = None
        self.segment_step: Optional[int] = None
        self.delay: Optional[int] = None
        self.rewards: List[float] = []
        self.op = op

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self.segment_step += 1
        self.rewards.append(reward)
        if self.segment_step == self.delay:
            # reset segment
            self.segment += 1
            self.segment_step = 0
            reward = self.op(self.rewards)
            # new delay
            self.delay = self.reward_delay.sample()
            self.rewards = []
        else:
            reward = None
        return (
            obs,
            reward,
            term,
            trunc,
            {
                **info,
                "delay": self.delay,
                "segment": self.segment,
                "segment_step": self.segment_step,
            },
        )

    def reset(self, *, seed=None, options=None):
        self.segment = 0
        self.segment_step = 0
        self.delay = self.reward_delay.sample()
        obs, info = super().reset(seed=seed, options=options)
        return obs, {
            **info,
            "delay": self.delay,
            "segment": self.segment,
            "segment_step": self.segment_step,
        }


class ZeroImputeMissingWrapper(gym.RewardWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)

    def reward(self, reward):
        if reward is None:
            return 0.0
        return reward
