import abc
import logging
import random
from typing import Callable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from drmdp import mathutils, metrics, optsol


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

        segment = self.segment
        segment_step = self.segment_step
        delay = self.delay
        if self.segment_step == self.delay - 1:
            # reset segment
            self.segment += 1
            self.segment_step = -1
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
                "delay": delay,
                "segment": segment,
                "segment_step": segment_step,
            },
        )

    def reset(self, *, seed=None, options=None):
        self.segment = 0
        self.segment_step = -1
        self.delay = self.reward_delay.sample()
        self.rewards = []
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


class LeastLfaMissingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        obs_encoding_wrapper: gym.ObservationWrapper,
        estimation_sample_size: int,
        use_bias: bool = False,
    ):
        super().__init__(env)
        if not isinstance(obs_encoding_wrapper.observation_space, gym.spaces.Box):
            raise ValueError(
                f"obs_wrapper space must of type Box. Got: {type(obs_encoding_wrapper)}"
            )
        if not isinstance(obs_encoding_wrapper.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"obs_wrapper space must of type Box. Got: {type(obs_encoding_wrapper)}"
            )
        self.obs_wrapper = obs_encoding_wrapper
        self.estimation_sample_size = estimation_sample_size
        self.use_bias = use_bias
        self.obs_buffer: List[np.ndarray] = []
        self.rew_buffer: List[np.ndarray] = []

        self.obs_dim = np.size(self.obs_wrapper.observation_space.sample())
        self.mdim = self.obs_dim * obs_encoding_wrapper.action_space.n + self.obs_dim
        self.weights = None
        self._obs_feats = None
        self._segment_features = None
        self.estimation_meta = {"use_bias": self.use_bias}

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        next_obs_feats = self.obs_wrapper.observation(next_obs)
        # Add s to action-specific columns
        # and s' to the last columns.
        start_index = action * self.obs_dim
        self._segment_features[start_index : start_index + self.obs_dim] += (
            self._obs_feats
        )
        self._segment_features[-self.obs_dim :] += next_obs_feats

        if self.weights is not None:
            # estimate
            feats = self._segment_features
            if self.use_bias:
                feats = np.concatenate([feats, np.array([1.0])])
            reward = np.dot(feats, self.weights)
            # reset for the next example
            self._segment_features = np.zeros(shape=(self.mdim))
        else:
            # Add example to buffer and
            # use aggregate reward.
            if info["segment_step"] == info["delay"] - 1:
                self.obs_buffer.append(self._segment_features)
                # aggregate reward
                self.rew_buffer.append(reward)
                # reset for the next segment
                self._segment_features = np.zeros(shape=(self.mdim))
            else:
                # zero impute until rewards are estimated
                reward = 0.0

            if len(self.obs_buffer) >= self.estimation_sample_size:
                # estimate rewards
                self.estimate_rewards()

        # For the next step
        self._obs_feats = next_obs_feats
        return next_obs, reward, term, trunc, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._obs_feats = self.obs_wrapper.observation(obs)
        # Init segment and step features array
        self._segment_features = np.zeros(shape=(self.mdim))
        return obs, info

    def estimate_rewards(self):
        """
        Estimate rewards with lstsq.
        """
        matrix = np.stack(self.obs_buffer)
        rewards = np.array(self.rew_buffer)
        if self.use_bias:
            matrix = np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=len(self.obs_buffer)), axis=-1),
                ],
                axis=1,
            )
        try:
            self.weights = optsol.solve_least_squares(matrix=matrix, rhs=rewards)
        except ValueError:
            # drop latest 5% of samples
            rng = np.random.default_rng()
            nsamples_drop = int(self.estimation_sample_size * 0.05)
            indices = rng.choice(
                np.arange(self.estimation_sample_size),
                self.estimation_sample_size - nsamples_drop,
                replace=False,
            )
            self.obs_buffer = np.asarray(self.obs_buffer)[indices].tolist()
            self.rew_buffer = np.asarray(self.rew_buffer)[indices].tolist()
            logging.info(
                "Failed estimation for %s. Dropping %d samples",
                self.env,
                nsamples_drop,
            )
        else:
            error = metrics.rmse(
                v_pred=np.dot(matrix, self.weights), v_true=rewards, axis=0
            )
            self.estimation_meta["sample"] = {"size": rewards.shape[0]}
            self.estimation_meta["error"] = {"rmse": error}
            self.estimation_meta["estimate"] = {
                "weights": self.weights.tolist(),
            }
            logging.info("Estimated rewards for %s. RMSE: %f", self.env, error)


class LeastBayesLfaMissingWrapper(gym.Wrapper):
    INTERVAL = "interval"
    DOUBLE = "double"

    def __init__(
        self,
        env: gym.Env,
        obs_encoding_wrapper: gym.ObservationWrapper,
        mode: str = DOUBLE,
        init_update_episodes: int = 10,
        use_bias: bool = False,
    ):
        super().__init__(env)
        if not isinstance(obs_encoding_wrapper.observation_space, gym.spaces.Box):
            raise ValueError(
                f"obs_wrapper space must of type Box. Got: {type(obs_encoding_wrapper)}"
            )
        if not isinstance(obs_encoding_wrapper.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"obs_wrapper space must of type Box. Got: {type(obs_encoding_wrapper)}"
            )
        if mode not in (self.INTERVAL, self.DOUBLE):
            pass
        self.obs_wrapper = obs_encoding_wrapper
        self.use_bias = use_bias
        self.mode = mode
        self.init_update_episodes = init_update_episodes
        self.update_episodes = init_update_episodes
        self.obs_buffer: List[np.ndarray] = []
        self.rew_buffer: List[np.ndarray] = []

        self.obs_dim = np.size(self.obs_wrapper.observation_space.sample())
        self.mdim = self.obs_dim * obs_encoding_wrapper.action_space.n + self.obs_dim
        self.mv_lst = None
        self._obs_feats = None
        self._segment_features = None
        self.estimation_meta = {"use_bias": self.use_bias}
        self.episodes = 0

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        next_obs_feats = self.obs_wrapper.observation(next_obs)
        # Add s to action-specific columns
        # and s' to the last columns.
        step_features = np.zeros(shape=(self.mdim))
        start_index = action * self.obs_dim
        step_features[start_index : start_index + self.obs_dim] += self._obs_feats
        step_features[-self.obs_dim :] += next_obs_feats
        self._segment_features += step_features

        # Add example to buffer and
        # use aggregate reward.
        if info["segment_step"] == info["delay"] - 1:
            self.obs_buffer.append(self._segment_features)
            # aggregate reward
            self.rew_buffer.append(reward)
            # reset for the next segment
            self._segment_features = np.zeros(shape=(self.mdim))

        if self.mv_lst is not None:
            # estimate
            feats = step_features
            if self.use_bias:
                feats = np.concatenate([feats, np.array([1.0])])
            reward = np.dot(feats, self.mv_lst.mean)
        else:
            # zero impute until rewards are estimated
            if info["segment_step"] != info["delay"] - 1:
                reward = 0.0
            # else, use aggregate reward

        if term or trunc:
            # Update estimate
            self.episodes += 1
            if self.episodes % self.update_episodes == 0:
                self.estimate_posterior()
                if self.mode == self.DOUBLE:
                    self.update_episodes *= 2
                elif self.mode == self.INTERVAL:
                    pass

        # For the next step
        self._obs_feats = next_obs_feats
        return next_obs, reward, term, trunc, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._obs_feats = self.obs_wrapper.observation(obs)
        # Init segment and step features array
        self._segment_features = np.zeros(shape=(self.mdim))
        return obs, info

    def estimate_posterior(self):
        """
        Estimate new posterior.
        """
        if len(self.obs_buffer) == 0:
            return

        # estimate rewards
        matrix = np.stack(self.obs_buffer, dtype=np.float64)
        rewards = np.array(self.rew_buffer, dtype=np.float64)
        if self.use_bias:
            matrix = np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=len(self.obs_buffer)), axis=-1),
                ],
                axis=1,
            )

        try:
            if self.mv_lst is None:
                # Frequentist prior
                self.mv_lst = optsol.MultivariateNormal.least_squares(
                    matrix=matrix, rhs=rewards
                )
            else:
                self.mv_lst = optsol.MultivariateNormal.bayes_linear_regression(
                    matrix=matrix, rhs=rewards, prior=self.mv_lst
                )

        except ValueError:
            logging.info(
                "Failed estimation for %s",
                self.env,
            )
        else:
            error = metrics.rmse(
                v_pred=np.dot(matrix, self.mv_lst.mean), v_true=rewards, axis=0
            )
            self.estimation_meta["sample"] = {"size": rewards.shape[0]}
            self.estimation_meta["error"] = {"rmse": error}
            self.estimation_meta["estimate"] = {
                "weights": self.mv_lst.mean.tolist(),
            }
            logging.info("Estimated rewards for %s. RMSE: %f", self.env, error)

            # Clear buffers for next data
            self.obs_buffer = []
            self.rew_buffer = []
