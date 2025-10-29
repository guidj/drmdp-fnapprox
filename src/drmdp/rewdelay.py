import abc
import logging
import random
import sys
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from drmdp import mathutils, metrics, optsol


class OptState(str, Enum):
    """
    Optimisation solver states.
    """

    UNSOLVED = "unsolved"
    SOLVED = "solved"


class RewardDelay(abc.ABC):
    """
    Abstract class for delayed reward config.
    """

    @abc.abstractmethod
    def sample(self) -> int: ...

    @abc.abstractmethod
    def range(self) -> Tuple[int, int]: ...

    @classmethod
    @abc.abstractmethod
    def id(cls) -> str: ...


class FixedDelay(RewardDelay):
    """
    Fixed window delays.
    """

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
    """
    Delays are sampled uniformly at random
    from a range of values.
    """

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
    """
    Delays are sampled from a clipped Poisson distribution
    """

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


class DataBuffer:
    """
    A data buffer, for storing inputs and
    labels.

    Has a `max_capacity` to limit memory usage.
    Either accumulates first or last samples.
    """

    ACC_FIRST = "FIRST"
    ACC_LASTEST = "LASTEST"

    def __init__(
        self,
        max_capacity: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        acc_mode: str = ACC_LASTEST,
    ):
        """
        Init.
        """
        self.max_capacity = max_capacity
        self.max_size_bytes = max_size_bytes
        self.acc_mode = acc_mode
        self.buffer: List[Any] = []

    def add(self, element: Any):
        """
        Adds data to buffer.
        """

        if self.max_capacity and self.max_size_bytes:
            safe_byte_limit = True
            safe_size_limit = True
            if list_size(self.buffer + [element]) >= self.max_size_bytes:
                if self.acc_mode == self.ACC_LASTEST:
                    while list_size(self.buffer + [element]) >= self.max_size_bytes:
                        self._pop_earliest()
                else:
                    safe_byte_limit = False

            if self.size() >= self.max_capacity:
                if self.acc_mode == self.ACC_LASTEST:
                    self._pop_earliest()
                else:
                    safe_size_limit = False

            if safe_byte_limit and safe_size_limit:
                self._append(element)

        elif self.max_size_bytes:
            if list_size(self.buffer + [element]) >= self.max_size_bytes:
                if self.acc_mode == self.ACC_LASTEST:
                    while list_size(self.buffer + [element]) >= self.max_size_bytes:
                        self._pop_earliest()
                    self._append(element)
                # else acc_mode == ACC_FIRST - do not add
            else:
                self._append(element)

        elif self.max_capacity:
            if self.size() >= self.max_capacity:
                if self.acc_mode == self.ACC_LASTEST:
                    self._pop_earliest()
                    self._append(element)
                # else mode == ACC_FIRST - do not add
            else:
                # Add new value
                self._append(element)
        else:
            # No limits
            self._append(element)

    def clear(self):
        """
        Empties buffer.
        """
        self.buffer = []

    def size(self):
        """
        Current buffer size.
        """
        return len(self.buffer)

    def size_bytes(self):
        """
        Current buffer size in bytes.
        """
        return list_size(self.buffer)

    def _pop_earliest(self):
        """
        Remove the First-In element in the list.
        """
        self.buffer.pop(0)

    def _append(self, element: Any):
        """
        Appends values to buffers.
        """
        self.buffer.append(element)


class DelayedRewardWrapper(gym.Wrapper):
    """
    Emits rewards following a delayed aggregation schedule.
    Rewards at the end of the reward window correspond
    to the sum of rewards in the window.
    In the remaining steps, no reward is emitted (`None`).
    """

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


class ZeroImputeMissingRewardWrapper(gym.RewardWrapper):
    """
    Missing rewards (`None`) are replaced with zero.
    """

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)

    def reward(self, reward):
        if reward is None:
            return 0.0
        return reward


class DiscretisedLeastLfaGenerativeRewardWrapper(gym.Wrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Least-Squares.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_encoding_wrapper: gym.ObservationWrapper,
        attempt_estimation_episode: int,
        use_bias: bool = False,
        max_buffer_size_bytes: Optional[int] = None,
    ):
        super().__init__(env)
        if not isinstance(obs_encoding_wrapper.observation_space, gym.spaces.Discrete):
            raise ValueError(
                f"obs_wrapper space must of type Discrete. Got: {type(obs_encoding_wrapper.observation_space)}"
            )
        if not isinstance(obs_encoding_wrapper.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"obs_wrapper space must of type Box. Got: {type(obs_encoding_wrapper.action_space)}"
            )
        self.obs_wrapper = obs_encoding_wrapper
        self.attempt_estimation_episode = attempt_estimation_episode
        self.episodes = 0
        self.use_bias = use_bias
        self.est_buffer = DataBuffer(max_size_bytes=max_buffer_size_bytes)

        self.nstates = obs_encoding_wrapper.observation_space.n
        self.nactions = obs_encoding_wrapper.action_space.n
        self.mdim = self.nstates * self.nactions
        self.weights = None
        self._obs_feats = None
        self._segment_features = None
        self.estimation_meta = {"use_bias": self.use_bias}
        self.rng = np.random.default_rng()

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        next_obs_feats = self.obs_wrapper.observation(next_obs)
        # Add s to action-specific columns
        # and s' to the last columns.
        self._segment_features[self._obs_feats * self.nactions + action] += 1

        if self.weights is not None:
            # estimate
            feats = self._segment_features
            if self.use_bias:
                feats = np.concatenate([feats, np.array([1.0])])
            reward = np.dot(feats, self.weights)
            # reset for the next example
            self._segment_features = np.zeros(shape=(self.mdim))
            est_state = OptState.SOLVED
        else:
            # Add example to buffer and
            # use aggregate reward.
            if info["segment_step"] == info["delay"] - 1:
                self.est_buffer.add((self._segment_features, reward))
                # reset for the next segment
                self._segment_features = np.zeros(shape=(self.mdim))
            else:
                # zero impute until rewards are estimated
                reward = 0.0
            est_state = OptState.UNSOLVED

        if term or trunc:
            self.episodes += 1
            if (
                self.weights is None
                and self.episodes >= self.attempt_estimation_episode
            ):
                # estimate rewards
                self.estimate_rewards()

        # For the next step
        self._obs_feats = next_obs_feats
        return (
            next_obs,
            reward,
            term,
            trunc,
            {"estimator": {"state": est_state}, **info},
        )

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
        if self.est_buffer.size() == 0:
            return

        obs_buffer, rew_buffer = zip(*self.est_buffer.buffer)
        matrix = np.stack(obs_buffer, dtype=np.float64)
        rewards = np.array(rew_buffer, dtype=np.float64)
        nexamples = rewards.shape[0]
        if self.use_bias:
            matrix = np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=nexamples), axis=-1),
                ],
                axis=1,
            )
        try:
            self.weights = optsol.solve_least_squares(matrix=matrix, rhs=rewards)
        except ValueError:
            # drop latest 5% of samples
            nexamples_drop = int(nexamples * 0.05)
            indices = self.rng.choice(
                np.arange(nexamples),
                nexamples - nexamples_drop,
                replace=False,
            )
            obs_buffer = matrix[indices].tolist()
            rew_buffer = rewards[indices].tolist()
            self.est_buffer.buffer = list(zip(obs_buffer, rew_buffer))

            logging.info(
                "%s - Failed estimation for %s. Dropping %d samples",
                type(self).__name__,
                self.env,
                nexamples_drop,
            )
        else:
            error = metrics.rmse(
                v_pred=np.dot(matrix, self.weights), v_true=rewards, axis=0
            )
            self.estimation_meta["sample"] = {"size": nexamples}
            self.estimation_meta["error"] = {"rmse": error}
            self.estimation_meta["estimate"] = {
                "weights": self.weights.tolist(),
            }
            logging.info(
                "%s - Estimated rewards for %s. RMSE: %f; # Samples: %d",
                type(self).__name__,
                self.env,
                error,
                nexamples,
            )


class LeastLfaGenerativeRewardWrapper(gym.Wrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Least-Squares.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_encoding_wrapper: gym.ObservationWrapper,
        attempt_estimation_episode: int,
        use_bias: bool = False,
        max_buffer_size_bytes: Optional[int] = None,
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
        self.attempt_estimation_episode = attempt_estimation_episode
        self.episodes = 0
        self.use_bias = use_bias
        self.est_buffer = DataBuffer(max_size_bytes=max_buffer_size_bytes)

        self.obs_dim = np.size(self.obs_wrapper.observation_space.sample())
        self.mdim = self.obs_dim * obs_encoding_wrapper.action_space.n + self.obs_dim
        self.weights = None
        self._obs_feats = None
        self._segment_features = None
        self.estimation_meta = {"use_bias": self.use_bias}
        self.rng = np.random.default_rng()

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
            est_state = OptState.SOLVED
        else:
            # Add example to buffer and
            # use aggregate reward.
            if info["segment_step"] == info["delay"] - 1:
                self.est_buffer.add((self._segment_features, reward))
                # reset for the next segment
                self._segment_features = np.zeros(shape=(self.mdim))
            else:
                # zero impute until rewards are estimated
                reward = 0.0
            est_state = OptState.UNSOLVED

        if term or trunc:
            self.episodes += 1
            if (
                self.weights is None
                and self.episodes >= self.attempt_estimation_episode
            ):
                # estimate rewards
                self.estimate_rewards()

        # For the next step
        self._obs_feats = next_obs_feats
        return (
            next_obs,
            reward,
            term,
            trunc,
            {"estimator": {"state": est_state}, **info},
        )

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
        if self.est_buffer.size() == 0:
            return

        obs_buffer, rew_buffer = zip(*self.est_buffer.buffer)
        matrix = np.stack(obs_buffer, dtype=np.float64)
        rewards = np.array(rew_buffer, dtype=np.float64)
        nexamples = rewards.shape[0]
        if self.use_bias:
            matrix = np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=nexamples), axis=-1),
                ],
                axis=1,
            )
        try:
            self.weights = optsol.solve_least_squares(matrix=matrix, rhs=rewards)
        except ValueError:
            # drop latest 5% of samples
            nexamples_drop = int(nexamples * 0.05)
            indices = self.rng.choice(
                np.arange(nexamples),
                nexamples - nexamples_drop,
                replace=False,
            )
            obs_buffer = matrix[indices].tolist()
            rew_buffer = rewards[indices].tolist()
            self.est_buffer.buffer = list(zip(obs_buffer, rew_buffer))
            logging.info(
                "%s - Failed estimation for %s. Dropping %d samples",
                type(self).__name__,
                self.env,
                nexamples_drop,
            )
        else:
            error = metrics.rmse(
                v_pred=np.dot(matrix, self.weights), v_true=rewards, axis=0
            )
            self.estimation_meta["sample"] = {"size": nexamples}
            self.estimation_meta["error"] = {"rmse": error}
            self.estimation_meta["estimate"] = {
                "weights": self.weights.tolist(),
            }
            logging.info(
                "%s - Estimated rewards for %s. RMSE: %f; # Samples: %d",
                type(self).__name__,
                self.env,
                error,
                nexamples,
            )


class BayesLeastLfaGenerativeRewardWrapper(gym.Wrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Bayesian Least-Squares.
    Rewards are first estimated after `init_attempt_estimation_episode`.
    After that, they are either updated following a doubling
    schedule or at fixed intervals.
    """

    INTERVAL = "interval"
    DOUBLE = "double"

    def __init__(
        self,
        env: gym.Env,
        obs_encoding_wrapper: gym.ObservationWrapper,
        mode: str = DOUBLE,
        init_attempt_estimation_episode: int = 10,
        use_bias: bool = False,
        max_buffer_size_bytes: Optional[int] = None,
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
        self.init_attempt_estimation_episode = init_attempt_estimation_episode
        self.update_episode = init_attempt_estimation_episode
        self.episodes = 0
        self.posterior_updates = 0
        self.est_buffer = DataBuffer(max_size_bytes=max_buffer_size_bytes)

        self.obs_dim = np.size(self.obs_wrapper.observation_space.sample())
        self.mdim = self.obs_dim * obs_encoding_wrapper.action_space.n + self.obs_dim
        self.mv_normal_rewards: Optional[optsol.MultivariateNormal] = None
        self._obs_feats = None
        self._segment_features = None
        self.estimation_meta = {"use_bias": self.use_bias}

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
            self.est_buffer.add((self._segment_features, reward))
            # Reset for the next segment
            self._segment_features = np.zeros(shape=(self.mdim))

        if self.mv_normal_rewards is not None:
            # Estimate rewards
            feats = step_features
            if self.use_bias:
                feats = np.concatenate([feats, np.array([1.0])])
            reward = np.dot(feats, self.mv_normal_rewards.mean)
            est_state = OptState.SOLVED
        else:
            # Zero impute until rewards are estimated
            if info["segment_step"] != info["delay"] - 1:
                reward = 0.0
            # else, use aggregate reward
            est_state = OptState.UNSOLVED

        if term or trunc:
            self.episodes += 1
            if self.episodes % self.update_episode == 0:
                self.estimate_rewards()
                # Schedule next update
                if self.mode == self.DOUBLE:
                    self.update_episode *= 2
                elif self.mode == self.INTERVAL:
                    pass

        # For the next step
        self._obs_feats = next_obs_feats
        return (
            next_obs,
            reward,
            term,
            trunc,
            {"estimator": {"state": est_state}, **info},
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._obs_feats = self.obs_wrapper.observation(obs)
        # Init segment and step features array
        self._segment_features = np.zeros(shape=(self.mdim))
        return obs, info

    def estimate_rewards(self):
        """
        Estimate new posterior.
        """
        if self.est_buffer.size() == 0:
            return

        obs_buffer, rew_buffer = zip(*self.est_buffer.buffer)
        matrix = np.stack(obs_buffer, dtype=np.float64)
        rewards = np.array(rew_buffer, dtype=np.float64)
        nexamples = rewards.shape[0]
        if self.use_bias:
            matrix = np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=nexamples), axis=-1),
                ],
                axis=1,
            )

        try:
            if self.mv_normal_rewards is None:
                # Frequentist prior
                self.mv_normal_rewards = optsol.MultivariateNormal.least_squares(
                    matrix=matrix, rhs=rewards
                )
            else:
                self.mv_normal_rewards = (
                    optsol.MultivariateNormal.bayes_linear_regression(
                        matrix=matrix, rhs=rewards, prior=self.mv_normal_rewards
                    )
                )
                self.posterior_updates += 1

        except ValueError:
            logging.info(
                "%s - Failed estimation for %s",
                type(self).__name__,
                self.env,
            )
        else:
            error = metrics.rmse(
                v_pred=np.dot(matrix, self.mv_normal_rewards.mean),
                v_true=rewards,
                axis=0,
            )
            self.estimation_meta["sample"] = {"size": nexamples}
            self.estimation_meta["error"] = {"rmse": error}
            self.estimation_meta["estimate"] = {
                "weights": self.mv_normal_rewards.mean.tolist(),
            }
            logging.info(
                "%s - %s rewards for %s. RMSE: %f; # Samples: %d",
                "Estimated" if self.posterior_updates == 0 else "Updated",
                type(self).__name__,
                self.env,
                error,
                nexamples,
            )

            # Clear buffers for next data
            self.est_buffer.clear()


class ConvexSolverGenerativeRewardWrapper(gym.Wrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Convex Optimisation,
    with constraints.
    The constraints are defined by using encoutered
    terminal states - the rewards for which are
    constrained to zero.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_encoding_wrapper: gym.ObservationWrapper,
        attempt_estimation_episode: int,
        use_bias: bool = False,
        max_buffer_size_bytes: Optional[int] = None,
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
        self.attempt_estimation_episode = attempt_estimation_episode
        self.episodes = 0
        self.use_bias = use_bias
        self.est_buffer = DataBuffer(max_size_bytes=max_buffer_size_bytes)
        self.tst_buffer = DataBuffer(max_size_bytes=max_buffer_size_bytes)

        self.obs_dim = np.size(self.obs_wrapper.observation_space.sample())
        self.mdim = self.obs_dim * self.obs_wrapper.action_space.n + self.obs_dim
        self.weights = None
        self._obs_feats = None
        self._segment_features = None
        self.estimation_meta = {"use_bias": self.use_bias}
        self.rng = np.random.default_rng()

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
            est_state = OptState.SOLVED
        else:
            # Add example to buffer and
            # use aggregate reward.
            if info["segment_step"] == info["delay"] - 1:
                self.est_buffer.add((self._segment_features, reward))
                # reset for the next segment
                self._segment_features = np.zeros(shape=(self.mdim))
            else:
                # zero impute until rewards are estimated
                reward = 0.0

            est_state = OptState.UNSOLVED

        # Add constraint for terminal state
        # if there is no estimate.
        if term and self.weights is None:
            # The action is not relevant here,
            # since every action leads to the same
            # transition.
            # But we represent every action for completeness.
            for ts_action in range(self.obs_wrapper.action_space.n):
                term_state = np.zeros(shape=(self.mdim))
                ts_idx = ts_action * self.obs_dim
                # To simplify the problem for the convex solver
                # we encode the (S,A) - not S'
                term_state[ts_idx : ts_idx + self.obs_dim] += next_obs_feats
                self.tst_buffer.add(term_state)

        if term or trunc:
            self.episodes += 1
            if (
                self.weights is None
                and self.episodes >= self.attempt_estimation_episode
            ):
                # estimate rewards
                self.estimate_rewards()

        # For the next step
        self._obs_feats = next_obs_feats
        return (
            next_obs,
            reward,
            term,
            trunc,
            {"estimator": {"state": est_state}, **info},
        )

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
        if self.est_buffer.size() == 0:
            return

        obs_buffer, rew_buffer = zip(*self.est_buffer.buffer)
        matrix = np.stack(obs_buffer, dtype=np.float64)
        rewards = np.array(rew_buffer, dtype=np.float64)
        term_states = (
            np.stack(self.tst_buffer.buffer, dtype=np.float64)
            if self.tst_buffer.size() > 0
            else []
        )
        nexamples = rewards.shape[0]
        if self.use_bias:
            matrix = np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(nexamples), axis=-1),
                ],
                axis=1,
            )
            if len(term_states) > 0:
                term_states = np.concatenate(
                    [
                        term_states,
                        np.expand_dims(np.ones(shape=len(term_states)), axis=-1),
                    ],
                    axis=1,
                )

        try:
            # Construct the problem.
            def constraint_fn(solution):
                return [term_state @ solution == 0 for term_state in term_states]

            self.weights = optsol.solve_convex_least_squares(
                matrix=matrix, rhs=rewards, constraint_fn=constraint_fn
            )
        except ValueError:
            # drop latest 5% of samples
            nexamples_drop = int(nexamples * 0.05)
            indices = self.rng.choice(
                np.arange(nexamples),
                self.attempt_estimation_episode - nexamples_drop,
                replace=False,
            )
            obs_buffer = matrix[indices].tolist()
            rew_buffer = rewards[indices].tolist()
            self.est_buffer.buffer = list(zip(obs_buffer, rew_buffer))
            logging.info(
                "%s - Failed estimation for %s. Dropping %d samples",
                type(self).__name__,
                self.env,
                nexamples_drop,
            )
        else:
            error = metrics.rmse(
                v_pred=np.dot(matrix, self.weights), v_true=rewards, axis=0
            )
            self.estimation_meta["sample"] = {"size": nexamples}
            self.estimation_meta["error"] = {"rmse": error}
            self.estimation_meta["estimate"] = {
                "weights": self.weights.tolist(),
                "constraints": len(term_states),
            }
            logging.info(
                "%s - Estimated rewards for %s. RMSE: %f; # Samples: %d, # Constraints: %d",
                type(self).__name__,
                self.env,
                error,
                nexamples,
                len(term_states),
            )


class BayesConvexSolverGenerativeRewardWrapper(gym.Wrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Bayesian Least-Squares.
    Rewards are first estimated after `init_attempt_estimation_episode`.
    After that, they are either updated following a doubling
    schedule or at fixed intervals.
    """

    INTERVAL = "interval"
    DOUBLE = "double"

    def __init__(
        self,
        env: gym.Env,
        obs_encoding_wrapper: gym.ObservationWrapper,
        mode: str = DOUBLE,
        init_attempt_estimation_episode: int = 10,
        use_bias: bool = False,
        max_buffer_size_bytes: Optional[int] = None,
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
        self.init_attempt_estimation_episode = init_attempt_estimation_episode
        self.update_episode = init_attempt_estimation_episode
        self.episodes = 0
        self.posterior_updates = 0
        self.est_buffer = DataBuffer(max_size_bytes=max_buffer_size_bytes)
        self.tst_buffer = DataBuffer(max_size_bytes=max_buffer_size_bytes)

        self.obs_dim = np.size(self.obs_wrapper.observation_space.sample())
        self.mdim = self.obs_dim * obs_encoding_wrapper.action_space.n + self.obs_dim
        self.mv_normal_rewards: Optional[optsol.MultivariateNormal] = None
        self._obs_feats = None
        self._segment_features = None
        self.estimation_meta = {"use_bias": self.use_bias}

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
            self.est_buffer.add((self._segment_features, reward))
            # reset for the next segment
            self._segment_features = np.zeros(shape=(self.mdim))

        if self.mv_normal_rewards is not None:
            # estimate
            feats = step_features
            if self.use_bias:
                feats = np.concatenate([feats, np.array([1.0])])
            reward = np.dot(feats, self.mv_normal_rewards.mean)
            est_state = OptState.SOLVED
        else:
            # zero impute until rewards are estimated
            if info["segment_step"] != info["delay"] - 1:
                reward = 0.0
            # else, use aggregate reward
            est_state = OptState.UNSOLVED

        # Add constraint for terminal state.
        if term:
            # The action is not relevant here,
            # since every action leads to the same
            # transition.
            # But we represent every action for completeness.
            for ts_action in range(self.obs_wrapper.action_space.n):
                term_state = np.zeros(shape=(self.mdim))
                ts_idx = ts_action * self.obs_dim
                # To simplify the problem for the convex solver
                # we encode the (S,A) - not S'
                term_state[ts_idx : ts_idx + self.obs_dim] += next_obs_feats
                self.tst_buffer.add(term_state)

        if term or trunc:
            self.episodes += 1
            if self.episodes % self.update_episode == 0:
                self.estimate_rewards()
                if self.mode == self.DOUBLE:
                    self.update_episode *= 2
                elif self.mode == self.INTERVAL:
                    pass

        # For the next step
        self._obs_feats = next_obs_feats
        return (
            next_obs,
            reward,
            term,
            trunc,
            {"estimator": {"state": est_state}, **info},
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._obs_feats = self.obs_wrapper.observation(obs)
        # Init segment and step features array
        self._segment_features = np.zeros(shape=(self.mdim))
        return obs, info

    def estimate_rewards(self):
        """
        Estimate new posterior.
        """
        if self.est_buffer.size() == 0:
            return

        obs_buffer, rew_buffer = zip(*self.est_buffer.buffer)
        matrix = np.stack(obs_buffer, dtype=np.float64)
        rewards = np.array(rew_buffer, dtype=np.float64)
        term_states = (
            np.stack(self.tst_buffer.buffer, dtype=np.float64)
            if self.tst_buffer.size() > 0
            else []
        )
        nexamples = rewards.shape[0]
        if self.use_bias:
            matrix = np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=nexamples), axis=-1),
                ],
                axis=1,
            )
            if len(term_states) > 0:
                term_states = np.concatenate(
                    [
                        term_states,
                        np.expand_dims(np.ones(shape=nexamples), axis=-1),
                    ],
                    axis=1,
                )

        try:
            if self.mv_normal_rewards is not None:
                # This is a posterior update,
                # increment counter.
                self.posterior_updates += 1

            # Construct the problem.
            def constraint_fn(solution):
                return [term_state @ solution == 0 for term_state in term_states]

            self.mv_normal_rewards = optsol.MultivariateNormal.convex_least_squares(
                matrix=matrix,
                rhs=rewards,
                constraint_fn=constraint_fn,
                warm_start_initial_guess=self.mv_normal_rewards.mean
                if self.mv_normal_rewards
                else None,
            )

        except ValueError as err:
            logging.info(
                "%s - Failed estimation for %s: \n%s",
                type(self).__name__,
                self.env,
                err,
            )
        else:
            error = metrics.rmse(
                v_pred=np.dot(matrix, self.mv_normal_rewards.mean),
                v_true=rewards,
                axis=0,
            )
            self.estimation_meta["sample"] = {"size": nexamples}
            self.estimation_meta["error"] = {"rmse": error}
            self.estimation_meta["estimate"] = {
                "weights": self.mv_normal_rewards.mean.tolist(),
                "constraints": len(term_states),
            }
            logging.info(
                "%s - %s rewards for %s. RMSE: %f; # Samples: %d, # Constraints: %d",
                "Estimated" if self.posterior_updates == 0 else "Updated",
                type(self).__name__,
                self.env,
                error,
                nexamples,
                self.tst_buffer.size(),
            )

            # Clear buffers for next data
            self.est_buffer.clear()


def list_size(xs: List[Any]) -> int:
    """
    Gets the size of a list in bytes.
    """
    return sys.getsizeof(xs) - sys.getsizeof([])
