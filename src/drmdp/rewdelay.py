import abc
import logging
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import (
    Any,
    Protocol,
)

import gymnasium as gym
import numpy as np

from drmdp import mathutils, metrics, optsol, rewardfeat

logger = logging.getLogger(__name__)


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
    def range(self) -> tuple[int, int]: ...

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

    def range(self) -> tuple[int, int]:
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

    def range(self) -> tuple[int, int]:
        return self.min_delay, self.max_delay

    @classmethod
    def id(cls):
        return "uniform"


class ClippedPoissonDelay(RewardDelay):
    """
    Delays are sampled from a clipped Poisson distribution
    """

    def __init__(
        self, lam: int, min_delay: int | None = None, max_delay: int | None = None
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

    def range(self) -> tuple[int, int]:
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
        max_capacity: int | None = None,
        max_size_bytes: int | None = None,
        acc_mode: str = ACC_LASTEST,
    ):
        """
        Init.
        """
        self.max_capacity = max_capacity
        self.max_size_bytes = max_size_bytes
        self.acc_mode = acc_mode
        self.buffer: list[Any] = []

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


class WindowedTaskSchedule:
    """
    Sets schedule for updates, using three types of schedules:
    1. Fixed interval (fixed)
    2. Exponentially growing interval (exponential): each window is
    `rho` times the previous one
    3. Doubling size (double)

    The first window spans episodes `init_update_ep` through
    `2 * init_update_ep`; the first update fires at its end. The
    initial estimate itself is gated by the estimation buffer, not by
    this schedule. Each later window grows from the previous one:
    fixed keeps it at `init_update_ep`, exponential multiplies it by
    `rho` (rounded up to whole episodes), double multiplies it by
    two.
    """

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    DOUBLE = "double"
    DEFAULT_RHO = 1.05

    def __init__(
        self,
        mode: str = EXPONENTIAL,
        init_update_ep: int = 10,
        rho: float = DEFAULT_RHO,
    ):
        """
        Instantiates the class for a given update schedule.
        """
        if mode not in (self.FIXED, self.EXPONENTIAL, self.DOUBLE):
            raise ValueError(
                f"Unsupported mode: {mode}. "
                f"Must be ({self.FIXED}, {self.EXPONENTIAL}, {self.DOUBLE})"
            )
        if mode == self.EXPONENTIAL and not (np.isfinite(rho) and rho > 1.0):
            raise ValueError(
                f"Exponential schedule requires a finite rho > 1.0. Got: {rho}"
            )
        if init_update_ep < 1:
            raise ValueError(
                f"init_update_ep must be a positive episode count. "
                f"Got: {init_update_ep}"
            )

        self.mode = mode
        self.init_update_ep = init_update_ep
        self.rho = rho
        self.curr_update_ep = init_update_ep
        self._done = False

        # the first window spans `init_update_ep` episodes; the first
        # update fires at its end
        self._window = init_update_ep
        self.next_update_ep = self.curr_update_ep + self._window

    def step(self, episode: int) -> None:
        """
        Updates the estimation window.
        """
        if episode == self.next_update_ep:
            # New window
            self.curr_update_ep = self.next_update_ep
            self._window = self._grow_window(self._window)
            self.next_update_ep = self.curr_update_ep + self._window
            # Reset window state.
            self._done = False

    def set_state(self, succ: bool) -> None:
        """
        Sets state for the current cycle.
        """

        self._done = succ

    @property
    def current_window_done(self) -> bool:
        """
        Returns true if the current cycle state is `False`.
        """
        return self._done

    def _grow_window(self, window: int) -> int:
        """
        Length of the window that follows one of `window` episodes.
        """
        if self.mode == self.FIXED:
            return self.init_update_ep
        if self.mode == self.EXPONENTIAL:
            # ceil keeps whole episodes and, for any rho > 1,
            # guarantees each window is at least one episode longer
            return int(np.ceil(window * self.rho))
        return window * 2


class SupportsName(Protocol):
    """
    Provides methods to get the name of the
    class and underlying (`unwrapped`) env.
    """

    env: gym.Env
    unwrapped: gym.Env

    def get_name(self):
        """
        Name and id of the class.
        """
        cls_name = type(self).__name__
        env_id = id(self)
        return f"{cls_name}(id={env_id})"

    def get_env_name(self):
        """
        Name and id of the underlying (`unwrapped`) environment.
        """
        cls_name = type(self.env.unwrapped).__name__
        env_id = id(self.unwrapped)
        return f"{cls_name}(id={env_id})"


class DelayedRewardWrapper(gym.Wrapper, SupportsName):
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
        self.segment: int | None = None
        self.segment_step: int | None = None
        self.delay: int | None = None
        self.rewards: list[float] = []
        self.op = op

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self.segment_step += 1
        self.rewards.append(reward)

        segment = self.segment
        segment_step = self.segment_step
        delay = self.delay
        final_segment_step = self.segment_step == self.delay - 1
        if final_segment_step:
            # reset segment
            self.segment += 1
            self.segment_step = -1
            reward = self.op(self.rewards)
            # new delay
            self.delay = self.reward_delay.sample()
            self.rewards = []
        else:
            reward = None
        if term and not final_segment_step:
            reward = self.op(self.rewards)
            self.rewards = []
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
                # Provide the next delay on the final step
                # and omit otherwise
                "next_delay": self.delay if final_segment_step else None,
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
            # Provide the next delay on the final step
            # and omit otherwise
            "next_delay": self.delay,
        }


class ImputeMissingRewardWrapper(gym.RewardWrapper, SupportsName):
    """
    Missing rewards (`None`) are replaced with zero.
    """

    def __init__(self, env: gym.Env, impute_value: float):
        super().__init__(env)
        self.impute_value = float(impute_value)

    def reward(self, reward):
        if reward is None:
            return self.impute_value
        return reward


class BaseGenerativeRewardWrapper(gym.Wrapper, SupportsName, abc.ABC):
    """
    Base class for generative reward wrappers.

    Handles common operations for reward estimation from aggregate rewards,
    including:
    - Feature computation and accumulation
    - Buffer management
    - Step and reset logic
    - Template methods for child-specific behavior
    """

    def __init__(
        self,
        env: gym.Env,
        feats_fn: rewardfeat.TransitionFeatureFn,
        intercept: bool = False,
        impute_value: float = 0.0,
        estimation_buffer_mult: int | None = None,
        reward_error_fn: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
    ):
        super().__init__(env)
        self._validate_action_space()

        self.feats_fn = feats_fn
        self.intercept = intercept
        self.impute_value = impute_value
        self.estimation_buffer_mult = estimation_buffer_mult
        self.reward_error_fn = reward_error_fn

        self.episodes = 0
        self.steps = 0
        self._start_monotonic = time.monotonic()
        self.mdim = feats_fn.feature_dim
        if self.mdim < 1:
            raise ValueError(f"feats_fn feature_dim must be positive. Got: {self.mdim}")
        # set by every concrete estimator; estimation events and
        # deferred attempts append to it
        self.estimation_meta: dict[str, Any] = {}
        self._latest_obs = None
        self._segment_features = None
        self.rng = np.random.default_rng()
        self.est_buffer = DataBuffer(
            max_capacity=self.mdim * estimation_buffer_mult
            if estimation_buffer_mult
            else None
        )

    def _validate_action_space(self):
        """Validate action space is Discrete."""
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError(
                f"Action space must be Discrete. Got: {type(self.action_space)}"
            )

    @abc.abstractmethod
    def _initialize_segment_features(self):
        """Initialize segment features array."""
        raise NotImplementedError

    def _accumulate_step_features(self, step_features: np.ndarray):
        """Add per-step features to the segment accumulator."""
        self._segment_features += step_features

    def _get_estimation_inputs(self, feats: np.ndarray) -> np.ndarray:
        """Prepare features for reward prediction, adding the intercept if needed."""
        if self.intercept:
            return np.concatenate([feats, np.array([1.0])])
        return feats

    def _add_intercept_to_matrix(
        self, matrix: np.ndarray, nexamples: int
    ) -> np.ndarray:
        """Append the intercept column to the design matrix when enabled."""
        if self.intercept:
            return np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=nexamples), axis=-1),
                ],
                axis=1,
            )
        return matrix

    @abc.abstractmethod
    def _has_estimate(self) -> bool:
        """Check if reward estimate exists."""
        raise NotImplementedError

    def _should_buffer_when_estimated(self) -> bool:
        """
        Whether to continue buffering segments even after getting an estimate.
        Default: False (stop buffering once we have weights).
        Override to True for continual learning (e.g., Bayesian updates).
        """
        return False

    @abc.abstractmethod
    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        """Get estimated reward from features."""
        raise NotImplementedError

    @abc.abstractmethod
    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        """Determine if estimation should be attempted."""
        raise NotImplementedError

    def _on_episode_end(self, term: bool, trunc: bool):
        """
        Handle episode end before estimation check.
        Override for windowed scheduling logic.
        """
        del term
        del trunc

    def _on_estimation_complete(self, success: bool):
        """
        Handle post-estimation logic.
        Override for windowed scheduling state updates.

        Args:
            success: Whether estimation succeeded (for windowed classes that return bool)
        """
        del success

    def _record_reward_error(self, weights: np.ndarray) -> None:
        """
        Measures the estimate's error against true rewards.

        One record per estimation event, when an evaluator is
        configured; `update_index` counts events within this run,
        1-based, so a Bayesian estimator's first record is its
        empirical prior. A failing measurement is logged and skipped:
        it must not fail the control run it instruments.
        """
        if self.reward_error_fn is None:
            return
        try:
            report = self.reward_error_fn(weights)
        except Exception as err:  # noqa: BLE001 - measurement must not abort control
            logger.warning(
                "%s - reward-error measurement failed: %s", self.get_name(), err
            )
            return
        self.estimation_meta.setdefault("reward_errors", []).append(
            {
                "update_index": len(self.estimation_meta["reward_errors"]) + 1,
                "episode": self.episodes,
                "steps": self.steps,
                "elapsed_s": round(time.monotonic() - self._start_monotonic, 6),
                **report,
            }
        )

    def _record_deferred_attempt(self, matrix: np.ndarray) -> None:
        """
        Records a scheduled attempt the gate deferred.

        The record exposes why the attempt did not solve: the buffer
        size and the nonzero-column count against the design width.
        """
        self.estimation_meta.setdefault("deferred_attempts", []).append(
            {
                "episode": self.episodes,
                "steps": self.steps,
                "buffer_rows": int(matrix.shape[0]),
                "nonzero_columns": optsol.matrix_factors_rank(matrix),
                "matrix_columns": int(matrix.shape[1]),
            }
        )

    def _on_terminal_state(self, transition: rewardfeat.Transition):
        """
        Handle terminal state.
        Override in child classes that need terminal state handling.
        """
        del transition

    def _get_estimator_info(self) -> dict[str, Any]:
        """
        Get estimator-specific info to include in step return.
        Override to add custom info.
        """
        return {}

    def _extract_buffer_data(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Extract and prepare data from estimation buffer.

        Returns:
            Tuple of (matrix, rewards, nexamples)
        """
        obs_buffer, rew_buffer = zip(*self.est_buffer.buffer)
        matrix = np.stack(obs_buffer, dtype=np.float64)
        rewards = np.array(rew_buffer, dtype=np.float64)
        nexamples = rewards.shape[0]
        return matrix, rewards, nexamples

    def _handle_estimation_failure(
        self, err: ValueError, matrix: np.ndarray, rewards: np.ndarray
    ):
        """Handle estimation failure by logging and dropping 5% of samples."""
        logger.debug(
            "%s - Failed estimation for %s: \n%s",
            self.get_name(),
            self.get_env_name(),
            err,
        )
        # Drop latest 5% of samples
        nexamples_dropped, (matrix, rewards) = drop_samples(
            frac=0.05, arrays=[matrix, rewards], rng=self.rng
        )
        obs_buffer = matrix.tolist()
        rew_buffer = rewards.tolist()
        self.est_buffer.buffer = list(zip(obs_buffer, rew_buffer))
        logger.debug(
            "%s - Dropped %d samples",
            self.get_name(),
            nexamples_dropped,
        )

    def _calculate_rmse(
        self, matrix: np.ndarray, weights: np.ndarray, rewards: np.ndarray
    ) -> float:
        """Calculate RMSE between predictions and true rewards."""
        return metrics.rmse(v_pred=np.dot(matrix, weights), v_true=rewards, axis=0)

    def _create_snapshot(
        self,
        nexamples: int,
        matrix: np.ndarray,
        weights: np.ndarray,
        error: float,
        **extra_fields,
    ) -> dict[str, Any]:
        """
        Create estimation snapshot for metadata tracking.

        Args:
            nexamples: Number of examples used
            matrix: Feature matrix
            weights: Estimated weights
            error: RMSE error
            **extra_fields: Additional fields to include in estimate dict

        Returns:
            Snapshot dictionary
        """
        snapshot = {
            "sample": {
                "size": nexamples,
                "factors_rank": optsol.matrix_factors_rank(matrix),
            },
            "error": {"rmse": error},
            "estimate": {
                "weights": weights.tolist(),
                **extra_fields,
            },
        }
        return snapshot

    @abc.abstractmethod
    def estimate_rewards(self):
        """Estimate rewards from buffer data."""
        raise NotImplementedError

    def step(self, action):
        previous_obs = self._latest_obs
        next_obs, reward, term, trunc, info = super().step(action)
        self.steps += 1
        transition = rewardfeat.Transition(
            observation=previous_obs,
            action=action,
            next_observation=next_obs,
            terminated=term,
            truncated=trunc,
            info=info,
        )

        step_features = np.asarray(self.feats_fn(transition), dtype=np.float64)
        if step_features.shape != (self.mdim,):
            raise ValueError(
                f"{self.get_name()} - feats_fn returned shape "
                f"{step_features.shape}, expected ({self.mdim},)"
            )
        if not self._has_estimate() or self._should_buffer_when_estimated():
            self._accumulate_step_features(step_features)

        partial_term = term and info["segment_step"] != info["delay"] - 1
        at_segment_end = info["segment_step"] == info["delay"] - 1

        # Buffer segment data at segment end (for continual learning)
        if self._should_buffer_when_estimated() and (at_segment_end or partial_term):
            self.est_buffer.add((self._segment_features, reward))
            self._segment_features = self._initialize_segment_features()

        if self._has_estimate():
            # Use estimated reward
            reward = self._get_estimated_reward(
                self._get_estimation_inputs(step_features)
            )
            est_state = OptState.SOLVED
        else:
            # Add example to buffer and use aggregate reward (one-time learning)
            if not self._should_buffer_when_estimated():
                if at_segment_end or partial_term:
                    self.est_buffer.add((self._segment_features, reward))
                    self._segment_features = self._initialize_segment_features()
                else:
                    reward = self.impute_value
            else:
                if not at_segment_end and not partial_term:
                    reward = self.impute_value
            est_state = OptState.UNSOLVED

        # Handle terminal state
        if term:
            self._on_terminal_state(transition)

        # Attempt estimation on episode end
        if term or trunc:
            self.episodes += 1
            self._on_episode_end(term, trunc)
            if self._should_attempt_estimation(term, trunc):
                result = self.estimate_rewards()
                # Handle windowed classes that return bool
                success = result if isinstance(result, bool) else True
                self._on_estimation_complete(success)

        # For the next step
        self._latest_obs = next_obs
        estimator_info = {"state": est_state, **self._get_estimator_info()}
        return (
            next_obs,
            reward,
            term,
            trunc,
            {"estimator": estimator_info, **info},
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._latest_obs = obs
        self._segment_features = self._initialize_segment_features()
        return obs, info


class DiscretisedLeastLfaGenerativeRewardWrapper(BaseGenerativeRewardWrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Least-Squares.
    Uses discrete observation space with count-based feature accumulation.
    """

    def __init__(
        self,
        env: gym.Env,
        feats_fn: rewardfeat.TransitionFeatureFn,
        attempt_estimation_episode: int,
        estimation_buffer_mult: int | None = None,
        impute_value: float = 0.0,
        check_factors: bool = False,
        intercept: bool = False,
    ):
        super().__init__(
            env=env,
            feats_fn=feats_fn,
            intercept=intercept,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.attempt_estimation_episode = attempt_estimation_episode
        self.check_factors = check_factors
        self.weights = None
        self.estimation_meta = {
            "intercept": intercept,
            "check_factors": check_factors,
            "snapshots": [],
            "deferred_attempts": [],
        }

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)  # type: ignore

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            (term or trunc)
            and self.weights is None
            and self.episodes >= self.attempt_estimation_episode
            and self.est_buffer.size() >= self.mdim
        )

    def estimate_rewards(self):
        """
        Estimate rewards with Least Squares.
        """
        matrix, rewards, nexamples = self._extract_buffer_data()

        # Check matrix rank if requested
        if self.check_factors:
            rank = optsol.matrix_factors_rank(matrix)
            if rank < matrix.shape[1]:
                self._record_deferred_attempt(matrix)
                return

        # Add the intercept column if needed
        matrix = self._add_intercept_to_matrix(matrix, nexamples)

        try:
            weights = optsol.solve_least_squares(matrix=matrix, rhs=rewards)
        except ValueError as err:
            self._handle_estimation_failure(err, matrix, rewards)
        else:
            self.weights = weights
            error = self._calculate_rmse(matrix, weights, rewards)
            snapshot = self._create_snapshot(nexamples, matrix, weights, error)
            self.estimation_meta["snapshots"].append(snapshot)
            logger.info(
                "%s - Estimated rewards for %s. RMSE: %f; No. Samples: %d",
                self.get_name(),
                self.get_env_name(),
                error,
                nexamples,
            )


class LeastLfaGenerativeRewardWrapper(BaseGenerativeRewardWrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Least-Squares.
    Uses Box observation space with additive feature accumulation.
    """

    def __init__(
        self,
        env: gym.Env,
        feats_fn: rewardfeat.TransitionFeatureFn,
        attempt_estimation_episode: int,
        estimation_buffer_mult: int | None = None,
        impute_value: float = 0.0,
        check_factors: bool = False,
        reward_error_fn: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
        intercept: bool = False,
    ):
        super().__init__(
            env=env,
            feats_fn=feats_fn,
            intercept=intercept,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
            reward_error_fn=reward_error_fn,
        )
        self.attempt_estimation_episode = attempt_estimation_episode
        self.check_factors = check_factors
        self.weights = None
        self.estimation_meta = {
            "intercept": intercept,
            "check_factors": check_factors,
            "snapshots": [],
            "reward_errors": [],
            "deferred_attempts": [],
        }

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)  # type: ignore

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            (term or trunc)
            and self.weights is None
            and self.episodes >= self.attempt_estimation_episode
            and self.est_buffer.size() >= self.mdim
        )

    def estimate_rewards(self):
        """
        Estimate rewards with Least Squares.
        """
        matrix, rewards, nexamples = self._extract_buffer_data()

        # Check matrix rank if requested
        if self.check_factors:
            rank = optsol.matrix_factors_rank(matrix)
            if rank < matrix.shape[1]:
                self._record_deferred_attempt(matrix)
                return

        # Add the intercept column if needed
        matrix = self._add_intercept_to_matrix(matrix, nexamples)

        try:
            weights = optsol.solve_least_squares(matrix=matrix, rhs=rewards)
        except ValueError as err:
            self._handle_estimation_failure(err, matrix, rewards)
        else:
            self.weights = weights
            error = self._calculate_rmse(matrix, weights, rewards)
            snapshot = self._create_snapshot(nexamples, matrix, weights, error)
            self.estimation_meta["snapshots"].append(snapshot)
            self._record_reward_error(weights)
            logger.info(
                "%s - Estimated rewards for %s. RMSE: %f; No. Samples: %d",
                self.get_name(),
                self.get_env_name(),
                error,
                nexamples,
            )


class BayesLeastLfaGenerativeRewardWrapper(BaseGenerativeRewardWrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated with Bayesian Least-Squares.
    The initial estimate is gated by the estimation buffer, not by
    episode count. After it, posterior updates follow the windowed
    schedule — exponential (the default), fixed, or doubling — set
    by `mode`.
    Uses Box observation space with continual learning (buffering continues after estimation).
    """

    def __init__(
        self,
        env: gym.Env,
        feats_fn: rewardfeat.TransitionFeatureFn,
        mode: str = WindowedTaskSchedule.EXPONENTIAL,
        init_attempt_estimation_episode: int = 10,
        schedule_rho: float = WindowedTaskSchedule.DEFAULT_RHO,
        estimation_buffer_mult: int | None = None,
        impute_value: float = 0.0,
        check_factors: bool = False,
        sample_weights: bool = False,
        reward_error_fn: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
        intercept: bool = False,
    ):
        super().__init__(
            env=env,
            feats_fn=feats_fn,
            intercept=intercept,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
            reward_error_fn=reward_error_fn,
        )
        self.mode = mode
        self.init_attempt_estimation_episode = init_attempt_estimation_episode
        self.check_factors = check_factors
        self.sample_weights = sample_weights
        self.windowed_task_schedule = WindowedTaskSchedule(
            mode=mode,
            init_update_ep=init_attempt_estimation_episode,
            rho=schedule_rho,
        )
        self.update_episode = init_attempt_estimation_episode
        self.posterior_updates = 0
        self.mv_normal_rewards: optsol.MultivariateNormal | None = None
        self._sampled_weights: np.ndarray | None = None
        self._sampled_posterior: optsol.MultivariateNormal | None = None
        self.estimation_meta: dict[str, Any] = {
            "intercept": intercept,
            "check_factors": check_factors,
            "snapshots": [],
            "reward_errors": [],
            "deferred_attempts": [],
        }

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def reset(self, *, seed=None, options=None):
        # Episode boundary: forget the held weights so the next
        # reward estimate draws a fresh posterior sample
        obs, info = super().reset(seed=seed, options=options)
        self._sampled_weights = None
        return obs, info

    def _has_estimate(self) -> bool:
        return self.mv_normal_rewards is not None

    def _should_buffer_when_estimated(self) -> bool:
        # Continue buffering for continual learning
        return True

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        if self.sample_weights:
            # One weight draw per episode (PSRL cadence): redraw only at
            # episode starts or when a continual update replaced the
            # posterior object, otherwise hold the sampled weights.
            posterior = self.mv_normal_rewards
            if (
                self._sampled_weights is None
                or self._sampled_posterior is not posterior
            ):
                self._sampled_weights = posterior.sample(self.rng)  # type: ignore[union-attr]
                self._sampled_posterior = posterior
            return float(np.dot(feats, self._sampled_weights))
        return float(np.dot(feats, self.mv_normal_rewards.mean))  # type: ignore[union-attr]

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            (term or trunc)
            and not self.windowed_task_schedule.current_window_done
            and self.est_buffer.size() >= self.mdim
        )

    def _on_episode_end(self, term: bool, trunc: bool):
        self.windowed_task_schedule.step(self.episodes)

    def _on_estimation_complete(self, success: bool):
        self.windowed_task_schedule.set_state(succ=success)

    def _get_estimator_info(self) -> dict[str, Any]:
        return {
            "posterior_updates": self.posterior_updates,
            "sample_weights": self.sample_weights,
        }

    def estimate_rewards(self) -> bool:
        """
        Estimate rewards or update posterior estimate
        of Least Squares.
        """
        matrix, rewards, nexamples = self._extract_buffer_data()

        # Check matrix rank if requested
        if self.check_factors:
            rank = optsol.matrix_factors_rank(matrix)
            if rank < matrix.shape[1]:
                self._record_deferred_attempt(matrix)
                return False

        # Add the intercept column if needed
        matrix = self._add_intercept_to_matrix(matrix, nexamples)

        try:
            if self.mv_normal_rewards is None:
                # Frequentist prior
                mv_normal = optsol.MultivariateNormal.least_squares(
                    matrix=matrix, rhs=rewards
                )
            else:
                mv_normal = optsol.MultivariateNormal.bayes_linear_regression(
                    matrix=matrix, rhs=rewards, prior=self.mv_normal_rewards
                )
                self.posterior_updates += 1 if mv_normal is not None else 0
        except ValueError as err:
            self._handle_estimation_failure(err, matrix, rewards)
            return False
        else:
            if mv_normal is not None:
                weights = mv_normal.mean
                error = self._calculate_rmse(matrix, weights, rewards)
                self.mv_normal_rewards = mv_normal
                snapshot = self._create_snapshot(nexamples, matrix, weights, error)
                self.estimation_meta["snapshots"].append(snapshot)
                self._record_reward_error(mv_normal.mean)

                logger.info(
                    "%s - %s rewards for %s. RMSE: %f; No. Samples: %d",
                    "Estimated" if self.posterior_updates == 0 else "Updated",
                    self.get_name(),
                    self.get_env_name(),
                    error,
                    nexamples,
                )

                # Clear buffers for next data
                self.est_buffer.clear()
        return True


class ConvexSolverGenerativeRewardWrapper(BaseGenerativeRewardWrapper):
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
    Uses Box observation space with additive feature accumulation.
    """

    def __init__(
        self,
        env: gym.Env,
        feats_fn: rewardfeat.TransitionFeatureFn,
        attempt_estimation_episode: int,
        estimation_buffer_mult: int | None = None,
        impute_value: float = 0.0,
        constraints_buffer_limit: int | None = None,
        terminal_constraint_fn: rewardfeat.TransitionFeatureFn | None = None,
        intercept: bool = False,
    ):
        super().__init__(
            env=env,
            feats_fn=feats_fn,
            intercept=intercept,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.attempt_estimation_episode = attempt_estimation_episode
        self.constraints_buffer_limit = constraints_buffer_limit
        self.terminal_constraint_fn = terminal_constraint_fn
        self.weights = None
        self.estimation_meta: dict[str, Any] = {
            "intercept": intercept,
            "snapshots": [],
        }
        # Terminal state buffer for constraints
        self.tst_buffer = DataBuffer(max_capacity=constraints_buffer_limit)

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)  # type: ignore

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            (term or trunc)
            and self.weights is None
            and self.episodes >= self.attempt_estimation_episode
            and self.est_buffer.size() >= self.mdim
        )

    def _on_terminal_state(self, transition: rewardfeat.Transition):
        # Add constraint for terminal state if there is no estimate yet
        if self.weights is None and self.terminal_constraint_fn is not None:
            self.tst_buffer.add(self.terminal_constraint_fn(transition))

    def estimate_rewards(self):
        """
        Estimate rewards with convex optimisation
        with constraints.
        """
        matrix, rewards, nexamples = self._extract_buffer_data()

        # Extract terminal states for constraints
        term_states = (
            np.stack(self.tst_buffer.buffer, dtype=np.float64)
            if self.tst_buffer.size() > 0
            else []
        )

        # Add the intercept column if needed
        matrix = self._add_intercept_to_matrix(matrix, nexamples)
        if self.intercept and len(term_states) > 0:
            term_states = np.concatenate(
                [
                    term_states,
                    np.expand_dims(np.ones(shape=len(term_states)), axis=-1),
                ],
                axis=1,
            )

        try:
            # Construct the problem with constraints
            def constraint_fn(solution):
                return [term_state @ solution == 0 for term_state in term_states]

            weights = optsol.solve_convex_least_squares(
                matrix=matrix, rhs=rewards, constraint_fn=constraint_fn
            )
        except ValueError as err:
            self._handle_estimation_failure(err, matrix, rewards)
        else:
            self.weights = weights
            error = self._calculate_rmse(matrix, weights, rewards)
            snapshot = self._create_snapshot(
                nexamples, matrix, weights, error, constraints=len(term_states)
            )
            self.estimation_meta["snapshots"].append(snapshot)

            logger.info(
                "%s - Estimated rewards for %s. RMSE: %f; No. Samples: %d, # Constraints: %d",
                self.get_name(),
                self.get_env_name(),
                error,
                nexamples,
                len(term_states),
            )


class RecurringConvexSolverGenerativeRewardWrapper(BaseGenerativeRewardWrapper):
    """
    The aggregate reward windows are used to
    estimate the underlying MDP rewards.

    Once estimated, the approximate rewards are used.
    Until then, the aggregate rewards are emitted when
    presented, and zero is used otherwise.

    Rewards are estimated convex Least-Squares.
    Each estimate uses the previous value as an initial guess.
    The initial estimate is gated by the estimation buffer, not by
    episode count. After it, recurring updates follow the windowed
    schedule — exponential (the default), fixed, or doubling.
    Uses Box observation space with continual learning and terminal state constraints.
    """

    def __init__(
        self,
        env: gym.Env,
        feats_fn: rewardfeat.TransitionFeatureFn,
        mode: str = WindowedTaskSchedule.EXPONENTIAL,
        init_attempt_estimation_episode: int = 10,
        schedule_rho: float = WindowedTaskSchedule.DEFAULT_RHO,
        estimation_buffer_mult: int | None = None,
        impute_value: float = 0.0,
        constraints_buffer_limit: int | None = None,
        terminal_constraint_fn: rewardfeat.TransitionFeatureFn | None = None,
        intercept: bool = False,
    ):
        super().__init__(
            env=env,
            feats_fn=feats_fn,
            intercept=intercept,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.mode = mode
        self.init_attempt_estimation_episode = init_attempt_estimation_episode
        self.constraints_buffer_limit = constraints_buffer_limit
        self.terminal_constraint_fn = terminal_constraint_fn
        self.windowed_task_schedule = WindowedTaskSchedule(
            mode=mode,
            init_update_ep=init_attempt_estimation_episode,
            rho=schedule_rho,
        )
        self.update_episode = init_attempt_estimation_episode
        self.posterior_updates = 0
        self.weights: np.ndarray | None = None
        self.estimation_meta: dict[str, Any] = {
            "intercept": intercept,
            "snapshots": [],
        }
        # Terminal state buffer for constraints
        self.tst_buffer = DataBuffer(max_capacity=constraints_buffer_limit)

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _should_buffer_when_estimated(self) -> bool:
        # Continue buffering for continual learning
        return True

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)  # type: ignore

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            (term or trunc)
            and not self.windowed_task_schedule.current_window_done
            and self.est_buffer.size() >= self.mdim
        )

    def _on_terminal_state(self, transition: rewardfeat.Transition):
        # Add constraint for terminal state
        if self.terminal_constraint_fn is not None:
            self.tst_buffer.add(self.terminal_constraint_fn(transition))

    def _on_episode_end(self, term: bool, trunc: bool):
        self.windowed_task_schedule.step(self.episodes)

    def _on_estimation_complete(self, success: bool):
        self.windowed_task_schedule.set_state(succ=success)

    def _get_estimator_info(self) -> dict[str, Any]:
        return {"posterior_updates": self.posterior_updates}

    def estimate_rewards(self) -> bool:
        """
        Estimate rewards with convex optimisation
        with constraints or update estimate
        using prior estimate as the initial
        guess.
        """
        matrix, rewards, nexamples = self._extract_buffer_data()

        # Extract terminal states for constraints
        term_states = (
            np.stack(self.tst_buffer.buffer, dtype=np.float64)
            if self.tst_buffer.size() > 0
            else np.array([])
        )

        # Add the intercept column if needed
        matrix = self._add_intercept_to_matrix(matrix, nexamples)
        if self.intercept and len(term_states) > 0:
            term_states = np.concatenate(
                [
                    term_states,
                    np.expand_dims(np.ones(shape=len(term_states)), axis=-1),
                ],
                axis=1,
            )

        try:
            # Construct the problem with constraints
            def constraint_fn(solution):
                return [term_state @ solution == 0 for term_state in term_states]

            weights = optsol.solve_convex_least_squares(
                matrix=matrix,
                rhs=rewards,
                constraint_fn=constraint_fn,
                warm_start_initial_guess=self.weights,
            )
        except ValueError as err:
            self._handle_estimation_failure(err, matrix, rewards)
            return False
        else:
            if self.weights is not None:
                # This is a posterior update
                self.posterior_updates += 1
            self.weights = weights
            error = self._calculate_rmse(matrix, weights, rewards)
            snapshot = self._create_snapshot(
                nexamples, matrix, weights, error, constraints=len(term_states)
            )
            self.estimation_meta["snapshots"].append(snapshot)

            logger.info(
                "%s - %s rewards for %s. RMSE: %f; No. Samples: %d, # Constraints: %d",
                "Estimated" if self.posterior_updates == 0 else "Updated",
                self.get_name(),
                self.get_env_name(),
                error,
                nexamples,
                self.tst_buffer.size(),
            )

            # Clear buffers for next data
            self.est_buffer.clear()
            self.tst_buffer.clear()
        return True


def list_size(xs: list[Any]) -> int:
    """
    Gets the size of a list in bytes.
    """
    return sys.getsizeof(xs) - sys.getsizeof([])


def drop_samples(
    frac: float, arrays: Sequence[np.ndarray], rng: np.random.Generator
) -> tuple[int, Sequence[np.ndarray]]:
    """
    Drops a fraction of examples of every arrays.
    Arrays are assumed to be of equal length in their
    most outer dimension.
    """
    first = next(iter(arrays))
    nexamples = len(first)
    nexamples_to_drop = int(nexamples * frac)
    indices = rng.choice(
        np.arange(nexamples),
        nexamples - nexamples_to_drop,
        replace=False,
    )
    return nexamples_to_drop, tuple(array[indices] for array in arrays)
