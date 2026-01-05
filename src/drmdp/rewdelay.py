import abc
import logging
import random
import sys
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import gymnasium as gym
import numpy as np

from drmdp import mathutils, metrics, optsol, transform


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


class WindowedTaskSchedule:
    """
    Sets schedule for updates, using two types of schedules:
    1. Fixed interval (fixed)
    2. Doubling size (double)
    """

    FIXED = "fixed"
    DOUBLE = "double"

    def __init__(self, mode: str, init_update_ep: int):
        """
        Instatiates the class for a given update schedule.
        """
        if mode not in (self.FIXED, self.DOUBLE):
            raise ValueError(
                f"Unsupported mode: {mode}. Must be ({self.FIXED}, {self.DOUBLE})"
            )

        self.mode = mode
        self.init_update_ep = init_update_ep
        self.curr_update_ep = init_update_ep
        self._done = False

        self.next_update_ep = (
            self.curr_update_ep * 2
            if self.mode == self.DOUBLE
            else self.curr_update_ep + self.init_update_ep
        )

    def step(self, episode: int) -> None:
        """
        Updates the estimation window.
        """
        if episode == self.next_update_ep:
            # New window
            self.curr_update_ep = self.next_update_ep
            self.next_update_ep = (
                self.curr_update_ep * 2
                if self.mode == self.DOUBLE
                else self.curr_update_ep + self.init_update_ep
            )
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
        ft_op: transform.FTOp,
        use_bias: bool = False,
        impute_value: float = 0.0,
        estimation_buffer_mult: Optional[int] = None,
    ):
        super().__init__(env)
        self._validate_observation_space(ft_op)
        self._validate_action_space(ft_op)

        self.ft_op = ft_op
        self.use_bias = use_bias
        self.impute_value = impute_value
        self.estimation_buffer_mult = estimation_buffer_mult

        self.episodes = 0
        self.mdim = self._compute_mdim(ft_op)
        self._latest_obs = None
        self._segment_features = None
        self.rng = np.random.default_rng()
        self.est_buffer = DataBuffer(
            max_capacity=self.mdim * estimation_buffer_mult
            if estimation_buffer_mult
            else None
        )

    def _validate_action_space(self, ft_op: transform.FTOp):
        """Validate action space is Discrete."""
        if not isinstance(ft_op.output_space.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"ft_op action space must be Discrete. "
                f"Got: {type(ft_op.output_space.action_space)}"
            )

    @abc.abstractmethod
    def _validate_observation_space(self, ft_op: transform.FTOp):
        """Validate observation space type (Box or Discrete)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_mdim(self, ft_op: transform.FTOp) -> int:
        """Compute feature dimension."""
        raise NotImplementedError

    @abc.abstractmethod
    def _initialize_segment_features(self):
        """Initialize segment features array."""
        raise NotImplementedError

    @abc.abstractmethod
    def _accumulate_step_features(self, latest_step_feats: transform.Example):
        """Accumulate features for the current step."""
        raise NotImplementedError

    def _get_estimation_features(self, feats: np.ndarray) -> np.ndarray:
        """Get features for reward estimation, adding bias if needed."""
        if self.use_bias:
            return np.concatenate([feats, np.array([1.0])])
        return feats

    @abc.abstractmethod
    def _has_estimate(self) -> bool:
        """Check if reward estimate exists."""
        raise NotImplementedError

    def _get_features_for_estimation(
        self, latest_step_feats: transform.Example
    ) -> np.ndarray:
        """
        Get features to use for reward estimation.
        Default: use accumulated segment features.
        Override to use latest step features instead (e.g., for Bayesian methods).
        """
        del latest_step_feats
        return self._segment_features

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

    def _on_terminal_state(self, next_obs, action: int = 0):
        """
        Handle terminal state.
        Override in child classes that need terminal state handling.
        """
        del next_obs
        del action

    def _get_estimator_info(self) -> Dict[str, Any]:
        """
        Get estimator-specific info to include in step return.
        Override to add custom info.
        """
        return {}

    def _extract_buffer_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
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

    def _add_bias_to_matrix(self, matrix: np.ndarray, nexamples: int) -> np.ndarray:
        """Add bias column to matrix if use_bias is True."""
        if self.use_bias:
            return np.concatenate(
                [
                    matrix,
                    np.expand_dims(np.ones(shape=nexamples), axis=-1),
                ],
                axis=1,
            )
        return matrix

    def _handle_estimation_failure(
        self, err: ValueError, matrix: np.ndarray, rewards: np.ndarray
    ):
        """Handle estimation failure by logging and dropping 5% of samples."""
        logging.debug(
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
        logging.debug(
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
    ) -> Dict[str, Any]:
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
        latest_step_feats = self.ft_op(transform.Example(self._latest_obs, action))
        self._accumulate_step_features(latest_step_feats)
        next_obs, reward, term, trunc, info = super().step(action)

        # Buffer segment data at segment end (for continual learning)
        if (
            self._should_buffer_when_estimated()
            and info["segment_step"] == info["delay"] - 1
        ):
            self.est_buffer.add((self._segment_features, reward))
            # Reset for the next segment
            self._segment_features = self._initialize_segment_features()

        if self._has_estimate():
            # Use estimated reward
            feats = self._get_features_for_estimation(latest_step_feats)
            feats = self._get_estimation_features(feats)
            reward = self._get_estimated_reward(feats)
            est_state = OptState.SOLVED
        else:
            # Add example to buffer and use aggregate reward (one-time learning)
            if not self._should_buffer_when_estimated():
                if info["segment_step"] == info["delay"] - 1:
                    self.est_buffer.add((self._segment_features, reward))
                    # Reset for the next segment
                    self._segment_features = self._initialize_segment_features()
                else:
                    # Impute until rewards are estimated
                    reward = self.impute_value
            else:
                # Impute when not at segment end
                if info["segment_step"] != info["delay"] - 1:
                    reward = self.impute_value
                # else, use aggregate reward
            est_state = OptState.UNSOLVED

        # Handle terminal state
        if term:
            self._on_terminal_state(next_obs, action=0)

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
        ft_op: transform.FTOp,
        attempt_estimation_episode: int,
        estimation_buffer_mult: Optional[int] = None,
        use_bias: bool = False,
        impute_value: float = 0.0,
        check_factors: bool = False,
    ):
        super().__init__(
            env=env,
            ft_op=ft_op,
            use_bias=use_bias,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.attempt_estimation_episode = attempt_estimation_episode
        self.check_factors = check_factors
        self.weights = None
        self.estimation_meta = {
            "use_bias": self.use_bias,
            "check_factors": check_factors,
            "snapshots": [],
        }

    def _validate_observation_space(self, ft_op: transform.FTOp):
        if not isinstance(ft_op.output_space.observation_space, gym.spaces.Discrete):
            raise ValueError(
                f"ft_op space must be Discrete. "
                f"Got: {type(ft_op.output_space.observation_space)}"
            )

    def _compute_mdim(self, ft_op: transform.FTOp) -> int:
        return ft_op.output_space.observation_space.n

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _accumulate_step_features(self, latest_step_feats: transform.Example):
        # Discretized: increment count at feature index
        self._segment_features[latest_step_feats.observation] += 1

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            self.weights is None
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
                return

        # Add bias column if needed
        matrix = self._add_bias_to_matrix(matrix, nexamples)

        try:
            weights = optsol.solve_least_squares(matrix=matrix, rhs=rewards)
        except ValueError as err:
            self._handle_estimation_failure(err, matrix, rewards)
        else:
            self.weights = weights
            error = self._calculate_rmse(matrix, weights, rewards)
            snapshot = self._create_snapshot(nexamples, matrix, weights, error)
            self.estimation_meta["snapshots"].append(snapshot)
            logging.info(
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
        ft_op: transform.FTOp,
        attempt_estimation_episode: int,
        estimation_buffer_mult: Optional[int] = None,
        use_bias: bool = False,
        impute_value: float = 0.0,
        check_factors: bool = False,
    ):
        super().__init__(
            env=env,
            ft_op=ft_op,
            use_bias=use_bias,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.attempt_estimation_episode = attempt_estimation_episode
        self.check_factors = check_factors
        self.weights = None
        self.estimation_meta = {
            "use_bias": use_bias,
            "check_factors": check_factors,
            "snapshots": [],
        }

    def _validate_observation_space(self, ft_op: transform.FTOp):
        if not isinstance(ft_op.output_space.observation_space, gym.spaces.Box):
            raise ValueError(
                f"ft_op space must be Box. "
                f"Got: {type(ft_op.output_space.observation_space)}"
            )

    def _compute_mdim(self, ft_op: transform.FTOp) -> int:
        return np.size(ft_op.output_space.observation_space.high)

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _accumulate_step_features(self, latest_step_feats: transform.Example):
        # Additive: sum feature vectors
        self._segment_features += latest_step_feats.observation

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            self.weights is None
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
                return

        # Add bias column if needed
        matrix = self._add_bias_to_matrix(matrix, nexamples)

        try:
            weights = optsol.solve_least_squares(matrix=matrix, rhs=rewards)
        except ValueError as err:
            self._handle_estimation_failure(err, matrix, rewards)
        else:
            self.weights = weights
            error = self._calculate_rmse(matrix, weights, rewards)
            snapshot = self._create_snapshot(nexamples, matrix, weights, error)
            self.estimation_meta["snapshots"].append(snapshot)
            logging.info(
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
    Rewards are first estimated after `init_attempt_estimation_episode`.
    After that, they are either updated following a doubling
    schedule or at fixed intervals.
    Uses Box observation space with continual learning (buffering continues after estimation).
    """

    def __init__(
        self,
        env: gym.Env,
        ft_op: transform.FTOp,
        mode: str = WindowedTaskSchedule.DOUBLE,
        init_attempt_estimation_episode: int = 10,
        estimation_buffer_mult: Optional[int] = None,
        use_bias: bool = False,
        impute_value: float = 0.0,
        check_factors: bool = False,
    ):
        super().__init__(
            env=env,
            ft_op=ft_op,
            use_bias=use_bias,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.mode = mode
        self.init_attempt_estimation_episode = init_attempt_estimation_episode
        self.check_factors = check_factors
        self.windowed_task_schedule = WindowedTaskSchedule(
            mode=mode, init_update_ep=init_attempt_estimation_episode
        )
        self.update_episode = init_attempt_estimation_episode
        self.posterior_updates = 0
        self.mv_normal_rewards: Optional[optsol.MultivariateNormal] = None
        self.estimation_meta: Dict[str, Any] = {
            "use_bias": use_bias,
            "check_factors": check_factors,
            "snapshots": [],
        }

    def _validate_observation_space(self, ft_op: transform.FTOp):
        if not isinstance(ft_op.output_space.observation_space, gym.spaces.Box):
            raise ValueError(
                f"ft_op space must be Box. "
                f"Got: {type(ft_op.output_space.observation_space)}"
            )

    def _compute_mdim(self, ft_op: transform.FTOp) -> int:
        return np.size(ft_op.output_space.observation_space.high)

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _accumulate_step_features(self, latest_step_feats: transform.Example):
        # Additive: sum feature vectors
        self._segment_features += latest_step_feats.observation

    def _has_estimate(self) -> bool:
        return self.mv_normal_rewards is not None

    def _get_features_for_estimation(
        self, latest_step_feats: transform.Example
    ) -> np.ndarray:
        # Use latest step features, not accumulated
        return latest_step_feats.observation

    def _should_buffer_when_estimated(self) -> bool:
        # Continue buffering for continual learning
        return True

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.mv_normal_rewards.mean)

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            not self.windowed_task_schedule.current_window_done
            and self.est_buffer.size() >= self.mdim
        )

    def _on_episode_end(self, term: bool, trunc: bool):
        self.windowed_task_schedule.step(self.episodes)

    def _on_estimation_complete(self, success: bool):
        self.windowed_task_schedule.set_state(succ=success)

    def _get_estimator_info(self) -> Dict[str, Any]:
        return {"posterior_updates": self.posterior_updates}

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
                return False

        # Add bias column if needed
        matrix = self._add_bias_to_matrix(matrix, nexamples)

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

                logging.info(
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
        ft_op: transform.FTOp,
        attempt_estimation_episode: int,
        estimation_buffer_mult: Optional[int] = None,
        use_bias: bool = False,
        impute_value: float = 0.0,
        constraints_buffer_limit: Optional[int] = None,
    ):
        super().__init__(
            env=env,
            ft_op=ft_op,
            use_bias=use_bias,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.attempt_estimation_episode = attempt_estimation_episode
        self.constraints_buffer_limit = constraints_buffer_limit
        self.weights = None
        self.estimation_meta = {"use_bias": use_bias, "snapshots": []}
        # Terminal state buffer for constraints
        self.tst_buffer = DataBuffer(max_capacity=constraints_buffer_limit)

    def _validate_observation_space(self, ft_op: transform.FTOp):
        if not isinstance(ft_op.output_space.observation_space, gym.spaces.Box):
            raise ValueError(
                f"ft_op space must be Box. "
                f"Got: {type(ft_op.output_space.observation_space)}"
            )

    def _compute_mdim(self, ft_op: transform.FTOp) -> int:
        return np.size(ft_op.output_space.observation_space.high)

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _accumulate_step_features(self, latest_step_feats: transform.Example):
        # Additive: sum feature vectors
        self._segment_features += latest_step_feats.observation

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            self.weights is None
            and self.episodes >= self.attempt_estimation_episode
            and self.est_buffer.size() >= self.mdim
        )

    def _on_terminal_state(self, next_obs, action: int = 0):
        # Add constraint for terminal state if there is no estimate yet
        if self.weights is None:
            self.tst_buffer.add(
                self.ft_op(transform.Example(next_obs, action)).observation
            )

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

        # Add bias column if needed
        matrix = self._add_bias_to_matrix(matrix, nexamples)
        if self.use_bias and len(term_states) > 0:
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

            logging.info(
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
    Rewards are first estimated after `init_attempt_estimation_episode`.
    After that, they are either updated following a doubling
    schedule or at fixed intervals.
    Uses Box observation space with continual learning and terminal state constraints.
    """

    def __init__(
        self,
        env: gym.Env,
        ft_op: transform.FTOp,
        mode: str = WindowedTaskSchedule.DOUBLE,
        init_attempt_estimation_episode: int = 10,
        estimation_buffer_mult: Optional[int] = None,
        use_bias: bool = False,
        impute_value: float = 0.0,
        constraints_buffer_limit: Optional[int] = None,
    ):
        super().__init__(
            env=env,
            ft_op=ft_op,
            use_bias=use_bias,
            impute_value=impute_value,
            estimation_buffer_mult=estimation_buffer_mult,
        )
        self.mode = mode
        self.init_attempt_estimation_episode = init_attempt_estimation_episode
        self.constraints_buffer_limit = constraints_buffer_limit
        self.windowed_task_schedule = WindowedTaskSchedule(
            mode=mode, init_update_ep=init_attempt_estimation_episode
        )
        self.update_episode = init_attempt_estimation_episode
        self.posterior_updates = 0
        self.weights: Optional[np.ndarray] = None
        self.estimation_meta: Dict[str, Any] = {
            "use_bias": self.use_bias,
            "snapshots": [],
        }
        # Terminal state buffer for constraints
        self.tst_buffer = DataBuffer(max_capacity=constraints_buffer_limit)

    def _validate_observation_space(self, ft_op: transform.FTOp):
        if not isinstance(ft_op.output_space.observation_space, gym.spaces.Box):
            raise ValueError(
                f"ft_op space must be Box. "
                f"Got: {type(ft_op.output_space.observation_space)}"
            )

    def _compute_mdim(self, ft_op: transform.FTOp) -> int:
        return np.size(ft_op.output_space.observation_space.high)

    def _initialize_segment_features(self):
        return np.zeros(shape=(self.mdim))

    def _accumulate_step_features(self, latest_step_feats: transform.Example):
        # Additive: sum feature vectors
        self._segment_features += latest_step_feats.observation

    def _has_estimate(self) -> bool:
        return self.weights is not None

    def _get_features_for_estimation(
        self, latest_step_feats: transform.Example
    ) -> np.ndarray:
        # Use latest step features, not accumulated
        return latest_step_feats.observation

    def _should_buffer_when_estimated(self) -> bool:
        # Continue buffering for continual learning
        return True

    def _get_estimated_reward(self, feats: np.ndarray) -> float:
        return np.dot(feats, self.weights)

    def _should_attempt_estimation(self, term: bool, trunc: bool) -> bool:
        return (
            not self.windowed_task_schedule.current_window_done
            and self.est_buffer.size() >= self.mdim
        )

    def _on_terminal_state(self, next_obs, action: int = 0):
        # Add constraint for terminal state
        self.tst_buffer.add(self.ft_op(transform.Example(next_obs, action)).observation)

    def _on_episode_end(self, term: bool, trunc: bool):
        self.windowed_task_schedule.step(self.episodes)

    def _on_estimation_complete(self, success: bool):
        self.windowed_task_schedule.set_state(succ=success)

    def _get_estimator_info(self) -> Dict[str, Any]:
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

        # Add bias column if needed
        matrix = self._add_bias_to_matrix(matrix, nexamples)
        if self.use_bias and len(term_states) > 0:
            term_states = np.concatenate(
                [
                    term_states,
                    np.expand_dims(np.ones(shape=nexamples), axis=-1),
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

            logging.info(
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


def list_size(xs: List[Any]) -> int:
    """
    Gets the size of a list in bytes.
    """
    return sys.getsizeof(xs) - sys.getsizeof([])


def drop_samples(
    frac: float, arrays: Sequence[np.ndarray], rng: np.random.Generator
) -> Tuple[int, Sequence[np.ndarray]]:
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
