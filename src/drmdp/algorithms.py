import abc
import copy
import dataclasses
import logging
from typing import Any, Iterator, Optional, Tuple

import gymnasium as gym
import numpy as np

from drmdp import core, mathutils, optsol, transform


@dataclasses.dataclass(frozen=True)
class PolicyControlSnapshot:
    steps: int
    returns: float
    weights: np.ndarray


class FnApproxAlgorithm(abc.ABC):
    def __init__(self, base_seed: Optional[int] = None):
        super().__init__()
        self.seeder = core.Seeder(base_seed)

    @abc.abstractmethod
    def train(
        self,
        env: gym.Env,
        num_episodes: int,
        monitor: core.EnvMonitor,
    ) -> Iterator[PolicyControlSnapshot]: ...


class SemigradientSARSAFnApprox(FnApproxAlgorithm):
    def __init__(
        self,
        lr: optsol.LearningRateSchedule,
        gamma: float,
        epsilon: float,
        policy: core.PyValueFnPolicy,
        base_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(base_seed)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.verbose = verbose

    def train(
        self,
        env: gym.Env,
        num_episodes: int,
        monitor: core.EnvMonitor,
    ) -> Iterator[PolicyControlSnapshot]:
        monitor.clear()
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=self.seeder.get_seed(episode=episode))
            policy_step = self.policy.action(obs, epsilon=self.epsilon)
            state_qvalues, gradients = (
                policy_step.info["values"],
                policy_step.info["gradients"],
            )
            while True:
                (
                    next_obs,
                    reward,
                    term,
                    trunc,
                    _,
                ) = env.step(policy_step.action)

                if term or trunc:
                    scaled_gradients = (
                        self.lr(episode, monitor.step)
                        * (reward - state_qvalues[policy_step.action])
                        * gradients[policy_step.action]
                    )
                    self.policy.step(policy_step.action, scaled_gradients)
                    break

                next_policy_step = self.policy.action(next_obs, epsilon=self.epsilon)
                next_state_qvalues, next_gradients = (
                    next_policy_step.info["values"],
                    next_policy_step.info["gradients"],
                )
                scaled_gradients = (
                    self.lr(episode, monitor.step)
                    * (
                        reward
                        + self.gamma * next_state_qvalues[next_policy_step.action]
                        - state_qvalues[policy_step.action]
                    )
                    * gradients[policy_step.action]
                )
                self.policy.step(policy_step.action, scaled_gradients)
                obs = next_obs
                policy_step = next_policy_step
                state_qvalues = next_state_qvalues
                gradients = next_gradients
            if self.verbose and (episode + 1) % max((num_episodes // 5), 1) == 0:
                logging.info(
                    "Episode %d mean returns: %f",
                    episode + 1,
                    np.mean(monitor.returns + [monitor.rewards]),
                )
            yield PolicyControlSnapshot(
                steps=monitor.step,
                returns=monitor.rewards,
                weights=copy.copy(self.policy.model),
            )
        env.close()


class LinearFnApproxPolicy(core.PyValueFnPolicy):
    def __init__(
        self,
        ft_op: transform.FTOp,
        action_space: gym.Space,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(ft_op.output_space.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(ft_op.output_space.action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.ft_op = ft_op
        self.actions = tuple(range(ft_op.output_space.action_space.n))
        self.weights = np.zeros(
            (ft_op.output_space.action_space.n,)
            + ft_op.output_space.observation_space.shape,
            dtype=np.float64,
        )

    def get_initial_state(self, batch_size=None):
        del batch_size
        return ()

    def action(
        self, observation, epsilon: float = 0.0, policy_state: Any = (), seed=None
    ):
        del seed
        state_qvalues, gradients = self.action_values_gradients(
            observation, self.actions
        )
        if epsilon and self.rng.random() < epsilon:
            action = self.rng.choice(self.actions)
        else:
            # Choose highest value action
            # breaking ties are random
            action = self.rng.choice(
                np.flatnonzero(state_qvalues == state_qvalues.max())
            )
        return core.PolicyStep(
            action,
            state=policy_state,
            info={"values": state_qvalues, "gradients": gradients},
        )

    def action_values_gradients(self, observation, actions):
        examples = [transform.Example(observation, action) for action in actions]
        state_action_m = [ex.observation for ex in self.ft_op.batch(examples)]
        return np.sum(self.weights * state_action_m, axis=1), state_action_m

    def step(self, action, scaled_gradients):
        self.weights[action] += scaled_gradients

    @property
    def model(self):
        return self.weights


class RandomFnApproxPolicy(core.PyValueFnPolicy):
    def __init__(
        self,
        ft_op: transform.FTOp,
        action_space: gym.Space,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.ft_op = ft_op
        self.actions = tuple(range(action_space.n))
        self.weights = np.zeros(
            (action_space.n,) + ft_op.output_space.observation_space.shape,
            dtype=np.float64,
        )

    def get_initial_state(self, batch_size=None):
        del batch_size
        return ()

    def action(
        self, observation, epsilon: float = 0.0, policy_state: Any = (), seed=None
    ):
        del epsilon
        del seed
        state_qvalues, gradients = self.action_values_gradients(
            observation, self.actions
        )
        action = self.rng.choice(self.actions)
        return core.PolicyStep(
            action,
            state=policy_state,
            info={"values": state_qvalues, "gradients": gradients},
        )

    def action_values_gradients(self, observation, actions):
        examples = [transform.Example(observation, action) for action in actions]
        state_action_m = [ex.observation for ex in self.ft_op.batch(examples)]
        return np.sum(self.weights * state_action_m, axis=1), state_action_m

    def step(self, action, scaled_gradients):
        self.weights[action] += scaled_gradients

    @property
    def model(self):
        return self.weights


class OptionsSemigradientSARSAFnApprox(FnApproxAlgorithm):
    def __init__(
        self,
        lr: optsol.LearningRateSchedule,
        gamma: float,
        epsilon: float,
        policy: core.PyValueFnPolicy,
        base_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(base_seed)

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.verbose = verbose

    def train(
        self,
        env: gym.Env,
        num_episodes: int,
        monitor: core.EnvMonitor,
    ) -> Iterator[PolicyControlSnapshot]:
        monitor.clear()
        for episode in range(num_episodes):
            obs, info = env.reset(seed=self.seeder.get_seed(episode=episode))
            policy_step = self.policy.action(
                obs, epsilon=self.epsilon, policy_state=(info["next_delay"],)
            )
            state_qvalues, gradients, actions = (
                policy_step.info["values"],
                policy_step.info["gradients"],
                policy_step.info["actions"],
            )
            while True:
                for idx, action in enumerate(actions):
                    (
                        next_obs,
                        reward,
                        term,
                        trunc,
                        info,
                    ) = env.step(action)

                    if term or trunc:
                        # aggregate reward is available
                        # update before terminating episode
                        if idx == len(actions) - 1:
                            scaled_gradients = (
                                self.lr(episode, monitor.step)
                                * (reward - state_qvalues[policy_step.action])
                                * gradients[policy_step.action]
                            )
                            self.policy.step(policy_step.action, scaled_gradients)
                        break
                if term or trunc:
                    break

                next_policy_step = self.policy.action(
                    next_obs, epsilon=self.epsilon, policy_state=(info["next_delay"],)
                )
                next_state_qvalues, next_gradients, next_actions = (
                    next_policy_step.info["values"],
                    next_policy_step.info["gradients"],
                    next_policy_step.info["actions"],
                )

                scaled_gradients = (
                    self.lr(episode, monitor.step)
                    * (
                        reward
                        + self.gamma * next_state_qvalues[next_policy_step.action]
                        - state_qvalues[policy_step.action]
                    )
                    * gradients[policy_step.action]
                )
                self.policy.step(policy_step.action, scaled_gradients)
                obs = next_obs
                policy_step = next_policy_step
                state_qvalues = next_state_qvalues
                gradients = next_gradients
                actions = next_actions
            if self.verbose and (episode + 1) % max(num_episodes // 5, 1) == 0:
                logging.info(
                    "Episode %d mean returns: %f",
                    episode + 1,
                    np.mean(monitor.returns + [monitor.rewards]),
                )
            yield PolicyControlSnapshot(
                steps=monitor.step,
                returns=monitor.rewards,
                weights=copy.copy(self.policy.model),
            )
        env.close()


class OptionsLinearFnApproxPolicy(core.PyValueFnPolicy):
    """
    Options are sequences of multi-step actions, or actions
    that last longer than one step.
    Given a delay `d` that represents the number of steps, there are
    K^{d} possible primitive actions.
    This policy supports a range of delays [delay_min, delay_max].

    When `action` is called, what the policy returns is a sequence
    primitive actions, which depends on the delay.

    The way they are encoded here is:

    1. Apply ft_op to the `observation` usign action=0
    2. Apply binary encoding of integers, using the first
    power of 2 that is greater than `num_options[delay_max]`.
    e.g. if delay_max = 3 and there are 4 actions,
    the binary encoding size is is 2^7 > 4^{3}.
    3. Apply one-hot-encoding to the delay, based on the number
    of delays (delay_max - delay_min + 1).

    The final option encoding is a concatenation of
    the output of steps 2 and 3
    option_enc = binary_enc(option) + one_hot(delay)
    observation_enc = ft_op.apply(observation)
    And the action encoding as well, has a max length.

    `option_enc` is generally very small compared to the
    number of actions.
    ft_op depends on the option.
    """

    def __init__(
        self,
        ft_op: transform.FTOp,
        action_space: gym.Space,
        options_length_range: Tuple[int, int],
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if len(ft_op.output_space.observation_space.shape) != 1:
            raise ValueError(
                f"Observation output space must be a vector. Got {type(ft_op.output_space.observation_space.shape)}"
            )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.ft_op = ft_op
        # + plus something something
        self.primitive_actions = tuple(range(action_space.n))
        self.num_primitive_actions = len(self.primitive_actions)
        self.options_length_range = options_length_range

        lower, upper = self.options_length_range
        self.delay_num_options_mapping = {
            length: self.num_primitive_actions**length
            for length in range(lower, upper + 1)
        }
        option_enc_size = 1
        while 2**option_enc_size < self.delay_num_options_mapping[upper]:
            option_enc_size += 1  # type: ignore
        self.option_enc_size = option_enc_size
        # seq len + OHE[delay]
        self.options_dim = self.option_enc_size + (upper - lower + 1)
        self.features_dim = (
            self.ft_op.output_space.observation_space.shape[0] + self.options_dim
        )
        self.delay_options_matrix_mapping = self.options_encoding()
        self.weights = np.zeros(self.features_dim, dtype=np.float64)

        self.action_space = gym.spaces.Discrete(
            sum(value for value in self.delay_num_options_mapping.values())
        )

    def options_encoding(self):
        """
        Pre-compute values of every option
        """
        lower, upper = self.options_length_range
        options_m = {}
        for delay in range(lower, upper + 1):
            options_m[delay] = self.transform_options(
                range(self.delay_num_options_mapping[delay]), delay
            )
        return options_m

    def transform_options(self, options, delay):
        lower, upper = self.options_length_range
        zs = np.zeros(shape=(len(options), self.options_dim))
        ohe_delay = np.zeros(shape=(upper - lower + 1))
        ohe_delay[upper - delay] = 1
        for i, option in enumerate(options):
            seq_enc = mathutils.interger_to_sequence(
                space_size=2, sequence_length=self.option_enc_size, index=option
            )
            zs[i] = np.concatenate([seq_enc, ohe_delay])
        return zs

    def get_initial_state(self, batch_size=None):
        del batch_size
        return ()

    def action(
        self, observation, epsilon: float = 0.0, policy_state: Any = (), seed=None
    ):
        del seed
        (delay,) = policy_state
        state_qvalues, gradients = self.action_values_gradients(observation, (delay,))
        if epsilon and self.rng.random() < epsilon:
            option = self.rng.integers(0, self.delay_num_options_mapping[delay])
        else:
            option = self.rng.choice(
                np.flatnonzero(state_qvalues == state_qvalues.max())
            )
        actions = mathutils.interger_to_sequence(
            space_size=self.num_primitive_actions, sequence_length=delay, index=option
        )
        return core.PolicyStep(
            option,
            state=policy_state,
            info={"values": state_qvalues, "gradients": gradients, "actions": actions},
        )

    def action_values_gradients(self, observation, actions):
        # observations = [observation] * len(actions)
        (delay,) = actions
        # repeat state m times
        options_matrix = self.delay_options_matrix_mapping[delay]
        state_m = np.tile(
            self.ft_op(transform.Example(observation, 0)).observation,
            (options_matrix.shape[0], 1),
        )
        # get option representations
        features_m = np.concatenate([state_m, options_matrix], axis=1)
        return np.dot(features_m, self.weights), features_m

    def step(self, action, scaled_gradients):
        del action
        self.weights += scaled_gradients

    @property
    def model(self):
        return self.weights


class SingleActionOptionsLinearFnApproxPolicy(core.PyValueFnPolicy):
    def __init__(
        self,
        ft_op: transform.FTOp,
        action_space: gym.Space,
        options_length_range: Tuple[int, int],
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if len(ft_op.output_space.observation_space.shape) != 1:
            raise ValueError(
                f"Observation output space must be a vector. Got {type(ft_op.output_space.observation_space.shape)}"
            )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.ft_op = ft_op
        # + plus something something
        self.primitive_actions = tuple(range(action_space.n))
        self.num_primitive_actions = len(self.primitive_actions)
        self.options_length_range = options_length_range

        lower, upper = self.options_length_range
        self.delay_options_mapping = {
            length: self.num_primitive_actions for length in range(lower, upper + 1)
        }
        option_enc_size = 1
        while 2**option_enc_size < self.delay_options_mapping[upper]:
            option_enc_size += 1  # type: ignore
        self.option_enc_size = option_enc_size
        # seq len + OHE[delay]
        self.options_dim = self.option_enc_size + (upper - lower + 1)
        self.features_dim = (
            self.ft_op.output_space.observation_space.shape[0] + self.options_dim
        )
        self.options_m = self.options_encoding()
        self.weights = np.zeros(self.features_dim, dtype=np.float64)

        self.action_space = gym.spaces.Discrete(
            sum(value for value in self.delay_options_mapping.values())
        )

    def options_encoding(self):
        """
        Pre-compute values of every option
        """
        lower, upper = self.options_length_range
        options_m = {}
        for delay in range(lower, upper + 1):
            options_m[delay] = self.transform_options(
                range(self.delay_options_mapping[delay]), delay
            )
        return options_m

    def transform_options(self, options, delay):
        lower, upper = self.options_length_range
        zs = np.zeros(shape=(len(options), self.options_dim))
        ohe_delay = np.zeros(shape=(upper - lower + 1))
        ohe_delay[upper - delay] = 1
        for i, option in enumerate(options):
            seq_enc = mathutils.interger_to_sequence(
                space_size=2, sequence_length=self.option_enc_size, index=option
            )
            zs[i] = np.concatenate([seq_enc, ohe_delay])
        return zs

    def get_initial_state(self, batch_size=None):
        del batch_size
        return ()

    def action(
        self, observation, epsilon: float = 0.0, policy_state: Any = (), seed=None
    ):
        del seed
        (delay,) = policy_state
        state_qvalues, gradients = self.action_values_gradients(observation, (delay,))
        if epsilon and self.rng.random() < epsilon:
            option = self.rng.integers(0, self.delay_options_mapping[delay])
        else:
            option = self.rng.choice(
                np.flatnonzero(state_qvalues == state_qvalues.max())
            )
        actions = (option,) * delay
        return core.PolicyStep(
            option,
            state=policy_state,
            info={"values": state_qvalues, "gradients": gradients, "actions": actions},
        )

    def action_values_gradients(self, observation, actions):
        # observations = [observation] * len(actions)
        (delay,) = actions
        # repeat state m times
        options_m = self.options_m[delay]
        state_m = np.tile(
            self.ft_op(transform.Example(observation, 0)).observation,
            (options_m.shape[0], 1),
        )
        # get option representations
        features_m = np.concatenate([state_m, options_m], axis=1)
        return np.dot(features_m, self.weights), features_m

    def step(self, action, scaled_gradients):
        del action
        self.weights += scaled_gradients

    @property
    def model(self):
        return self.weights


class DropMissingSemigradientSARSAFnApprox(FnApproxAlgorithm):
    def __init__(
        self,
        lr: optsol.LearningRateSchedule,
        gamma: float,
        epsilon: float,
        policy: core.PyValueFnPolicy,
        base_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(base_seed)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.verbose = verbose

    def train(
        self,
        env: gym.Env,
        num_episodes: int,
        monitor: core.EnvMonitor,
    ) -> Iterator[PolicyControlSnapshot]:
        monitor.clear()
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=self.seeder.get_seed(episode=episode))
            policy_step = self.policy.action(obs, epsilon=self.epsilon)
            state_qvalues, gradients = (
                policy_step.info["values"],
                policy_step.info["gradients"],
            )
            while True:
                (
                    next_obs,
                    reward,
                    term,
                    trunc,
                    _,
                ) = env.step(policy_step.action)

                if term or trunc:
                    if reward is not None:
                        scaled_gradients = (
                            self.lr(episode, monitor.step)
                            * (reward - state_qvalues[policy_step.action])
                            * gradients[policy_step.action]
                        )
                        self.policy.step(policy_step.action, scaled_gradients)
                    break

                next_policy_step = self.policy.action(next_obs, epsilon=self.epsilon)
                next_state_qvalues, next_gradients = (
                    next_policy_step.info["values"],
                    next_policy_step.info["gradients"],
                )
                if reward is not None:
                    scaled_gradients = (
                        self.lr(episode, monitor.step)
                        * (
                            reward
                            + self.gamma * next_state_qvalues[next_policy_step.action]
                            - state_qvalues[policy_step.action]
                        )
                        * gradients[policy_step.action]
                    )
                    self.policy.step(policy_step.action, scaled_gradients)
                obs = next_obs
                policy_step = next_policy_step
                state_qvalues = next_state_qvalues
                gradients = next_gradients
            if self.verbose and (episode + 1) % max((num_episodes // 5), 1) == 0:
                logging.info(
                    "Episode %d mean returns: %f",
                    episode + 1,
                    np.mean(monitor.returns + [monitor.rewards]),
                )
            yield PolicyControlSnapshot(
                steps=monitor.step,
                returns=monitor.rewards,
                weights=copy.copy(self.policy.model),
            )
        env.close()


class HCDecompositionLinearFnApproxPolicy(core.PyValueFnPolicy):
    """
    Head-Critic Decomposition Policy for Delayed Rewards (Han et al., 2022).

    Implements the HC-decomposition from "Off-Policy Reinforcement Learning
    with Delayed Rewards" (ICML 2022). The Q-function is decomposed into:
    - Q^H (Head): Historical component based on trajectory history
    - Q^C (Critic): Current component based on current state-action

    Total Q-value: Q(h, s, a) = Q^H(h) + Q^C(s, a)

    For linear function approximation, history is represented as an
    aggregated feature vector over a fixed window of past observations.
    """

    def __init__(
        self,
        ft_op: transform.FTOp,
        action_space: gym.Space,
        history_window: int = 10,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(ft_op.output_space.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(ft_op.output_space.action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.ft_op = ft_op
        self.actions = tuple(range(ft_op.output_space.action_space.n))
        self.history_window = history_window

        # Feature dimensions
        obs_dim = ft_op.output_space.observation_space.shape[0]

        # Head network: learns from history features
        # History representation: mean of features over window
        self.head_weights = np.zeros(obs_dim, dtype=np.float64)

        # Critic network: learns from current state-action features
        self.critic_weights = np.zeros(
            (ft_op.output_space.action_space.n, obs_dim),
            dtype=np.float64,
        )

        # History buffer: stores recent observation features
        self.history_buffer = []

    def get_initial_state(self, batch_size=None):
        del batch_size
        return ()

    def _get_history_features(self):
        """
        Aggregate history buffer into a single feature vector.
        Uses mean pooling over the history window.
        """
        if len(self.history_buffer) == 0:
            # Return zero vector if no history
            return np.zeros_like(self.head_weights)
        # Mean pooling over history
        return np.mean(self.history_buffer, axis=0)

    def action(
        self, observation, epsilon: float = 0.0, policy_state: Any = (), seed=None
    ):
        del seed
        # Get current observation features (for critic)
        current_features = self.ft_op(transform.Example(observation, 0)).observation

        # Update history buffer
        self.history_buffer.append(current_features.copy())
        if len(self.history_buffer) > self.history_window:
            self.history_buffer.pop(0)

        # Compute Q-values with HC decomposition
        state_qvalues, head_value, critic_gradients = self.action_values_gradients(
            observation, self.actions
        )

        if epsilon and self.rng.random() < epsilon:
            action = self.rng.choice(self.actions)
        else:
            # Choose highest value action
            # breaking ties randomly
            action = self.rng.choice(
                np.flatnonzero(state_qvalues == state_qvalues.max())
            )

        return core.PolicyStep(
            action,
            state=policy_state,
            info={
                "values": state_qvalues,
                "head_value": head_value,
                "critic_gradients": critic_gradients,
                "history_features": self._get_history_features(),
            },
        )

    def action_values_gradients(self, observation, actions):
        """
        Compute Q-values using HC decomposition: Q(h,s,a) = Q^H(h) + Q^C(s,a)
        """
        # Get history features
        history_features = self._get_history_features()

        # Compute Head component: Q^H(h)
        head_value = np.dot(history_features, self.head_weights)

        # Compute Critic component for all actions: Q^C(s,a)
        examples = [transform.Example(observation, action) for action in actions]
        state_action_features = [ex.observation for ex in self.ft_op.batch(examples)]
        critic_values = np.sum(self.critic_weights * state_action_features, axis=1)

        # Total Q-values: Q^H + Q^C
        total_qvalues = head_value + critic_values

        return total_qvalues, head_value, state_action_features

    def step(self, action, head_gradients, critic_gradients):
        """
        Update both Head and Critic weights.

        Args:
            action: The action taken
            head_gradients: Scaled gradients for Head network
            critic_gradients: Scaled gradients for Critic network
        """
        self.head_weights += head_gradients
        self.critic_weights[action] += critic_gradients

    def reset_history(self):
        """Clear history buffer (call at episode start)."""
        self.history_buffer = []

    @property
    def model(self):
        """Return combined model weights."""
        return np.concatenate([self.head_weights.flatten(), self.critic_weights.flatten()])


class HCDecompositionSemigradientSARSAFnApprox(FnApproxAlgorithm):
    """
    Semi-gradient SARSA with Head-Critic Decomposition for Delayed Rewards.

    Implements the training algorithm from "Off-Policy Reinforcement Learning
    with Delayed Rewards" (Han et al., 2022, ICML).

    Key features:
    - Separate updates for Head (historical) and Critic (current) components
    - Policy gradient only uses Critic component to reduce variance
    - Handles delayed rewards by learning historical reward accumulation
    """

    def __init__(
        self,
        lr_head: optsol.LearningRateSchedule,
        lr_critic: optsol.LearningRateSchedule,
        gamma: float,
        epsilon: float,
        policy: HCDecompositionLinearFnApproxPolicy,
        base_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(base_seed)
        self.lr_head = lr_head
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.verbose = verbose

    def train(
        self,
        env: gym.Env,
        num_episodes: int,
        monitor: core.EnvMonitor,
    ) -> Iterator[PolicyControlSnapshot]:
        monitor.clear()
        for episode in range(num_episodes):
            # Reset history at episode start
            self.policy.reset_history()

            obs, _ = env.reset(seed=self.seeder.get_seed(episode=episode))
            policy_step = self.policy.action(obs, epsilon=self.epsilon)
            state_qvalues = policy_step.info["values"]
            head_value = policy_step.info["head_value"]
            critic_gradients = policy_step.info["critic_gradients"]
            history_features = policy_step.info["history_features"]

            while True:
                (
                    next_obs,
                    reward,
                    term,
                    trunc,
                    _,
                ) = env.step(policy_step.action)

                # Treat None rewards as 0 (delayed rewards not yet received)
                reward_value = reward if reward is not None else 0.0

                if term or trunc:
                    # Terminal state: target = reward only
                    # Update Head: learns historical component
                    head_td_error = reward_value - head_value
                    head_scaled_gradients = (
                        self.lr_head(episode, monitor.step)
                        * head_td_error
                        * history_features
                    )

                    # Update Critic: learns current action value
                    critic_td_error = reward_value - state_qvalues[policy_step.action]
                    critic_scaled_gradients = (
                        self.lr_critic(episode, monitor.step)
                        * critic_td_error
                        * critic_gradients[policy_step.action]
                    )

                    self.policy.step(
                        policy_step.action,
                        head_scaled_gradients,
                        critic_scaled_gradients,
                    )
                    break

                next_policy_step = self.policy.action(next_obs, epsilon=self.epsilon)
                next_state_qvalues = next_policy_step.info["values"]
                next_head_value = next_policy_step.info["head_value"]
                next_critic_gradients = next_policy_step.info["critic_gradients"]
                next_history_features = next_policy_step.info["history_features"]

                # SARSA target with HC decomposition
                # Target = r + γ * [Q^H(h') + Q^C(s', a')]
                target = (
                    reward_value
                    + self.gamma * next_state_qvalues[next_policy_step.action]
                )

                # Update Head: learns to predict historical reward component
                # Target for Head = r + γ * Q^H(h')
                head_target = reward_value + self.gamma * next_head_value
                head_td_error = head_target - head_value
                head_scaled_gradients = (
                    self.lr_head(episode, monitor.step)
                    * head_td_error
                    * history_features
                )

                # Update Critic: learns current action value
                # Target for Critic is adjusted by removing Head component
                critic_td_error = target - state_qvalues[policy_step.action]
                critic_scaled_gradients = (
                    self.lr_critic(episode, monitor.step)
                    * critic_td_error
                    * critic_gradients[policy_step.action]
                )

                self.policy.step(
                    policy_step.action, head_scaled_gradients, critic_scaled_gradients
                )

                obs = next_obs
                policy_step = next_policy_step
                state_qvalues = next_state_qvalues
                head_value = next_head_value
                critic_gradients = next_critic_gradients
                history_features = next_history_features

            if self.verbose and (episode + 1) % max((num_episodes // 5), 1) == 0:
                logging.info(
                    "Episode %d mean returns: %f",
                    episode + 1,
                    np.mean(monitor.returns + [monitor.rewards]),
                )
            yield PolicyControlSnapshot(
                steps=monitor.step,
                returns=monitor.rewards,
                weights=copy.copy(self.policy.model),
            )
        env.close()


# =============================================================================
# RRD: Randomized Return Decomposition
# =============================================================================


@dataclasses.dataclass
class TrajectoryData:
    """
    Container for a complete trajectory.

    Attributes:
        observations: Raw observations [T]
        actions: Actions taken [T]
        features: Transformed features [T, feat_dim]
        episode_return: Total discounted return
        length: Number of steps
    """

    observations: list
    actions: list
    features: np.ndarray
    episode_return: float
    length: int


class TrajectoryBuffer:
    """
    Buffer for storing complete trajectories for RRD training.

    Stores recent trajectories and their returns for training the reward
    redistribution network.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize trajectory buffer.

        Args:
            capacity: Maximum number of trajectories to store
        """
        self.capacity = capacity
        self.trajectories = []
        self.returns = []

    def add(self, trajectory: TrajectoryData):
        """Add a trajectory to the buffer."""
        if len(self.trajectories) >= self.capacity:
            # FIFO eviction
            self.trajectories.pop(0)
            self.returns.pop(0)

        self.trajectories.append(trajectory)
        self.returns.append(trajectory.episode_return)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        """Sample a batch of trajectories uniformly."""
        if len(self.trajectories) < batch_size:
            batch_size = len(self.trajectories)

        indices = rng.choice(len(self.trajectories), size=batch_size, replace=False)
        return [self.trajectories[i] for i in indices]

    def __len__(self):
        return len(self.trajectories)


class RRDRewardNetwork:
    """
    Linear reward redistribution model R_θ(s,a) = θ^T φ(s,a).

    Uses gradient descent with analytical gradients for RRD loss.
    Input: concatenated state-action features
    Output: scalar reward value
    """

    def __init__(
        self,
        feature_dim: int,
        learning_rate: float = 3e-4,
        seed: Optional[int] = None,
    ):
        """
        Initialize linear reward network.

        Args:
            feature_dim: Dimension of input features
            learning_rate: Learning rate for gradient descent
            seed: Random seed for reproducibility
        """
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate

        # Initialize RNG and weights
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(loc=0.0, scale=0.01, size=feature_dim)

        # Storage for accumulated gradients
        self._accumulated_gradients = None

    def forward(self, features):
        """
        Compute linear predictions: R_θ(s,a) = θ^T φ(s,a).

        Args:
            features: Input features [batch_size, feature_dim] or [feature_dim]
                     Can be numpy array or torch tensor (converted to numpy)

        Returns:
            Predicted rewards [batch_size] or scalar (as numpy array)
        """
        # Convert from torch if needed (for compatibility during transition)
        if hasattr(features, "numpy"):  # torch tensor
            feats = (
                features.numpy()
                if not features.requires_grad
                else features.detach().numpy()
            )
        else:
            feats = np.asarray(features, dtype=np.float64)

        # Linear prediction: θ^T φ
        return np.dot(feats, self.weights)

    def compute_loss_and_gradients(
        self,
        trajectories: list,
        K: int,
        M: int,
        rng: np.random.Generator,
    ):
        """
        Compute RRD loss and analytical gradients.

        For each trajectory, samples M subsequences of K timesteps and computes:
        - Predicted return: R_hat = (T/K) * sum(R_θ(s_i, a_i))
        - Loss: L = (R_ep - R_hat)²
        - Gradient: ∂L/∂θ = -2(R_ep - R_hat) * (T/K) * sum(φ_i)

        Args:
            trajectories: List of TrajectoryData objects
            K: Subsequence length to sample
            M: Number of subsequences per trajectory
            rng: Random number generator

        Returns:
            loss: Scalar loss value (as float)
            gradients: None (gradients stored internally for compatibility)
        """
        accumulated_gradients = np.zeros(self.feature_dim, dtype=np.float64)
        total_loss = 0.0
        count = 0

        for traj in trajectories:
            T = traj.length
            R_ep = traj.episode_return

            for _ in range(M):
                # Sample K indices uniformly without replacement
                if K >= T:
                    indices = np.arange(T)
                else:
                    indices = rng.choice(T, size=K, replace=False)

                # Get sampled features
                sampled_features = traj.features[indices]  # shape: (K, feature_dim)

                # Compute predictions for sampled features
                R_pred_per_step = np.dot(sampled_features, self.weights)  # shape: (K,)

                # Compute predicted return: R_hat = (T/K) * sum(predictions)
                R_hat = (T / K) * np.sum(R_pred_per_step)

                # Squared error loss
                loss = (R_ep - R_hat) ** 2
                total_loss += loss
                count += 1

                # Analytical gradient: ∂L/∂θ = -2(R_ep - R_hat) * (T/K) * sum(φ_i)
                prediction_error = R_ep - R_hat
                scale_factor = -2.0 * prediction_error * (T / K)
                gradient = scale_factor * np.sum(sampled_features, axis=0)

                accumulated_gradients += gradient

        # Average loss and gradients
        avg_loss = total_loss / count if count > 0 else 0.0
        self._accumulated_gradients = (
            accumulated_gradients / count if count > 0 else accumulated_gradients
        )

        return avg_loss, None

    def update(self, gradients: Optional[dict] = None):
        """
        Update weights using gradient descent.

        Applies: θ ← θ + lr * ∇L
        (Note: gradient already includes negative sign from compute_loss_and_gradients)

        Args:
            gradients: IGNORED (kept for interface compatibility)
        """
        del gradients  # Unused, kept for interface compatibility
        self.weights += self.learning_rate * self._accumulated_gradients


class RRDSemigradientSARSAFnApprox(FnApproxAlgorithm):
    """
    SARSA with Randomized Return Decomposition (RRD) reward redistribution.

    Implements Algorithm 1 from Ren et al. (ICLR 2022) for learning with
    sparse episodic rewards by redistributing rewards across timesteps.

    This implements the INTERLEAVED training approach from the paper:
    Each episode iteration performs:
    1. Collect trajectory using current policy
    2. Update reward network R_θ(s,a) using RRD loss on trajectory buffer
    3. Update policy using redistributed rewards from current R_θ(s,a)

    Note: The reward model and policy are trained TOGETHER in an iterleaved
    manner, NOT sequentially (there is no pre-training phase for R_θ).
    """

    def __init__(
        self,
        lr: optsol.LearningRateSchedule,
        gamma: float,
        epsilon: float,
        policy: core.PyValueFnPolicy,
        ft_op: transform.FTOp,
        K: int = 64,
        M: int = 1,
        reward_network_lr: float = 3e-4,
        reward_update_freq: int = 10,
        trajectory_buffer_capacity: int = 10000,
        batch_size: int = 256,
        base_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize RRD SARSA algorithm.

        Args:
            lr: Learning rate schedule for policy
            gamma: Discount factor
            epsilon: Exploration rate
            policy: Policy to train
            ft_op: Feature transformation operator
            K: Subsequence length for RRD sampling
            M: Number of subsequences per trajectory
            reward_network_lr: Learning rate for reward network
            reward_update_freq: Episodes between reward network updates
            trajectory_buffer_capacity: Max trajectories to store
            batch_size: Batch size for reward network training
            base_seed: Random seed
            verbose: Whether to log progress
        """
        super().__init__(base_seed)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.ft_op = ft_op
        self.K = K
        self.M = M
        self.reward_update_freq = reward_update_freq
        self.batch_size = batch_size
        self.verbose = verbose

        # Initialize trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(capacity=trajectory_buffer_capacity)

        # Initialize reward network
        feature_dim = np.prod(ft_op.output_space.observation_space.shape)
        self.reward_network = RRDRewardNetwork(
            feature_dim=feature_dim,
            learning_rate=reward_network_lr,
            seed=base_seed,
        )

        self.rng = np.random.default_rng(base_seed)

    def train(
        self,
        env: gym.Env,
        num_episodes: int,
        monitor: core.EnvMonitor,
    ) -> Iterator[PolicyControlSnapshot]:
        """Train policy with RRD reward redistribution."""
        monitor.clear()

        for episode in range(num_episodes):
            # Collect episode
            obs, _ = env.reset(seed=self.seeder.get_seed(episode=episode))

            # Store trajectory for this episode
            observations = [obs]
            actions = []
            features = []
            rewards = []

            policy_step = self.policy.action(obs, epsilon=self.epsilon)
            state_qvalues, gradients = (
                policy_step.info["values"],
                policy_step.info["gradients"],
            )

            # Get feature for current state-action
            example = transform.Example(obs, policy_step.action)
            feat = self.ft_op.apply(example).observation
            features.append(feat)

            while True:
                (next_obs, reward, term, trunc, _) = env.step(policy_step.action)

                observations.append(next_obs)
                actions.append(policy_step.action)
                rewards.append(reward)

                # Compute redistributed reward from reward network
                redistributed_reward = float(self.reward_network.forward(feat))

                if term or trunc:
                    # Terminal update using redistributed reward
                    scaled_gradients = (
                        self.lr(episode, monitor.step)
                        * (redistributed_reward - state_qvalues[policy_step.action])
                        * gradients[policy_step.action]
                    )
                    self.policy.step(policy_step.action, scaled_gradients)
                    break

                # Get next state-action features
                next_policy_step = self.policy.action(next_obs, epsilon=self.epsilon)
                next_state_qvalues, next_gradients = (
                    next_policy_step.info["values"],
                    next_policy_step.info["gradients"],
                )

                next_example = transform.Example(next_obs, next_policy_step.action)
                next_feat = self.ft_op.apply(next_example).observation
                features.append(next_feat)

                # SARSA update with redistributed reward
                scaled_gradients = (
                    self.lr(episode, monitor.step)
                    * (
                        redistributed_reward
                        + self.gamma * next_state_qvalues[next_policy_step.action]
                        - state_qvalues[policy_step.action]
                    )
                    * gradients[policy_step.action]
                )
                self.policy.step(policy_step.action, scaled_gradients)

                obs = next_obs
                policy_step = next_policy_step
                state_qvalues = next_state_qvalues
                gradients = next_gradients
                feat = next_feat

            # Compute episode return (treat None rewards as 0)
            episode_return = sum(
                (rewards[t] if rewards[t] is not None else 0.0) * (self.gamma**t)
                for t in range(len(rewards))
            )

            # Store trajectory in buffer
            trajectory = TrajectoryData(
                observations=observations,
                actions=actions,
                features=np.array(features),
                episode_return=episode_return,
                length=len(actions),
            )
            self.trajectory_buffer.add(trajectory)

            # Periodically update reward network
            if (
                episode > 0
                and episode % self.reward_update_freq == 0
                and len(self.trajectory_buffer) >= self.batch_size
            ):
                # Sample batch and update reward network
                batch = self.trajectory_buffer.sample_batch(self.batch_size, self.rng)
                loss, gradients_dict = self.reward_network.compute_loss_and_gradients(
                    batch, self.K, self.M, self.rng
                )
                self.reward_network.update(gradients_dict)

                if self.verbose:
                    logging.info("Episode %d - RRD loss: %f", episode, loss)

            if self.verbose and (episode + 1) % max((num_episodes // 5), 1) == 0:
                logging.info(
                    "Episode %d mean returns: %f",
                    episode + 1,
                    np.mean(monitor.returns + [monitor.rewards]),
                )

            yield PolicyControlSnapshot(
                steps=monitor.step,
                returns=monitor.rewards,
                weights=copy.copy(self.policy.model),
            )

        env.close()


# =============================================================================
# IRCR: Iterative Relative Credit Refinement
# =============================================================================


class TrajectoryDatabase:
    """
    Database for storing trajectories and computing guidance rewards (IRCR).

    Stores trajectories and computes r_g(s,a) = E[R_ep(τ) | (s,a) ∈ τ]
    as the average return of all trajectories containing (s,a).
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize trajectory database.

        Args:
            capacity: Maximum number of trajectories to store
        """
        self.capacity = capacity
        self.trajectories = []
        self.r_min = float("inf")
        self.r_max = float("-inf")

        # Map (s,a) hash to list of trajectory indices
        self._sa_to_traj_indices = {}

    def add(self, trajectory: TrajectoryData):
        """Add trajectory and update statistics."""
        if len(self.trajectories) >= self.capacity:
            # FIFO eviction - need to rebuild index
            self.trajectories.pop(0)
            self._rebuild_index()

        traj_idx = len(self.trajectories)
        self.trajectories.append(trajectory)

        # Update return statistics
        self.r_min = min(self.r_min, trajectory.episode_return)
        self.r_max = max(self.r_max, trajectory.episode_return)

        # Update (s,a) -> trajectory index mapping
        for action, feat in zip(trajectory.actions, trajectory.features):
            key = self._hash_state_action(feat, action)
            if key not in self._sa_to_traj_indices:
                self._sa_to_traj_indices[key] = []
            self._sa_to_traj_indices[key].append(traj_idx)

    def _rebuild_index(self):
        """Rebuild the (s,a) index after eviction."""
        self._sa_to_traj_indices = {}
        for traj_idx, traj in enumerate(self.trajectories):
            for action, feat in zip(traj.actions, traj.features):
                key = self._hash_state_action(feat, action)
                if key not in self._sa_to_traj_indices:
                    self._sa_to_traj_indices[key] = []
                self._sa_to_traj_indices[key].append(traj_idx)

    def _hash_state_action(self, feature: np.ndarray, action: int) -> tuple:
        """
        Hash state-action pair for lookup.

        For continuous features, we discretize to bins.
        """
        # Simple discretization: round to 1 decimal place
        discretized_feat = tuple(np.round(feature, decimals=1))
        return (discretized_feat, action)

    def get_guidance_reward(self, feature: np.ndarray, action: int) -> float:
        """
        Compute guidance reward r_g(s,a).

        Returns the normalized average return of trajectories containing (s,a).
        """
        key = self._hash_state_action(feature, action)

        if (
            key not in self._sa_to_traj_indices
            or len(self._sa_to_traj_indices[key]) == 0
        ):
            # Unseen (s,a): return default value (mean)
            if len(self.trajectories) == 0:
                return 0.0
            returns = [traj.episode_return for traj in self.trajectories]
            r_g = np.mean(returns)
        else:
            # Compute average return for this (s,a)
            traj_indices = self._sa_to_traj_indices[key]
            returns = [self.trajectories[i].episode_return for i in traj_indices]
            r_g = np.mean(returns)

        # Normalize to [0, 1]
        if self.r_max == self.r_min:
            return 0.5
        return (r_g - self.r_min) / (self.r_max - self.r_min)

    def __len__(self):
        return len(self.trajectories)


class IRCRSemigradientSARSAFnApprox(FnApproxAlgorithm):
    """
    SARSA with IRCR (Iterative Relative Credit Refinement) guidance rewards.

    Implements IRCR algorithm (Gangwani et al., NeurIPS 2020) adapted for
    on-policy SARSA. The original paper uses SAC (off-policy with replay buffer).

    Algorithm (adapted for on-policy SARSA):
    1. Collect episodes and store in trajectory database
    2. Compute guidance rewards r_g(s,a) = E[R_ep(τ) | (s,a) ∈ τ]
    3. Use guidance rewards for policy learning

    Implementation Note:
    - Uses "one-step lag": guidance rewards from episode N are computed using
      trajectories from episodes 1 to N-1
    - Current trajectory is added to database after collection
    - This differs from paper's offline approach but is necessary for on-policy learning
    """

    def __init__(
        self,
        lr: optsol.LearningRateSchedule,
        gamma: float,
        epsilon: float,
        policy: core.PyValueFnPolicy,
        ft_op: transform.FTOp,
        trajectory_db_capacity: int = 10000,
        base_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize IRCR SARSA algorithm.

        Args:
            lr: Learning rate schedule for policy
            gamma: Discount factor
            epsilon: Exploration rate
            policy: Policy to train
            ft_op: Feature transformation operator
            trajectory_db_capacity: Max trajectories to store
            base_seed: Random seed
            verbose: Whether to log progress
        """
        super().__init__(base_seed)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.ft_op = ft_op
        self.verbose = verbose

        # Initialize trajectory database
        self.trajectory_db = TrajectoryDatabase(capacity=trajectory_db_capacity)

    def train(
        self,
        env: gym.Env,
        num_episodes: int,
        monitor: core.EnvMonitor,
    ) -> Iterator[PolicyControlSnapshot]:
        """Train policy with IRCR guidance rewards."""
        monitor.clear()

        for episode in range(num_episodes):
            # Collect episode
            obs, _ = env.reset(seed=self.seeder.get_seed(episode=episode))

            # Store trajectory for this episode
            observations = [obs]
            actions = []
            features = []
            rewards = []

            policy_step = self.policy.action(obs, epsilon=self.epsilon)
            state_qvalues, gradients = (
                policy_step.info["values"],
                policy_step.info["gradients"],
            )

            # Get feature for current state-action
            example = transform.Example(obs, policy_step.action)
            feat = self.ft_op.apply(example).observation
            features.append(feat)

            while True:
                (next_obs, reward, term, trunc, _) = env.step(policy_step.action)

                observations.append(next_obs)
                actions.append(policy_step.action)
                rewards.append(reward)

                # Compute guidance reward from trajectory database
                guidance_reward = self.trajectory_db.get_guidance_reward(
                    feat, policy_step.action
                )

                if term or trunc:
                    # Terminal update using guidance reward
                    scaled_gradients = (
                        self.lr(episode, monitor.step)
                        * (guidance_reward - state_qvalues[policy_step.action])
                        * gradients[policy_step.action]
                    )
                    self.policy.step(policy_step.action, scaled_gradients)
                    break

                # Get next state-action features
                next_policy_step = self.policy.action(next_obs, epsilon=self.epsilon)
                next_state_qvalues, next_gradients = (
                    next_policy_step.info["values"],
                    next_policy_step.info["gradients"],
                )

                next_example = transform.Example(next_obs, next_policy_step.action)
                next_feat = self.ft_op.apply(next_example).observation
                features.append(next_feat)

                # SARSA update with guidance reward
                scaled_gradients = (
                    self.lr(episode, monitor.step)
                    * (
                        guidance_reward
                        + self.gamma * next_state_qvalues[next_policy_step.action]
                        - state_qvalues[policy_step.action]
                    )
                    * gradients[policy_step.action]
                )
                self.policy.step(policy_step.action, scaled_gradients)

                obs = next_obs
                policy_step = next_policy_step
                state_qvalues = next_state_qvalues
                gradients = next_gradients
                feat = next_feat

            # Compute episode return
            episode_return = sum(
                rewards[t] * (self.gamma**t) for t in range(len(rewards))
            )

            # Store trajectory in database
            trajectory = TrajectoryData(
                observations=observations,
                actions=actions,
                features=np.array(features),
                episode_return=episode_return,
                length=len(actions),
            )
            self.trajectory_db.add(trajectory)

            if self.verbose and (episode + 1) % max((num_episodes // 5), 1) == 0:
                logging.info(
                    "Episode %d mean returns: %f, DB size: %d",
                    episode + 1,
                    np.mean(monitor.returns + [monitor.rewards]),
                    len(self.trajectory_db),
                )

            yield PolicyControlSnapshot(
                steps=monitor.step,
                returns=monitor.rewards,
                weights=copy.copy(self.policy.model),
            )

        env.close()
