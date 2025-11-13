import abc
import copy
import dataclasses
import logging
from typing import Any, Iterator, Optional, Tuple

import gymnasium as gym
import numpy as np

from drmdp import core, feats, mathutils, optsol


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
                    self.policy.update(scaled_gradients)
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
                self.policy.update(scaled_gradients)
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
        feat_transform: feats.FeatTransform,
        action_space: gym.Space,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.feat_transform = feat_transform
        self.weights = np.zeros(feat_transform.output_shape, dtype=np.float64)
        self.actions = tuple(range(action_space.n))

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
        observations = [observation] * len(actions)
        state_action_m = self.feat_transform.batch_transform(observations, actions)
        return np.dot(state_action_m, self.weights), state_action_m

    def update(self, scaled_gradients):
        self.weights += scaled_gradients

    @property
    def model(self):
        return self.weights


class RandomFnApproxPolicy(core.PyValueFnPolicy):
    def __init__(
        self,
        feat_transform: feats.FeatTransform,
        action_space: gym.Space,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.feat_transform = feat_transform
        self.weights = np.zeros(feat_transform.output_shape, dtype=np.float64)
        self.actions = tuple(range(action_space.n))

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
        observations = [observation] * len(actions)
        state_action_m = self.feat_transform.batch_transform(observations, actions)
        return np.dot(state_action_m, self.weights), state_action_m

    def update(self, scaled_gradients):
        self.weights += scaled_gradients

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
                            self.policy.update(scaled_gradients)
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
                self.policy.update(scaled_gradients)
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

    1. Apply feat_transform to the `observation` usign action=0
    2. Apply binary encoding of integers, using the first
    power of 2 that is greater than `num_options[delay_max]`.
    e.g. if delay_max = 3 and there are 4 actions,
    the binary encoding size is is 2^7 > 4^{3}.
    3. Apply one-hot-encoding to the delay, based on the number
    of delays (delay_max - delay_min + 1).

    The final option encoding is a concatenation of
    the output of steps 2 and 3
    option_enc = binary_enc(option) + one_hot(delay)
    observation_enc = feat_transform.transform(observation)
    And the action encoding as well, has a max length.

    `option_enc` is generally very small compared to the
    number of actions.
    feat_transform depends on the option.
    """

    def __init__(
        self,
        feat_transform: feats.FeatTransform,
        action_space: gym.Space,
        options_length_range: Tuple[int, int],
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.feat_transform = feat_transform
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
        self.features_dim = self.feat_transform.output_shape + self.options_dim
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
            self.feat_transform.transform(observation, 0), (options_matrix.shape[0], 1)
        )
        # get option representations
        features_m = np.concatenate([state_m, options_matrix], axis=1)
        return np.dot(features_m, self.weights), features_m

    def update(self, scaled_gradients):
        self.weights += scaled_gradients

    @property
    def model(self):
        return self.weights


class SingleActionOptionsLinearFnApproxPolicy(core.PyValueFnPolicy):
    def __init__(
        self,
        feat_transform: feats.FeatTransform,
        action_space: gym.Space,
        options_length_range: Tuple[int, int],
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"This policy only supports discrete action spaces. Got {type(action_space)}"
            )
        super().__init__(action_space, emit_log_probability, seed)
        self.feat_transform = feat_transform
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
        self.features_dim = self.feat_transform.output_shape + self.options_dim
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
            self.feat_transform.transform(observation, 0), (options_m.shape[0], 1)
        )
        # get option representations
        features_m = np.concatenate([state_m, options_m], axis=1)
        return np.dot(features_m, self.weights), features_m

    def update(self, scaled_gradients):
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
                        self.policy.update(scaled_gradients)
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
                    self.policy.update(scaled_gradients)
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
