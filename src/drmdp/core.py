"""
This module defines core abstractions and types.
"""

import abc
import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

NestedArray = Union[Mapping, np.ndarray]
TimeStep = Tuple[ObsType, SupportsFloat, bool, bool, Mapping[str, Any]]
InitState = Tuple[ObsType, Mapping[str, Any]]
RenderType = Optional[Union[RenderFrame, Sequence[RenderFrame]]]
StateTransition = Mapping[int, Sequence[Tuple[float, int, float, bool]]]
# Type: Mapping[state, Mapping[action, Sequence[Tuple[prob, next_state, reward, terminated]]]]
EnvTransition = Mapping[int, StateTransition]
MutableStateTransition = Dict[int, List[Tuple[float, int, float, bool]]]
MutableEnvTransition = Dict[int, MutableStateTransition]
MapsToIntId = Callable[[Any], int]


@dataclasses.dataclass(frozen=True)
class PolicyStep:
    """
    Output of a policy's action function.
    Encapsulates the chosen action and policy state.
    """

    action: ActType
    state: Any
    info: Mapping[str, Any]


class PyPolicy(abc.ABC):
    """
    Base class for python policies.
    """

    def __init__(
        self,
        action_space: gym.Space,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        self.action_space = action_space
        self.emit_log_probability = emit_log_probability
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        """Returns an initial state usable by the policy.

        Args:
          batch_size: An optional batch size.

        Returns:
          An initial policy state.
        """

    @abc.abstractmethod
    def action(
        self,
        observation: ObsType,
        epsilon: float = 0.0,
        policy_state: Any = (),
    ) -> PolicyStep:
        """Implementation of `action`.

        Args:
          observation: An observation.
          policy_state: An Array, or a nested dict, list or tuple of Arrays
            representing the previous policy state.
          seed: Seed to use when choosing action. Impl specific.

        Returns:
          A `PolicyStep` named tuple containing:
            `action`: The policy's chosen action.
            `state`: A policy state to be fed into the next call to action.
            `info`: Optional side information such as action log probabilities.
        """


class PyValueFnPolicy(PyPolicy):
    @abc.abstractmethod
    def action_values_gradients(self, observation: ObsType, actions: Sequence[ActType]):
        """
        Value for multiple actions
        """

    @abc.abstractmethod
    def step(self, action: ActType, scaled_gradients):
        """
        Updates the policy's value
        """

    @property
    @abc.abstractmethod
    def model(self):
        """
        Model backing the policy.
        """


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Configuration parameters for an experiment.
    """

    name: str
    args: Optional[Mapping[str, Any]]
    feats_spec: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class ProblemSpec:
    """
    Configuration for delayed, aggregate (and anonymous) reward experiments.
    """

    policy_type: str
    reward_mapper: Mapping[str, Any]
    delay_config: Optional[Mapping[str, Any]]
    epsilon: float
    gamma: float
    learning_rate_config: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """
    Configuration for experiment run.
    """

    num_runs: int
    episodes_per_run: int
    log_episode_frequency: int
    use_seed: bool
    output_dir: str


@dataclasses.dataclass(frozen=True)
class Experiment:
    """
    Experiments definition.
    """

    env_spec: EnvSpec
    problem_spec: ProblemSpec
    epochs: int


@dataclasses.dataclass(frozen=True)
class ExperimentInstance:
    """
    A single experiment task.
    """

    exp_id: str
    instance_id: int
    experiment: Experiment
    run_config: RunConfig
    context: Optional[Mapping[str, Any]]


class EnvMonitorWrapper(gym.Wrapper):
    """
    Tracks the returns and steps for an environment.
    """

    def __init__(self, env):
        super().__init__(env)
        self.mon = EnvMonitor()

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self.mon.rewards += reward
        self.mon.step += 1
        return obs, reward, term, trunc, info

    def reset(self, *, seed=None, options=None):
        self.mon.reset()
        return super().reset(seed=seed, options=options)


class EnvMonitor:
    """
    Monitors episode returns and steps.
    """

    def __init__(self):
        self.returns: List[float] = []
        self.steps: List[int] = []
        self.rewards: float = 0
        self.step: int = 0

    def reset(self):
        """
        Stack values to track new episode.
        """
        if self.step > 0:
            self.returns.append(self.rewards)
            self.steps.append(self.step)
        self.rewards = 0.0
        self.step = 0

    def clear(self):
        """
        Clear monitored data.
        """
        self.returns = []
        self.steps = []
        self.step = 0
        self.rewards = 0.0


class Seeder:
    MAX_INS = 1000
    MAX_EPS = 100_000

    def __init__(self, instance: Optional[int] = None):
        self.instance = instance

    def get_seed(self, episode: int) -> Optional[int]:
        """
        For a given instance (seed), generated
        episode specific seeds consistently.
        """
        if self.instance is not None:
            return (self.MAX_INS * self.instance + 1) * (self.MAX_EPS + episode + 1)
        return self.instance


@dataclasses.dataclass
class ProxiedEnv:
    """
    An env and its proxy.
    """

    env: gym.Env
    proxy: gym.Env
