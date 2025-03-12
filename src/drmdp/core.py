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


@dataclasses.dataclass(frozen=True)
class TrajectoryStep:
    """
    A trajectory step for training RL agents.
    """

    observation: ObsType
    action: ActType
    policy_info: Mapping[str, Any]
    terminated: bool
    truncated: bool
    reward: float
    info: Mapping[str, Any]

    @staticmethod
    def from_transition(
        time_step: TimeStep,
        action_step: PolicyStep,
        next_time_step: TimeStep,
    ) -> "TrajectoryStep":
        """
        Builds a trajectory step given a state and action.
        """
        obs, _, terminated, truncated, _ = time_step
        _, next_reward, _, _, _ = next_time_step

        return TrajectoryStep(
            observation=obs,
            action=action_step.action,
            policy_info=action_step.info,
            terminated=terminated,
            truncated=truncated,
            reward=next_reward,
            info={},
        )


class PyPolicy(abc.ABC):
    """
    Base class for python policies.
    """

    def __init__(
        self,
        emit_log_probability: bool = False,
        seed: Optional[int] = None,
    ):
        self.emit_log_probability = emit_log_probability
        self.seed = seed

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
        policy_state: Any = (),
        seed: Optional[int] = None,
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


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Configuration parameters for an experiment.
    """

    name: str
    args: Optional[Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class ProblemSpec:
    """
    Configuration for delayed, aggregate (and anonymous) reward experiments.
    """

    policy_type: str
    traj_mapping_method: str
    algorithm: str
    algorithm_args: Mapping[str, Any]
    delay_config: Mapping[str, Any]
    epsilon: float
    gamma: float
    learning_rate_config: Mapping[str, Any]
    feats_spec: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """
    Configuration for experiment run.
    """

    num_runs: int
    episodes_per_run: int
    log_episode_frequency: int
    output_dir: str


@dataclasses.dataclass(frozen=True)
class Experiment:
    """
    Experiments definition.
    """

    env_spec: EnvSpec
    problem_spec: ProblemSpec


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
