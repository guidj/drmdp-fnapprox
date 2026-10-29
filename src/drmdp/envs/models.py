"""Transition models exposed explicitly by environment adapters."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

type TransitionOutcome = tuple[float, int, float, bool]


@dataclasses.dataclass(frozen=True)
class DiscreteTransitionModel:
    """A finite model in the provider's observation and action representation.

    Outcomes contain probability, successor state ID, reward, and termination.
    Terminal state IDs identify states from which no episode takes an action;
    a self-loop alone does not imply termination. Observation arrays retain
    their dtype and need not encode state IDs or coordinates.

    The provider guarantees agreement with runtime transitions. Models omit
    info events and truncation, describing the MDP rather than a time limit.
    """

    observations: Mapping[int, np.ndarray]
    transition: Mapping[int, Mapping[int, Sequence[TransitionOutcome]]]
    terminal_state_ids: frozenset[int]


@runtime_checkable
class DiscreteTransitionModelProvider(Protocol):
    """Opt-in contract for enumeration in the environment's representation.

    Concrete environment and wrapper classes must declare this method rather
    than inherit or forward it: a subclass may change runtime semantics
    without updating an inherited model. Return None when agreement between
    the model and runtime transitions is not established.
    """

    def discrete_transition_model(self) -> DiscreteTransitionModel | None:
        """Returns the model matching this environment's runtime transitions."""
        ...
