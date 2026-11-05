"""
Task oracles for the estimation campaigns.

An oracle is the policy that maximises the environment's own task return,
not the shaped reward the estimators are measured against: reaching the
grid exit, the flag, or the upright acrobot. Estimation campaigns use it
to sample trajectories from a goal-directed occupancy instead of an
exploration-only one; the exploration rate is applied by the caller.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from drmdp.envs import models

GRID_ORACLE_ENV_PREFIX = "GridWorld-"
MOUNTAIN_CAR_ENV = "MountainCar-v0"
ACROBOT_ENV = "Acrobot-v1"

# action indices of the gym environments
MOUNTAIN_CAR_LEFT = 0
MOUNTAIN_CAR_RIGHT = 2
ACROBOT_NEGATIVE_TORQUE = 0
ACROBOT_POSITIVE_TORQUE = 2

GRAVITY = 9.8

_VALUE_ITERATION_SWEEPS = 2000
_VALUE_ITERATION_TOLERANCE = 1e-9


class TaskOracle(Protocol):
    """Maps an observation to a task-maximising action."""

    def action(self, observation: Any) -> int:
        """Greedy action of the task objective at `observation`."""
        ...


class GridOracle:
    """
    Greedy policy of the grid MDP's own reward, by value iteration.

    Uses the environment's declared transition model, so the oracle is
    exact when the model agrees with the runtime transitions. Values are
    undiscounted, matching the step and cliff rewards the grid defines.
    """

    def __init__(self, model: models.DiscreteTransitionModel):
        self._state_ids: dict[tuple[int, ...], int] = {
            tuple(int(coordinate) for coordinate in observation): state_id
            for state_id, observation in model.observations.items()
        }
        self._values, self._greedy_actions = _value_iteration(model)

    def action(self, observation: Any) -> int:
        return self._greedy_actions[self._state_id(observation)]

    def value(self, observation: Any) -> float:
        """Optimal undiscounted return from `observation`."""
        return self._values[self._state_id(observation)]

    def _state_id(self, observation: Any) -> int:
        key = tuple(int(coordinate) for coordinate in np.asarray(observation).ravel())
        return self._state_ids[key]


class MountainCarOracle:
    """
    Push in the direction of the velocity.

    The mountain car gains energy only from forces aligned with its
    motion, so this controller pumps the car up the valley wall until it
    crosses the flag at position $0.5$.
    """

    def action(self, observation: Any) -> int:
        velocity = float(np.asarray(observation).ravel()[1])
        return MOUNTAIN_CAR_RIGHT if velocity >= 0.0 else MOUNTAIN_CAR_LEFT


class AcrobotOracle:
    """
    Energy pump for the acrobot's own termination condition.

    Applies torque at the elbow in the direction of the joint velocity
    while the system energy sits below the energy of the upright
    configuration, and in the opposite direction above it, so the tip
    reaches the height at which the episode terminates.
    """

    def __init__(self, env: gym.Env):
        base = env.unwrapped
        self._mass_1 = float(getattr(base, "LINK_MASS_1", 1.0))
        self._mass_2 = float(getattr(base, "LINK_MASS_2", 1.0))
        self._length_1 = float(getattr(base, "LINK_LENGTH_1", 1.0))
        self._com_1 = float(getattr(base, "LINK_COM_POS_1", 0.5))
        self._com_2 = float(getattr(base, "LINK_COM_POS_2", 0.5))
        self._inertia_1 = float(getattr(base, "LINK_MOI", 1.0))
        self._inertia_2 = float(getattr(base, "LINK_MOI", 1.0))
        self._gravity = GRAVITY
        self._upright_energy = (
            self._mass_1 * self._com_1 + self._mass_2 * self._length_1
        ) * self._gravity + self._mass_2 * self._com_2 * self._gravity

    def action(self, observation: Any) -> int:
        cos_1, sin_1, cos_2, sin_2, velocity_1, velocity_2 = (
            float(value) for value in np.asarray(observation).ravel()
        )
        energy = self._energy(
            math.atan2(sin_1, cos_1),
            math.atan2(sin_2, cos_2),
            velocity_1,
            velocity_2,
        )
        if (self._upright_energy - energy) * velocity_2 > 0.0:
            return ACROBOT_POSITIVE_TORQUE
        return ACROBOT_NEGATIVE_TORQUE

    def _energy(
        self, angle_1: float, angle_2: float, velocity_1: float, velocity_2: float
    ) -> float:
        coupling = self._inertia_2 + self._mass_2 * (
            self._com_2**2 + self._length_1 * self._com_2 * math.cos(angle_2)
        )
        mass_11 = (
            self._inertia_1
            + self._inertia_2
            + self._mass_1 * self._com_1**2
            + self._mass_2
            * (
                self._length_1**2
                + self._com_2**2
                + 2.0 * self._length_1 * self._com_2 * math.cos(angle_2)
            )
        )
        mass_22 = self._inertia_2 + self._mass_2 * self._com_2**2
        kinetic = 0.5 * (
            mass_11 * velocity_1**2
            + 2.0 * coupling * velocity_1 * velocity_2
            + mass_22 * velocity_2**2
        )
        potential = -(
            self._mass_1 * self._com_1 + self._mass_2 * self._length_1
        ) * self._gravity * math.cos(angle_1) - (
            self._mass_2 * self._com_2 * self._gravity * math.cos(angle_1 + angle_2)
        )
        return kinetic + potential


def task_oracle(env_name: str, env: gym.Env) -> TaskOracle:
    """
    Task oracle of a supported environment.

    Raises for environments without a declared oracle so that a campaign
    configured with one fails visibly instead of behaving randomly.
    """
    if env_name.startswith(GRID_ORACLE_ENV_PREFIX):
        model = _declared_transition_model(env)
        if model is None:
            raise ValueError(
                f"{env_name!r} exposes no transition model; the grid oracle is undefined"
            )
        return GridOracle(model)
    if env_name == MOUNTAIN_CAR_ENV:
        return MountainCarOracle()
    if env_name == ACROBOT_ENV:
        return AcrobotOracle(env)
    raise ValueError(f"no task oracle for environment {env_name!r}")


def _declared_transition_model(env: gym.Env) -> models.DiscreteTransitionModel | None:
    """
    Transition model declared by the innermost environment that provides it.

    Gymnasium 1.3 wrappers do not forward attribute access, so the chain is
    walked explicitly; the method is read from the class to avoid picking
    up a forwarded attribute.
    """
    provider = Callable[[Any], models.DiscreteTransitionModel | None]
    probe: Any = env
    while probe is not None:
        declared: provider | None = getattr(
            type(probe), "discrete_transition_model", None
        )
        if declared is not None:
            return declared(probe)
        probe = getattr(probe, "env", None)
    return None


def _value_iteration(
    model: models.DiscreteTransitionModel,
) -> tuple[dict[int, float], dict[int, int]]:
    """Optimal values and greedy actions of the undiscounted task return."""
    state_ids = sorted(model.transition)
    values = {state_id: 0.0 for state_id in state_ids}
    actions = {state_id: 0 for state_id in state_ids}
    for _ in range(_VALUE_ITERATION_SWEEPS):
        change = 0.0
        for state_id in state_ids:
            if state_id in model.terminal_state_ids:
                continue
            immediate, best_action = _best_outcome(
                model.transition[state_id], values, model.terminal_state_ids
            )
            change = max(change, abs(immediate - values[state_id]))
            values[state_id] = immediate
            actions[state_id] = best_action
        if change < _VALUE_ITERATION_TOLERANCE:
            break
    return values, actions


def _best_outcome(
    outcomes: Mapping[int, Sequence[models.TransitionOutcome]],
    values: Mapping[int, float],
    terminal_state_ids: frozenset[int],
) -> tuple[float, int]:
    """Best (value, action) over the actions of one state."""
    best_value = -math.inf
    best_action = 0
    for action, action_outcomes in outcomes.items():
        expected = 0.0
        for probability, next_state_id, reward, terminated in action_outcomes:
            next_value = (
                0.0
                if terminated or next_state_id in terminal_state_ids
                else values[next_state_id]
            )
            expected += probability * (reward + next_value)
        if expected > best_value:
            best_value = expected
            best_action = int(action)
    return best_value, best_action
