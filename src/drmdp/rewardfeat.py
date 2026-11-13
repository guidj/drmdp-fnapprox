"""
Reward features computed from a transition.

A `TransitionFeatureFn` maps one transition to the reward-feature vector
the estimators fit. It replaces the estimator-side `FTOp` pipeline: the
estimators see features, not transformed examples, so a feature may depend
on the successor observation and on the termination flag.

The vector is a concatenation of named blocks. A block either delegates its
map to an `FTOp` specification declared in the hook args, or computes its
entries from transition fields directly.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from drmdp import transform

# rlplg's grid environments order actions LEFT, RIGHT, UP, DOWN
# (`rlplg/environments/gridworld.py:31-34`) and move one cell in that
# direction, leaving the position unchanged at a wall. Neither convention is
# derivable from the observation space, so the table is stated here and both
# grid features use it.
GRIDWORLD_MOVES: tuple[tuple[int, ...], ...] = (
    (0, -1),  # LEFT
    (0, 1),  # RIGHT
    (-1, 0),  # UP
    (1, 0),  # DOWN
)


@dataclasses.dataclass(frozen=True)
class Transition:
    """
    One environment step: the observation acted on, the action, and the
    transition's outcome.
    """

    observation: Any
    action: int
    next_observation: Any
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


class TransitionFeatureFn(Protocol):
    """
    Reward features for one transition.
    """

    @property
    def feature_dim(self) -> int: ...

    def __call__(self, transition: Transition) -> np.ndarray: ...


class FTOpFeatures:
    """Adapts an FTOp's discrete index or Box vector to reward features.

    Successor specifications compile their operator on a single-action
    space. The fixed action then carries no successor-action information.
    """

    def __init__(self, ft_op: transform.FTOp, successor: bool = False):
        self.ft_op = ft_op
        self.successor = successor
        self._space = ft_op.output_space.observation_space
        if isinstance(self._space, gym.spaces.Discrete):
            self._feature_dim = int(self._space.n)
        elif isinstance(self._space, gym.spaces.Box):
            self._feature_dim = int(np.prod(self._space.shape))
        else:
            raise TypeError("FTOp reward features require Discrete or Box output")

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def __call__(self, transition: Transition) -> np.ndarray:
        if self.successor:
            example = transform.Example(transition.next_observation, 0)
        else:
            example = transform.Example(transition.observation, transition.action)
        observation = self.ft_op(example).observation
        if isinstance(self._space, gym.spaces.Discrete):
            features = np.zeros(self.feature_dim, dtype=np.float64)
            index = int(observation) - int(self._space.start)
            if not 0 <= index < self.feature_dim:
                raise ValueError(
                    f"FTOp returned an out-of-range discrete index: {observation}"
                )
            features[index] = 1.0
            return features
        return np.asarray(observation, dtype=np.float64).reshape(self.feature_dim)


class CurrentAndSuccessorFeatures:
    r"""
    A current block followed by a successor block.

    With ``terminal_gate`` the successor block is multiplied by
    $\mathbf{1}[\neg\text{terminated}]$: a potential-based reward pays no
    successor term on the terminal step (`PotentialShapingWrapper.step` sets
    the successor potential to zero when the episode ends), and a terminal
    successor's potential need not vanish.
    """

    def __init__(
        self,
        current: TransitionFeatureFn,
        successor: TransitionFeatureFn,
        terminal_gate: bool = False,
    ):
        self.current = current
        self.successor = successor
        self.terminal_gate = terminal_gate

    @property
    def feature_dim(self) -> int:
        return self.current.feature_dim + self.successor.feature_dim

    def __call__(self, transition: Transition) -> np.ndarray:
        successor = self.successor(transition)
        if self.terminal_gate and transition.terminated:
            successor = np.zeros_like(successor)
        return np.concatenate([self.current(transition), successor])


class ActionTerminationFeatures:
    """
    Action indicators for nonterminal steps followed by action indicators
    for terminal steps, so a terminal reward per action is representable.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    @property
    def feature_dim(self) -> int:
        return 2 * self.num_actions

    def __call__(self, transition: Transition) -> np.ndarray:
        features = np.zeros(self.feature_dim, dtype=np.float64)
        offset = self.num_actions if transition.terminated else 0
        features[offset + transition.action] = 1.0
        return features


class NonterminalActionFeatures:
    """
    Action indicators gated by the nonterminal flag, for a reward that is
    zero at termination.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    @property
    def feature_dim(self) -> int:
        return self.num_actions

    def __call__(self, transition: Transition) -> np.ndarray:
        features = np.zeros(self.feature_dim, dtype=np.float64)
        if not transition.terminated:
            features[transition.action] = 1.0
        return features


class NonterminalConstantFeatures:
    """
    A unit on nonterminal steps, zero on the terminal one.

    Represents a per-step base reward that stops at termination: with weight
    $c$, the prediction pays $c$ on every nonterminal step and $0$ on the
    terminal one. One column, shared by every action.
    """

    @property
    def feature_dim(self) -> int:
        return 1

    def __call__(self, transition: Transition) -> np.ndarray:
        return np.array([0.0 if transition.terminated else 1.0], dtype=np.float64)


class ConcatenatedFeatures:
    """
    The concatenation of transition-feature blocks.
    """

    def __init__(self, blocks: tuple[TransitionFeatureFn, ...]):
        self.blocks = blocks

    @property
    def feature_dim(self) -> int:
        return sum(block.feature_dim for block in self.blocks)

    def __call__(self, transition: Transition) -> np.ndarray:
        return np.concatenate([block(transition) for block in self.blocks])


class ConstantAndSuccessorBasisFeatures:
    """
    A constant followed by a terminal-gated basis expansion of the
    successor observation.
    """

    def __init__(self, basis: Callable[[Any], np.ndarray], basis_dim: int):
        self.basis = basis
        self.basis_dim = basis_dim

    @property
    def feature_dim(self) -> int:
        return 1 + self.basis_dim

    def __call__(self, transition: Transition) -> np.ndarray:
        gate = 0.0 if transition.terminated else 1.0
        features = np.zeros(self.feature_dim, dtype=np.float64)
        features[0] = 1.0
        features[1:] = gate * self.basis(transition.next_observation)
        return features


class ConstantAndEventFeatures:
    """
    A constant followed by an indicator of an event read from `info`.
    """

    def __init__(self, event_key: str):
        self.event_key = event_key

    @property
    def feature_dim(self) -> int:
        return 2

    def __call__(self, transition: Transition) -> np.ndarray:
        if self.event_key not in transition.info:
            raise KeyError(
                f"info has no {self.event_key!r} entry; the event feature "
                "requires the environment to emit it"
            )
        features = np.zeros(2, dtype=np.float64)
        features[0] = 1.0
        features[1] = 1.0 if transition.info[self.event_key] else 0.0
        return features


class RawObservationFeatures:
    """Delegates to a feature function over the raw observation prefix.

    An observation wrapper may append coordinates to the raw observation.
    Feature functions that describe the raw environment read only the first
    ``raw_dim`` entries of the observation and of its successor; the appended
    block follows the raw dimensions.
    """

    def __init__(self, feats_fn: TransitionFeatureFn, raw_dim: int):
        if raw_dim < 1:
            raise ValueError(f"raw_dim must be positive. Got: {raw_dim}")
        self.feats_fn = feats_fn
        self.raw_dim = int(raw_dim)

    @property
    def feature_dim(self) -> int:
        return self.feats_fn.feature_dim

    def __call__(self, transition: Transition) -> np.ndarray:
        return self.feats_fn(
            dataclasses.replace(
                transition,
                observation=np.asarray(transition.observation)[: self.raw_dim],
                next_observation=np.asarray(transition.next_observation)[
                    : self.raw_dim
                ],
            )
        )


class StateBinsActionFeatures:
    """
    A one-hot over the joint cell of a uniform partition of the current
    observation, selected by the action block.

    The partition covers the observation box: each dimension is split into
    `bins[idx]` equal-width cells, the joint cell index is row-major over
    dimensions. Each action block has one coordinate per partition cell.
    """

    def __init__(
        self,
        bins: Sequence[int],
        low: Sequence[float] | np.ndarray,
        high: Sequence[float] | np.ndarray,
        num_actions: int,
    ):
        if len(bins) != len(low) or len(bins) != len(high):
            raise ValueError(
                f"bins must match the observation dimensionality: "
                f"bins={bins}, low={low}, high={high}"
            )
        if any(count < 1 for count in bins):
            raise ValueError(f"every bin count must be positive. Got: {bins}")
        self.bins = tuple(int(count) for count in bins)
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        if np.any(self.high <= self.low):
            raise ValueError(
                f"every observation bound must satisfy high > low. "
                f"Got low={low}, high={high}"
            )
        self.num_actions = num_actions
        self.num_cells = int(np.prod(self.bins))

    @property
    def feature_dim(self) -> int:
        return self.num_actions * self.num_cells

    def __call__(self, transition: Transition) -> np.ndarray:
        features = np.zeros(self.feature_dim, dtype=np.float64)
        cell = self._cell_index(transition.observation)
        features[transition.action * self.num_cells + cell] = 1.0
        return features

    def _cell_index(self, observation: Any) -> int:
        values = np.asarray(observation, dtype=np.float64)
        widths = (self.high - self.low) / self.bins
        indices = np.floor((values - self.low) / widths).astype(np.int64)
        # The inclusive upper endpoint belongs to the final partition cell.
        indices = np.minimum(np.maximum(indices, 0), np.asarray(self.bins) - 1)
        return int(np.ravel_multi_index(indices, self.bins))


class _GridWorldNominalFeatures:
    """
    Nominal successor under `rlplg`'s GridWorld move semantics: the action's
    unit step added to the position, clamped to the observation box.

    Both the action order and the clamp come from the environment, not from
    the observation space, so a feature built on this class describes that
    environment's dynamics rather than a generic grid. Subclasses supply the
    encoding of the nominal successor.
    """

    def __init__(
        self, low: Sequence[int] | np.ndarray, high: Sequence[int] | np.ndarray
    ):
        low_values = np.asarray(low, dtype=np.float64)
        high_values = np.asarray(high, dtype=np.float64)
        if low_values.shape != (2,) or high_values.shape != (2,):
            raise ValueError("GridWorld bounds must both have shape (2,)")
        if not (
            np.all(np.isfinite(low_values))
            and np.all(np.isfinite(high_values))
            and np.all(low_values == np.round(low_values))
            and np.all(high_values == np.round(high_values))
            and np.all(high_values >= low_values)
        ):
            raise ValueError(
                "GridWorld bounds must be finite integers with high >= low"
            )
        self.low = low_values.astype(np.int64)
        self.high = high_values.astype(np.int64)

    def _nominal(self, observation: Any, action: int) -> np.ndarray:
        position = np.asarray(observation, dtype=np.int64)
        nominal: np.ndarray = np.clip(
            position + np.asarray(GRIDWORLD_MOVES[action]), self.low, self.high
        )
        return nominal


class GridWorldNominalTransitionAnomalyFeatures(_GridWorldNominalFeatures):
    """
    A constant followed by an indicator that the successor observation
    deviates from the nominal move.

    Any successor other than the clamped unit step - a teleport to a
    constructed start after entering a cliff, for example - is an anomaly.
    This assumes the move semantics, not a reward value or a layout.
    """

    @property
    def feature_dim(self) -> int:
        return 2

    def __call__(self, transition: Transition) -> np.ndarray:
        nominal = self._nominal(transition.observation, transition.action)
        successor = np.asarray(transition.next_observation, dtype=np.int64)
        features = np.zeros(self.feature_dim, dtype=np.float64)
        features[0] = 1.0
        features[1] = 1.0 if np.any(successor != nominal) else 0.0
        return features


class GridWorldNominalNextCellFeatures(_GridWorldNominalFeatures):
    """
    A constant followed by a one-hot over the nominal next cell.

    The cell index is row-major over the full observation box, so the one-hot
    covers every cell of the grid, reachable or not; cells never reached by a
    nominal move keep a structurally zero column.
    """

    def __init__(
        self, low: Sequence[int] | np.ndarray, high: Sequence[int] | np.ndarray
    ):
        super().__init__(low=low, high=high)
        self.cell_sizes = tuple(self.high - self.low + 1)
        self.num_cells = int(np.prod(self.cell_sizes))

    @property
    def feature_dim(self) -> int:
        return 1 + self.num_cells

    def __call__(self, transition: Transition) -> np.ndarray:
        nominal = self._nominal(transition.observation, transition.action)
        cell = int(np.ravel_multi_index(nominal - self.low, self.cell_sizes))
        features = np.zeros(self.feature_dim, dtype=np.float64)
        features[0] = 1.0
        features[1 + cell] = 1.0
        return features


def transition_feature_fn(env: gym.Env, spec: Mapping[str, Any]) -> TransitionFeatureFn:
    """
    Builds a transition-feature function from a specification.
    """
    name = spec["name"]
    args = dict(spec.get("args") or {})
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown transition feature {name!r}. Must be one of: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name](env, **args)


def current_features_spec(
    current_spec: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """
    Specification for the current state-action block alone.
    """
    return {"name": "current-features", "args": {"current_spec": list(current_spec)}}


def current_and_successor_features_spec(
    current_spec: Sequence[Mapping[str, Any]],
    successor_spec: Sequence[Mapping[str, Any]],
    terminal_gate: bool = False,
) -> Mapping[str, Any]:
    """
    Specification for a current block followed by a successor block.

    With ``terminal_gate`` the successor block is zeroed on terminated
    transitions, the gate a potential-based reward applies to its successor
    term.
    """
    return {
        "name": "current-and-successor-features",
        "args": {
            "current_spec": list(current_spec),
            "successor_spec": list(successor_spec),
            "terminal_gate": terminal_gate,
        },
    }


def concatenated_features_spec(
    blocks: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """
    Specification for the concatenation of named transition-feature blocks.
    """
    return {"name": "concatenated-features", "args": {"blocks": list(blocks)}}


def _current_features(
    env: gym.Env, current_spec: Sequence[Mapping[str, Any]]
) -> TransitionFeatureFn:
    return FTOpFeatures(transform.transform_pipeline(env, current_spec))


def _current_and_successor_features(
    env: gym.Env,
    current_spec: Sequence[Mapping[str, Any]],
    successor_spec: Sequence[Mapping[str, Any]],
    terminal_gate: bool = False,
) -> TransitionFeatureFn:
    for spec in successor_spec:
        if (spec.get("args") or {}).get("hash_dim"):
            raise ValueError(
                "a successor block must not set hash_dim: bucket collisions "
                "would merge distinct successor states"
            )
    pipe = transform.Pipeline.from_specs(
        input_space=transform.ExampleSpace(
            observation_space=env.observation_space,
            action_space=gym.spaces.Discrete(1),
        ),
        specs=successor_spec,
    )
    return CurrentAndSuccessorFeatures(
        current=_current_features(env, current_spec),
        successor=FTOpFeatures(pipe, successor=True),
        terminal_gate=terminal_gate,
    )


def _concatenated_features(
    env: gym.Env,
    blocks: Sequence[Mapping[str, Any]],
) -> TransitionFeatureFn:
    """Concatenation of named transition-feature blocks."""
    if not blocks:
        raise ValueError("concatenated-features requires at least one block")
    return ConcatenatedFeatures(
        tuple(transition_feature_fn(env, block) for block in blocks)
    )


def _nonterminal_constant_features(env: gym.Env) -> TransitionFeatureFn:
    del env
    return NonterminalConstantFeatures()


def _action_and_termination_features(
    env: gym.Env,
) -> TransitionFeatureFn:
    return ActionTerminationFeatures(_num_actions(env))


def _nonterminal_action_features(env: gym.Env) -> TransitionFeatureFn:
    return NonterminalActionFeatures(_num_actions(env))


def _num_actions(env: gym.Env) -> int:
    """Validates the action indexing used by indicator features."""
    if (
        not isinstance(env.action_space, gym.spaces.Discrete)
        or env.action_space.start != 0
    ):
        raise TypeError("Reward features require a zero-based Discrete action space")
    return int(env.action_space.n)


def _constant_and_successor_basis_features(
    env: gym.Env, basis_spec: Mapping[str, Any]
) -> TransitionFeatureFn:
    del env
    basis, basis_dim = _basis(basis_spec)
    return ConstantAndSuccessorBasisFeatures(basis=basis, basis_dim=basis_dim)


def _constant_and_event_features(env: gym.Env, event_key: str) -> TransitionFeatureFn:
    """Constant and an indicator of the `event_key` entry of `info`."""
    del env
    return ConstantAndEventFeatures(event_key=event_key)


def _state_bins_action_features(
    env: gym.Env, bins: Sequence[int], raw_dim: int | None = None
) -> TransitionFeatureFn:
    """Joint-cell partition of the observation box, one action block each."""
    space = _box_observation_space(env)
    low = space.low if raw_dim is None else space.low[:raw_dim]
    high = space.high if raw_dim is None else space.high[:raw_dim]
    features = StateBinsActionFeatures(
        bins=bins,
        low=low,
        high=high,
        num_actions=_num_actions(env),
    )
    return features if raw_dim is None else RawObservationFeatures(features, raw_dim)


def _gridworld_nominal_transition_anomaly_features(
    env: gym.Env,
    low: Sequence[int] | None = None,
    high: Sequence[int] | None = None,
    raw_dim: int | None = None,
) -> TransitionFeatureFn:
    """Uses reachable GridWorld bounds unless coordinates are overridden."""
    low_bound, high_bound = _gridworld_bounds(env, low, high, raw_dim=raw_dim)
    features = GridWorldNominalTransitionAnomalyFeatures(low_bound, high_bound)
    return features if raw_dim is None else RawObservationFeatures(features, raw_dim)


def _gridworld_nominal_next_cell_features(
    env: gym.Env,
    low: Sequence[int] | None = None,
    high: Sequence[int] | None = None,
    raw_dim: int | None = None,
) -> TransitionFeatureFn:
    """Uses reachable GridWorld bounds unless coordinates are overridden."""
    low_bound, high_bound = _gridworld_bounds(env, low, high, raw_dim=raw_dim)
    features = GridWorldNominalNextCellFeatures(low_bound, high_bound)
    return features if raw_dim is None else RawObservationFeatures(features, raw_dim)


def _gridworld_bounds(
    env: gym.Env,
    low: Sequence[int] | None,
    high: Sequence[int] | None,
    raw_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    space = _integer_box_observation_space(env)
    if raw_dim is not None:
        space = gym.spaces.Box(
            low=space.low[:raw_dim], high=space.high[:raw_dim], dtype=np.int64
        )
    if space.shape != (2,):
        raise ValueError(f"GridWorld moves are 2-dimensional: shape={space.shape}")
    if _num_actions(env) != len(GRIDWORLD_MOVES):
        raise ValueError("GridWorld features require four actions")
    default_high = space.high
    if env.has_wrapper_attr("grid_shape"):
        default_high = space.low + np.asarray(env.get_wrapper_attr("grid_shape")) - 1
    return (
        np.asarray(space.low if low is None else low),
        np.asarray(default_high if high is None else high),
    )


def _box_observation_space(env: gym.Env) -> gym.spaces.Box:
    space = env.observation_space
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(
            f"Reward features require a Box observation space. Got: {type(space)}"
        )
    return space


def _integer_box_observation_space(env: gym.Env) -> gym.spaces.Box:
    """A Box observation space whose bounds are whole numbers."""
    space = _box_observation_space(env)
    if not (
        np.allclose(space.low, np.round(space.low))
        and np.allclose(space.high, np.round(space.high))
    ):
        raise ValueError(
            f"grid features require whole-number observation bounds. "
            f"Got low={space.low}, high={space.high}"
        )
    return space


def _basis(
    spec: Mapping[str, Any],
) -> tuple[Callable[[Any], np.ndarray], int]:
    name = spec["name"]
    args = dict(spec.get("args") or {})
    coordinate = int(args.get("coordinate", 0))
    if name == "sine":
        frequency = float(args["frequency"])
        return (
            lambda obs: np.array(
                [np.sin(frequency * float(np.asarray(obs)[coordinate]))],
                dtype=np.float64,
            ),
            1,
        )
    if name == "fourier":
        order = int(args["order"])
        low = float(args["low"])
        high = float(args["high"])
        if order < 1:
            raise ValueError(f"fourier order must be positive. Got: {order}")
        if high <= low:
            raise ValueError(
                f"fourier bounds must satisfy high > low. Got: {low}, {high}"
            )

        def fourier(obs: Any) -> np.ndarray:
            scaled = (float(np.asarray(obs)[coordinate]) - low) / (high - low)
            scaled = min(max(scaled, 0.0), 1.0)
            harmonics = [
                value
                for idx in range(1, order + 1)
                for value in (
                    np.cos(np.pi * idx * scaled),
                    np.sin(np.pi * idx * scaled),
                )
            ]
            return np.array(harmonics, dtype=np.float64)

        return fourier, 2 * order
    if name == "uniform-bins":
        count = int(args["count"])
        low = float(args["low"])
        high = float(args["high"])

        def bins(obs: Any) -> np.ndarray:
            value = float(np.asarray(obs)[coordinate])
            width = (high - low) / count
            index = int(np.floor((value - low) / width))
            index = min(max(index, 0), count - 1)
            features = np.zeros(count, dtype=np.float64)
            features[index] = 1.0
            return features

        return bins, count
    raise ValueError(
        f"Unknown basis {name!r}. Must be one of: sine, fourier, uniform-bins"
    )


_BUILDERS: dict[str, Callable[..., TransitionFeatureFn]] = {
    "current-features": _current_features,
    "current-and-successor-features": _current_and_successor_features,
    "concatenated-features": _concatenated_features,
    "nonterminal-constant-features": _nonterminal_constant_features,
    "action-and-termination-features": _action_and_termination_features,
    "nonterminal-action-features": _nonterminal_action_features,
    "constant-and-successor-basis-features": _constant_and_successor_basis_features,
    "constant-and-event-features": _constant_and_event_features,
    "state-bins-action-features": _state_bins_action_features,
    "gridworld-nominal-transition-anomaly-features": (
        _gridworld_nominal_transition_anomaly_features
    ),
    "gridworld-nominal-next-cell-features": _gridworld_nominal_next_cell_features,
}
