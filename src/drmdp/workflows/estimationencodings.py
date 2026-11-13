"""
Transition-feature encodings of the reward-estimation campaign.

The builders describe the map from a transition $(s, a, s')$ to the
feature row the estimator regresses aggregate window sums against. Each
builder returns a `transition_feats_spec` dict consumed by
`task.estimator_feats_fn`.

Encodings differ in the knowledge they assume: reward structure (the
functional form of the per-step reward), dynamics structure (the nominal
transition rule), general basis expansions, and the production tile maps.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from drmdp import rewardfeat


def mc_potential_curr_succ() -> Mapping[str, Any]:
    """Current and terminal-gated successor potential of the shaped MountainCar reward."""
    return potential_curr_succ(raw_dim=2, observation_dim=3)


def acrobot_potential_curr_succ() -> Mapping[str, Any]:
    """Current and successor potential of the shaped Acrobot reward.

    The nonterminal-gated unit carries the base reward that stops at
    termination: Acrobot pays $-1$ on nonterminal steps and $0$ on the
    terminal one, so the reward needs no constant column.
    """
    return rewardfeat.concatenated_features_spec(
        [
            potential_curr_succ(raw_dim=6, observation_dim=7, constant=False),
            {"name": "nonterminal-constant-features", "args": {}},
        ]
    )


def gw_potential_curr_succ() -> Mapping[str, Any]:
    """Current and successor potential of the shaped GridWorld reward.

    The teleport anomaly block carries the constant and the cliff payment:
    the base reward is $-1$ per move and $-100$ when the move enters a cliff
    cell, which is exactly the step whose successor deviates from the nominal
    clamped unit move.
    """
    return rewardfeat.concatenated_features_spec(
        [
            potential_curr_succ(raw_dim=2, observation_dim=3, constant=False),
            gridworld_transition_anomaly(raw_dim=2),
        ]
    )


def potential_curr_succ(
    raw_dim: int, observation_dim: int, constant: bool = True
) -> Mapping[str, Any]:
    r"""Current and successor blocks over the appended potential entry.

    The row is $[\Phi(s), 1 \\mid \\Phi(s')]$ with the successor block zeroed
    on terminated transitions: the shaping wrapper pays no successor term
    there, and a terminal successor's potential need not vanish (a
    MountainCar overshoot past the goal, an Acrobot tip above the goal
    height, a GridWorld teleport to the start cell). With ``constant`` the
    current block appends the constant ``1`` — a feature rather than a
    fit-time intercept because the estimator fits window sums: only a
    feature accumulates to the window length. Encodings whose remaining
    block supplies the constant (or replaces it with a gated unit) set
    ``constant=False``. The discount weights the successor entry, so one
    weight vector per arm of the discount sweep.
    """
    current_spec = [keep_potential(raw_dim=raw_dim, observation_dim=observation_dim)]
    if constant:
        current_spec.append({"name": "constant-ft", "args": {}})
    return rewardfeat.current_and_successor_features_spec(
        current_spec=current_spec,
        successor_spec=[
            keep_potential(raw_dim=raw_dim, observation_dim=observation_dim)
        ],
        terminal_gate=True,
    )


def keep_potential(raw_dim: int, observation_dim: int) -> Mapping[str, Any]:
    """Keeps the appended potential entry, dropping every other dimension."""
    dropped = [idx for idx in range(observation_dim) if idx != raw_dim]
    return {
        "name": "drop-observation-dims-ft",
        "args": {"axis_dims": {0: dropped}},
    }


def keep_raw_dims(raw_dim: int, observation_dim: int) -> Mapping[str, Any]:
    """Removes an appended observation block, keeping the raw dimensions.

    Encodings that describe the raw environment are prefixed with this
    operator, so their meaning does not depend on the appended coordinates.
    """
    return {
        "name": "drop-observation-dims-ft",
        "args": {"axis_dims": {0: list(range(raw_dim, observation_dim))}},
    }


def mc_fourier_succ(order: int = 5) -> Mapping[str, Any]:
    """Constant and terminal-gated Fourier harmonics of the successor position."""
    return {
        "name": "constant-and-successor-basis-features",
        "args": {
            "basis_spec": {
                "name": "fourier",
                "args": {
                    "order": order,
                    "coordinate": 0,
                    "low": -1.2,
                    "high": 0.6,
                },
            }
        },
    }


def mc_succ_bins(count: int = 54) -> Mapping[str, Any]:
    """Constant and terminal-gated successor-position bins."""
    return {
        "name": "constant-and-successor-basis-features",
        "args": {
            "basis_spec": {
                "name": "uniform-bins",
                "args": {
                    "count": count,
                    "low": -1.2,
                    "high": 0.6,
                    "coordinate": 0,
                },
            }
        },
    }


def sa_bins(bins: Sequence[int], raw_dim: int | None = None) -> Mapping[str, Any]:
    """Joint-cell uniform partition of the observation, one block per action."""
    args: dict[str, Any] = {"bins": list(bins)}
    if raw_dim is not None:
        args["raw_dim"] = raw_dim
    return {"name": "state-bins-action-features", "args": args}


def current_tiles(
    tiling_dim: int, raw_dim: int | None = None, observation_dim: int | None = None
) -> Mapping[str, Any]:
    """The current state-action tile map of the control pipeline."""
    current_spec: list[Mapping[str, Any]] = []
    if raw_dim is not None:
        current_spec.append(
            keep_raw_dims(
                raw_dim=raw_dim,
                observation_dim=_observation_dim(raw_dim, observation_dim),
            )
        )
    current_spec.append(
        {"name": "tile-observation-action-ft", "args": {"tiling_dim": tiling_dim}}
    )
    return rewardfeat.current_features_spec(current_spec=current_spec)


def current_and_successor_tiles(
    tiling_dim: int, raw_dim: int | None = None, observation_dim: int | None = None
) -> Mapping[str, Any]:
    """Current tiles followed by successor tiles, the production map."""
    tiles = {"name": "tile-observation-action-ft", "args": {"tiling_dim": tiling_dim}}
    prefix: list[Mapping[str, Any]] = []
    if raw_dim is not None:
        prefix.append(
            keep_raw_dims(
                raw_dim=raw_dim,
                observation_dim=_observation_dim(raw_dim, observation_dim),
            )
        )
    return rewardfeat.current_and_successor_features_spec(
        current_spec=[*prefix, tiles],
        successor_spec=[*prefix, tiles],
    )


def _observation_dim(raw_dim: int, observation_dim: int | None) -> int:
    """Total observation dimension, defaulting to the raw one."""
    return raw_dim if observation_dim is None else observation_dim


def action_termination() -> Mapping[str, Any]:
    """Action indicators split by the termination flag."""
    return {"name": "action-and-termination-features", "args": {}}


def gridworld_transition_anomaly(raw_dim: int = 2) -> Mapping[str, Any]:
    """Constant and nominal-move anomaly indicator."""
    return {
        "name": "gridworld-nominal-transition-anomaly-features",
        "args": {"raw_dim": raw_dim},
    }


def gridworld_next_cell(raw_dim: int = 2) -> Mapping[str, Any]:
    """Constant and nominal-next-cell one-hot."""
    return {
        "name": "gridworld-nominal-next-cell-features",
        "args": {"raw_dim": raw_dim},
    }


def gridworld_sa_ohe(
    dead_ohe_indices: Sequence[int],
    raw_dim: int | None = None,
    observation_dim: int | None = None,
) -> Mapping[str, Any]:
    """One-hot over live state-action pairs, dead columns dropped."""
    current_spec: list[Mapping[str, Any]] = []
    if raw_dim is not None:
        current_spec.append(
            keep_raw_dims(
                raw_dim=raw_dim,
                observation_dim=_observation_dim(raw_dim, observation_dim),
            )
        )
    current_spec.extend(
        [
            {"name": "flat-grid-observation-action-ft", "args": {}},
            {
                "name": "drop-observation-dims-ft",
                "args": {"axis_dims": {0: dead_ohe_indices}},
            },
        ]
    )
    return rewardfeat.current_features_spec(current_spec=current_spec)
