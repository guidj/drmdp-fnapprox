"""
Experiment specifications for the estimation-only encoding campaign.

One estimation experiment per (environment, encoding, method, delay, noise
multiplier): the estimators receive aggregate window sums under a random
policy and are scored against the true per-step rewards, without policy
learning. The encodings are the ones measured in the 2026-10-28 encoding
analyses: reward-structure encodings, the general transition recipe, the
production feature maps, and uniform-bin space partitions. Dynamics structure
means that an encoding assumes an environment's transition rule, not its
reward values.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from drmdp import rewardfeat, rewdelay
from drmdp.workflows import controlexps

EPSILON = 0.1
GAMMA = 1.0
LEARNING_RATE_SPEC = {
    "name": "constant",
    "args": {"initial_lr": 0.01},
}
ATTEMPT_ESTIMATION_EPISODE = 10
ESTIMATION_BUFFER_MULT = 100

# estimation-episode caps: one-shot LEAST stops at its first estimation
# event, so the cap matters for the encodings that defer or update
# continually
EPISODE_CAPS: Mapping[str, int] = {
    "MountainCar-v0": 500,
    "Acrobot-v1": 500,
    "GridWorld-MINES": 500,
}

DELAY_CONFIGS: tuple[Mapping[str, Any], ...] = (
    {"name": "uniform", "args": {"min_delay": 2, "max_delay": 4}},
    {"name": "uniform", "args": {"min_delay": 10, "max_delay": 12}},
)
NOISE_STD_MULTIPLIERS: Sequence[float] = (0.0, 1.0)

# reward-noise unit: the modal per-step reward magnitude, the convention of
# the gaussian illustration family
REFERENCE_STD = 1.0


def estimation_experiment_specs() -> Sequence[Mapping[str, Any]]:
    """
    Estimation-only experiment specs.

    One spec per (env, encoding, noise multiplier): the encoding label
    and knowledge class are recorded in the experiment metadata, and the
    problem specs hold the method and delay axes. `feats_specs` is empty:
    no control pipeline runs in this campaign.
    """
    specs: list[Mapping[str, Any]] = []
    for env_name, template in _env_templates().items():
        for noise_std_multiplier in NOISE_STD_MULTIPLIERS:
            env_args = dict(template["base_args"])
            env_args["reward_noise"] = {
                "name": "relative-gaussian",
                "args": {
                    "std_multiplier": noise_std_multiplier,
                    "reference_std": REFERENCE_STD,
                },
            }
            for label, knowledge, transition_feats_spec in template["encodings"]:
                specs.append(
                    {
                        "name": env_name,
                        "args": env_args,
                        "feats_specs": [[]],
                        "problem_specs": _problem_specs(transition_feats_spec),
                        "metadata": {
                            "encoding": label,
                            "knowledge": knowledge,
                            "noise_std_multiplier": noise_std_multiplier,
                        },
                        "epochs": 1,
                    }
                )
    return tuple(specs)


def episode_cap(env_name: str) -> int:
    """Estimation-episode cap of an environment."""
    if env_name not in EPISODE_CAPS:
        raise ValueError(
            f"no episode cap declared for {env_name!r}. Known: {sorted(EPISODE_CAPS)}"
        )
    return EPISODE_CAPS[env_name]


def expected_run_count(num_runs: int) -> int:
    """Total experiment instances of the campaign, for logging and tests."""
    encodings = sum(
        len(template["encodings"]) for template in _env_templates().values()
    )
    methods = len(_mapper_axis({}))
    return (
        encodings * methods * len(DELAY_CONFIGS) * len(NOISE_STD_MULTIPLIERS) * num_runs
    )


def _mc_structural_sine() -> Mapping[str, Any]:
    """Constant and terminal-gated sine of successor position."""
    return {
        "name": "constant-and-successor-basis-features",
        "args": {
            "basis_spec": {
                "name": "sine",
                "args": {"frequency": 3.0, "coordinate": 0},
            }
        },
    }


def _mc_fourier_succ(order: int = 5) -> Mapping[str, Any]:
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


def _mc_succ_bins(count: int = 54) -> Mapping[str, Any]:
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


def _sa_bins(bins: Sequence[int]) -> Mapping[str, Any]:
    """Joint-cell uniform partition of the observation, one block per action."""
    return {
        "name": "state-bins-action-features",
        "args": {"bins": list(bins)},
    }


def _current_tiles(tiling_dim: int) -> Mapping[str, Any]:
    """The current state-action tile map of the control pipeline."""
    return rewardfeat.current_features_spec(
        current_spec=[
            {"name": "tile-observation-action-ft", "args": {"tiling_dim": tiling_dim}}
        ]
    )


def _current_and_successor_tiles(tiling_dim: int) -> Mapping[str, Any]:
    """Current tiles followed by successor tiles, the production map."""
    return rewardfeat.current_and_successor_features_spec(
        current_spec=[
            {"name": "tile-observation-action-ft", "args": {"tiling_dim": tiling_dim}}
        ],
        successor_spec=[
            {"name": "tile-observation-action-ft", "args": {"tiling_dim": tiling_dim}}
        ],
    )


def _action_termination() -> Mapping[str, Any]:
    """Action indicators split by the termination flag."""
    return {"name": "action-and-termination-features", "args": {}}


def _nonterminal_action() -> Mapping[str, Any]:
    """Action indicators gated by the nonterminal flag."""
    return {"name": "nonterminal-action-features", "args": {}}


def _gw_transition_anomaly() -> Mapping[str, Any]:
    """Constant and nominal-move anomaly indicator."""
    return {
        "name": "gridworld-nominal-transition-anomaly-features",
        "args": {},
    }


def _gw_next_cell() -> Mapping[str, Any]:
    """Constant and nominal-next-cell one-hot."""
    return {
        "name": "gridworld-nominal-next-cell-features",
        "args": {},
    }


def _gw_sa_ohe(grid: Sequence[str]) -> Mapping[str, Any]:
    """One-hot over live state-action pairs, dead columns dropped."""
    dead_ohe_indices = controlexps.grid_dead_ohe_indices(grid)
    return rewardfeat.current_features_spec(
        current_spec=[
            {"name": "flat-grid-observation-action-ft", "args": {}},
            {
                "name": "drop-observation-dims-ft",
                "args": {"axis_dims": {0: dead_ohe_indices}},
            },
        ]
    )


ESTIMATION_CHECK_MODES: Sequence[str] = (
    rewdelay.ESTIMATION_CHECK_NONE,
    rewdelay.ESTIMATION_CHECK_FACTORS,
    rewdelay.ESTIMATION_CHECK_FULL_RANK,
)


def _least_args(
    transition_feats_spec: Mapping[str, Any], estimation_check: str
) -> Mapping[str, Any]:
    return {
        "attempt_estimation_episode": ATTEMPT_ESTIMATION_EPISODE,
        "transition_feats_spec": transition_feats_spec,
        "impute_value": 0,
        "estimation_buffer_mult": ESTIMATION_BUFFER_MULT,
        "intercept": False,
        "estimation_check": estimation_check,
    }


def _bayes_args(
    transition_feats_spec: Mapping[str, Any], estimation_check: str
) -> Mapping[str, Any]:
    return {
        "init_attempt_estimation_episode": ATTEMPT_ESTIMATION_EPISODE,
        "mode": "exponential",
        "transition_feats_spec": transition_feats_spec,
        "impute_value": 0,
        "estimation_buffer_mult": ESTIMATION_BUFFER_MULT,
        "intercept": False,
        "sample_weights": False,
        "estimation_check": estimation_check,
    }


def _mapper_axis(
    transition_feats_spec: Mapping[str, Any],
) -> Sequence[tuple[str, Mapping[str, Any]]]:
    """(mapper_name, mapper_args) for each (estimator, check mode) pair."""
    entries: list[tuple[str, Mapping[str, Any]]] = []
    for estimation_check in ESTIMATION_CHECK_MODES:
        entries.append(
            ("least-lfa", _least_args(transition_feats_spec, estimation_check))
        )
        entries.append(
            (
                "bayes-least-lfa",
                _bayes_args(transition_feats_spec, estimation_check),
            )
        )
    return tuple(entries)


def _problem_specs(
    transition_feats_spec: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    """Estimator × check-mode × delay axis."""
    specs = []
    for delay_config in DELAY_CONFIGS:
        for mapper_name, mapper_args in _mapper_axis(transition_feats_spec):
            specs.append(
                {
                    "policy_type": "markovian",
                    "reward_mapper": {"name": mapper_name, "args": mapper_args},
                    "delay_config": delay_config,
                    "epsilon": EPSILON,
                    "gamma": GAMMA,
                    "learning_rate_config": LEARNING_RATE_SPEC,
                }
            )
    return tuple(specs)


def _mc_encodings() -> Sequence[tuple[str, str, Mapping[str, Any]]]:
    """(label, knowledge class, transition-feature spec) for MountainCar."""
    return (
        ("structural-sine", "reward structure", _mc_structural_sine()),
        ("fourier-succ-o5", "general", _mc_fourier_succ()),
        ("succ-bins-54", "general", _mc_succ_bins()),
        ("sa-bins-4x2", "general", _sa_bins((4, 2))),
        ("sa-bins-18x4", "general", _sa_bins((18, 4))),
        ("current-tiles-td3", "env-specific", _current_tiles(3)),
        ("currsucc-tiles-td3", "env-specific", _current_and_successor_tiles(3)),
    )


def _acrobot_encodings() -> Sequence[tuple[str, str, Mapping[str, Any]]]:
    """(label, knowledge class, transition-feature spec) for Acrobot."""
    return (
        ("action-termination", "general", _action_termination()),
        ("nonterminal-action", "reward structure", _nonterminal_action()),
        ("sa-bins-2x6", "general", _sa_bins((2, 2, 2, 2, 2, 2))),
    )


def _gw_encodings() -> Sequence[tuple[str, str, Mapping[str, Any]]]:
    """(label, knowledge class, transition-feature spec) for GridWorld-MINES."""
    return (
        ("transition-anomaly", "dynamics structure", _gw_transition_anomaly()),
        ("next-cell-ohe", "dynamics structure", _gw_next_cell()),
        ("sa-ohe", "env-specific", _gw_sa_ohe(controlexps.MINES_GW_GRID)),
        ("tiles-td3", "env-specific", _current_tiles(3)),
    )


def _env_templates() -> Mapping[str, Any]:
    """Env construction args and encodings, keyed by env name."""
    return {
        "MountainCar-v0": {
            "base_args": {
                "max_episode_steps": 2500,
                "reward_shaping": {
                    "name": "mountain-car-height",
                    "args": {"scale": 1.0},
                },
            },
            "encodings": _mc_encodings(),
        },
        "Acrobot-v1": {
            "base_args": {
                "max_episode_steps": 500,
                "reward_shaping": {
                    "name": "action-cost",
                    "args": {"action_costs": [0.0, 0.5, 1.0]},
                },
            },
            "encodings": _acrobot_encodings(),
        },
        "GridWorld-MINES": {
            "base_args": {
                "grid": controlexps.MINES_GW_GRID,
                "max_episode_steps": 200,
            },
            "encodings": _gw_encodings(),
        },
    }
