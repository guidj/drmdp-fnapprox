"""
Experiment specifications for the estimation-only encoding campaign.

One estimation experiment per (environment, encoding, method, delay, noise
multiplier): the estimators receive aggregate window sums from the
environment's task oracle mixed with a uniform action draw at the campaign's
exploration rate, and are scored against the true per-step rewards, without
policy learning. The encodings cover reward structure, the general transition
recipe, the production feature maps, and uniform-bin space partitions.
Dynamics structure means that an encoding assumes an environment's transition
rule, not its reward values.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from drmdp import constants, rewdelay
from drmdp.workflows import controlexps, estimationencodings

# exploration rate of the behaviour policy: the oracle acts on all other
# steps, so the occupancy is goal-directed with an exploration floor
EXPLORATION_RATE = 0.2
POLICY_TYPE = "oracle"
GAMMA = 1.0
LEARNING_RATE_SPEC = {
    "name": "constant",
    "args": {"initial_lr": 0.01},
}
ATTEMPT_ESTIMATION_EPISODE = 10

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


def estimation_experiment_specs(
    exploration_rate: float = EXPLORATION_RATE,
    policy_type: str = POLICY_TYPE,
) -> Sequence[Mapping[str, Any]]:
    """
    Estimation-only experiment specs.

    One spec per (env, encoding, noise multiplier): the encoding label
    and knowledge class are recorded in the experiment metadata, and the
    problem specs hold the method and delay axes. `feats_specs` is empty:
    no control pipeline runs in this campaign. `policy_type` and
    `exploration_rate` set the behaviour policy of every problem spec.
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
                        "problem_specs": _problem_specs(
                            transition_feats_spec, exploration_rate, policy_type
                        ),
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


def _gw_sa_ohe(grid: Sequence[str]) -> Mapping[str, Any]:
    """One-hot over live state-action pairs, dead columns dropped."""
    return estimationencodings.gridworld_sa_ohe(controlexps.grid_dead_ohe_indices(grid))


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
        "estimation_buffer_mult": constants.DEFAULT_ESTIMATION_BUFFER_MULT,
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
        "estimation_buffer_mult": constants.DEFAULT_ESTIMATION_BUFFER_MULT,
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
    exploration_rate: float,
    policy_type: str,
) -> Sequence[Mapping[str, Any]]:
    """Estimator × check-mode × delay axis."""
    specs = []
    for delay_config in DELAY_CONFIGS:
        for mapper_name, mapper_args in _mapper_axis(transition_feats_spec):
            specs.append(
                {
                    "policy_type": policy_type,
                    "reward_mapper": {"name": mapper_name, "args": mapper_args},
                    "delay_config": delay_config,
                    "epsilon": exploration_rate,
                    "gamma": GAMMA,
                    "learning_rate_config": LEARNING_RATE_SPEC,
                }
            )
    return tuple(specs)


def _mc_encodings() -> Sequence[tuple[str, str, Mapping[str, Any]]]:
    """(label, knowledge class, transition-feature spec) for MountainCar."""
    return (
        (
            "structural-sine",
            "reward structure",
            estimationencodings.mc_structural_sine(),
        ),
        ("fourier-succ-o5", "general", estimationencodings.mc_fourier_succ()),
        ("succ-bins-54", "general", estimationencodings.mc_succ_bins()),
        ("sa-bins-4x2", "general", estimationencodings.sa_bins((4, 2))),
        ("sa-bins-18x4", "general", estimationencodings.sa_bins((18, 4))),
        ("current-tiles-td3", "env-specific", estimationencodings.current_tiles(3)),
        (
            "currsucc-tiles-td3",
            "env-specific",
            estimationencodings.current_and_successor_tiles(3),
        ),
    )


def _acrobot_encodings() -> Sequence[tuple[str, str, Mapping[str, Any]]]:
    """(label, knowledge class, transition-feature spec) for Acrobot."""
    return (
        (
            "action-termination",
            "general",
            estimationencodings.action_termination(),
        ),
        (
            "nonterminal-action",
            "reward structure",
            estimationencodings.nonterminal_action(),
        ),
        ("sa-bins-2x6", "general", estimationencodings.sa_bins((2, 2, 2, 2, 2, 2))),
    )


def _gw_encodings() -> Sequence[tuple[str, str, Mapping[str, Any]]]:
    """(label, knowledge class, transition-feature spec) for GridWorld-MINES."""
    return (
        (
            "transition-anomaly",
            "dynamics structure",
            estimationencodings.gridworld_transition_anomaly(),
        ),
        (
            "next-cell-ohe",
            "dynamics structure",
            estimationencodings.gridworld_next_cell(),
        ),
        ("sa-ohe", "env-specific", _gw_sa_ohe(controlexps.MINES_GW_GRID)),
        ("tiles-td3", "env-specific", estimationencodings.current_tiles(3)),
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
