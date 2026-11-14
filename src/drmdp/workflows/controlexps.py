import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from drmdp import constants, rewardfeat, rewdelay
from drmdp.envs import gridutils, lunar_lander
from drmdp.workflows import estimationencodings

EPSILON = 0.1
MAX_STEPS_PER_EPISODE_GEM = 10_000
LEARNING_RATE_SPEC: Mapping[str, Any] = {
    "name": "constant",
    "args": {"initial_lr": 0.01},
}
# Numerator of the dimension-derived learning-rate rules. The rate is the head
# divided by the representation's size, so a larger representation takes
# proportionally smaller steps.
LUNAR_LANDER_BASE_STEP_SIZE = 0.1
# Rule the lunar template resolves into its learning rate. The options are the
# keys of LUNAR_LANDER_LEARNING_RATE_RULES and their relative performance is
# measured in agents/plans/2026-11-08-lunar-lander-illustration.md.
LUNAR_LANDER_LEARNING_RATE_RULE: str = "dim"
LUNAR_LANDER_TILING_DIM = 8
LUNAR_LANDER_NUM_TILINGS = 32
LUNAR_LANDER_HASH_DIM = 2048
LUNAR_LANDER_MAX_EPISODE_STEPS = 1000
MINES_GW_GRID = [
    "ooooxooooooo",
    "oooooooooxoo",
    "oxoooooxoooo",
    "oooooxoooooo",
    "ooxooooooxoo",
    "sxxxxxxxxxxg",
]
MAX_OPTIONS_DELAY = 4
DEFAULT_UNIFORM_DELAY_RANGE = 5
DEFAULT_IMPUTE_VALUE = 0
DEFAULT_DISCOUNT_FACTORS = (1.0, 0.99, 0.98)


def default_delay_config() -> list[dict[str, Any]]:
    """Uniform delay ranges for the control experiment families."""
    delay_configs = []
    for min_delay, max_delay in ((2, 4), (6, 8), (10, 12)):
        delay_configs.append(
            {
                "name": "uniform",
                "args": {"min_delay": min_delay, "max_delay": max_delay},
            },
        )
    return delay_configs


def _delay_min(delay_config: Mapping[str, Any]) -> int:
    """Extracts the lower bound delay from a delay config."""
    args = delay_config["args"]
    if delay_config["name"] == "fixed":
        return int(args["delay"])
    return int(args["min_delay"])


def learning_rate_from_dimension(base_step_size: float, observation_dim: int) -> float:
    """Step size divided by the observation width."""
    if observation_dim < 1:
        raise ValueError(f"observation_dim must be positive. Got: {observation_dim}")
    return base_step_size / observation_dim


def learning_rate_from_active_features(
    base_step_size: float, num_tilings: int
) -> float:
    """Step size divided by the number of active binary features.

    A tile-coded state-action row activates one tile per tiling, so the count of
    active features is the number of tilings and the squared feature norm is that
    same number.
    """
    if num_tilings < 1:
        raise ValueError(f"num_tilings must be positive. Got: {num_tilings}")
    return base_step_size / num_tilings


def learning_rate_fixed(base_step_size: float, dimension: int) -> float:
    """The control families' constant rate, independent of the dimension."""
    del base_step_size, dimension
    return float(LEARNING_RATE_SPEC["args"]["initial_lr"])


LUNAR_LANDER_LEARNING_RATE_RULES = {
    "dim": learning_rate_from_dimension,
    "actives": learning_rate_from_active_features,
    "fixed": learning_rate_fixed,
}


def lunar_lander_learning_rate_config(
    rule: str | None = None,
    num_tilings: int = LUNAR_LANDER_NUM_TILINGS,
    base_step_size: float = LUNAR_LANDER_BASE_STEP_SIZE,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolves the lunar rule into a learning-rate config and its provenance.

    Returns the config a problem spec carries and the record of the rule that
    produced it: the rule name, the dimension it resolved against and the
    resulting rate.
    """
    selected = LUNAR_LANDER_LEARNING_RATE_RULE if rule is None else rule
    if selected not in LUNAR_LANDER_LEARNING_RATE_RULES:
        raise ValueError(
            f"Unknown lunar learning-rate rule {selected!r}. "
            f"Must be one of: {sorted(LUNAR_LANDER_LEARNING_RATE_RULES)}"
        )
    observation_dim = lunar_lander.OBSERVATION_SIZE
    dimension = num_tilings if selected == "actives" else observation_dim
    rate = LUNAR_LANDER_LEARNING_RATE_RULES[selected](base_step_size, dimension)
    config = {"name": "constant", "args": {"initial_lr": rate}}
    provenance = {
        "learning_rate_rule": selected,
        "learning_rate": rate,
        "observation_dim": observation_dim,
        "dimension_used": dimension,
        "num_tilings": num_tilings,
        "base_step_size": base_step_size,
    }
    return config, provenance


def least_specs(
    attempt_estimation_episodes: Sequence[int],
    transition_feats_specs: Sequence[Mapping[str, Any]],
    delay_configs: Sequence[Mapping[str, Any]] = default_delay_config(),
    discounts: Sequence[float] = DEFAULT_DISCOUNT_FACTORS,
    estimation_check: str = "none",
    impute_value: float = DEFAULT_IMPUTE_VALUE,
    learning_rate_config: Mapping[str, Any] = LEARNING_RATE_SPEC,
) -> Sequence[Mapping[str, Any]]:
    """
    Least Squares specs.
    """
    specs = []
    for (
        delay_config,
        gamma,
        transition_feats_spec,
        attempt_estimation_episode,
    ) in itertools.product(
        delay_configs, discounts, transition_feats_specs, attempt_estimation_episodes
    ):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "least-lfa",
                    "args": {
                        "attempt_estimation_episode": attempt_estimation_episode,
                        "transition_feats_spec": transition_feats_spec,
                        "impute_value": impute_value,
                        "estimation_buffer_mult": (
                            constants.DEFAULT_ESTIMATION_BUFFER_MULT
                        ),
                        "intercept": False,
                        "estimation_check": estimation_check,
                    },
                },
                "delay_config": delay_config,
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": learning_rate_config,
            },
        )
    return tuple(specs)


def bayes_least_specs(
    init_attempt_estimation_episodes: Sequence[int],
    transition_feats_specs: Sequence[Mapping[str, Any]],
    delay_configs: Sequence[Mapping[str, Any]] = default_delay_config(),
    discounts: Sequence[float] = DEFAULT_DISCOUNT_FACTORS,
    impute_value: float = DEFAULT_IMPUTE_VALUE,
    sample_weights: bool = False,
    error_variance: str = rewdelay.ERROR_VARIANCE_UNIT,
    learning_rate_config: Mapping[str, Any] = LEARNING_RATE_SPEC,
) -> Sequence[Mapping[str, Any]]:
    """
    Bayesian linear regression specs.

    The mapper name carries the error-variance mode: aggregation labels a
    run by its mapper name alone, so the NIG mode needs a name of its own
    to stay apart from the unit-variance estimator.
    """
    error_variance = rewdelay.resolve_error_variance(error_variance)
    mapper_name = (
        "blade-td-nig"
        if error_variance == rewdelay.ERROR_VARIANCE_NIG
        else "bayes-least-lfa"
    )
    specs = []
    for (
        delay_config,
        gamma,
        transition_feats_spec,
        init_attempt_estimation_episode,
    ) in itertools.product(
        delay_configs,
        discounts,
        transition_feats_specs,
        init_attempt_estimation_episodes,
    ):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": mapper_name,
                    "args": {
                        "init_attempt_estimation_episode": init_attempt_estimation_episode,
                        "mode": "exponential",
                        "transition_feats_spec": transition_feats_spec,
                        "impute_value": impute_value,
                        "estimation_buffer_mult": (
                            constants.DEFAULT_ESTIMATION_BUFFER_MULT
                        ),
                        "intercept": False,
                        "sample_weights": sample_weights,
                        "error_variance": error_variance,
                    },
                },
                "delay_config": delay_config,
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": learning_rate_config,
            },
        )
    return tuple(specs)


def common_problem_specs(
    delay_configs: Sequence[dict[str, Any]] = default_delay_config(),
    discounts: Sequence[float] = DEFAULT_DISCOUNT_FACTORS,
    impute_value: float = DEFAULT_IMPUTE_VALUE,
    include_options: bool = True,
    learning_rate_config: Mapping[str, Any] = LEARNING_RATE_SPEC,
):
    """
    Specs that apply to every env.
    """
    specs = []
    for gamma in discounts:
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {"name": "identity", "args": None},
                "delay_config": None,
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": learning_rate_config,
            },
        )
        for delay_config in delay_configs:
            specs.extend(
                [
                    {
                        "policy_type": "drop-missing",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": delay_config,
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": learning_rate_config,
                    },
                    {
                        "policy_type": "markovian",
                        "reward_mapper": {
                            "name": "impute-missing",
                            "args": {"impute_value": impute_value},
                        },
                        "delay_config": delay_config,
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": learning_rate_config,
                    },
                ]
            )
            min_delay = _delay_min(delay_config)
            if include_options and min_delay <= MAX_OPTIONS_DELAY:
                fixed_delay_config = {
                    "name": "uniform",
                    "args": {
                        "min_delay": min_delay,
                        "max_delay": min_delay,
                    },
                }
                specs.extend(
                    [
                        {
                            "policy_type": "options",
                            "reward_mapper": {
                                "name": "identity",
                                "args": None,
                            },
                            "delay_config": delay_config,
                            "epsilon": EPSILON,
                            "gamma": gamma,
                            "learning_rate_config": learning_rate_config,
                        },
                        {
                            "policy_type": "single-action-options",
                            "reward_mapper": {
                                "name": "identity",
                                "args": None,
                            },
                            "delay_config": fixed_delay_config,
                            "epsilon": EPSILON,
                            "gamma": gamma,
                            "learning_rate_config": learning_rate_config,
                        },
                    ]
                )
    return tuple(specs)


def electric_motor_experiment_specs() -> Sequence[Mapping[str, Any]]:
    """
    Control experiment specs.
    """
    specs = [
        {
            "name": "Finite-CC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [
                    {
                        "name": "splice-tile-observation-action-ft",
                        "args": {"tiling_dim": 4},
                    }
                ]
            ],
            "problem_specs": common_problem_specs(impute_value=0)
            + least_specs(
                attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-CC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "problem_specs": common_problem_specs(impute_value=0)
            + least_specs(
                attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [
                    {
                        "name": "splice-tile-observation-action-ft",
                        "args": {"tiling_dim": 3},
                    }
                ]
            ],
            "problem_specs": common_problem_specs(impute_value=2)
            + least_specs(
                attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ]
            ],
            "problem_specs": common_problem_specs(impute_value=24)
            + least_specs(
                attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "problem_specs": common_problem_specs(impute_value=0)
            + least_specs(
                attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ]
            ],
            "problem_specs": common_problem_specs(impute_value=1)
            + least_specs(
                attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=[
                    rewardfeat.current_and_successor_features_spec(
                        current_spec=[
                            {"name": "scale-observation-ft", "args": None},
                            {"name": "action-segment-observation-ft", "args": None},
                        ],
                        successor_spec=[{"name": "scale-observation-ft", "args": None}],
                    )
                ],
            ),
            "epochs": 1,
        },
    ]
    return tuple(specs)


def grid_dead_ohe_indices(grid: Sequence[str], nactions: int = 4) -> list[int]:
    """
    Returns OHE indices of dead state-action columns.

    Dead states are cliffs, terminal/goal cells, and cells
    unreachable from the start.  Each dead state produces
    ``nactions`` dead columns in the flat-grid-observation-action-ft
    layout (``action * nstates + state_idx``).
    """
    char_map = {
        "o": gridutils.CELL_OPEN,
        "s": gridutils.CELL_START,
        "g": gridutils.CELL_GOAL,
        "x": gridutils.CELL_CLIFF,
    }
    grid_arr = np.array([[char_map[ch] for ch in row] for row in grid], dtype=np.int8)
    exits = list(zip(*np.where(grid_arr == gridutils.CELL_GOAL)))
    return gridutils.grid_dead_ohe_indices(grid_arr, exits=exits, nactions=nactions)


def gaussian_illustration_experiment_specs(
    mc_tiling_dim: int = 4,
    acrobot_tiling_dim: int = 3,
    acrobot_hash_dim: int = 8192,
    gw_tiling_dim: int = 5,
    noise_std_multipliers: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> Sequence[Mapping[str, Any]]:
    """Illustration specs with unit-magnitude Gaussian reward noise.

    Produces one spec per (env, standard-deviation multiplier) combination.
    The sweep value m sets the noise standard deviation directly:
    ``sigma = std_multiplier * reference_std`` with ``reference_std`` 1.0 (the
    modal per-step reward magnitude, 1 for all three environments), so the
    sweep value, the config value, and the sampled standard deviation are the
    same number m and no variance/square-root pair sits between them. The
    sweep follows the bsuite convention of noise std as a magnitude ladder on
    unit-scale rewards (Osband et al., 2019: std factors {0.1, 0.3, 1.0, 3.0,
    10.0}); the full post-shaping range is rejected as the unit because
    GridWorld's is dominated by the -100 cliff penalty (std 25-200 against
    modal rewards of -1). BLADE-TD consumes the posterior mean: the sweep
    compares reward estimators, so the posterior weight draw is switched off
    and its reward estimate is fixed, as LEAST's.

    Every environment carries a PBRS shaping whose discount follows the
    problem spec's gamma through `task.create_env`, an observation wrapper
    that appends the shaped potential, and control features computed on the
    raw dimensions only.
    """
    mc_est_feats: Sequence[Mapping[str, Any]] = [
        estimationencodings.mc_potential_curr_succ()
    ]
    acrobot_est_feats: Sequence[Mapping[str, Any]] = [
        estimationencodings.acrobot_potential_curr_succ()
    ]
    gw_est_feats: Sequence[Mapping[str, Any]] = [
        estimationencodings.gw_potential_curr_succ()
    ]

    env_templates: list[Mapping[str, Any]] = [
        {
            "name": "MountainCar-v0",
            "base_args": {
                "max_episode_steps": 2500,
                "reward_shaping": {
                    "name": "mountain-car-height-pbrs",
                    "args": {"scale": 1.0},
                },
                "reward_obs": {
                    "name": "mountain-car-height-pbrs-potential",
                    "args": {},
                },
            },
            # unit = modal per-step reward magnitude: the base reward is -1
            # per step and the PBRS term r' = -1 + gamma*Phi(s') - Phi(s) is
            # bounded by 0.21 at gamma = 1 (|x' - x| <= 0.07,
            # |d sin(3x)/dx| <= 3)
            "reference_std": 1.0,
            "feats_specs": [
                [
                    estimationencodings.keep_raw_dims(raw_dim=2, observation_dim=3),
                    {
                        "name": "tile-observation-action-ft",
                        "args": {"tiling_dim": mc_tiling_dim},
                    },
                ]
            ],
            "problem_specs_fn": lambda: (
                common_problem_specs(include_options=False)
                + least_specs(
                    attempt_estimation_episodes=(10,),
                    estimation_check="factors",
                    transition_feats_specs=mc_est_feats,
                )
                + bayes_least_specs(
                    init_attempt_estimation_episodes=(10,),
                    transition_feats_specs=mc_est_feats,
                )
            ),
            "epochs": 10,
        },
        {
            "name": "Acrobot-v1",
            "base_args": {
                "max_episode_steps": 500,
                "reward_shaping": {
                    "name": "acrobot-tip-height-pbrs",
                    "args": {"scale": 1.0},
                },
                "reward_obs": {"name": "acrobot-tip-height-pbrs-potential", "args": {}},
            },
            # unit = modal per-step reward magnitude: the base reward is -1 on
            # non-terminal steps and 0 on the terminal one, and the PBRS term
            # r' = -1*1[not term] + gamma*Phi(s') - Phi(s) is bounded by the
            # potential's range
            "reference_std": 1.0,
            "feats_specs": [
                [
                    estimationencodings.keep_raw_dims(raw_dim=6, observation_dim=7),
                    {
                        "name": "tile-observation-action-ft",
                        "args": {
                            "tiling_dim": acrobot_tiling_dim,
                            "hash_dim": acrobot_hash_dim,
                        },
                    },
                ]
            ],
            "problem_specs_fn": lambda: (
                common_problem_specs(include_options=False)
                + least_specs(
                    attempt_estimation_episodes=(10,),
                    estimation_check="factors",
                    transition_feats_specs=acrobot_est_feats,
                )
                + bayes_least_specs(
                    init_attempt_estimation_episodes=(10,),
                    transition_feats_specs=acrobot_est_feats,
                )
            ),
            "epochs": 10,
        },
        {
            "name": "GridWorld-MINES",
            "base_args": {
                "grid": MINES_GW_GRID,
                "max_episode_steps": 200,
                "reward_shaping": {
                    "name": "gridworld-distance-pbrs",
                    "args": {"scale": 1.0, "exponent": 2.0},
                },
                "reward_obs": {"name": "gridworld-distance-pbrs-potential", "args": {}},
            },
            # unit = modal per-step reward magnitude: the base reward is -1 per
            # move and -100 on a cliff entry, so the modal magnitude is 1 and
            # the cliff carries a separate indicator coordinate
            "reference_std": 1.0,
            "feats_specs": [
                [
                    estimationencodings.keep_raw_dims(raw_dim=2, observation_dim=3),
                    {
                        "name": "tile-observation-action-ft",
                        "args": {"tiling_dim": gw_tiling_dim},
                    },
                ]
            ],
            "problem_specs_fn": lambda: (
                common_problem_specs(include_options=False)
                + least_specs(
                    attempt_estimation_episodes=(10,),
                    estimation_check="factors",
                    transition_feats_specs=gw_est_feats,
                )
                + bayes_least_specs(
                    init_attempt_estimation_episodes=(10,),
                    transition_feats_specs=gw_est_feats,
                )
            ),
            "epochs": 5,
        },
        _lunar_lander_template(),
    ]

    specs: list[Mapping[str, Any]] = []
    for noise_std_multiplier in noise_std_multipliers:
        for template in env_templates:
            env_args = dict(template["base_args"])
            env_args["reward_noise"] = {
                "name": "relative-gaussian",
                "args": {
                    "std_multiplier": noise_std_multiplier,
                    "reference_std": template["reference_std"],
                },
            }
            specs.append(
                {
                    "name": template["name"],
                    "args": env_args,
                    "feats_specs": template["feats_specs"],
                    "problem_specs": template["problem_specs_fn"](),
                    "metadata": template.get("metadata"),
                    "epochs": template["epochs"],
                }
            )
    return tuple(specs)


def _lunar_lander_template(
    learning_rate_rule: str | None = None,
) -> Mapping[str, Any]:
    """The illustration family's LunarLander-v3 template.

    The reward is the environment's own, with no shaping wrapper, and the
    observation carries the reward basis so the per-step reward is a linear
    function of the emitted features. The estimator fits the family's
    transition encoding, which reads that basis, under the ``factors`` gate.

    The gate admits a fit only once every column is nonzero. The landing column
    is nonzero only after a transition in which the policy came to rest, so the
    first fit is deferred until a landing occurs; deferred attempts are recorded
    in the run's estimation events.
    """
    learning_rate_config, provenance = lunar_lander_learning_rate_config(
        rule=learning_rate_rule
    )
    encoding: Sequence[Mapping[str, Any]] = [estimationencodings.lunar_lander_reward()]

    def problem_specs_fn() -> Sequence[Mapping[str, Any]]:
        specs: list[Mapping[str, Any]] = list(
            common_problem_specs(
                include_options=False,
                learning_rate_config=learning_rate_config,
            )
        )
        specs += list(
            least_specs(
                attempt_estimation_episodes=(10,),
                estimation_check="factors",
                transition_feats_specs=encoding,
                learning_rate_config=learning_rate_config,
            )
        )
        specs += list(
            bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=encoding,
                learning_rate_config=learning_rate_config,
            )
        )
        return specs

    return {
        "name": "LunarLander-v3",
        "base_args": {
            "max_episode_steps": LUNAR_LANDER_MAX_EPISODE_STEPS,
            "wrapper": constants.LUNAR_LANDER_REWARD_OBS,
        },
        # unit = modal per-step reward magnitude: the potential difference and
        # the fuel cost are of order 1 per step, and the +/-100 terminal reward
        # is a separate branch the terminal columns carry
        "reference_std": 1.0,
        "feats_specs": [
            [
                {
                    "name": "splice-tile-observation-action-ft",
                    "args": {
                        "tiling_dim": LUNAR_LANDER_TILING_DIM,
                        "num_tilings": LUNAR_LANDER_NUM_TILINGS,
                        "hash_dim": LUNAR_LANDER_HASH_DIM,
                    },
                }
            ]
        ],
        "problem_specs_fn": problem_specs_fn,
        "metadata": provenance,
        "epochs": 10,
    }


def load_solvable_grids(
    path: str,
    min_passes: int = 4,
) -> Sequence[Mapping[str, Any]]:
    """
    Loads grid specs from a solvable grids JSON file, filtering by
    minimum number of passed solvability trials.
    """
    with open(path, "r") as readable:
        entries: list[Mapping[str, Any]] = json.load(readable)
    passing_grids = [entry for entry in entries if entry["passes"] >= min_passes]
    return passing_grids


def grid_experiments_specs(
    grid_specs: Sequence[Mapping[str, Any]],
    max_episode_steps: int = 200,
) -> Sequence[Mapping[str, Any]]:
    """
    Control experiment specs from pre-validated grid environments.
    """
    specs: list[Mapping[str, Any]] = []
    for entry in grid_specs:
        nrows, ncols = entry["size"]
        dead_ohe_indices = entry["dead_ohe"]
        tiling_dim = math.ceil(max(nrows, ncols) / 2)
        episode_steps = entry.get("max_episode_steps", max_episode_steps)
        specs.append(
            {
                "name": entry["env_name"],
                "args": {
                    "grid": entry["grid"],
                    "max_episode_steps": episode_steps,
                },
                "metadata": {
                    "size": [nrows, ncols],
                    "seed": entry["seed"],
                    "distance": entry["bfs_distance"],
                    "num_cliffs": entry.get("actual_cliffs"),
                },
                "feats_specs": [
                    [
                        {
                            "name": "tile-observation-action-ft",
                            "args": {"tiling_dim": tiling_dim},
                        }
                    ]
                ],
                "problem_specs": common_problem_specs(include_options=False)
                + least_specs(
                    attempt_estimation_episodes=(10,),
                    estimation_check="factors",
                    transition_feats_specs=[
                        rewardfeat.current_features_spec(
                            current_spec=[
                                {
                                    "name": "flat-grid-observation-action-ft",
                                    "args": {},
                                },
                                {
                                    "name": "drop-observation-dims-ft",
                                    "args": {"axis_dims": {0: dead_ohe_indices}},
                                },
                            ]
                        )
                    ],
                )
                + bayes_least_specs(
                    init_attempt_estimation_episodes=(10,),
                    transition_feats_specs=[
                        rewardfeat.current_features_spec(
                            current_spec=[
                                {
                                    "name": "flat-grid-observation-action-ft",
                                    "args": {},
                                },
                                {
                                    "name": "drop-observation-dims-ft",
                                    "args": {"axis_dims": {0: dead_ohe_indices}},
                                },
                            ]
                        )
                    ],
                ),
                "epochs": 5,
            }
        )
    return tuple(specs)
