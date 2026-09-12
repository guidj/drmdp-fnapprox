import itertools
import json
import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from drmdp import mathutils
from drmdp.envs import gridutils

EPSILON = 0.1
MAX_STEPS_PER_EPISODE_GEM = 10_000
LEARNING_RATE_SPEC = {
    "name": "constant",
    "args": {"initial_lr": 0.01},
}
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
DEFAULT_DISCOUNT_FACTORS = (1.0, 0.99)


def default_delay_config() -> List[Dict[str, Any]]:
    delay_configs = []
    for lam in (2, 5, 7):
        lb, ub = mathutils.poisson_exact_confidence_interval(observed_value=lam)
        delay_configs.append(
            {
                "name": "clipped-poisson",
                "args": {"lam": lam, "min_delay": max(2, lb), "max_delay": ub},
            },
        )
    return delay_configs


def _delay_min(delay_config: Mapping[str, Any]) -> int:
    """Extracts the lower bound delay from a delay config."""
    args = delay_config["args"]
    if delay_config["name"] == "fixed":
        return int(args["delay"])
    return int(args["min_delay"])


def least_specs(
    attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Sequence[Mapping[str, Any]]],
    delay_configs: Sequence[Mapping[str, Any]] = default_delay_config(),
    discounts: Sequence[float] = DEFAULT_DISCOUNT_FACTORS,
    use_next_state: bool = True,
    check_factors: bool = False,
    impute_value: float = DEFAULT_IMPUTE_VALUE,
) -> Sequence[Mapping[str, Any]]:
    """
    Least Squares specs.
    """
    specs = []
    for (
        delay_config,
        gamma,
        feats_spec,
        attempt_estimation_episode,
    ) in itertools.product(
        delay_configs, discounts, feats_specs, attempt_estimation_episodes
    ):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "least-lfa",
                    "args": {
                        "attempt_estimation_episode": attempt_estimation_episode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                        "impute_value": impute_value,
                        "estimation_buffer_mult": 25,
                        "use_next_state": use_next_state,
                        "check_factors": check_factors,
                    },
                },
                "delay_config": delay_config,
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": LEARNING_RATE_SPEC,
            },
        )
    return tuple(specs)


def bayes_least_specs(
    init_attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Sequence[Mapping[str, Any]]],
    delay_configs: Sequence[Mapping[str, Any]] = default_delay_config(),
    discounts: Sequence[float] = DEFAULT_DISCOUNT_FACTORS,
    impute_value: float = DEFAULT_IMPUTE_VALUE,
) -> Sequence[Mapping[str, Any]]:
    """
    Bayesian linear regression specs.
    """
    specs = []
    for (
        delay_config,
        gamma,
        feats_spec,
        init_attempt_estimation_episode,
    ) in itertools.product(
        delay_configs, discounts, feats_specs, init_attempt_estimation_episodes
    ):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_attempt_estimation_episode": init_attempt_estimation_episode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                        "impute_value": impute_value,
                        "estimation_buffer_mult": 25,
                    },
                },
                "delay_config": delay_config,
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": LEARNING_RATE_SPEC,
            },
        )
    return tuple(specs)


def common_problem_specs(
    delay_configs: Sequence[Dict[str, Any]] = default_delay_config(),
    discounts: Sequence[float] = DEFAULT_DISCOUNT_FACTORS,
    impute_value: float = DEFAULT_IMPUTE_VALUE,
    include_options: bool = True,
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
                "learning_rate_config": LEARNING_RATE_SPEC,
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
                        "learning_rate_config": LEARNING_RATE_SPEC,
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
                        "learning_rate_config": LEARNING_RATE_SPEC,
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
                            "learning_rate_config": LEARNING_RATE_SPEC,
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
                            "learning_rate_config": LEARNING_RATE_SPEC,
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
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
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
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
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
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
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
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
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
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
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
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ]
                ],
            ),
            "epochs": 1,
        },
    ]
    return tuple(specs)


def _grid_dead_ohe_indices(grid: Sequence[str], nactions: int = 4) -> List[int]:
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


def illustration_experiment_specs(
    mc_tiling_dim: int = 4,
    acrobot_tiling_dim: int = 3,
    acrobot_hash_dim: int = 8192,
    gw_tiling_dim: int = 5,
) -> Sequence[Mapping[str, Any]]:
    """
    Illustration experiment specs for Mountain Car, Acrobot, and GridWorld.

    Reward shaping creates non-constant rewards so estimation
    quality matters.  Control and estimation encodings are
    independent — control uses the canonical tile coding,
    estimation uses count-based encodings matched to the
    reward function's structure.
    """
    mc_est_feats: Sequence[Sequence[Mapping[str, Any]]] = [
        [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
    ]
    acrobot_est_feats: Sequence[Sequence[Mapping[str, Any]]] = [
        [
            {
                "name": "drop-observation-dims-ft",
                "args": {"axis_dims": {0: [1, 2, 3, 4, 5]}},
            },
            {
                "name": "tile-observation-action-ft",
                "args": {"tiling_dim": 1, "num_tilings": 1},
            },
        ]
    ]
    gw_dead_ohe = _grid_dead_ohe_indices(MINES_GW_GRID)
    gw_est_feats: Sequence[Sequence[Mapping[str, Any]]] = [
        [
            {"name": "flat-grid-observation-action-ft", "args": {}},
            {
                "name": "drop-observation-dims-ft",
                "args": {"axis_dims": {0: gw_dead_ohe}},
            },
        ]
    ]

    specs: List[Mapping[str, Any]] = [
        {
            "name": "MountainCar-v0",
            "args": {
                "max_episode_steps": 2500,
                "reward_shaping": {
                    "name": "mountain-car-height",
                    "args": {"scale": 1.0},
                },
            },
            "feats_specs": [
                [
                    {
                        "name": "tile-observation-action-ft",
                        "args": {"tiling_dim": mc_tiling_dim},
                    }
                ]
            ],
            "problem_specs": common_problem_specs(include_options=False)
            + least_specs(
                attempt_estimation_episodes=(10,),
                check_factors=True,
                feats_specs=mc_est_feats,
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=mc_est_feats,
            ),
            "epochs": 10,
        },
        {
            "name": "Acrobot-v1",
            "args": {
                "max_episode_steps": 500,
                "reward_shaping": {
                    "name": "action-cost",
                    "args": {"action_costs": [0.0, 0.5, 1.0]},
                },
            },
            "feats_specs": [
                [
                    {
                        "name": "tile-observation-action-ft",
                        "args": {
                            "tiling_dim": acrobot_tiling_dim,
                            "hash_dim": acrobot_hash_dim,
                        },
                    }
                ]
            ],
            "problem_specs": common_problem_specs(include_options=False)
            + least_specs(
                attempt_estimation_episodes=(10,),
                check_factors=True,
                feats_specs=acrobot_est_feats,
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=acrobot_est_feats,
            ),
            "epochs": 10,
        },
        {
            "name": "GridWorld-MINES",
            "args": {
                "grid": MINES_GW_GRID,
                "max_episode_steps": 200,
            },
            "feats_specs": [
                [
                    {
                        "name": "tile-observation-action-ft",
                        "args": {"tiling_dim": gw_tiling_dim},
                    }
                ]
            ],
            "problem_specs": common_problem_specs(include_options=False)
            + least_specs(
                attempt_estimation_episodes=(10,),
                use_next_state=False,
                check_factors=True,
                feats_specs=gw_est_feats,
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=gw_est_feats,
            ),
            "epochs": 5,
        },
    ]
    return tuple(specs)


def load_solvable_grids(
    path: str,
    min_passes: int = 4,
) -> Sequence[Mapping[str, Any]]:
    """
    Loads grid specs from a solvable grids JSON file, filtering by
    minimum number of passed solvability trials.
    """
    with open(path, "r") as readable:
        entries: List[Mapping[str, Any]] = json.load(readable)
    passing_grids = [entry for entry in entries if entry["passes"] >= min_passes]
    return passing_grids


def grid_experiments_specs(
    grid_specs: Sequence[Mapping[str, Any]],
    max_episode_steps: int = 200,
) -> Sequence[Mapping[str, Any]]:
    """
    Control experiment specs from pre-validated grid environments.
    """
    specs: List[Mapping[str, Any]] = []
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
                    use_next_state=False,
                    check_factors=True,
                    feats_specs=[
                        [
                            {
                                "name": "flat-grid-observation-action-ft",
                                "args": {},
                            },
                            {
                                "name": "drop-observation-dims-ft",
                                "args": {"axis_dims": {0: dead_ohe_indices}},
                            },
                        ]
                    ],
                )
                + bayes_least_specs(
                    init_attempt_estimation_episodes=(10,),
                    feats_specs=[
                        [
                            {
                                "name": "flat-grid-observation-action-ft",
                                "args": {},
                            },
                            {
                                "name": "drop-observation-dims-ft",
                                "args": {"axis_dims": {0: dead_ohe_indices}},
                            },
                        ]
                    ],
                ),
                "epochs": 5,
            }
        )
    return tuple(specs)
