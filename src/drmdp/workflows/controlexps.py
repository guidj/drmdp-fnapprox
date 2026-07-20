import itertools
from typing import Any, List, Mapping, Optional, Sequence, Tuple

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
DEFAULT_IMPUTE_VALUE = 0


def least_specs(
    attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Sequence[Mapping[str, Any]]],
    delays: Sequence[int] = (2, 4, 6, 8),
    discounts: Sequence[float] = (1.0, 0.99),
    use_next_state: bool = True,
    check_factors: bool = False,
    impute_value: float = DEFAULT_IMPUTE_VALUE,
) -> Sequence[Mapping[str, Any]]:
    """
    Least Squares specs.
    """
    specs = []
    for delay, gamma, feats_spec, attempt_estimation_episode in itertools.product(
        delays, discounts, feats_specs, attempt_estimation_episodes
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
                "delay_config": poisson_delay_config(delay),
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": LEARNING_RATE_SPEC,
            },
        )
    return tuple(specs)


def bayes_least_specs(
    init_attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Sequence[Mapping[str, Any]]],
    delays: Sequence[int] = (2, 4, 6, 8),
    discounts: Sequence[float] = (1.0, 0.99),
    impute_value: float = DEFAULT_IMPUTE_VALUE,
) -> Sequence[Mapping[str, Any]]:
    """
    Bayesian linear regression specs.
    """
    specs = []
    for delay, gamma, feats_spec, init_attempt_estimation_episode in itertools.product(
        delays, discounts, feats_specs, init_attempt_estimation_episodes
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
                "delay_config": poisson_delay_config(delay),
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": LEARNING_RATE_SPEC,
            },
        )
    return tuple(specs)


def common_problem_specs(
    delays: Sequence[int] = (2, 4, 6, 8),
    discounts: Sequence[float] = (1.0, 0.99),
    impute_value: float = DEFAULT_IMPUTE_VALUE,
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

        for delay in delays:
            specs.extend(
                [
                    {
                        "policy_type": "drop-missing",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": poisson_delay_config(delay),
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
                        "delay_config": poisson_delay_config(delay),
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": LEARNING_RATE_SPEC,
                    },
                ]
            )

            if delay <= MAX_OPTIONS_DELAY:
                specs.extend(
                    [
                        # These configs are memory intensive
                        # even with moderate delays.
                        # Limit upper bound
                        {
                            "policy_type": "options",
                            "reward_mapper": {"name": "identity", "args": None},
                            "delay_config": poisson_delay_config(
                                delay, max_delay=delay
                            ),
                            "epsilon": EPSILON,
                            "gamma": gamma,
                            "learning_rate_config": LEARNING_RATE_SPEC,
                        },
                        {
                            "policy_type": "single-action-options",
                            "reward_mapper": {"name": "identity", "args": None},
                            "delay_config": poisson_delay_config(
                                delay, max_delay=delay
                            ),
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


def grid_experiments_specs(
    dimensions: Sequence[Tuple[int, int]] = ((25, 25),),
    num_grids: int = 6,
    cliff_ratio: float = 0.25,
    min_distance: int = 3,
) -> Sequence[Mapping[str, Any]]:
    """
    Control experiment specs for generated grid environments.
    """
    specs: List[Mapping[str, Any]] = []
    for nrows, ncols in dimensions:
        num_cliffs = int(cliff_ratio * nrows * ncols)
        seed = 0
        for _ in range(num_grids):
            while True:
                grid, start, end = gridutils.create_grid(
                    size=(nrows, ncols), num_cliffs=num_cliffs, seed=seed
                )
                distance = gridutils.grid_bfs(grid, source=start, target=end)
                if distance >= min_distance:
                    break
                seed += 1
            gid = gridutils.grid_id(size=(nrows, ncols), seed=seed)
            specs.append(
                {
                    "name": f"grid-{gid}",
                    "args": {
                        "grid": gridutils.grid_to_strings(grid),
                        "max_episode_steps": 200,
                    },
                    "metadata": {
                        "size": [nrows, ncols],
                        "seed": seed,
                        "distance": distance,
                        "num_cliffs": num_cliffs,
                    },
                    "feats_specs": [
                        [
                            {
                                "name": "tile-observation-action-ft",
                                "args": {"tiling_dim": 7},
                            }
                        ]
                    ],
                    "problem_specs": common_problem_specs(impute_value=1)
                    + least_specs(
                        attempt_estimation_episodes=(10,),
                        use_next_state=False,
                        check_factors=True,
                        feats_specs=[
                            [
                                {
                                    "name": "flat-grid-observation-action-ft",
                                    "args": {},
                                }
                            ]
                        ],
                    )
                    + bayes_least_specs(
                        init_attempt_estimation_episodes=(10,),
                        feats_specs=[
                            [
                                {
                                    "name": "tile-observation-action-ft",
                                    "args": {"tiling_dim": 7},
                                }
                            ]
                        ],
                    ),
                    "epochs": 5,
                }
            )
            seed += 1
    return tuple(specs)


def poisson_delay_config(lam: int, max_delay: Optional[int] = None):
    """
    Natural Poisson bounds:
    low, lambda, high
    0 2 5
    0 3 7
    1 4 8
    1 5 10
    2 6 11
    2 7 13
    3 8 14
    """
    lb, _ = mathutils.poisson_exact_confidence_interval(lam)
    return {
        "name": "clipped-poisson",
        "args": {"lam": lam, "min_delay": max(2, lb), "max_delay": max_delay},
    }
