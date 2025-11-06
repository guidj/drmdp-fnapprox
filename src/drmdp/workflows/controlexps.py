import itertools
from typing import Any, Mapping, Sequence

from drmdp import mathutils

EPSILON = 0.2
MAX_STEPS_PER_EPISODE_GEM = 200
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


def discrete_least_specs(
    attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Mapping[str, Any]],
    delays: Sequence[int] = (2, 4, 6),
    discounts: Sequence[float] = (1.0, 0.99),
):
    """
    Discretised Least Squares specs.
    """
    specs = []
    for delay, gamma, feats_spec, attempt_estimation_episode in itertools.product(
        delays, discounts, feats_specs, attempt_estimation_episodes
    ):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "discrete-least-lfa",
                    "args": {
                        "attempt_estimation_episode": attempt_estimation_episode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
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


def least_specs(
    attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Mapping[str, Any]],
    delays: Sequence[int] = (2, 4, 6),
    discounts: Sequence[float] = (1.0, 0.99),
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


def bayes_least_specs(
    init_attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Mapping[str, Any]],
    delays: Sequence[int] = (2, 4, 6),
    discounts: Sequence[float] = (1.0, 0.99),
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


def cvlps_specs(
    attempt_estimation_episodes: Sequence[int],
    feats_specs: Sequence[Mapping[str, Any]],
    delays: Sequence[int] = (2, 4, 6),
    discounts: Sequence[float] = (1.0, 0.99),
) -> Sequence[Mapping[str, Any]]:
    """
    Constrained optimisation specs.
    """
    specs = []
    for delay, gamma, feats_spec, attempt_estimation_episode in itertools.product(
        delays, discounts, feats_specs, attempt_estimation_episodes
    ):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "cvlps",
                    "args": {
                        "attempt_estimation_episode": attempt_estimation_episode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
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
    delays: Sequence[int] = (2, 4, 6), discounts: Sequence[float] = (1.0, 0.99)
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
                        "reward_mapper": {"name": "zero-impute", "args": None},
                        "delay_config": poisson_delay_config(delay),
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": LEARNING_RATE_SPEC,
                    },
                    {
                        "policy_type": "options",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": {
                            "name": "clipped-poisson",
                            "args": {"delay": 2},
                        },
                        "epsilon": EPSILON,
                        "gamma": 1.0,
                        "learning_rate_config": LEARNING_RATE_SPEC,
                    },
                    {
                        "policy_type": "single-action-options",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": poisson_delay_config(delay),
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": LEARNING_RATE_SPEC,
                    },
                ]
            )
    return tuple(specs)


def experiment_specs() -> Sequence[Mapping[str, Any]]:
    """
    Control experiment specs.
    """
    specs = [
        {
            "name": "Finite-CC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 4}}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "cluster-c", "args": {"num_clusters": 100}}],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "scale", "args": None}],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "scale", "args": None}],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[{"name": "scale", "args": None}],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-CC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "cluster-c", "args": {"num_clusters": 100}}],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 3}}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "cluster-c", "args": {"num_clusters": 100}}],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "cluster-c", "args": {"num_clusters": 100}}],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "flat-grid-coord", "args": None}],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                    {"name": "tiles", "args": {"tiling_dim": 6}},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "IceWorld-v0",
            "args": {"map_name": "8x8"},
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 6}},
            ],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "flat-grid-coord", "args": None}],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "tiles", "args": {"tiling_dim": 6}}],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "tiles", "args": {"tiling_dim": 6}}],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[{"name": "tiles", "args": {"tiling_dim": 6}}],
            ),
            "epochs": 5,
        },
        {
            "name": "GridWorld-v0",
            "args": {"grid": MINES_GW_GRID, "max_episode_steps": 200},
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 8}},
            ],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "flat-grid-coord", "args": None}],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "tiles", "args": {"tiling_dim": 8}}],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "tiles", "args": {"tiling_dim": 8}}],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[{"name": "tiles", "args": {"tiling_dim": 8}}],
            ),
            "epochs": 5,
        },
        {
            "name": "Finite-CC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "cluster-c", "args": {"num_clusters": 100}},
                ],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "scale", "args": None}],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[{"name": "scale", "args": None}],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[{"name": "scale", "args": None}],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-CC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "cluster-c", "args": {"num_clusters": 100}},
                ],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "cluster-c", "args": {"num_clusters": 100}},
                ],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "cluster-c", "args": {"num_clusters": 100}},
                ],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": common_problem_specs()
            + discrete_least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "cluster-c", "args": {"num_clusters": 100}},
                ],
            )
            + least_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + cvlps_specs(
                attempt_estimation_episodes=(50,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feats_specs=[
                    {"name": "scale", "args": None},
                ],
            ),
            "epochs": 1,
        },
    ]
    return tuple(specs)


def poisson_delay_config(lam: int):
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
    return {"name": "clipped-poisson", "args": {"lam": lam, "min_delay": max(2, lb)}}
