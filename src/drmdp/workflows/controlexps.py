import itertools
from typing import Any, Mapping, Sequence

EPSILON = 0.2


def least_specs(
    attempt_estimation_episode: int, feats_spec: Mapping[str, Any]
) -> Sequence[Mapping[str, Any]]:
    """
    Least Squares specs.
    """
    specs = []
    for delay, gamma in itertools.product((2, 4, 6), (1.0, 0.99)):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "least-lfa",
                    "args": {
                        "attempt_estimation_episode": attempt_estimation_episode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": delay}},
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
        )
    return tuple(specs)


def bayes_least_specs(
    init_attempt_estimation_episode: int,
    feats_spec: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    """
    Bayesian linear regression specs.
    """
    specs = []
    for delay, gamma in itertools.product((2, 4, 6), (1.0, 0.99)):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "bayes-least-lfa",
                    "args": {
                        "init_attempt_estimation_episode": init_attempt_estimation_episode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": delay}},
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
        )
    return tuple(specs)


def cvlps_specs(
    attempt_estimation_episode: int, feats_spec: Mapping[str, Any]
) -> Sequence[Mapping[str, Any]]:
    """
    Constrained optimisation specs.
    """
    specs = []
    for delay, gamma in itertools.product((2, 4, 6), (1.0, 0.99)):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {
                    "name": "cvlps",
                    "args": {
                        "attempt_estimation_episode": attempt_estimation_episode,
                        "feats_spec": feats_spec,
                        "use_bias": False,
                    },
                },
                "delay_config": {"name": "fixed", "args": {"delay": delay}},
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": {
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            },
        )
    return tuple(specs)


def common_problem_specs():
    """
    Specs that apply to every env.
    """
    specs = []
    lr_config = {
        "name": "constant",
        "args": {"initial_lr": 0.01},
    }
    for gamma in (1.0, 0.99):
        specs.append(
            {
                "policy_type": "markovian",
                "reward_mapper": {"name": "identity", "args": None},
                "delay_config": None,
                "epsilon": EPSILON,
                "gamma": gamma,
                "learning_rate_config": lr_config,
            },
        )

        for delay in (2, 4, 6):
            specs.extend(
                [
                    {
                        "policy_type": "drop-missing",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": {"name": "fixed", "args": {"delay": delay}},
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": lr_config,
                    },
                    {
                        "policy_type": "markovian",
                        "reward_mapper": {"name": "zero-impute", "args": None},
                        "delay_config": {"name": "fixed", "args": {"delay": delay}},
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": lr_config,
                    },
                    {
                        "policy_type": "options",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": {"name": "fixed", "args": {"delay": 2}},
                        "epsilon": EPSILON,
                        "gamma": 1.0,
                        "learning_rate_config": lr_config,
                    },
                    {
                        "policy_type": "single-action-options",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": {"name": "fixed", "args": {"delay": delay}},
                        "epsilon": EPSILON,
                        "gamma": gamma,
                        "learning_rate_config": lr_config,
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
                "max_episode_steps": 200,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 4}}],
            "problem_specs": common_problem_specs()
            + least_specs(2000, feats_spec={"name": "scale", "args": None})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(2000, feats_spec={"name": "scale", "args": None}),
            "epochs": 1,
        },
        {
            "name": "Finite-CC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": 200,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
            "problem_specs": common_problem_specs()
            + least_specs(2000, {"name": "scale", "args": None})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(2000, {"name": "scale", "args": None}),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": 200,
            },
            "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 3}}],
            "problem_specs": common_problem_specs()
            + least_specs(2000, {"name": "scale", "args": None})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(2000, {"name": "scale", "args": None}),
            "epochs": 1,
        },
        {
            "name": "Finite-SC-ShuntDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": 200,
            },
            "feats_specs": [{"name": "scale", "args": None}],
            "problem_specs": common_problem_specs()
            + least_specs(2000, {"name": "scale", "args": None})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(2000, {"name": "scale", "args": None}),
            "epochs": 1,
        },
        {
            "name": "Finite-TC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": 0.0,
                "max_episode_steps": 200,
            },
            "feats_specs": [{"name": "tiles", "args": {"tiling_dim": 3}}],
            "problem_specs": common_problem_specs()
            + least_specs(2000, {"name": "scale", "args": None})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "scale", "args": None},
            )
            + cvlps_specs(2000, {"name": "scale", "args": None}),
            "epochs": 1,
        },
        {
            "name": "RedGreen-v0",
            "args": None,
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 6}},
            ],
            "problem_specs": common_problem_specs()
            + least_specs(1000, {"name": "tiles", "args": {"tiling_dim": 6}})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + cvlps_specs(1000, {"name": "tiles", "args": {"tiling_dim": 6}}),
            "epochs": 100,
        },
        {
            "name": "IceWorld-v0",
            "args": None,
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 6}},
            ],
            "problem_specs": common_problem_specs()
            + least_specs(1000, {"name": "tiles", "args": {"tiling_dim": 6}})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + cvlps_specs(1000, {"name": "tiles", "args": {"tiling_dim": 6}}),
            "epochs": 100,
        },
        {
            "name": "MountainCar-v0",
            "args": {
                "max_episode_steps": 2500,
            },
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 6}},
            ],
            "problem_specs": common_problem_specs()
            + least_specs(5000, {"name": "tiles", "args": {"tiling_dim": 6}})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "tiles", "args": {"tiling_dim": 6}},
            )
            + cvlps_specs(5000, {"name": "tiles", "args": {"tiling_dim": 6}}),
            "epochs": 10,
        },
        {
            "name": "GridWorld-v0",
            "args": {"max_episode_steps": 200},
            "feats_specs": [
                {"name": "tiles", "args": {"tiling_dim": 8}},
            ],
            "problem_specs": common_problem_specs()
            + least_specs(5000, {"name": "tiles", "args": {"tiling_dim": 8}})
            + bayes_least_specs(
                init_attempt_estimation_episode=10,
                feats_spec={"name": "tiles", "args": {"tiling_dim": 8}},
            )
            + cvlps_specs(5000, {"name": "tiles", "args": {"tiling_dim": 8}}),
            "epochs": 10,
        },
    ]
    return tuple(specs)
