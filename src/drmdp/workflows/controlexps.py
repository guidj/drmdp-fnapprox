from typing import Any, Mapping, Sequence


def least_specs(estimation_sample_size: int):
    return (
        {
            "policy_type": "markovian",
            "reward_mapper": {
                "name": "least-lfa",
                "args": {"estimation_sample_size": estimation_sample_size},
            },
            "delay_config": {"name": "fixed", "args": {"delay": 2}},
            "epsilon": 0.2,
            "gamma": 1.0,
            "learning_rate_config": {
                "name": "constant",
                "args": {"initial_lr": 0.01},
            },
        },
        {
            "policy_type": "markovian",
            "reward_mapper": {
                "name": "least-lfa",
                "args": {"estimation_sample_size": estimation_sample_size},
            },
            "delay_config": {"name": "fixed", "args": {"delay": 4}},
            "epsilon": 0.2,
            "gamma": 1.0,
            "learning_rate_config": {
                "name": "constant",
                "args": {"initial_lr": 0.01},
            },
        },
    )


COMMON_PROBLEM_SPECS = (
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": None,
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "zero-impute", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 2}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "zero-impute", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 4}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "zero-impute", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 6}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "markovian",
        "reward_mapper": {"name": "zero-impute", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 8}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 2}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 4}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 6}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 8}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "single-action-options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 2}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "single-action-options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 4}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "single-action-options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 6}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "single-action-options",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 8}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
)

SPECS: Sequence[Mapping[str, Any]] = (
    {
        "name": "Finite-CC-PMSM-v0",
        "args": {
            "pos_enforcement": True,
            "penalty_gamma": 1.0,
            "violation_reward": 0.0,
            "max_episode_steps": 1000,
        },
        "feats_specs": [{"name": "scale", "args": None}],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
    {
        "name": "Finite-TC-PermExDc-v0",
        "args": {
            "pos_enforcement": True,
            "penalty_gamma": 1.0,
            "violation_reward": 0.0,
            "max_episode_steps": 1000,
        },
        "feats_specs": [{"name": "scale", "args": None}],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
    {
        "name": "Finite-CC-SeriesDc-v0",
        "args": {
            "pos_enforcement": True,
            "penalty_gamma": 1.0,
            "violation_reward": 0.0,
            "max_episode_steps": 1000,
        },
        "feats_specs": [{"name": "scale", "args": None}],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
    {
        "name": "Finite-TC-ShuntDc-v0",
        "args": {
            "pos_enforcement": True,
            "penalty_gamma": 1.0,
            "violation_reward": 0.0,
            "max_episode_steps": 1000,
        },
        "feats_specs": [{"name": "scale", "args": None}],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
    {
        "name": "Finite-CC-SCIM-v0",
        "args": {
            "pos_enforcement": True,
            "penalty_gamma": 1.0,
            "violation_reward": 0.0,
            "max_episode_steps": 1000,
        },
        "feats_specs": [{"name": "scale", "args": None}],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
    {
        "name": "MountainCar-v0",
        "args": {
            "max_episode_steps": 2500,
        },
        "feats_specs": [
            {"name": "scale", "args": None},
            {
                "name": "gaussian-mix",
                "args": {"n_components": (384 // 3), "covariance_type": "diag"},
            },
        ],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
    {
        "name": "RedGreen-v0",
        "args": None,
        "feats_specs": [
            {"name": "random", "args": {"enc_size": 32}},
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(20_000),
    },
    {
        "name": "IceWorld-v0",
        "args": None,
        "feats_specs": [
            {"name": "random", "args": {"enc_size": 64}},
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
    {
        "name": "GridWorld-v0",
        "args": None,
        "feats_specs": [
            {"name": "random", "args": {"enc_size": 64}},
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS + least_specs(50_000),
    },
)
