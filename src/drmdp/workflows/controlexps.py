from typing import Any, Mapping, Sequence


def least_specs(estimation_sample_size: int, feats_spec: Mapping[str, Any]):
    return (
        {
            "policy_type": "markovian",
            "reward_mapper": {
                "name": "least-lfa",
                "args": {
                    "estimation_sample_size": estimation_sample_size,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
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
                "args": {
                    "estimation_sample_size": estimation_sample_size,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
            },
            "delay_config": {"name": "fixed", "args": {"delay": 4}},
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
                "args": {
                    "estimation_sample_size": estimation_sample_size,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
            },
            "delay_config": {"name": "fixed", "args": {"delay": 6}},
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
                "args": {
                    "estimation_sample_size": estimation_sample_size,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
            },
            "delay_config": {"name": "fixed", "args": {"delay": 8}},
            "epsilon": 0.2,
            "gamma": 1.0,
            "learning_rate_config": {
                "name": "constant",
                "args": {"initial_lr": 0.01},
            },
        },
    )


def bayes_least_specs(feats_spec: Mapping[str, Any]):
    return (
        {
            "policy_type": "markovian",
            "reward_mapper": {
                "name": "least-bayes-lfa",
                "args": {
                    "init_update_episodes": 10,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
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
                "name": "least-bayes-lfa",
                "args": {
                    "init_update_episodes": 10,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
            },
            "delay_config": {"name": "fixed", "args": {"delay": 4}},
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
                "name": "least-bayes-lfa",
                "args": {
                    "init_update_episodes": 10,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
            },
            "delay_config": {"name": "fixed", "args": {"delay": 6}},
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
                "name": "least-bayes-lfa",
                "args": {
                    "init_update_episodes": 10,
                    "feats_spec": feats_spec,
                    "use_bias": False,
                },
            },
            "delay_config": {"name": "fixed", "args": {"delay": 8}},
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
        "policy_type": "drop-missing",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 2}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "drop-missing",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 4}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "drop-missing",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 6}},
        "epsilon": 0.2,
        "gamma": 1.0,
        "learning_rate_config": {"name": "constant", "args": {"initial_lr": 0.01}},
    },
    {
        "policy_type": "drop-missing",
        "reward_mapper": {"name": "identity", "args": None},
        "delay_config": {"name": "fixed", "args": {"delay": 8}},
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
        "name": "Finite-CC-PermExDc-v0",
        "args": {
            "reward_fn": "pos-enf",
            "penalty_gamma": 1.0,
            "constraint_violation_reward": 0.0,
            "max_episode_steps": 200,
        },
        "feats_specs": [{"name": "spliced-tiles", "args": {"tiling_dim": 4}}],
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(10_000, feats_spec={"name": "scale", "args": None})
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
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
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(10_000, {"name": "scale", "args": None})
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
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
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(10_000, {"name": "scale", "args": None})
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
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
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(10_000, {"name": "scale", "args": None})
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
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
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(10_000, {"name": "scale", "args": None})
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
        "epochs": 1,
    },
    {
        "name": "Finite-TC-ShuntDc-v0",
        "args": {
            "reward_fn": "pos-enf",
            "penalty_gamma": 1.0,
            "constraint_violation_reward": 0.0,
            "max_episode_steps": 200,
        },
        "feats_specs": [{"name": "scale", "args": None}],
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(10_000, {"name": "scale", "args": None})
        + bayes_least_specs(feats_spec={"name": "scale", "args": None}),
        "epochs": 1,
    },
    {
        "name": "RedGreen-v0",
        "args": None,
        "feats_specs": [
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(1000, {"name": "tiles", "args": {"tiling_dim": 6}})
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 6}}),
        "epochs": 100,
    },
    {
        "name": "IceWorld-v0",
        "args": None,
        "feats_specs": [
            {"name": "tiles", "args": {"tiling_dim": 6}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(1000, {"name": "tiles", "args": {"tiling_dim": 6}})
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 6}}),
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
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(5000, {"name": "tiles", "args": {"tiling_dim": 6}})
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 6}}),
        "epochs": 10,
    },
    {
        "name": "GridWorld-v0",
        "args": {"max_episode_steps": 200},
        "feats_specs": [
            {"name": "tiles", "args": {"tiling_dim": 8}},
        ],
        "problem_specs": COMMON_PROBLEM_SPECS
        + least_specs(5000, {"name": "tiles", "args": {"tiling_dim": 8}})
        + bayes_least_specs(feats_spec={"name": "tiles", "args": {"tiling_dim": 8}}),
        "epochs": 10,
    },
)
