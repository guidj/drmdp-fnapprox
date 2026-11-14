"""
Spec families for the methods campaign.

The campaign compares methods of handling the delayed aggregate reward
(least squares, Bayesian linear regression, the convex CVLPS variants)
while a policy learns. `experiment_specs` enumerates the environment
configurations (Finite-CC/SC/TC PermExDc/ShuntDc) and, for each, the
feature specifications and the delayed-reward handling method
specifications. Launch through `job.py`.
"""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from drmdp import constants

MAX_STEPS = 10_000


def discrete_least_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
    estimation_buffer_multiples: Sequence[int | None] = (
        constants.DEFAULT_ESTIMATION_BUFFER_MULT,
    ),
):
    """
    Discretised Least Squares specs.
    """
    specs = []
    for attempt_episode, feat_spec, est_buffer_mult in itertools.product(
        attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "discrete-least-lfa",
                "args": {
                    "attempt_estimation_episode": attempt_episode,
                    "transition_feats_spec": {
                        "name": "current-features",
                        "args": {"current_spec": feat_spec},
                    },
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def least_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
    estimation_buffer_multiples: Sequence[int | None] = (
        constants.DEFAULT_ESTIMATION_BUFFER_MULT,
    ),
):
    """
    Least Squares specs.
    """
    specs = []
    for attempt_episode, feat_spec, est_buffer_mult in itertools.product(
        attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "least-lfa",
                "args": {
                    "attempt_estimation_episode": attempt_episode,
                    "transition_feats_spec": {
                        "name": "current-features",
                        "args": {"current_spec": feat_spec},
                    },
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def bayes_least_specs(
    init_attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
    estimation_buffer_multiples: Sequence[int | None] = (
        constants.DEFAULT_ESTIMATION_BUFFER_MULT,
    ),
):
    """
    Bayesian linear regression specs.
    """
    specs = []
    for init_attempt_episode, feat_spec, est_buffer_mult in itertools.product(
        init_attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "bayes-least-lfa",
                "args": {
                    "init_attempt_estimation_episode": init_attempt_episode,
                    "mode": "exponential",
                    "transition_feats_spec": {
                        "name": "current-features",
                        "args": {"current_spec": feat_spec},
                    },
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def cvlps_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
    estimation_buffer_multiples: Sequence[int | None] = (
        constants.DEFAULT_ESTIMATION_BUFFER_MULT,
    ),
    constraints_buffer_limit: int | None = 100,
):
    """
    Constrained optimisation specs.
    """
    specs = []
    for attempt_episode, feat_spec, est_buffer_mult in itertools.product(
        attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "cvlps",
                "args": {
                    "attempt_estimation_episode": attempt_episode,
                    "transition_feats_spec": {
                        "name": "current-features",
                        "args": {"current_spec": feat_spec},
                    },
                    "constraints_buffer_limit": constraints_buffer_limit,
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def recurring_cvlps(
    init_attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
    estimation_buffer_multiples: Sequence[int | None] = (
        constants.DEFAULT_ESTIMATION_BUFFER_MULT,
    ),
    constraints_buffer_limit: int | None = 100,
):
    """
    Recurring convex linear estimation specs.
    """
    specs = []
    for init_attempt_episode, feat_spec, est_buffer_mult in itertools.product(
        init_attempt_estimation_episodes, feat_specs, estimation_buffer_multiples
    ):
        specs.append(
            {
                "name": "recurring-cvlps",
                "args": {
                    "init_attempt_estimation_episode": init_attempt_episode,
                    "mode": "exponential",
                    "transition_feats_spec": {
                        "name": "current-features",
                        "args": {"current_spec": feat_spec},
                    },
                    "constraints_buffer_limit": constraints_buffer_limit,
                    "estimation_buffer_mult": est_buffer_mult,
                },
            }
        )
    return specs


def experiment_specs() -> Sequence[Mapping[str, Any]]:
    """
    Returns experiments configurations.
    """
    return (
        {
            "name": "Finite-CC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10.0,
                "max_episode_steps": MAX_STEPS,
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
            "methods": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
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
                "max_episode_steps": MAX_STEPS,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "methods": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
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
                "max_episode_steps": MAX_STEPS,
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
            "methods": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
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
                "max_episode_steps": MAX_STEPS,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ],
            ],
            "methods": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
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
                "max_episode_steps": MAX_STEPS,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "methods": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
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
                "max_episode_steps": MAX_STEPS,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ]
            ],
            "methods": least_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            ),
            "epochs": 1,
        },
    )
