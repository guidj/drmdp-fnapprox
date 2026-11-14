"""
Spec families for the bias-variance campaign.

The campaign decomposes each reward estimator's prediction error into
bias and variance: `experiment_specs` enumerates the environment
configurations (Finite-CC/SC/TC PermExDc/ShuntDc) with the least-lfa and
bayes-least-lfa estimator specifications, and the job replicates each
estimator across seeds on one fixed sample dataset. Launch through
`job.py`.
"""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from drmdp import constants

MAX_STEPS_PER_EPISODE_GEM = 10_000


def least_lfa_specs(
    attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    """Generate specs for least-lfa estimator."""
    specs: list[Mapping[str, Any]] = []
    for attempt_episode, feat_spec in itertools.product(
        attempt_estimation_episodes, feat_specs
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
                    "estimation_buffer_mult": (
                        constants.DEFAULT_ESTIMATION_BUFFER_MULT
                    ),
                },
            }
        )
    return specs


def bayes_least_lfa_specs(
    init_attempt_estimation_episodes: Sequence[int],
    feat_specs: Sequence[Sequence[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    """Generate specs for bayes-least-lfa estimator."""
    specs: list[Mapping[str, Any]] = []
    for init_attempt_episode, feat_spec in itertools.product(
        init_attempt_estimation_episodes, feat_specs
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
                    "estimation_buffer_mult": (
                        constants.DEFAULT_ESTIMATION_BUFFER_MULT
                    ),
                },
            }
        )
    return specs


def experiment_specs() -> Sequence[Mapping[str, Any]]:
    """
    Returns experiment configurations.
    """
    return (
        {
            "name": "Finite-CC-PermExDc-v0",
            "args": {
                "reward_fn": "pos-enf",
                "penalty_gamma": 1.0,
                "constraint_violation_reward": -10,
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
            "methods": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
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
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "methods": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
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
                "constraint_violation_reward": -10,
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
            "methods": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
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
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ],
            ],
            "methods": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
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
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": False,
            },
            "feats_specs": [
                [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
            ],
            "methods": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
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
                "constraint_violation_reward": -10,
                "max_episode_steps": MAX_STEPS_PER_EPISODE_GEM,
                "emit_state": True,
            },
            "feats_specs": [
                [
                    {"name": "scale-observation-ft", "args": None},
                    {"name": "action-segment-observation-ft", "args": None},
                ]
            ],
            "methods": least_lfa_specs(
                attempt_estimation_episodes=(10,),
                feat_specs=[
                    [
                        {"name": "scale-observation-ft", "args": None},
                        {"name": "action-segment-observation-ft", "args": None},
                    ],
                ],
            )
            + bayes_least_lfa_specs(
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
