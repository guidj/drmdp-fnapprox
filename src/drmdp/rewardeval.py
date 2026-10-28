"""
Measures the error of estimated per-step rewards against true rewards.

Discrete environments — those exposing a `transition` table and a
`states_mapping` (GridWorld) — are enumerated: every state and action
is scored, with the true reward taken from the transition entry's
probability-weighted reward (the expected reward; deterministic tables
have a single probability-1 entry). Continuous environments are sampled
with a random policy on the clean proxy env: `num_samples` steps per
call, resetting at episode ends.

Feature computation reuses the estimator's own transition-feature
function, so the error is measured in the representation the estimator
fits and no second feature implementation can drift from it.

True rewards come from the proxy env: shaped and un-noised, matching
the target the estimator infers from the delayed, noisy training
rewards.
"""

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from drmdp import metrics, rewardfeat

DEFAULT_NUM_SAMPLES = 10_000


def true_reward_error_fn(
    proxy_env: gym.Env,
    feats_fn: rewardfeat.TransitionFeatureFn,
    intercept: bool = False,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int | None = None,
) -> Callable[[np.ndarray], Mapping[str, Any]]:
    """
    Returns a callable measuring reward-estimate error against true rewards.

    The callable takes the estimator's weights and returns
    `{"rmse": float, "num_samples": int}`. Discrete envs score every
    state-action pair, with the feature matrix computed once; continuous
    envs draw `num_samples` fresh random-policy samples per call. With
    `seed`, every call draws the same samples, so successive estimates
    are scored on a fixed evaluation set.

    Args:
        proxy_env: clean env (un-noised rewards) used as the source of
            true rewards and of random-policy samples.
        feats_fn: the estimator's transition-feature function.
        intercept: append the intercept column the estimator fits.
        num_samples: random-policy steps per call on continuous envs.
        seed: seeds the sampler's action draws and first reset, for
            reproducible measurements; unseeded by default.
    """
    is_discrete = supports_discrete_enumeration(proxy_env)
    # the enumeration is a property of the env; compute it once
    discrete_cache: tuple[tuple[np.ndarray, np.ndarray], ...] = ()

    def measure(weights: np.ndarray) -> Mapping[str, Any]:
        nonlocal discrete_cache
        if is_discrete:
            if not discrete_cache:
                discrete_cache = (
                    discrete_state_action_rewards(
                        proxy_env=proxy_env, feats_fn=feats_fn, intercept=intercept
                    ),
                )
            features, true_rewards = discrete_cache[0]
        else:
            features, true_rewards = random_policy_rewards(
                proxy_env=proxy_env,
                feats_fn=feats_fn,
                intercept=intercept,
                num_samples=num_samples,
                seed=seed,
            )
        predicted = np.dot(features, weights)
        return {
            "rmse": metrics.rmse(v_pred=predicted, v_true=true_rewards, axis=0),
            "num_samples": len(true_rewards),
        }

    return measure


def discrete_state_action_rewards(
    proxy_env: gym.Env,
    feats_fn: rewardfeat.TransitionFeatureFn,
    intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Features and true rewards for every state-action pair of a discrete env.

    Rows are ordered state-major, action-minor: the row for state id
    `s` and action `a` sits at index `s * num_actions + a`, matching the
    `states_mapping` ids. Rewards and next states are read from the
    env's transition table, which is start-independent — cliff falls
    target the start given at construction — so one table serves every
    episode.

    The transition table carries no `info`, so a feature that reads an
    env event from `info` cannot be scored through enumeration; such a
    feature raises at evaluation time rather than being scored zero.
    """
    transition = proxy_env.get_wrapper_attr("transition")
    states_mapping: Mapping[tuple[int, int], int] = proxy_env.get_wrapper_attr(
        "states_mapping"
    )
    position_by_id = {
        state_id: position for position, state_id in states_mapping.items()
    }
    action_space: gym.spaces.Discrete = proxy_env.action_space  # type: ignore[assignment]
    num_actions = int(action_space.n)
    features: list[np.ndarray] = []
    true_rewards: list[float] = []
    for state_id in sorted(transition):
        obs = np.array(position_by_id[state_id], dtype=np.int64)
        for action in range(num_actions):
            entries = transition[state_id][action]
            _, next_state_id, _, terminated = max(entries, key=lambda entry: entry[0])
            next_obs = np.array(position_by_id[next_state_id], dtype=np.int64)
            features.append(
                step_features(
                    feats_fn=feats_fn,
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    terminated=bool(terminated),
                    intercept=intercept,
                )
            )
            # expected reward over the transition's outcomes;
            # a deterministic table has a single probability-1 entry
            true_rewards.append(
                float(sum(prob * step_reward for prob, _, step_reward, _ in entries))
            )
    return np.array(features), np.array(true_rewards)


def random_policy_rewards(
    proxy_env: gym.Env,
    feats_fn: rewardfeat.TransitionFeatureFn,
    intercept: bool = False,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Features and true rewards from random-policy steps on the proxy env.

    Resets at episode ends; with `seed`, the action draws and the first
    reset are seeded, making the sample reproducible.
    """
    if seed is not None:
        proxy_env.action_space.seed(seed)
        obs, _ = proxy_env.reset(seed=seed)
    else:
        obs, _ = proxy_env.reset()
    features: list[np.ndarray] = []
    true_rewards: list[float] = []
    for _ in range(num_samples):
        action = proxy_env.action_space.sample()
        next_obs, reward, term, trunc, info = proxy_env.step(action)
        features.append(
            step_features(
                feats_fn=feats_fn,
                obs=obs,
                action=action,
                next_obs=next_obs,
                terminated=term,
                truncated=trunc,
                info=info,
                intercept=intercept,
            )
        )
        true_rewards.append(float(reward))
        obs = next_obs
        if term or trunc:
            obs, _ = proxy_env.reset()
    return np.array(features), np.array(true_rewards)


def step_features(
    feats_fn: rewardfeat.TransitionFeatureFn,
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    terminated: bool = False,
    truncated: bool = False,
    info: Mapping[str, Any] | None = None,
    intercept: bool = False,
) -> np.ndarray:
    """
    Estimator features for one step, computed by the estimator's own hook.

    `intercept` appends the intercept column the estimator fits, so the
    scoring basis matches the fitted one.
    """
    transition = rewardfeat.Transition(
        observation=obs,
        action=action,
        next_observation=next_obs,
        terminated=terminated,
        truncated=truncated,
        info=info if info is not None else {},
    )
    features = np.asarray(feats_fn(transition), dtype=np.float64)
    if intercept:
        features = np.concatenate([features, np.array([1.0])])
    return features


def supports_discrete_enumeration(env: gym.Env) -> bool:
    """
    Whether the env exposes the transition table and state mapping that
    discrete enumeration requires.
    """
    return bool(
        env.has_wrapper_attr("transition") and env.has_wrapper_attr("states_mapping")
    )
