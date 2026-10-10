"""
Measures the error of estimated per-step rewards against true rewards.

Discrete environments — those exposing a `transition` table and a
`states_mapping` (GridWorld) — are enumerated: every state and action
is scored, with the true reward taken from the transition entry's
probability-weighted reward (the expected reward; deterministic tables
have a single probability-1 entry). Continuous environments are sampled
with a random policy on the clean proxy env: `num_samples` steps per
call, resetting at episode ends.

Feature computation mirrors the generative reward wrappers, so the
error is measured in the estimator's own representation: `ft_op` on
(obs, action), next-state concatenation when the estimator uses it,
and a bias column when the estimator fits one.

True rewards come from the proxy env: shaped and un-noised, matching
the target the estimator infers from the delayed, noisy training
rewards. For a shaped discrete env the transition-table rewards would
need shaping applied; no such env is in the experiment families.
"""

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from drmdp import metrics, transform

DEFAULT_NUM_SAMPLES = 10_000


def true_reward_error_fn(
    proxy_env: gym.Env,
    ft_op: transform.FTOp,
    use_bias: bool = False,
    use_next_state: bool = False,
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
        ft_op: the estimator's feature operator.
        use_bias: append a bias column, as the estimator does.
        use_next_state: concatenate next-state features, as the estimator
            does.
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
                        proxy_env=proxy_env,
                        ft_op=ft_op,
                        use_bias=use_bias,
                        use_next_state=use_next_state,
                    ),
                )
            features, true_rewards = discrete_cache[0]
        else:
            features, true_rewards = random_policy_rewards(
                proxy_env=proxy_env,
                ft_op=ft_op,
                use_bias=use_bias,
                use_next_state=use_next_state,
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
    ft_op: transform.FTOp,
    use_bias: bool = False,
    use_next_state: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Features and true rewards for every state-action pair of a discrete env.

    Rows are ordered state-major, action-minor: the row for state id
    `s` and action `a` sits at index `s * num_actions + a`, matching the
    `states_mapping` ids. Rewards and next states are read from the
    env's transition table, which is start-independent — cliff falls
    target the start given at construction — so one table serves every
    episode.
    """
    transition = proxy_env.get_wrapper_attr("transition")
    states_mapping: Mapping[tuple[int, int], int] = proxy_env.get_wrapper_attr(
        "states_mapping"
    )
    position_by_id = {
        state_id: position for position, state_id in states_mapping.items()
    }
    num_actions = proxy_env.action_space.n
    features: list[np.ndarray] = []
    true_rewards: list[float] = []
    for state_id in sorted(transition):
        obs = np.array(position_by_id[state_id], dtype=np.int64)
        for action in range(num_actions):
            entries = transition[state_id][action]
            _, next_state_id, _, _ = max(entries, key=lambda entry: entry[0])
            next_obs = np.array(position_by_id[next_state_id], dtype=np.int64)
            features.append(
                step_features(
                    ft_op=ft_op,
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    use_bias=use_bias,
                    use_next_state=use_next_state,
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
    ft_op: transform.FTOp,
    use_bias: bool = False,
    use_next_state: bool = False,
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
        next_obs, reward, term, trunc, _ = proxy_env.step(action)
        features.append(
            step_features(
                ft_op=ft_op,
                obs=obs,
                action=action,
                next_obs=next_obs,
                use_bias=use_bias,
                use_next_state=use_next_state,
            )
        )
        true_rewards.append(float(reward))
        obs = next_obs
        if term or trunc:
            obs, _ = proxy_env.reset()
    return np.array(features), np.array(true_rewards)


def step_features(
    ft_op: transform.FTOp,
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    use_bias: bool = False,
    use_next_state: bool = False,
) -> np.ndarray:
    """
    Estimator features for one step, mirroring the reward wrappers.

    `use_next_state` concatenates next-state features (computed with
    action 0, as the wrappers do); `use_bias` appends the constant
    column the estimator fits.
    """
    step_feats = np.asarray(
        ft_op(transform.Example(obs, action)).observation, dtype=np.float64
    )
    if use_next_state:
        next_feats = np.asarray(
            ft_op(transform.Example(next_obs, 0)).observation, dtype=np.float64
        )
        step_feats = np.concatenate([step_feats, next_feats])
    if use_bias:
        step_feats = np.concatenate([step_feats, np.array([1.0])])
    return step_feats


def supports_discrete_enumeration(env: gym.Env) -> bool:
    """
    Whether the env exposes the transition table and state mapping that
    discrete enumeration requires.
    """
    return bool(
        env.has_wrapper_attr("transition") and env.has_wrapper_attr("states_mapping")
    )
