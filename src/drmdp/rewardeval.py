"""Measures estimated per-step rewards against rewards from the clean proxy.

Environments that explicitly provide a deterministic transition model are
enumerated over nonterminal state-action pairs. A TimeLimit wrapper is
permitted: the table metric describes the underlying MDP, not the time-limit
horizon. Other wrappers and environments are sampled with a random policy
on the clean proxy, retaining each realized successor, termination flag,
and reward from the same step. Sampling resets at episode ends.

Feature computation reuses the estimator's transition-feature function.
Proxy rewards include shaping but exclude training reward noise.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from drmdp import metrics, rewardfeat
from drmdp.envs import models

DEFAULT_NUM_SAMPLES = 10_000


def true_reward_error_fn(
    proxy_env: gym.Env,
    feats_fn: rewardfeat.TransitionFeatureFn,
    intercept: bool = False,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int | None = None,
) -> Callable[[np.ndarray], Mapping[str, Any]]:
    """Returns a callable scoring estimator weights against proxy rewards.

    Declared deterministic models score every nonterminal state-action pair,
    caching the feature matrix. Other environments draw `num_samples`
    random-policy steps per call. With a seed, calls draw the same samples
    so differences between measurements reflect the weights.
    """
    enumerate_model = supports_discrete_enumeration(proxy_env)
    discrete_cache: tuple[np.ndarray, np.ndarray] | None = None

    def measure(weights: np.ndarray) -> Mapping[str, Any]:
        nonlocal discrete_cache
        if enumerate_model:
            if discrete_cache is None:
                discrete_cache = discrete_state_action_rewards(
                    proxy_env=proxy_env, feats_fn=feats_fn, intercept=intercept
                )
            features, true_rewards = discrete_cache
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
    """Enumerates deterministic, nonterminal state-action pairs.

    Rows are state-major in ascending state ID order, then action-minor.
    Terminal states come from environment metadata, not self-loop inference.
    The table contains no info or time-limit truncation flags. Features that
    require an info event raise rather than assigning a missing event zero.

    Wrappers that may change rewards, actions, or observations must use
    sampling unless their model is supported explicitly.
    """
    model = _discrete_model(proxy_env)
    if model is None:
        raise ValueError("Enumeration requires a declared transition model")
    features: list[np.ndarray] = []
    true_rewards: list[float] = []
    for state_id in sorted(model.transition):
        if state_id in model.terminal_state_ids:
            continue
        obs = model.observations[state_id]
        for action, entries in sorted(model.transition[state_id].items()):
            _, next_state_id, reward, terminated = _deterministic_outcome(entries)
            next_obs = model.observations[next_state_id]
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
            true_rewards.append(float(reward))
    if not features:
        raise ValueError("Transition model has no nonterminal state-action pairs")
    return np.stack(features), np.array(true_rewards)


def random_policy_rewards(
    proxy_env: gym.Env,
    feats_fn: rewardfeat.TransitionFeatureFn,
    intercept: bool = False,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples features and rewards together, resetting at episode ends."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
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
    return np.stack(features), np.array(true_rewards)


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
    """Computes the estimator's features, including its intercept if enabled."""
    transition = rewardfeat.Transition(
        observation=obs,
        action=action,
        next_observation=next_obs,
        terminated=terminated,
        truncated=truncated,
        info=info if info is not None else {},
    )
    features = np.asarray(feats_fn(transition), dtype=np.float64)
    if features.shape != (feats_fn.feature_dim,):
        raise ValueError(f"Reward features returned shape {features.shape}")
    if intercept:
        features = np.concatenate([features, np.array([1.0])])
    return features


def supports_discrete_enumeration(env: gym.Env) -> bool:
    """Whether the proxy explicitly provides a deterministic transition model."""
    model = _discrete_model(env)
    if model is None:
        return False
    try:
        for actions in model.transition.values():
            for entries in actions.values():
                _deterministic_outcome(entries)
    except ValueError:
        return False
    return True


def _discrete_model(env: gym.Env) -> models.DiscreteTransitionModel | None:
    # A time limit changes the horizon, not the underlying MDP transitions.
    while (
        isinstance(env, gym.wrappers.TimeLimit) and type(env) is gym.wrappers.TimeLimit
    ):
        env = env.env
    # Forwarded or inherited providers may describe different runtime semantics.
    if "discrete_transition_model" not in vars(type(env)):
        return None
    if isinstance(env, models.DiscreteTransitionModelProvider):
        return env.discrete_transition_model()
    return None


def _deterministic_outcome(
    entries: Sequence[tuple[float, int, float, bool]],
) -> tuple[float, int, float, bool]:
    realized = [entry for entry in entries if entry[0] != 0.0]
    if len(realized) != 1 or realized[0][0] != 1.0:
        raise ValueError("Enumeration requires deterministic transition outcomes")
    return realized[0]
