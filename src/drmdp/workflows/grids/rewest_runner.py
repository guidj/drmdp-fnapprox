import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import joblib
import numpy as np
import ray
import tqdm.auto

from drmdp import envs, metrics, rewdelay, task, transform
from drmdp.workflows import controlexps

REWARD_EVAL_SAMPLES = 25_000


class RewardStoreWrapper(gym.Wrapper):
    """Captures per-step rewards and observations into buffers."""

    def __init__(self, env: gym.Env, buffer_size: int):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.buffer: List[float] = []
        self.obs_buffer: List[np.ndarray] = []
        self.action_buffer: List[int] = []
        self.solver_state: Dict[str, Any] = {"solution_found_step": None}
        self.steps_counter = 0
        self._prev_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = super().step(action)
        if len(self.buffer) < self.buffer_size:
            assert self._prev_obs is not None
            self.buffer.append(reward)
            self.obs_buffer.append(self._prev_obs)
            self.action_buffer.append(action)
        self._prev_obs = obs
        if (
            self.solver_state["solution_found_step"] is None
            and "estimator" in info
            and info["estimator"]["state"] == rewdelay.OptState.SOLVED
        ):
            self.solver_state["solution_found_step"] = self.steps_counter
        self.steps_counter += 1
        return obs, reward, term, trunc, info


def run_reward_estimation(
    grid_spec: Mapping[str, Any],
    method: str,
    method_args: Mapping[str, Any],
    delay_config: Mapping[str, Any],
    gamma: float,
    num_episodes: int,
    seed: int,
    epsilon: float = controlexps.EPSILON,
    mean_return_window: Optional[int] = None,
) -> Mapping[str, Any]:
    """
    Run a single reward estimation experiment.

    Returns per-step true rewards, predicted rewards,
    solver state, and episode returns.
    """
    env_name = grid_spec["env_name"]
    env_args = {
        "grid": grid_spec["grid"],
        "max_episode_steps": grid_spec.get("max_episode_steps", 200),
    }
    nrows, ncols = grid_spec["size"]
    tiling_dim = math.ceil(max(nrows, ncols) / 2)
    dead_ohe_indices = grid_spec["dead_ohe"]

    mdim = nrows * ncols * 4 - len(dead_ohe_indices)

    control_feats_spec = [
        {"name": "tile-observation-action-ft", "args": {"tiling_dim": tiling_dim}}
    ]

    mapping_spec: Dict[str, Any]
    if method == "identity":
        mapping_spec = {"name": method, "args": None}
    elif method == "impute-missing":
        mapping_spec = {"name": method, "args": {**method_args}}
    else:
        rewest_feats_spec = [
            {"name": "flat-grid-observation-action-ft", "args": {}},
            {
                "name": "drop-observation-dims-ft",
                "args": {"axis_dims": {0: dead_ohe_indices}},
            },
        ]
        mapping_spec = {
            "name": method,
            "args": {**method_args, "feats_spec": rewest_feats_spec},
        }

    env = envs.make(env_name=env_name, **env_args)
    proxy_env = envs.make(env_name=env_name, **env_args)
    env, monitor = task.monitor_wrapper(env)

    true_store = RewardStoreWrapper(env, buffer_size=REWARD_EVAL_SAMPLES)
    env = true_store
    true_buffer = true_store.buffer
    true_obs_buffer = true_store.obs_buffer
    true_action_buffer = true_store.action_buffer

    rew_delay = task.reward_delay_distribution(delay_config)
    env = task.delay_wrapper(env, rew_delay)
    reward_mapper_wrapper = task.reward_mapper(
        env, proxy_env=proxy_env, mapping_spec=mapping_spec
    )
    env = reward_mapper_wrapper

    env = RewardStoreWrapper(env, buffer_size=REWARD_EVAL_SAMPLES)
    pred_buffer = env.buffer
    solver_state = env.solver_state

    ft_op = transform.transform_pipeline(env=env, specs=control_feats_spec)
    lr = task.learning_rate(**controlexps.LEARNING_RATE_SPEC)  # type: ignore
    algorithm = task.create_algorithm(
        env=env,
        ft_op=ft_op,
        delay_reward=rew_delay,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        policy_type="markovian",
        base_seed=seed,
    )

    episode_returns = []
    results = algorithm.train(env=env, num_episodes=num_episodes, monitor=monitor)
    for snapshot in results:
        episode_returns.append(snapshot.returns)

    env.close()

    r_true = np.array(true_buffer, dtype=np.float64)
    r_pred = np.array(pred_buffer, dtype=np.float64)
    obs_true = np.array(true_obs_buffer)
    min_size = min(len(r_true), len(r_pred))
    r_true = r_true[-min_size:]
    r_pred = r_pred[-min_size:]
    obs_true = obs_true[-min_size:]

    all_rmse = metrics.rmse(v_true=r_true, v_pred=r_pred, axis=0)
    solution_step = solver_state["solution_found_step"]
    post_est_rmse = (
        metrics.rmse(
            v_true=r_true[solution_step:], v_pred=r_pred[solution_step:], axis=0
        )
        if solution_step is not None
        else np.nan
    )

    obs_arr = np.array(true_obs_buffer)
    action_arr = np.array(true_action_buffer)
    sa_pairs = set()
    for obs_idx in range(len(obs_arr)):
        obs_key = tuple(obs_arr[obs_idx].flat)
        sa_pairs.add((obs_key, action_arr[obs_idx]))
    coverage_pct = len(sa_pairs) / mdim if mdim > 0 else 0.0

    factors_coverage_pct = None
    estimation_meta = None
    if hasattr(reward_mapper_wrapper, "estimation_meta"):
        estimation_meta = reward_mapper_wrapper.estimation_meta
        snapshots = estimation_meta.get("snapshots", [])
        if snapshots:
            last_rank = snapshots[-1].get("sample", {}).get("factors_rank")
            if last_rank is not None:
                factors_coverage_pct = last_rank / mdim

    episode_returns_arr = np.array(episode_returns)
    return {
        "env_name": env_name,
        "method": method,
        "delay_config": delay_config,
        "gamma": gamma,
        "seed": seed,
        "epsilon": epsilon,
        "num_episodes": num_episodes,
        "max_episode_steps": grid_spec.get("max_episode_steps", 200),
        "grid_size": [nrows, ncols],
        "r_true": r_true,
        "r_pred": r_pred,
        "obs_true": obs_true,
        "all_rmse": all_rmse,
        "post_est_rmse": post_est_rmse,
        "solution_step": solution_step,
        "episode_returns": episode_returns_arr,
        "mean_return": float(np.mean(episode_returns_arr)),
        "return_variance": float(np.var(episode_returns_arr)),
        "coverage_pct": coverage_pct,
        "factors_coverage_pct": factors_coverage_pct,
    }


def run_parallel(
    run_configs: Sequence[Mapping[str, Any]],
    num_workers: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """Run experiments in parallel using joblib."""
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    results = joblib.Parallel(n_jobs=num_workers, backend="loky")(
        joblib.delayed(run_reward_estimation)(**config)
        for config in tqdm.auto.tqdm(run_configs, desc="Submitting")
    )
    return list(results)


def run_sequential(
    run_configs: Sequence[Mapping[str, Any]],
    verbose: bool = True,
) -> List[Mapping[str, Any]]:
    """Run experiments sequentially (for debugging)."""
    results: List[Mapping[str, Any]] = []
    iterator = (
        tqdm.auto.tqdm(run_configs, desc="Experiments") if verbose else run_configs
    )
    for config in iterator:
        results.append(run_reward_estimation(**config))
    return results


_run_reward_estimation_remote = ray.remote(run_reward_estimation)


def run_ray(
    run_configs: Sequence[Mapping[str, Any]],
    cluster_uri: Optional[str] = None,
) -> List[Mapping[str, Any]]:
    """Run experiments as Ray tasks, optionally against a remote cluster."""
    shuffled_configs = list(run_configs)
    np.random.shuffle(shuffled_configs)  # type: ignore

    # Inside a `ray job submit` entrypoint, Ray already sets RAY_ADDRESS and
    # the job's own runtime_env (working_dir + py_modules) is automatically
    # inherited by this process and any .remote() calls it makes -- passing
    # an overlapping runtime_env here would conflict with that ambient one.
    # Only ad-hoc local runs (no surrounding job) need to ship py_modules
    # themselves.
    runtime_env = None if "RAY_ADDRESS" in os.environ else {"py_modules": ["src/drmdp"]}
    with ray.init(cluster_uri, runtime_env=runtime_env) as context:
        print(f"Ray context: {context}")
        refs = [
            _run_reward_estimation_remote.remote(**config)
            for config in shuffled_configs
        ]
        unfinished = refs
        results: List[Mapping[str, Any]] = []
        with tqdm.auto.tqdm(total=len(refs), desc="Experiments") as progress:
            while unfinished:
                finished, unfinished = ray.wait(unfinished)
                results.extend(ray.get(finished))
                progress.update(len(finished))
    return results
