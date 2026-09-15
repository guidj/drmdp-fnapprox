"""Tests for task.py — env creation, reward mapping, experiment generation."""

from typing import Sequence

import numpy as np
import pytest

from drmdp import algorithms, core, envs, rewdelay, task, transform
from drmdp.workflows import controlexps


class TestCreateEnv:
    def test_env_and_proxy_are_independent(self):
        result = task.create_env("MountainCar-v0", {"max_episode_steps": 200})
        obs_env, _ = result.env.reset(seed=0)
        obs_proxy, _ = result.proxy.reset(seed=0)
        np.testing.assert_array_equal(obs_env, obs_proxy)
        result.env.step(0)
        obs_proxy2, _ = result.proxy.reset(seed=0)
        np.testing.assert_array_equal(obs_proxy, obs_proxy2)
        result.env.close()
        result.proxy.close()

    def test_creates_gridworld_with_correct_obs(self):
        result = task.create_env(
            "GridWorld-MINES",
            {"grid": controlexps.MINES_GW_GRID, "max_episode_steps": 200},
        )
        obs, _ = result.env.reset()
        assert obs.shape == (2,)
        assert obs[0] >= 0 and obs[1] >= 0
        result.env.close()
        result.proxy.close()

    def test_none_args_creates_functional_env(self):
        result = task.create_env("MountainCar-v0", None)
        obs, _ = result.env.reset(seed=0)
        assert obs.shape == (2,)
        _, rew, _, _, _ = result.env.step(0)
        assert rew == -1.0
        result.env.close()
        result.proxy.close()

    def test_reward_shaping_mountain_car(self):
        result = task.create_env(
            "MountainCar-v0",
            {
                "max_episode_steps": 200,
                "reward_shaping": {
                    "name": "mountain-car-height",
                    "args": {"scale": 1.0},
                },
            },
        )
        obs, _ = result.env.reset(seed=0)
        _, rew, _, _, _ = result.env.step(0)
        assert rew != -1.0
        result.env.close()
        result.proxy.close()

    def test_reward_shaping_action_cost(self):
        result = task.create_env(
            "Acrobot-v1",
            {
                "max_episode_steps": 500,
                "reward_shaping": {
                    "name": "action-cost",
                    "args": {"action_costs": [0.0, 0.5, 1.0]},
                },
            },
        )
        obs, _ = result.env.reset(seed=0)
        _, rew1, _, _, _ = result.env.step(1)
        assert rew1 == pytest.approx(-0.5)
        result.env.close()
        result.proxy.close()

    def test_unknown_shaping_raises(self):
        with pytest.raises(ValueError, match="Unknown reward shaping"):
            task.create_env(
                "MountainCar-v0",
                {
                    "max_episode_steps": 200,
                    "reward_shaping": {"name": "nonexistent", "args": {}},
                },
            )


class TestRewardMapper:
    def _make_env(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        proxy = envs.make("MountainCar-v0", max_episode_steps=200)
        return env, proxy

    def test_identity(self):
        env, proxy = self._make_env()
        result = task.reward_mapper(
            env, proxy_env=proxy, mapping_spec={"name": "identity", "args": None}
        )
        assert result is env
        env.close()
        proxy.close()

    def test_impute_missing_replaces_none_with_zero(self):
        env, proxy = self._make_env()
        env, _ = task.monitor_wrapper(env)
        delay = rewdelay.FixedDelay(delay=3)
        env = task.delay_wrapper(env, delay)
        result = task.reward_mapper(
            env,
            proxy_env=proxy,
            mapping_spec={"name": "impute-missing", "args": {"impute_value": 0}},
        )
        result.reset(seed=0)
        _, rew, _, _, _ = result.step(0)
        assert rew == 0.0
        result.close()
        proxy.close()

    def test_least_lfa_returns_imputed_before_estimation(self):
        env, proxy = self._make_env()
        env, _ = task.monitor_wrapper(env)
        delay = rewdelay.FixedDelay(delay=3)
        env = task.delay_wrapper(env, delay)
        result = task.reward_mapper(
            env,
            proxy_env=proxy,
            mapping_spec={
                "name": "least-lfa",
                "args": {
                    "attempt_estimation_episode": 10,
                    "use_bias": False,
                    "impute_value": 0,
                    "estimation_buffer_mult": 25,
                    "use_next_state": False,
                    "check_factors": False,
                    "feats_spec": [
                        {
                            "name": "tile-observation-action-ft",
                            "args": {"tiling_dim": 2},
                        }
                    ],
                },
            },
        )
        result.reset(seed=0)
        _, rew, _, _, _ = result.step(0)
        assert rew == 0.0
        assert result.get_wrapper_attr("weights") is None
        result.close()
        proxy.close()

    def test_bayes_least_lfa_has_no_estimate_initially(self):
        env, proxy = self._make_env()
        env, _ = task.monitor_wrapper(env)
        delay = rewdelay.FixedDelay(delay=3)
        env = task.delay_wrapper(env, delay)
        result = task.reward_mapper(
            env,
            proxy_env=proxy,
            mapping_spec={
                "name": "bayes-least-lfa",
                "args": {
                    "init_attempt_estimation_episode": 10,
                    "use_bias": False,
                    "impute_value": 0,
                    "estimation_buffer_mult": 25,
                    "feats_spec": [
                        {
                            "name": "tile-observation-action-ft",
                            "args": {"tiling_dim": 2},
                        }
                    ],
                },
            },
        )
        result.reset(seed=0)
        _, rew, _, _, _ = result.step(0)
        assert rew == 0.0
        assert result.get_wrapper_attr("mv_normal_rewards") is None
        result.close()
        proxy.close()

    def test_unknown_mapper_raises(self):
        env, proxy = self._make_env()
        with pytest.raises(ValueError, match="Unknown mapping_method"):
            task.reward_mapper(
                env,
                proxy_env=proxy,
                mapping_spec={"name": "nonexistent", "args": {}},
            )
        env.close()
        proxy.close()


class TestRewardDelayDistribution:
    def test_fixed(self):
        result = task.reward_delay_distribution({"name": "fixed", "args": {"delay": 3}})
        assert isinstance(result, rewdelay.FixedDelay)
        assert result.sample() == 3

    def test_uniform(self):
        result = task.reward_delay_distribution(
            {"name": "uniform", "args": {"min_delay": 2, "max_delay": 5}}
        )
        assert isinstance(result, rewdelay.UniformDelay)
        for _ in range(20):
            assert 2 <= result.sample() <= 5

    def test_clipped_poisson(self):
        result = task.reward_delay_distribution(
            {"name": "clipped-poisson", "args": {"lam": 3, "min_delay": 2}}
        )
        assert isinstance(result, rewdelay.ClippedPoissonDelay)

    def test_none_config(self):
        assert task.reward_delay_distribution(None) is None

    def test_unknown_delay_raises(self):
        with pytest.raises(ValueError, match="Unknown delay type"):
            task.reward_delay_distribution({"name": "nonexistent", "args": {}})


class TestLearningRate:
    def test_constant(self):
        lr = task.learning_rate(name="constant", args={"initial_lr": 0.01})
        assert lr.initial_lr == pytest.approx(0.01)
        assert lr.schedule() == pytest.approx(0.01)

    def test_missing_initial_lr_raises(self):
        with pytest.raises(ValueError, match="Missing `initial_lr`"):
            task.learning_rate(name="constant", args={})

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown lr"):
            task.learning_rate(name="nonexistent", args={"initial_lr": 0.01})


class TestBundle:
    def test_exact_division(self):
        result = task.bundle([1, 2, 3, 4], bundle_size=2)
        assert result == [[1, 2], [3, 4]]

    def test_remainder(self):
        result = task.bundle([1, 2, 3, 4, 5], bundle_size=2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_bundle(self):
        result = task.bundle([1, 2, 3], bundle_size=10)
        assert result == [[1, 2, 3]]

    def test_size_one(self):
        result = task.bundle([1, 2, 3], bundle_size=1)
        assert result == [[1], [2], [3]]

    def test_empty(self):
        result = task.bundle([], bundle_size=2)
        assert result == []

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            task.bundle([1], bundle_size=0)


class TestGenerateExperimentInstances:
    def test_generates_correct_count(self):
        experiment = core.Experiment(
            env_spec=core.EnvSpec(
                name="MountainCar-v0",
                args={"max_episode_steps": 200},
                feats_spec=[],
            ),
            problem_spec=core.ProblemSpec(
                policy_type="markovian",
                reward_mapper={"name": "identity", "args": None},
                delay_config=None,
                epsilon=0.1,
                gamma=0.99,
                learning_rate_config={
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            ),
            epochs=1,
        )
        instances = list(
            task.generate_experiments_instances(
                experiments=[experiment],
                num_runs=3,
                num_episodes_per_epoch=100,
                log_episode_frequency=10,
                use_seed=True,
                output_dir="/tmp/test",
                task_prefix="test",
                export_model=False,
            )
        )
        assert len(instances) == 3
        assert all(inst.run_config.episodes_per_run == 100 for inst in instances)
        assert all(inst.instance_id == idx for idx, inst in enumerate(instances))


class TestCreateAlgorithm:
    def _make_algo(self, policy_type, delay_reward=None):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        mon_env, mon = task.monitor_wrapper(env)
        if delay_reward:
            mon_env = task.delay_wrapper(mon_env, delay_reward)
        ft_op = transform.transform_pipeline(
            env=mon_env,
            specs=[{"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}],
        )
        algo = task.create_algorithm(
            env=mon_env,
            ft_op=ft_op,
            delay_reward=delay_reward,
            lr=task.learning_rate(name="constant", args={"initial_lr": 0.01}),
            gamma=0.99,
            epsilon=0.2,
            policy_type=policy_type,
            base_seed=0,
        )
        return algo, mon_env, mon

    def test_markovian_trains_and_updates_weights(self):
        algo, env, mon = self._make_algo("markovian")
        snapshots = list(algo.train(env=env, num_episodes=3, monitor=mon))
        _assert_training_snapshots(snapshots, expected_count=3)
        assert mon.step > 0
        env.close()

    def test_drop_missing_trains_with_delayed_rewards(self):
        delay = rewdelay.FixedDelay(delay=3)
        algo, env, mon = self._make_algo("drop-missing", delay_reward=delay)
        snapshots = list(algo.train(env=env, num_episodes=3, monitor=mon))
        _assert_training_snapshots(snapshots, expected_count=3)
        assert mon.step > 0
        env.close()

    def test_uniform_random_trains(self):
        algo, env, mon = self._make_algo("uniform-random")
        snapshots = list(algo.train(env=env, num_episodes=3, monitor=mon))
        _assert_training_snapshots(snapshots, expected_count=3)
        assert mon.step > 0
        env.close()

    def test_unknown_policy_type_raises(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        ft_op = transform.transform_pipeline(
            env=env,
            specs=[{"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}],
        )
        with pytest.raises(ValueError, match="Unknown policy_type"):
            task.create_algorithm(
                env=env,
                ft_op=ft_op,
                delay_reward=None,
                lr=task.learning_rate(name="constant", args={"initial_lr": 0.01}),
                gamma=0.99,
                epsilon=0.1,
                policy_type="nonexistent",
            )
        env.close()


def _assert_training_snapshots(
    snapshots: Sequence[algorithms.PolicyControlSnapshot],
    expected_count: int,
):
    """Verify invariant properties that every training run must satisfy."""
    assert len(snapshots) == expected_count
    for snapshot in snapshots:
        assert isinstance(snapshot, algorithms.PolicyControlSnapshot)
        assert snapshot.steps > 0
        assert isinstance(snapshot.weights, np.ndarray)
        assert np.isfinite(snapshot.returns)

    shapes = [snap.weights.shape for snap in snapshots]
    assert len(set(shapes)) == 1, f"Inconsistent weight shapes: {set(shapes)}"

    assert not np.all(snapshots[-1].weights == 0)
