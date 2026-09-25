"""Tests for rewest.py — job creation and reward store wrapper."""

import gymnasium as gym

from drmdp.workflows import rewest


class TestCreateExpInstance:
    def test_episodes_scale_with_epochs(self):
        job_spec = rewest.JobSpec(
            env_name="MountainCar-v0",
            env_args={"max_episode_steps": 200},
            control_feats_spec=[
                {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
            ],
            rewest_method="least-lfa",
            rewest_args={
                "attempt_estimation_episode": 10,
                "feats_spec": [],
                "estimation_buffer_mult": 25,
            },
            reward_delay=3,
            num_episodes=100,
            use_bias=False,
            epochs=3,
            turn=0,
        )
        instance = rewest.create_exp_instance(job_spec)
        assert instance.run_config.episodes_per_run == 300

    def test_delay_wired_as_poisson(self):
        job_spec = rewest.JobSpec(
            env_name="MountainCar-v0",
            env_args={"max_episode_steps": 200},
            control_feats_spec=[],
            rewest_method="least-lfa",
            rewest_args={},
            reward_delay=5,
            num_episodes=100,
            use_bias=False,
            epochs=1,
            turn=0,
        )
        instance = rewest.create_exp_instance(job_spec)
        delay_config = instance.experiment.problem_spec.delay_config
        assert delay_config["name"] == "clipped-poisson"
        assert delay_config["args"]["lam"] == 5

    def test_turn_becomes_instance_id(self):
        job_spec = rewest.JobSpec(
            env_name="MountainCar-v0",
            env_args={},
            control_feats_spec=[],
            rewest_method="least-lfa",
            rewest_args={},
            reward_delay=3,
            num_episodes=100,
            use_bias=False,
            epochs=1,
            turn=7,
        )
        instance = rewest.create_exp_instance(job_spec)
        assert instance.instance_id == 7


class TestPoissonDelayConfig:
    def test_min_delay_at_least_two(self):
        for lam in (2, 3, 5, 7, 10):
            config = rewest.poisson_delay_config(lam=lam)
            assert config["args"]["min_delay"] >= 2

    def test_lam_determines_distribution_parameter(self):
        config = rewest.poisson_delay_config(lam=8)
        assert config["args"]["lam"] == 8
        assert config["name"] == "clipped-poisson"


class TestRewardStoreWrapper:
    def test_stores_rewards(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        store = rewest.RewardStoreWrapper(env, buffer_size=100)
        store.reset(seed=0)
        for _ in range(10):
            store.step(store.action_space.sample())
        assert len(store.buffer) == 10
        assert store.steps_counter == 10
        store.close()

    def test_respects_buffer_limit(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        store = rewest.RewardStoreWrapper(env, buffer_size=5)
        store.reset(seed=0)
        for _ in range(20):
            store.step(store.action_space.sample())
        assert len(store.buffer) == 5
        assert store.steps_counter == 20
        store.close()
