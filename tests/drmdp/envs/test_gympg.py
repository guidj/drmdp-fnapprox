import numpy as np
import pytest

from drmdp import constants
from drmdp.envs import gympg


class TestMakeAcrobot:
    def test_obs_shape_and_action_count(self):
        env = gympg.make("Acrobot-v1")
        assert env.observation_space.shape == (6,)
        assert env.action_space.n == 3
        env.close()

    def test_reset_returns_valid_obs(self):
        env = gympg.make("Acrobot-v1", max_episode_steps=500)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (6,)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_step_returns_negative_reward(self):
        env = gympg.make("Acrobot-v1", max_episode_steps=500)
        env.reset(seed=0)
        _, rew, _, _, _ = env.step(0)
        assert rew == -1.0
        env.close()


class TestMakeMountainCar:
    def test_obs_shape_and_action_count(self):
        env = gympg.make("MountainCar-v0")
        assert env.observation_space.shape == (2,)
        assert env.action_space.n == 3
        env.close()

    def test_step_returns_negative_reward(self):
        env = gympg.make("MountainCar-v0", max_episode_steps=200)
        env.reset(seed=0)
        _, rew, _, _, _ = env.step(0)
        assert rew == -1.0
        env.close()


class TestMakeGridWorld:
    def test_obs_is_grid_coordinates(self):
        grid = ["sog"]
        env = gympg.make("GridWorld-test", grid=grid)
        obs, _ = env.reset()
        assert obs.shape == (2,)
        assert obs[0] >= 0 and obs[1] >= 0
        env.close()

    def test_step_produces_valid_transition(self):
        grid = ["sog"]
        env = gympg.make("GridWorld-test", grid=grid, max_episode_steps=50)
        env.reset(seed=0)
        obs, rew, _, _, _ = env.step(0)
        assert obs.shape == (2,)
        assert isinstance(rew, (int, float))
        env.close()


class TestMakeRedGreen:
    def test_step_and_reset(self):
        env = gympg.make("RedGreen-v0", max_episode_steps=50)
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] > 0
        obs2, _, _, _, _ = env.step(0)
        assert obs2.shape == obs.shape
        env.close()


class TestMakeWithWrapper:
    def test_scale_wrapper_normalises_obs(self):
        env = gympg.make(
            "MountainCar-v0", wrapper=constants.SCALE, max_episode_steps=200
        )
        obs, _ = env.reset(seed=0)
        assert np.all(obs >= 0) and np.all(obs <= 1)
        env.step(0)
        obs2, _, _, _, _ = env.step(2)
        assert np.all(obs2 >= 0) and np.all(obs2 <= 1)
        env.close()


class TestUnknownEnv:
    def test_raises_value_error(self):
        with pytest.raises(ValueError):
            gympg.make("NonexistentEnv-v0")
