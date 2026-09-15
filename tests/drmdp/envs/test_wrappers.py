import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from sklearn import cluster, mixture, model_selection

from drmdp import constants, envs
from drmdp.envs import wrappers
from drmdp.workflows import controlexps


class BoxEnv(gym.Env):
    """
    Terminates on `term_steps`.
    The observation is a vector with the
    step count, with possible values {0, 1, 2, ... term_steps}
    """

    def __init__(self, dim: int, term_steps: int = 2):
        if dim < 1:
            raise ValueError("`dim` must >= 2")

        self.observation_space = spaces.Box(low=0, high=2, shape=(dim,))
        self.action_space = spaces.Discrete(2)
        self.dim = dim
        self.step_count = 0
        self.term_steps = term_steps

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.ones(self.dim) * self.step_count
        reward = 1.0

        terminated = self.step_count >= self.term_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        del seed
        del options
        return np.ones(self.dim) * self.step_count, {}


def test_random_binary_obs_wrapper():
    wrapped_env = wrappers.RandomBinaryObsWrapper(BoxEnv(dim=1), enc_size=4)
    assert wrapped_env.enc_size == 4
    assert wrapped_env.observation_space == spaces.Box(
        low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.int64
    )

    obs_1 = wrapped_env.observation(np.array([0]))
    np.testing.assert_array_equal(np.size(obs_1), 4)
    np.testing.assert_array_equal(obs_1, getattr(wrapped_env, "_representations")[(0,)])
    assert np.min(obs_1) <= 0
    assert np.max(obs_1) <= 1
    assert np.sum(obs_1) <= 4

    obs_2 = wrapped_env.observation(np.array([1]))
    np.testing.assert_array_equal(np.size(obs_2), 4)
    np.testing.assert_array_equal(obs_2, getattr(wrapped_env, "_representations")[(1,)])
    assert np.min(obs_2) <= 0
    assert np.max(obs_2) <= 1
    assert np.sum(obs_2) <= 4


def test_scale_obs_wrapper():
    wrapped_env = wrappers.ScaleObsWrapper(BoxEnv(dim=1))
    assert wrapped_env.num_actions == 2
    assert wrapped_env.obs_dim == 1
    assert wrapped_env.observation_space == spaces.Box(
        low=0, high=1, shape=(1,), dtype=np.float64
    )

    np.testing.assert_array_equal(wrapped_env.observation(np.array([0])), 0)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1])), 0.5)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([2])), 1)


def test_gaussian_mix_obs_wrapper():
    wrapped_env = wrappers.GaussianMixObsWrapper(
        BoxEnv(dim=1),
        param_grid={
            "n_components": [2],
            "covariance_type": ["spherical"],
        },
        sample_steps=1000,
        random_state=113,
    )
    assert wrapped_env.param_grid == {
        "n_components": [2],
        "covariance_type": ["spherical"],
    }
    assert wrapped_env.sample_steps == 1000
    assert isinstance(wrapped_env.grid_search, model_selection.GridSearchCV)
    assert isinstance(wrapped_env.estimator, mixture.GaussianMixture)
    assert wrapped_env.obs_dim == 2
    assert wrapped_env.observation_space == spaces.Box(
        low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float64
    )

    obs_1 = wrapped_env.observation(np.array([0]))
    assert np.shape(obs_1) == (2,)
    np.testing.assert_allclose(np.sum(obs_1), 1)

    obs_2 = wrapped_env.observation(np.array([0]))
    assert np.shape(obs_2) == (2,)
    np.testing.assert_allclose(np.sum(obs_2), 1)


def test_cluster_centroid_obs_wrapper():
    wrapped_env = wrappers.ClusterCentroidObsWrapper(
        BoxEnv(dim=1), num_clusters=10, sample_steps=1000, seed=137
    )
    assert wrapped_env.num_clusters == 10
    assert wrapped_env.sample_steps == 1000
    assert wrapped_env.obs_dim == 10
    assert isinstance(wrapped_env.estimator, cluster.KMeans)
    assert wrapped_env.observation_space == gym.spaces.Discrete(10)


def test_flat_grid_coord_obs_wrapper():
    wrapped_env = wrappers.FlatGridCoordObsWrapper(BoxEnv(dim=1))
    assert wrapped_env.observation_space == gym.spaces.Discrete(2)
    obs, _ = wrapped_env.reset()
    np.testing.assert_array_equal(obs, 0)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([0])), 0)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1])), 1)

    wrapped_env = wrappers.FlatGridCoordObsWrapper(BoxEnv(dim=2))
    assert wrapped_env.observation_space == gym.spaces.Discrete(4)
    obs, _ = wrapped_env.reset()
    np.testing.assert_array_equal(obs, 0)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([0, 0])), 0)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([0, 1])), 1)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1, 0])), 2)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1, 1])), 3)

    wrapped_env = wrappers.FlatGridCoordObsWrapper(BoxEnv(dim=3))
    assert wrapped_env.observation_space == gym.spaces.Discrete(8)
    obs, _ = wrapped_env.reset()
    np.testing.assert_array_equal(obs, 0)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([0, 0, 0])), 0)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([0, 0, 1])), 1)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([0, 1, 0])), 2)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([0, 1, 1])), 3)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1, 0, 0])), 4)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1, 0, 1])), 5)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1, 1, 0])), 6)
    np.testing.assert_array_equal(wrapped_env.observation(np.array([1, 1, 1])), 7)


class TestPotentialShapingWrapper:
    def test_non_terminal_applies_shaping(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        shaped = wrappers.MountainCarHeightShaping(env, gamma=0.99)
        obs, _ = shaped.reset(seed=0)
        _, rew, _, _, _ = shaped.step(0)
        assert rew != -1.0
        shaped.close()

    def test_shaping_value(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        shaped = wrappers.MountainCarHeightShaping(env, gamma=1.0)
        obs, _ = shaped.reset(seed=0)
        prev_pot = np.sin(3.0 * obs[0])
        obs_next, rew, _, _, _ = shaped.step(0)
        next_pot = np.sin(3.0 * obs_next[0])
        expected = -1.0 + (next_pot - prev_pot)
        assert rew == pytest.approx(expected, abs=1e-6)
        shaped.close()


class TestAcrobotTipHeightShaping:
    def test_applies_shaping(self):
        env = gym.make("Acrobot-v1", max_episode_steps=500)
        shaped = wrappers.AcrobotTipHeightShaping(env, gamma=0.99)
        shaped.reset(seed=0)
        _, rew, _, _, _ = shaped.step(0)
        assert rew != -1.0
        shaped.close()


class TestAdditiveShapingWrapper:
    def test_mountain_car_height_bonus(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        shaped = wrappers.MountainCarHeightBonus(env, scale=2.0)
        shaped.reset(seed=0)
        obs_next, rew, _, _, _ = shaped.step(0)
        expected = -1.0 + 2.0 * np.sin(3.0 * obs_next[0])
        assert rew == pytest.approx(expected, abs=1e-6)
        shaped.close()

    def test_scale_zero_gives_original(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        shaped = wrappers.MountainCarHeightBonus(env, scale=0.0)
        shaped.reset(seed=0)
        _, rew, _, _, _ = shaped.step(0)
        assert rew == pytest.approx(-1.0)
        shaped.close()

    def test_acrobot_tip_height_bonus(self):
        env = gym.make("Acrobot-v1", max_episode_steps=500)
        shaped = wrappers.AcrobotTipHeightBonus(env, scale=1.0)
        shaped.reset(seed=0)
        obs_next, rew, _, _, _ = shaped.step(0)
        ct1, st1, ct2, st2 = obs_next[0], obs_next[1], obs_next[2], obs_next[3]
        cos_sum = ct1 * ct2 - st1 * st2
        expected = -1.0 + (-(ct1 + cos_sum))
        assert rew == pytest.approx(expected, abs=1e-6)
        shaped.close()


class TestActionCostShapingWrapper:
    def test_per_action_cost(self):
        env = gym.make("Acrobot-v1", max_episode_steps=500)
        shaped = wrappers.ActionCostShapingWrapper(env, action_costs=[0.0, 0.5, 1.0])
        shaped.reset(seed=0)
        _, r0, _, _, _ = shaped.step(0)
        shaped.reset(seed=0)
        _, r1, _, _, _ = shaped.step(1)
        shaped.reset(seed=0)
        _, r2, _, _, _ = shaped.step(2)
        assert r0 == pytest.approx(-1.0)
        assert r1 == pytest.approx(-0.5)
        assert r2 == pytest.approx(0.0)
        shaped.close()


class TestWrapFunction:
    def test_none_returns_env(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        assert wrappers.wrap(env, wrapper=None) is env
        env.close()

    def test_scale(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        wrapped = wrappers.wrap(env, wrapper=constants.SCALE)
        assert isinstance(wrapped, wrappers.ScaleObsWrapper)
        obs, _ = wrapped.reset(seed=0)
        assert np.all(obs >= 0) and np.all(obs <= 1)
        wrapped.close()

    def test_tiles(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        wrapped = wrappers.wrap(env, wrapper=constants.TILES, tiling_dim=4)
        obs, _ = wrapped.reset(seed=0)
        assert np.all((obs == 0) | (obs == 1))
        active_tiles = np.sum(obs)
        assert active_tiles > 0
        wrapped.close()

    def test_unknown_raises(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        with pytest.raises(ValueError, match="unknown"):
            wrappers.wrap(env, wrapper="nonexistent")
        env.close()

    def test_flat_grid_coord(self):
        env = envs.make(
            "GridWorld-MINES",
            grid=controlexps.MINES_GW_GRID,
            max_episode_steps=200,
        )
        wrapped = wrappers.wrap(env, wrapper=constants.FLAT_GRID_COORD, ohe=False)
        obs, _ = wrapped.reset(seed=0)
        assert isinstance(obs, (int, np.integer))
        assert obs >= 0
        wrapped.close()


class TestTilesObsWrapper:
    def test_output_is_binary(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        wrapped = wrappers.TilesObsWrapper(env, tiling_dim=4)
        obs, _ = wrapped.reset(seed=0)
        assert np.all((obs == 0) | (obs == 1))
        wrapped.close()

    def test_active_tile_count_equals_num_tilings(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        wrapped = wrappers.TilesObsWrapper(env, tiling_dim=4)
        obs, _ = wrapped.reset(seed=0)
        num_tilings = wrapped.tiles.num_tilings
        assert np.sum(obs) == num_tilings
        wrapped.close()

    def test_different_states_produce_different_tiles(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        wrapped = wrappers.TilesObsWrapper(env, tiling_dim=4)
        wrapped.reset(seed=0)
        obs1, _, _, _, _ = wrapped.step(0)
        for _ in range(20):
            wrapped.step(2)
        obs2, _, _, _, _ = wrapped.step(2)
        assert not np.array_equal(obs1, obs2)
        wrapped.close()

    def test_observation_space_shape(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        wrapped = wrappers.TilesObsWrapper(env, tiling_dim=3)
        obs, _ = wrapped.reset(seed=0)
        assert obs.shape == wrapped.observation_space.shape
        wrapped.close()

    def test_step_returns_valid_obs(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        wrapped = wrappers.TilesObsWrapper(env, tiling_dim=4)
        wrapped.reset(seed=0)
        obs, rew, term, trunc, info = wrapped.step(0)
        assert np.all((obs == 0) | (obs == 1))
        assert isinstance(rew, float)
        wrapped.close()
