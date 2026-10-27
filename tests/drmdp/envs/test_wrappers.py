from collections.abc import Mapping, Sequence

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from rlplg.environments import gridworld
from sklearn import cluster, mixture, model_selection

from drmdp import constants, envs
from drmdp.envs import gympg, wrappers
from drmdp.workflows import controlexps

DISTANCE_GRID = ["sog", "xxx"]


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
    np.testing.assert_array_equal(obs_1, wrapped_env._representations[(0,)])
    assert np.min(obs_1) <= 0
    assert np.max(obs_1) <= 1
    assert np.sum(obs_1) <= 4

    obs_2 = wrapped_env.observation(np.array([1]))
    np.testing.assert_array_equal(np.size(obs_2), 4)
    np.testing.assert_array_equal(obs_2, wrapped_env._representations[(1,)])
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
        _obs, _ = shaped.reset(seed=0)
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
        # the goal-referenced offset cancels in the difference at gamma=1
        expected = -1.0 + (next_pot - prev_pot)
        assert rew == pytest.approx(expected, abs=1e-6)
        shaped.close()

    def test_discount_weights_successor_potential(self):
        shaped = ShapedPotentialStub([-1.0, 2.0, 100.0], gamma=0.5)
        shaped.reset()
        _, rew, _, _, _ = shaped.step(0)
        # r + gamma*Phi(s') - Phi(s), with Phi drawn from the observations
        assert rew == pytest.approx(-1.0 + 0.5 * 2.0 - (-1.0))
        shaped.close()

    def test_scale_multiplies_shaping_term(self):
        shaped = ShapedPotentialStub([-1.0, 2.0, 100.0], gamma=1.0, scale=3.0)
        shaped.reset()
        _, rew, _, _, _ = shaped.step(0)
        assert rew == pytest.approx(-1.0 + 3.0 * (2.0 - (-1.0)))
        shaped.close()

    def test_scale_zero_reproduces_base_reward(self):
        shaped = ShapedPotentialStub([-1.0, 2.0, 100.0], gamma=1.0, scale=0.0)
        shaped.reset()
        _, rew, _, _, _ = shaped.step(0)
        assert rew == pytest.approx(-1.0)
        shaped.close()

    def test_terminal_transition_uses_zero_potential(self):
        # Phi drops to zero at the terminal state, so the terminating step
        # pays the previous potential, not the terminal observation's value
        shaped = ShapedPotentialStub([-1.0, 2.0, 100.0], gamma=1.0)
        shaped.reset()
        shaped.step(0)
        _, rew, terminated, _, _ = shaped.step(0)
        assert terminated
        assert rew == pytest.approx(-1.0 - 2.0)
        shaped.close()


class TestMountainCarHeightShaping:
    def test_goal_potential_is_goal_height(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        assert wrappers.MountainCarHeightShaping(env).goal_potential == pytest.approx(
            np.sin(1.5)
        )
        env.close()


class TestAcrobotTipHeightShaping:
    def test_applies_shaping(self):
        env = gym.make("Acrobot-v1", max_episode_steps=500)
        shaped = wrappers.AcrobotTipHeightShaping(env, gamma=0.99)
        shaped.reset(seed=0)
        _, rew, _, _, _ = shaped.step(0)
        assert rew != -1.0
        shaped.close()

    def test_goal_potential_is_goal_height(self):
        env = gym.make("Acrobot-v1", max_episode_steps=500)
        assert wrappers.AcrobotTipHeightShaping(env).goal_potential == pytest.approx(
            1.0
        )
        env.close()


class TestGridWorldDistanceShaping:
    def test_distance_table_holds_geodesic_moves_to_the_exit(self):
        shaped = wrappers.GridWorldDistanceShaping(_distance_env())
        assert shaped.max_distance == pytest.approx(2.0)
        assert shaped.distances[(0, 2)] == 0
        assert shaped.distances[(0, 1)] == 1
        assert shaped.distances[(0, 0)] == 2
        shaped.close()

    def test_cliff_cells_take_the_largest_distance(self):
        shaped = wrappers.GridWorldDistanceShaping(_distance_env())
        assert shaped.distances[(1, 0)] == 2
        shaped.close()

    def test_affine_potential_pays_a_constant_reward_on_a_shortest_path(self):
        # Phi = -d, so every step that closes the distance by one pays -1 + scale
        shaped = wrappers.GridWorldDistanceShaping(
            _distance_env(), gamma=1.0, scale=1.0, exponent=1.0
        )
        shaped.reset(seed=0)
        _, first, _, _, _ = shaped.step(gridworld.RIGHT)
        _, second, terminated, _, _ = shaped.step(gridworld.RIGHT)
        assert terminated
        assert first == pytest.approx(0.0)
        assert second == pytest.approx(0.0)
        shaped.close()

    def test_graded_potential_pays_more_further_from_the_exit(self):
        shaped = wrappers.GridWorldDistanceShaping(
            _distance_env(), gamma=1.0, scale=1.0, exponent=2.0
        )
        shaped.reset(seed=0)
        _, first, _, _, _ = shaped.step(gridworld.RIGHT)
        _, second, terminated, _, _ = shaped.step(gridworld.RIGHT)
        assert terminated
        # Phi = -2*(d/2)**2: -2 at d=2, -0.5 at d=1, 0 at the exit
        assert first == pytest.approx(-1.0 - 0.5 + 2.0)
        assert second == pytest.approx(-1.0 + 0.5)
        shaped.close()

    def test_discount_weights_the_successor_potential(self):
        shaped = wrappers.GridWorldDistanceShaping(
            _distance_env(), gamma=0.9, scale=1.0, exponent=1.0
        )
        shaped.reset(seed=0)
        _, reward, _, _, _ = shaped.step(gridworld.RIGHT)
        assert reward == pytest.approx(-1.0 + 0.9 * (-1.0) - (-2.0))
        shaped.close()

    def test_cliff_fall_adds_the_potential_drop_to_the_penalty(self):
        shaped = wrappers.GridWorldDistanceShaping(
            _distance_env(), gamma=1.0, scale=1.0, exponent=2.0
        )
        shaped.reset(seed=0)
        shaped.step(gridworld.RIGHT)
        obs, reward, terminated, _, _ = shaped.step(gridworld.DOWN)
        assert not terminated
        np.testing.assert_array_equal(obs, np.array([0, 0]))
        assert reward == pytest.approx(-100.0 - 2.0 + 0.5)
        shaped.close()

    def test_scale_zero_reproduces_the_base_reward(self):
        shaped = wrappers.GridWorldDistanceShaping(
            _distance_env(), gamma=1.0, scale=0.0, exponent=2.0
        )
        shaped.reset(seed=0)
        _, reward, _, _, _ = shaped.step(gridworld.RIGHT)
        assert reward == pytest.approx(-1.0)
        shaped.close()

    def test_requires_grid_geometry(self):
        env = gym.make("MountainCar-v0", max_episode_steps=10)
        with pytest.raises(ValueError):
            wrappers.GridWorldDistanceShaping(env)
        env.close()


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
        obs, rew, _term, _trunc, _info = wrapped.step(0)
        assert np.all((obs == 0) | (obs == 1))
        assert isinstance(rew, float)
        wrapped.close()


class TestGaussianRewardNoiseWrapper:
    def test_noise_alters_reward(self):
        env = BoxEnv(dim=1, term_steps=100)
        noisy = wrappers.ConstantGaussianNoiseReward(env, std_multiplier=1.0, seed=42)
        noisy.reset()
        rewards = []
        for _ in range(50):
            _, rew, term, trunc, _ = noisy.step(0)
            if term or trunc:
                break
            rewards.append(rew)
        assert not all(r == rewards[0] for r in rewards)

    def test_zero_multiplier_gives_original(self):
        env = BoxEnv(dim=1, term_steps=100)
        noisy = wrappers.ConstantGaussianNoiseReward(env, std_multiplier=0.0, seed=42)
        noisy.reset()
        _, rew, _, _, _ = noisy.step(0)
        assert rew == pytest.approx(1.0)

    def test_mean_noise_near_zero(self):
        env = BoxEnv(dim=1, term_steps=600)
        noisy = wrappers.ConstantGaussianNoiseReward(env, std_multiplier=1.0, seed=7)
        noisy.reset()
        noise_samples = []
        for _ in range(500):
            _, rew, term, trunc, _ = noisy.step(0)
            if term or trunc:
                break
            noise_samples.append(rew - 1.0)
        assert abs(np.mean(noise_samples)) < 0.3

    def test_noise_is_clipped(self):
        env = BoxEnv(dim=1, term_steps=600)
        noisy = wrappers.ConstantGaussianNoiseReward(
            env, std_multiplier=1.0, clip_std=1.96, seed=0
        )
        noisy.reset()
        sigma = 1.0
        bound = 1.96 * sigma
        for _ in range(500):
            _, rew, term, trunc, _ = noisy.step(0)
            if term or trunc:
                break
            noise = rew - 1.0
            assert noise >= -bound - 1e-10
            assert noise <= bound + 1e-10

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError):
            wrappers.ConstantGaussianNoiseReward(
                BoxEnv(dim=1, term_steps=100), std_multiplier=-0.5
            )

    def test_terminal_step_has_no_noise(self):
        env = BoxEnv(dim=1, term_steps=1)
        noisy = wrappers.ConstantGaussianNoiseReward(env, std_multiplier=5.0, seed=42)
        noisy.reset()
        _, rew, term, _, _ = noisy.step(0)
        assert term is True
        assert rew == pytest.approx(1.0)


class TestConstantGaussianNoiseReward:
    def test_reference_std_is_one(self):
        wrapper = wrappers.ConstantGaussianNoiseReward.__new__(
            wrappers.ConstantGaussianNoiseReward
        )
        assert wrapper._reference_std(np.array([0.0]), 0) == 1.0
        assert wrapper._reference_std(np.array([1.0, 2.0]), 1) == 1.0

    def test_applies_noise_on_real_env(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        noisy = wrappers.ConstantGaussianNoiseReward(env, std_multiplier=1.0, seed=42)
        noisy.reset(seed=0)
        rewards = []
        for _ in range(50):
            _, rew, _, _, _ = noisy.step(0)
            rewards.append(rew)
        assert not all(r == rewards[0] for r in rewards)
        noisy.close()


class TestRewardRangeGaussianNoiseReward:
    def test_reference_std_is_the_configured_std(self):
        wrapper = wrappers.RewardRangeGaussianNoiseReward.__new__(
            wrappers.RewardRangeGaussianNoiseReward
        )
        wrapper.reference_std = 4.0
        assert wrapper._reference_std(np.array([0.0]), 0) == 4.0

    def test_noise_std_scales_with_reference_std(self):
        # the empirical std over many seeded steps approximates
        # std_multiplier * reference_std for each reference
        stds = {}
        for reference_std in (1.0, 4.0):
            env = BoxEnv(dim=1, term_steps=2_000)
            noisy = wrappers.RewardRangeGaussianNoiseReward(
                env, reference_std=reference_std, std_multiplier=1.0, seed=42
            )
            noisy.reset()
            samples = []
            for _ in range(1_000):
                _, rew, term, trunc, _ = noisy.step(0)
                if term or trunc:
                    break
                samples.append(rew - 1.0)
            stds[reference_std] = np.std(samples)
        assert abs(stds[4.0] - 4.0) < 0.8
        assert abs(stds[1.0] - 1.0) < 0.3
        # the std ratio tracks the reference ratio
        assert 3.0 < stds[4.0] / stds[1.0] < 5.0

    def test_zero_multiplier_gives_original(self):
        env = BoxEnv(dim=1, term_steps=100)
        noisy = wrappers.RewardRangeGaussianNoiseReward(
            env, reference_std=100.0, std_multiplier=0.0, seed=42
        )
        noisy.reset()
        _, rew, _, _, _ = noisy.step(0)
        assert rew == pytest.approx(1.0)

    def test_noise_is_clipped(self):
        env = BoxEnv(dim=1, term_steps=600)
        std_multiplier, reference_std = 1.0, 4.0
        noisy = wrappers.RewardRangeGaussianNoiseReward(
            env,
            reference_std=reference_std,
            std_multiplier=std_multiplier,
            seed=0,
        )
        noisy.reset()
        bound = 1.96 * std_multiplier * reference_std
        for _ in range(500):
            _, rew, term, trunc, _ = noisy.step(0)
            if term or trunc:
                break
            noise = rew - 1.0
            assert noise >= -bound - 1e-10
            assert noise <= bound + 1e-10

    def test_terminal_step_has_no_noise(self):
        env = BoxEnv(dim=1, term_steps=1)
        noisy = wrappers.RewardRangeGaussianNoiseReward(
            env, reference_std=100.0, std_multiplier=5.0, seed=42
        )
        noisy.reset()
        _, rew, term, _, _ = noisy.step(0)
        assert term is True
        assert rew == pytest.approx(1.0)

    def test_same_seed_reproduces_noise(self):
        noisy_a = wrappers.RewardRangeGaussianNoiseReward(
            BoxEnv(dim=1, term_steps=100),
            reference_std=2.0,
            std_multiplier=1.0,
            seed=7,
        )
        noisy_b = wrappers.RewardRangeGaussianNoiseReward(
            BoxEnv(dim=1, term_steps=100),
            reference_std=2.0,
            std_multiplier=1.0,
            seed=7,
        )
        noisy_a.reset()
        noisy_b.reset()
        rewards_a, rewards_b = [], []
        for _ in range(50):
            _, rew_a, term_a, _, _ = noisy_a.step(0)
            _, rew_b, term_b, _, _ = noisy_b.step(0)
            rewards_a.append(rew_a)
            rewards_b.append(rew_b)
            if term_a or term_b:
                break
        assert rewards_a == rewards_b


class TestStdMultiplier:
    @pytest.mark.parametrize("std_multiplier", (0.0, 0.25, 0.5, 1.0))
    @pytest.mark.parametrize("reference_std", (1.0, 2.0))
    def test_noise_std_is_multiplier_times_reference(
        self, std_multiplier: float, reference_std: float
    ):
        expected = std_multiplier * reference_std
        env = BoxEnv(dim=1, term_steps=20_000)
        noisy = wrappers.RewardRangeGaussianNoiseReward(
            env,
            reference_std=reference_std,
            std_multiplier=std_multiplier,
            clip_std=10.0,
            seed=0,
        )
        noisy.reset()
        samples = []
        for _ in range(10_000):
            _, rew, term, trunc, _ = noisy.step(0)
            if term or trunc:
                break
            samples.append(rew - 1.0)
        assert np.std(samples) == pytest.approx(expected, abs=0.08 + 0.08 * expected)

    def test_zero_multiplier_leaves_every_reward_unchanged(self):
        env = BoxEnv(dim=1, term_steps=500)
        noisy = wrappers.RewardRangeGaussianNoiseReward(
            env, reference_std=3.0, std_multiplier=0.0, seed=11
        )
        noisy.reset()
        for _ in range(100):
            _, rew, term, trunc, _ = noisy.step(0)
            if term or trunc:
                break
            assert rew == pytest.approx(1.0)


class TestNormalizeNoiseArgs:
    def test_legacy_variance_multiplier_becomes_std_multiplier(self):
        normalized = wrappers.normalize_noise_args({"scale": 0.25, "reward_range": 1.0})
        assert normalized == {"std_multiplier": 0.5, "reference_std": 1.0}

    def test_std_multiplier_args_pass_through(self):
        args = {
            "std_multiplier": 0.5,
            "reference_std": 2.0,
            "clip_std": 1.96,
            "seed": 3,
        }
        assert wrappers.normalize_noise_args(args) == args

    def test_both_shapes_raise(self):
        with pytest.raises(TypeError):
            wrappers.normalize_noise_args({"scale": 1.0, "std_multiplier": 1.0})

    def test_both_reference_shapes_raise(self):
        with pytest.raises(TypeError):
            wrappers.normalize_noise_args({"reward_range": 1.0, "reference_std": 1.0})

    def test_unknown_arg_raises(self):
        with pytest.raises(TypeError):
            wrappers.normalize_noise_args({"noise_magnitude": 1.0})

    def test_negative_legacy_scale_raises(self):
        with pytest.raises(ValueError):
            wrappers.normalize_noise_args({"scale": -1.0})

    @pytest.mark.parametrize("multiplier", (0.0, 0.25, 0.5, 1.0, 2.0))
    def test_legacy_config_samples_as_the_multiplier(self, multiplier: float):
        legacy = wrappers.normalize_noise_args(
            {"scale": multiplier**2, "reward_range": 2.0}
        )
        direct = {"std_multiplier": multiplier, "reference_std": 2.0}
        assert _noise_samples(legacy, seed=0) == _noise_samples(direct, seed=0)

    def test_both_shapes_sample_identically_for_one_config(self):
        legacy = wrappers.normalize_noise_args({"scale": 0.0625, "reward_range": 1.0})
        direct = {"std_multiplier": 0.25, "reference_std": 1.0}
        assert _noise_samples(legacy, seed=5) == _noise_samples(direct, seed=5)


class PotentialStubEnv(gym.Env):
    """Deterministic env whose observation is the planned potential.

    The observation at reset is ``potentials[0]`` and after the n-th step
    ``potentials[n]``, so a wrapper reading ``obs[0]`` as its potential
    sees a chosen sequence.
    """

    def __init__(self, potentials: Sequence[float], term_steps: int):
        self.potentials = tuple(potentials)
        self.term_steps = term_steps
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(1,))
        self.action_space = spaces.Discrete(1)
        self.step_count = 0

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.array([self.potentials[self.step_count]])
        terminated = self.step_count >= self.term_steps
        return obs, -1.0, terminated, False, {}

    def reset(self, seed=None, options=None):
        del seed, options
        self.step_count = 0
        return np.array([self.potentials[0]]), {}


class ShapedPotentialStub(wrappers.PotentialShapingWrapper):
    """Potential-based shaper that reads the potential from the observation."""

    def __init__(self, potentials: Sequence[float], term_steps: int = 2, **kwargs):
        super().__init__(PotentialStubEnv(potentials, term_steps), **kwargs)

    def _potential(self, obs: np.ndarray) -> float:
        return float(obs[0])


def _distance_env() -> gym.Env:
    """GridWorld with the agent as a position vector and a fixed start."""
    size, cliffs, exits, start = gridworld.parse_grid_from_text(DISTANCE_GRID)
    return gympg.GridWorldObsAsVectorWrapper(
        gridworld.GridWorld(size, cliffs, exits, start)
    )


def _noise_samples(args: Mapping[str, float], seed: int) -> list[float]:
    """Reward offsets of a `relative-gaussian` wrapper built from `args`."""
    noisy = wrappers.RewardRangeGaussianNoiseReward(
        BoxEnv(dim=1, term_steps=100), seed=seed, **args
    )
    noisy.reset()
    samples = []
    for _ in range(50):
        _, rew, term, trunc, _ = noisy.step(0)
        if term or trunc:
            break
        samples.append(rew - 1.0)
    return samples
