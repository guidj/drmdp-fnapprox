import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn import cluster, mixture, model_selection

from drmdp.envs import wrappers


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
