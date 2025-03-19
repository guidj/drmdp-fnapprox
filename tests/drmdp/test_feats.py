import gymnasium as gym
import numpy as np

from drmdp import feats


def test_gaussian_mix_transform():
    # Create a simple environment with Box observation space
    env = gym.make("MountainCar-v0")

    # Initialize transform with some basic parameters
    params = {"n_components": 3, "random_state": 42}
    transform = feats.GaussianMixFeatTransform(env, sample_steps=100, **params)

    # Test single transform
    obs = env.observation_space.sample()
    action = env.action_space.sample()
    result = transform.transform(obs, action)

    # Check output shape and type
    assert result.shape == (transform.output_shape,)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32 or result.dtype == np.float64

    # Check that only one section has active features
    action_section_size = transform.obs_dim
    for i in range(env.action_space.n):
        section = result[i * action_section_size : (i + 1) * action_section_size]
        if i == action:
            assert np.any(section > 0)
        else:
            assert np.all(section == 0)
    np.testing.assert_almost_equal(np.sum(result), 1.0, decimal=5)


def test_gaussian_mix_batch_transform():
    # Create a simple environment with Box observation space
    env = gym.make("MountainCar-v0")

    # Initialize transform with some basic parameters
    params = {"n_components": 3, "random_state": 42}
    transform = feats.GaussianMixFeatTransform(env, sample_steps=100, **params)

    # Create batch of observations and actions
    batch_size = 5
    observations = np.array([env.observation_space.sample() for _ in range(batch_size)])
    actions = np.random.randint(0, env.action_space.n, size=batch_size)

    # Get batch transform result
    batch_result = transform.batch_transform(observations, actions)

    # Compare with individual transforms
    individual_results = np.zeros((batch_size, transform.output_shape))
    for i in range(batch_size):
        individual_results[i] = transform.transform(observations[i], actions[i])

    # Results should be equal
    np.testing.assert_array_almost_equal(batch_result, individual_results)

    # Check output shape
    assert batch_result.shape == (batch_size, transform.output_shape)
    # Test RndBinaryTransform


def test_rnd_binary_transform():
    env = gym.make("MountainCar-v0")
    transform = feats.RandomBinaryFeatTransform(env, enc_size=10)

    # Test single transform
    obs = env.observation_space.sample()
    action = env.action_space.sample()
    result = transform.transform(obs, action)
    assert result.shape == (transform.output_shape,)
    assert np.all((result == 0) | (result == 1))  # Binary values
    # Test that at least one value is positive
    assert np.any(result > 0)

    # Check that only one section has active features
    action_section_size = transform.obs_dim
    for i in range(env.action_space.n):
        section = result[i * action_section_size : (i + 1) * action_section_size]
        if i == action:
            assert np.any(section > 0)
        else:
            assert np.all(section == 0)
    assert np.sum(result) <= transform.obs_dim


def test_rnd_binary_batch_transform():
    env = gym.make("MountainCar-v0")
    transform = feats.RandomBinaryFeatTransform(env, enc_size=10)

    # Test batch transform
    batch_size = 5
    observations = [env.observation_space.sample() for _ in range(batch_size)]
    actions = [env.action_space.sample() for _ in range(batch_size)]
    batch_result = transform.batch_transform(observations, actions)

    # Compare with individual transforms
    individual_results = np.zeros((batch_size, transform.output_shape))
    for i in range(batch_size):
        individual_results[i] = transform.transform(observations[i], actions[i])
    np.testing.assert_array_equal(batch_result, individual_results)


def test_scale_obs_ohe_act_transform():
    env = gym.make("MountainCar-v0")
    transform = feats.ScaleFeatTransform(env)

    # Test single transform
    obs = env.observation_space.sample()
    action = env.action_space.sample()
    result = transform.transform(obs, action)
    assert result.shape == (transform.output_shape,)

    action_section_size = transform.obs_dim
    for i in range(env.action_space.n):
        section = result[i * action_section_size : (i + 1) * action_section_size]
        if i == action:
            assert np.any(section > 0)
        else:
            assert np.all(section == 0)
    # Scaled values should be between 0 and 1
    assert np.sum(result) <= transform.obs_dim


def test_scale_obs_ohe_act_batch_transform():
    env = gym.make("MountainCar-v0")
    transform = feats.ScaleFeatTransform(env)

    # Test batch transform
    batch_size = 5
    observations = [env.observation_space.sample() for _ in range(batch_size)]
    actions = [env.action_space.sample() for _ in range(batch_size)]
    batch_result = transform.batch_transform(observations, actions)

    # Compare with individual transforms
    individual_results = np.zeros((batch_size, transform.output_shape))
    for i in range(batch_size):
        individual_results[i] = transform.transform(observations[i], actions[i])
    np.testing.assert_array_almost_equal(batch_result, individual_results)


def test_tile_transform():
    env = gym.make("MountainCar-v0")
    transform = feats.TileFeatTransform(env, tiling_dim=8, num_tilings=16)

    # Test single transform
    obs = env.observation_space.sample()
    action = env.action_space.sample()
    result = transform.transform(obs, action)
    assert result.shape == (transform.output_shape,)
    assert np.all((result == 0) | (result == 1))  # Binary values
    assert np.sum(result) == transform.num_tilings

    # Test with hashing
    transform_hashed = feats.TileFeatTransform(
        env, tiling_dim=8, num_tilings=16, hash_dim=4
    )
    result_hashed = transform_hashed.transform(obs, action)
    assert result_hashed.shape == (transform_hashed.output_shape,)
    # Test that at least one value is positive
    assert np.any(result_hashed > 0)
    # First output is 1...16, which is evenly
    assert np.all(result_hashed == 16 // 4)

    # Second output
    result_hashed = transform_hashed.transform(env.observation_space.sample(), action)
    assert result_hashed.shape == (transform_hashed.output_shape,)
    # Test that at least one value is positive
    assert np.any(result_hashed > 0)
    assert np.all(result_hashed <= 16)


def test_tile_batch_transform():
    env = gym.make("MountainCar-v0")
    transform = feats.TileFeatTransform(env, tiling_dim=2, num_tilings=1)

    # Test batch transform
    batch_size = 5
    observations = [env.observation_space.sample() for _ in range(batch_size)]
    actions = [env.action_space.sample() for _ in range(batch_size)]
    batch_result = transform.batch_transform(observations, actions)

    # Compare with individual transforms
    individual_results = np.zeros((batch_size, transform.output_shape))
    for i in range(batch_size):
        individual_results[i] = transform.transform(observations[i], actions[i])
    np.testing.assert_array_equal(batch_result, individual_results)

    # Test with hashing
    transform_hashed = feats.TileFeatTransform(
        env, tiling_dim=8, num_tilings=16, hash_dim=4
    )
    batch_result_hashed = transform_hashed.batch_transform(observations, actions)
    individual_results_hashed = np.zeros((batch_size, transform_hashed.output_shape))
    for i in range(batch_size):
        individual_results_hashed[i] = transform_hashed.transform(
            observations[i], actions[i]
        )
    np.testing.assert_array_equal(batch_result_hashed, individual_results_hashed)
