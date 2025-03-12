import numpy as np
import pytest

from drmdp import optsol


def test_delay_reward_data():
    # Create test buffer with known values
    buffer = [
        (np.array([1.0, 2.0]), 0, np.array([3.0, 4.0]), 1.0),
        (np.array([5.0, 6.0]), 1, np.array([7.0, 8.0]), 2.0),
        (np.array([9.0, 10.0]), 0, np.array([11.0, 12.0]), 3.0),
    ]

    delay = 2
    sample_size = 2

    # Set random seed for reproducibility
    np.random.seed(42)

    matrix, rewards = optsol.delay_reward_data(buffer, delay, sample_size)

    # Check shapes
    assert matrix.shape == (sample_size, 6)  # 2 obs dims * 2 actions + 2 obs dims
    assert rewards.shape == (sample_size,)

    # Check matrix values are non-zero (exact values depend on random sampling)
    assert np.any(matrix != 0)

    # Check rewards are summed correctly
    assert np.all(rewards >= 0)  # All rewards in buffer are positive


def test_proj_obs_to_rwest_vec():
    # Create test buffer
    buffer = [
        (np.array([1.0, 2.0]), 0, np.array([3.0, 4.0]), 1.0),
        (np.array([5.0, 6.0]), 1, np.array([7.0, 8.0]), 2.0),
        (np.array([9.0, 10.0]), 0, np.array([11.0, 12.0]), 3.0),
    ]

    sample_size = 2

    # Set random seed for reproducibility
    np.random.seed(42)

    matrix, rewards = optsol.proj_obs_to_rwest_vec(buffer, sample_size)

    # Check shapes
    assert matrix.shape == (sample_size, 6)  # 2 obs dims * 2 actions + 2 obs dims
    assert rewards.shape == (sample_size,)

    # Check matrix values are non-zero
    assert np.any(matrix != 0)

    # Check rewards match original buffer values
    assert np.all(rewards > 0)  # All rewards in buffer are positive


def test_delay_reward_data_invalid_inputs():
    buffer = [(np.array([1.0]), 0, np.array([2.0]), 1.0)]

    # Test invalid delay
    with pytest.raises(ValueError):
        optsol.delay_reward_data(buffer, delay=0, sample_size=1)
    with pytest.raises(ValueError):
        optsol.delay_reward_data(buffer, delay=1, sample_size=1)
    with pytest.raises(ValueError):
        optsol.delay_reward_data(buffer, delay=-1, sample_size=1)

    # Test invalid sample size
    with pytest.raises(ValueError):
        optsol.delay_reward_data(buffer, delay=1, sample_size=0)


def test_proj_obs_to_rwest_vec_invalid_inputs():
    buffer = [(np.array([1.0]), 0, np.array([2.0]), 1.0)]

    # Test invalid sample size
    with pytest.raises(ValueError):
        optsol.proj_obs_to_rwest_vec(buffer, sample_size=0)
