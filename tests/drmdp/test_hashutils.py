import numpy as np
import pytest

from drmdp import hashtutils


def test_hashtrick_empty():
    # Test with empty input
    xs = np.zeros(10)
    result = hashtutils.hashtrick(xs, dim=5)
    assert np.array_equal(result, np.zeros(5))


def test_hashtrick_single_value():
    # Test with single 1 value
    xs = np.zeros(10)
    xs[3] = 1
    result = hashtutils.hashtrick(xs, dim=5)
    expected = np.zeros(5)
    expected[3] = 1
    assert np.array_equal(result, expected)


def test_hashtrick_collision():
    # Test hash collision (5 % 3 = 2, 2 % 3 = 2)
    xs = np.zeros(10)
    xs[5] = 1
    xs[2] = 1
    result = hashtutils.hashtrick(xs, dim=3)
    expected = np.zeros(3)
    expected[2] = 2  # Two values hash to same bucket
    assert np.array_equal(result, expected)


def test_hashtrick_full_range():
    # Test with 1s across full range
    xs = np.zeros(10)
    xs[1] = 1  # 1 % 4 = 1
    xs[4] = 1  # 4 % 4 = 0
    xs[7] = 1  # 7 % 4 = 3
    result = hashtutils.hashtrick(xs, dim=4)
    expected = np.array([1, 1, 0, 1])  # 1->1, 4->0, 7->3
    assert np.array_equal(result, expected)


def test_hashtrick_invalid_input():
    # Test with invalid inputs
    with pytest.raises(ValueError):
        hashtutils.hashtrick(np.zeros(10), dim=0)  # dim must be positive

    with pytest.raises(ValueError):
        hashtutils.hashtrick(np.zeros(10), dim=-1)  # dim must be positive
