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


class TestMatrixFactorsRank:
    def test_all_columns_nonzero(self):
        matrix = np.array([[1, 2], [3, 4]])
        assert optsol.matrix_factors_rank(matrix) == 2

    def test_zero_column(self):
        matrix = np.array([[1, 0], [3, 0]])
        assert optsol.matrix_factors_rank(matrix) == 1

    def test_sparse_with_all_columns_covered(self):
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert optsol.matrix_factors_rank(matrix) == 3


class TestMatrixNumericalRank:
    def test_full_rank(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert optsol.matrix_numerical_rank(matrix) == 2

    def test_rank_deficient_duplicate_rows(self):
        matrix = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        assert optsol.matrix_numerical_rank(matrix) == 1

    def test_rank_deficient_linearly_dependent_columns(self):
        col_a = np.array([1.0, 2.0, 3.0, 4.0])
        col_b = np.array([5.0, 6.0, 7.0, 8.0])
        col_c = col_a + col_b
        matrix = np.column_stack([col_a, col_b, col_c])
        assert optsol.matrix_numerical_rank(matrix) == 2

    def test_identity_full_rank(self):
        matrix = np.eye(5)
        assert optsol.matrix_numerical_rank(matrix) == 5

    def test_sparse_tile_coding_full_rank(self):
        rng = np.random.default_rng(42)
        nrows, ncols = 200, 50
        matrix = np.zeros((nrows, ncols))
        for idx in range(nrows):
            active = rng.choice(ncols, size=4, replace=False)
            matrix[idx, active] = rng.uniform(0.5, 2.0, size=4)
        assert optsol.matrix_numerical_rank(matrix) == ncols

    def test_factors_rank_passes_but_numerical_rank_detects_deficiency(self):
        matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [1.0, 1.0, 2.0],
            ]
        )
        assert optsol.matrix_factors_rank(matrix) == 3
        assert optsol.matrix_numerical_rank(matrix) < 3


class TestMultivariateNormal:
    def test_least_squares_pseudo(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rhs = np.array([1.0, 2.0, 3.0])
        result = optsol.MultivariateNormal.least_squares(matrix, rhs, inverse="pseudo")
        assert result is not None
        np.testing.assert_allclose(result.mean, [1.0, 2.0], atol=1e-6)
        assert result.cov.shape == (2, 2)

    def test_least_squares_exact(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rhs = np.array([1.0, 2.0, 3.0])
        result = optsol.MultivariateNormal.least_squares(matrix, rhs, inverse="exact")
        assert result is not None
        np.testing.assert_allclose(result.mean, [1.0, 2.0], atol=1e-6)
        assert result.cov.shape == (2, 2)

    def test_least_squares_unknown_inverse_raises(self):
        with pytest.raises(ValueError, match="Unknown inverse"):
            optsol.MultivariateNormal.least_squares(
                np.eye(2), np.array([1.0, 2.0]), inverse="bad"
            )

    def test_bayes_linear_regression(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([3.0, 4.0])
        prior = optsol.MultivariateNormal(
            mean=np.array([0.0, 0.0]),
            cov=np.eye(2) * 10.0,
        )
        result = optsol.MultivariateNormal.bayes_linear_regression(matrix, rhs, prior)
        assert result is not None
        assert result.mean.shape == (2,)
        assert result.cov.shape == (2, 2)

    def test_bayes_updates_toward_data(self):
        matrix = np.eye(3) * 2.0
        rhs = np.array([10.0, 20.0, 30.0])
        prior = optsol.MultivariateNormal(
            mean=np.zeros(3),
            cov=np.eye(3),
        )
        result = optsol.MultivariateNormal.bayes_linear_regression(matrix, rhs, prior)
        assert result is not None
        for idx in range(3):
            assert abs(result.mean[idx]) > abs(prior.mean[idx])


class TestSolveConvexLeastSquares:
    def test_unconstrained(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rhs = np.array([1.0, 2.0, 3.0])
        result = optsol.solve_convex_least_squares(
            matrix, rhs, constraint_fn=lambda var: []
        )
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-4)

    def test_with_non_negative_constraint(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([3.0, -1.0])
        result = optsol.solve_convex_least_squares(
            matrix, rhs, constraint_fn=lambda var: [var >= 0]
        )
        assert result[0] >= -1e-6
        assert result[1] >= -1e-6


def test_streaming_mean_estimator():
    xs = np.random.rand(100_000)
    estimator = optsol.StreamingMean()
    assert estimator.count == 0
    assert estimator.mean is None

    for val in xs:
        estimator.add(val)
    assert estimator.count == 100_000
    np.testing.assert_almost_equal(estimator.mean, 0.5, decimal=2)
