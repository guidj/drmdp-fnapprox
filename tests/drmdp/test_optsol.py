import warnings

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

    def test_sample_matches_multivariate_normal(self):
        # Posterior-like distribution at the production dimension (d=216)
        dim = 216
        rng = np.random.default_rng(5)

        def tile_rows(num_rows):
            matrix = np.zeros((num_rows, dim))
            for idx in range(num_rows):
                columns = rng.choice(dim, size=8, replace=False)
                matrix[idx, columns] = 1.0
            rhs = -rng.integers(2, 14, num_rows) + rng.normal(scale=2.0, size=num_rows)
            return matrix, rhs

        matrix, rhs = tile_rows(256)
        prior = optsol.MultivariateNormal.least_squares(matrix, rhs, inverse="pseudo")
        matrix, rhs = tile_rows(5400)
        posterior = optsol.MultivariateNormal.bayes_linear_regression(
            matrix, rhs, prior
        )
        # the reference sampler is only exact for PSD covariances
        assert np.linalg.eigvalsh(posterior.cov).min() > 0

        num_samples = 20000
        draw_rng = np.random.default_rng(7)
        samples = np.empty((num_samples, dim))
        for idx in range(num_samples):
            samples[idx] = posterior.sample(draw_rng)
        ref_rng = np.random.default_rng(11)
        reference = ref_rng.multivariate_normal(
            posterior.mean, posterior.cov, size=num_samples
        )

        def max_z_score(estimate, target, std_error):
            return np.max(np.abs(estimate - target) / std_error)

        # empirical means within Monte-Carlo error of the analytic mean
        mean_std_error = np.sqrt(np.diag(posterior.cov) / num_samples)
        assert max_z_score(samples.mean(axis=0), posterior.mean, mean_std_error) < 6.0
        assert max_z_score(reference.mean(axis=0), posterior.mean, mean_std_error) < 6.0

        # empirical covariance entries within Monte-Carlo error of the analytic cov
        cov_std_error = np.sqrt(
            (
                np.outer(np.diag(posterior.cov), np.diag(posterior.cov))
                + posterior.cov * posterior.cov
            )
            / num_samples
        )
        assert (
            max_z_score(np.cov(samples, rowvar=False), posterior.cov, cov_std_error)
            < 6.0
        )
        assert (
            max_z_score(np.cov(reference, rowvar=False), posterior.cov, cov_std_error)
            < 6.0
        )

        # variance along a feature direction matches feats @ cov @ feats
        feats = np.zeros(dim)
        feats[rng.choice(dim, size=8, replace=False)] = 1.0
        target_variance = feats @ posterior.cov @ feats
        variance_std_error = target_variance * np.sqrt(2.0 / num_samples)
        assert (
            abs(np.var(samples @ feats, ddof=1) - target_variance)
            < 6.0 * variance_std_error
        )
        assert (
            abs(np.var(reference @ feats, ddof=1) - target_variance)
            < 6.0 * variance_std_error
        )

    def test_sample_non_psd_covariance_clips(self):
        # a small negative eigenvalue must not warn and must yield finite
        # draws; at this magnitude numpy's own sampler emits a
        # `covariance is not symmetric positive-semidefinite` warning and
        # silently samples an absolutized variant instead
        dim = 6
        rng = np.random.default_rng(3)
        base = rng.normal(size=(dim, dim))
        psd = base @ base.T + 0.5 * np.eye(dim)
        eig_values, eig_vectors = np.linalg.eigh(psd)
        eig_values[0] = -1e-6
        cov = (eig_vectors * eig_values) @ eig_vectors.T
        mv = optsol.MultivariateNormal(mean=rng.normal(size=dim), cov=cov)

        draw_rng = np.random.default_rng(4)
        num_samples = 200
        samples = np.empty((num_samples, dim))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for idx in range(num_samples):
                samples[idx] = mv.sample(draw_rng)
        assert np.all(np.isfinite(samples))
        np.testing.assert_allclose(samples.mean(axis=0), mv.mean, atol=1.0)

    def test_sample_singular_covariance_clips(self):
        # rank-deficient posteriors (e.g. identical-feature segments)
        # have exact zero eigenvalues; draws must stay finite and silent,
        # with full variance along the row space and ~none (the clipped
        # 1e-12) along null directions
        dim = 6
        rng = np.random.default_rng(6)
        direction = rng.normal(size=dim)
        unit = direction / np.linalg.norm(direction)
        orthogonal = rng.normal(size=dim)
        orthogonal -= (orthogonal @ unit) * unit
        orthogonal /= np.linalg.norm(orthogonal)
        cov = np.outer(direction, direction)
        mv = optsol.MultivariateNormal(mean=np.zeros(dim), cov=cov)

        draw_rng = np.random.default_rng(9)
        num_samples = 4000
        samples = np.empty((num_samples, dim))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for idx in range(num_samples):
                samples[idx] = mv.sample(draw_rng)
        assert np.all(np.isfinite(samples))

        row_target = float(unit @ cov @ unit)
        row_variance = float(np.var(samples @ unit, ddof=1))
        row_std_error = row_target * np.sqrt(2.0 / num_samples)
        assert abs(row_variance - row_target) < 6.0 * row_std_error

        null_variance = float(np.var(samples @ orthogonal, ddof=1))
        assert null_variance < 1e-8

    def test_sample_factor_cached(self):
        # the PSD factor is computed once per posterior, not per draw
        mv = optsol.MultivariateNormal(mean=np.zeros(4), cov=np.eye(4))
        rng = np.random.default_rng(0)
        first = mv.sample(rng)
        assert first.shape == (4,)
        assert mv._factor is not None
        factor = mv._factor
        second = mv.sample(rng)
        assert second.shape == (4,)
        assert mv._factor is factor


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
