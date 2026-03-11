"""Tests for bias-variance decomposition in metrics module."""

import numpy as np
import pytest

from drmdp import metrics


class TestBiasVarianceDecomposition:
    """Test bias-variance decomposition functions."""

    def test_bias_zero_when_unbiased(self):
        """Test that bias is zero when predictions are centered on true value."""
        true_value = 5.0
        predictions = np.array([4.8, 5.1, 4.9, 5.2, 5.0])
        b = metrics.bias(predictions, true_value)
        assert abs(b) < 0.1  # Should be close to zero

    def test_bias_positive_when_overestimate(self):
        """Test that bias is positive when predictions overestimate."""
        true_value = 5.0
        predictions = np.array([6.0, 6.5, 7.0, 5.5, 6.0])
        b = metrics.bias(predictions, true_value)
        assert b > 0

    def test_bias_negative_when_underestimate(self):
        """Test that bias is negative when predictions underestimate."""
        true_value = 5.0
        predictions = np.array([4.0, 3.5, 4.5, 3.0, 4.0])
        b = metrics.bias(predictions, true_value)
        assert b < 0

    def test_variance_zero_for_constant_predictions(self):
        """Test that variance is zero when all predictions are identical."""
        predictions = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        v = metrics.variance(predictions)
        assert v == pytest.approx(0.0)

    def test_variance_positive_for_varying_predictions(self):
        """Test that variance is positive when predictions vary."""
        predictions = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        v = metrics.variance(predictions)
        assert v > 0

    def test_mse_equals_squared_error_for_constant_predictions(self):
        """Test MSE for constant predictions."""
        true_value = 5.0
        predictions = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        m = metrics.mse(predictions, true_value)
        expected_mse = (3.0 - 5.0) ** 2
        assert m == pytest.approx(expected_mse)

    def test_bias_variance_decomposition_identity(self):
        """Test that bias^2 + variance = MSE."""
        true_value = 10.0
        predictions = np.array([8.0, 9.5, 11.0, 10.5, 9.0, 12.0, 8.5])

        decomp = metrics.bias_variance_decomposition(predictions, true_value)

        # Verify the mathematical identity
        assert decomp["bias_squared"] == pytest.approx(decomp["bias"] ** 2)
        assert decomp["mse"] == pytest.approx(
            decomp["bias_squared"] + decomp["variance"], abs=1e-6
        )
        assert decomp["verification_error"] < 1e-6

    def test_bias_variance_decomposition_high_bias_low_variance(self):
        """Test decomposition for high bias, low variance case."""
        true_value = 0.0
        # All predictions far from true value but close to each other
        predictions = np.array([5.0, 5.1, 4.9, 5.0, 5.0])

        decomp = metrics.bias_variance_decomposition(predictions, true_value)

        # Should have high bias (mean is around 5)
        assert abs(decomp["bias"]) > 4.5
        # Should have low variance (predictions are similar)
        assert decomp["variance"] < 0.1
        # Most of error should come from bias
        assert decomp["bias_squared"] > 0.9 * decomp["mse"]

    def test_bias_variance_decomposition_low_bias_high_variance(self):
        """Test decomposition for low bias, high variance case."""
        true_value = 5.0
        # Predictions centered on true value but widely spread
        predictions = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        decomp = metrics.bias_variance_decomposition(predictions, true_value)

        # Should have low bias (mean is exactly 5)
        assert abs(decomp["bias"]) < 0.01
        # Should have high variance (predictions are spread out)
        assert decomp["variance"] > 5.0
        # Most of error should come from variance
        assert decomp["variance"] > 0.9 * decomp["mse"]

    def test_bias_variance_decomposition_all_keys_present(self):
        """Test that decomposition returns all expected keys."""
        predictions = np.array([1.0, 2.0, 3.0])
        true_value = 2.0

        decomp = metrics.bias_variance_decomposition(predictions, true_value)

        expected_keys = {
            "bias",
            "variance",
            "bias_squared",
            "mse",
            "verification_error",
        }
        assert set(decomp.keys()) == expected_keys

    def test_bias_variance_decomposition_single_prediction(self):
        """Test decomposition with only one prediction (edge case)."""
        predictions = np.array([3.0])
        true_value = 5.0

        decomp = metrics.bias_variance_decomposition(predictions, true_value)

        # Bias should be the difference
        assert decomp["bias"] == pytest.approx(-2.0)
        # Variance should be NaN or 0 with single sample
        # (ddof=1 makes variance undefined for n=1)
        assert np.isnan(decomp["variance"]) or decomp["variance"] == 0
