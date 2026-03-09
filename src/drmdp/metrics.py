from typing import Mapping

import numpy as np


def rmse(v_pred: np.ndarray, v_true: np.ndarray, axis: int) -> float:
    if np.shape(v_pred) != np.shape(v_true):
        raise ValueError(
            f"Tensors have different shapes: {np.shape(v_pred)} != {np.shape(v_true)}"
        )
    delta = v_pred - v_true
    sq = delta * delta  # np.power(delta, 2)
    sqsum = np.sum(sq, axis=axis) / np.shape(v_pred)[axis]
    sqsqrt: float = np.sqrt(sqsum)
    return sqsqrt


def bias(v_pred: np.ndarray, v_true: float) -> float:
    """Compute bias: mean(predictions) - true_value"""
    return float(np.mean(v_pred) - v_true)


def variance(v_pred: np.ndarray) -> float:
    """Compute variance of predictions (population variance for bias-variance decomposition)"""
    return float(np.var(v_pred, ddof=0))


def mse(v_pred: np.ndarray, v_true: float) -> float:
    """Compute mean squared error"""
    return float(np.mean((v_pred - v_true) ** 2))


def bias_variance_decomposition(
    v_pred: np.ndarray, v_true: float
) -> Mapping[str, float]:
    """Compute full bias-variance decomposition with verification.

    For a set of predictions and a true value, computes:
    - bias = mean(predictions) - true_value
    - variance = var(predictions) [population variance, ddof=0]
    - mse = mean((predictions - true_value)^2)

    Also verifies the mathematical identity: bias^2 + variance = mse

    Note: Uses population variance (ddof=0) to ensure the bias-variance
    decomposition identity holds exactly: MSE = Bias² + Variance

    Args:
        v_pred: Array of predictions from multiple runs
        v_true: True value

    Returns:
        Dictionary containing bias, variance, bias_squared, mse, and verification_error
    """
    bias_ = bias(v_pred, v_true)
    var_ = variance(v_pred)
    mse_ = mse(v_pred, v_true)
    return {
        "bias": bias_,
        "variance": var_,
        "bias_squared": bias_**2,
        "mse": mse_,
        "verification_error": abs(bias_**2 + var_ - mse_),
    }
