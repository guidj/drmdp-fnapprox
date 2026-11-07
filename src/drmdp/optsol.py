import abc
import dataclasses
from collections.abc import Callable, Sequence
from typing import Optional, Protocol

import cvxpy as cp
import cvxpy.error
import gymnasium as gym
import numpy as np
from scipy import linalg

from drmdp import dataproc


class WeightPosterior(Protocol):
    """
    A posterior over linear-regression weights and its error scale.

    Implemented by `MultivariateNormal`, which fixes the observation error
    variance, and by `NormalInverseGamma`, which estimates it.
    """

    @property
    def mean(self) -> np.ndarray: ...

    @property
    def cov(self) -> np.ndarray: ...

    def sample(self, rng: np.random.Generator) -> np.ndarray: ...

    def update(
        self, matrix: np.ndarray, rhs: np.ndarray
    ) -> "WeightPosterior | None": ...


@dataclasses.dataclass
class MultivariateNormal:
    """
    A multivariate normal distribution.
    """

    mean: np.ndarray
    cov: np.ndarray
    _factor: np.ndarray | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Samples from the posterior using a cached PSD factorization.

        Factorizing once per posterior object makes per-episode weight draws
        cheap: each draw costs O(d^2) via mean + factor @ z instead of
        numpy's per-call O(d^3) SVD. The factor satisfies
        factor @ factor.T == cov for PSD covariances, so draws match
        Generator.multivariate_normal in distribution; eigenvalues below
        epsilon (numerical round-off from pseudo-inverse posteriors) are
        clipped, i.e. the nearest PSD covariance is sampled.
        """
        if self._factor is None:
            self._factor = psd_factor(self.cov)
        standard_normal = rng.standard_normal(self.mean.shape[0])
        return self.mean + self._factor @ standard_normal

    @staticmethod
    def perturb_covariance_matrix(cov, noise: float = 1e-6):
        eig_values, eig_matrix = np.linalg.eig(cov)
        perturbed_eig_values = np.maximum(
            eig_values, np.array([noise] * len(eig_values))
        )
        return eig_matrix * np.diag(perturbed_eig_values) * np.linalg.inv(eig_matrix)

    @classmethod
    def least_squares(
        cls, matrix, rhs, inverse: str = "pseudo"
    ) -> Optional["MultivariateNormal"]:
        """
        Least-squares estimation: mean and covariance.
        """
        if inverse == "exact":
            inverse_op = linalg.inv
        elif inverse == "pseudo":
            inverse_op = linalg.pinv
        else:
            raise ValueError(f"Unknown inverse: {inverse}")

        coeff = solve_least_squares(matrix, rhs)
        try:
            # Σᵦ = (Σᵦ^-1 + X'X)^-1 * σ²
            # Sigma^2 is the variance of the error term
            # Here, we assume sigma^2 = 1
            cov = inverse_op(np.matmul(matrix.T, matrix))
        except linalg.LinAlgError as err:
            if "Singular matrix" in err.args[0]:
                return None
            raise
        return MultivariateNormal(coeff, cov)

    @classmethod
    def convex_least_squares(
        cls,
        matrix,
        rhs,
        constraint_fn: Callable[[cp.Variable], Sequence],
        warm_start_initial_guess: np.ndarray | None = None,
        inverse: str = "pseudo",
    ) -> Optional["MultivariateNormal"]:
        """
        Least-squares estimation: mean and covariance.
        """
        if inverse == "exact":
            inverse_op = linalg.inv
        elif inverse == "pseudo":
            inverse_op = linalg.pinv
        else:
            raise ValueError(f"Unknown inverse: {inverse}")

        coeff = solve_convex_least_squares(
            matrix=matrix,
            rhs=rhs,
            constraint_fn=constraint_fn,
            warm_start_initial_guess=warm_start_initial_guess,
        )
        try:
            # Σᵦ = (Σᵦ^-1 + X'X)^-1 * σ²
            # Sigma^2 is the variance of the error term
            # Here, we assume sigma^2 = 1
            cov = inverse_op(np.matmul(matrix.T, matrix))
        except linalg.LinAlgError as err:
            if "Singular matrix" in err.args[0]:
                return None
            raise
        return MultivariateNormal(coeff, cov)

    @classmethod
    def bayes_linear_regression(
        cls, matrix, rhs, prior: "MultivariateNormal"
    ) -> Optional["MultivariateNormal"]:
        """
        Bayesian least-squares estimation: mean and covariance.
        """
        matrix = matrix.astype(prior.mean.dtype)
        rhs = rhs.astype(prior.mean.dtype)
        try:
            # Σᵦ_new = (Σᵦ^-1 + X'X)^-1 * σ²
            # μᵦ_new = (Σᵦ^-1 + X'X)^-1 * (Σᵦ^-1*μᵦ + X'Y)
            # Sigma^2 is the variance of the error term
            # Here, we assume sigma^2 = 1
            inv_prior_sigma = linalg.pinv(prior.cov)
            cov = linalg.pinv(inv_prior_sigma + np.matmul(matrix.T, matrix))
            coeff = np.matmul(
                cov, (np.matmul(inv_prior_sigma, prior.mean) + np.matmul(matrix.T, rhs))
            )
            return MultivariateNormal(coeff, cov)
        except linalg.LinAlgError as err:
            raise ValueError("Failed Bayesian estimation") from err

    def update(
        self, matrix: np.ndarray, rhs: np.ndarray
    ) -> "MultivariateNormal | None":
        """
        Posterior after folding in one more batch; `self` is the prior.
        """
        return MultivariateNormal.bayes_linear_regression(
            matrix=matrix, rhs=rhs, prior=self
        )


@dataclasses.dataclass
class InverseGamma:
    """
    The marginal posterior over the observation variance.

    The density is `p(sigma^2) ∝ (sigma^2)^-(shape+1) exp(-scale/sigma^2)`.
    """

    shape: float
    scale: float

    @property
    def variance(self) -> float:
        """
        The plug-in variance `scale/shape`.

        It is the maximum-likelihood estimate from `2*shape` residuals and the
        scale of the marginal Student-t over the weights, not the mean of the
        Inverse-Gamma density.
        """
        if self.shape <= 0.0:
            raise ValueError(f"Inverse-Gamma shape must be positive. Got {self.shape}")
        return self.scale / self.shape

    @property
    def degrees_of_freedom(self) -> float:
        return 2.0 * self.shape


@dataclasses.dataclass
class NormalInverseGamma:
    """
    Conjugate posterior over weights with an estimated error variance.

    `mean` and `lam` parameterise the conditional Gaussian
    `beta | sigma^2 ~ N(mean, sigma^2 lam^+)`, with `lam` the unscaled
    information accumulator; `ig` is the Inverse-Gamma posterior of
    `sigma^2`. Integrating out the variance gives a multivariate Student-t
    with `2*ig.shape` degrees of freedom, location `mean` and scale
    `ig.variance * lam^+`, which is what `cov` reports.
    """

    mean: np.ndarray
    lam: np.ndarray
    ig: InverseGamma
    _factor: np.ndarray | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )
    _cov_scale: np.ndarray | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def cov(self) -> np.ndarray:
        """The Student-t scale matrix `ig.variance * lam^+`."""
        return self.ig.variance * self._information_inverse

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draws from the marginal Student-t posterior over the weights.

        The draw is the Gaussian scale mixture of the conjugate model: a
        variance is drawn from the Inverse-Gamma posterior, then the weights
        from the Gaussian conditional. As the degrees of freedom grow the
        draw approaches `N(mean, cov)`. The factor of `lam^+` is cached per
        posterior, so a draw costs O(d^2) after the first.
        """
        if self._factor is None:
            self._factor = psd_factor(self._information_inverse)
        standard_normal = rng.standard_normal(self.mean.shape[0])
        scale = float(np.sqrt(self._sigma2_draw(rng)))
        return self.mean + scale * (self._factor @ standard_normal)

    @classmethod
    def least_squares(cls, matrix: np.ndarray, rhs: np.ndarray) -> "NormalInverseGamma":
        """
        Posterior from a first batch under the improper prior.

        The Inverse-Gamma prior is `shape = 0, scale = 0` and the information
        prior is zero, so the shape is half the batch row count and the scale
        is half the residual sum of squares of the least-squares fit.
        """
        mean = solve_least_squares(matrix, rhs)
        lam = matrix.T @ matrix
        residual_sum_squares = float(rhs @ rhs - mean @ (lam @ mean))
        return cls(
            mean=mean,
            lam=lam,
            ig=InverseGamma(
                shape=matrix.shape[0] / 2.0,
                scale=max(residual_sum_squares, 0.0) / 2.0,
            ),
        )

    @classmethod
    def bayes_linear_regression(
        cls,
        matrix: np.ndarray,
        rhs: np.ndarray,
        prior: "NormalInverseGamma",
    ) -> "NormalInverseGamma":
        """
        Sequentially updates the posterior with one aggregate batch.
        """
        matrix = matrix.astype(prior.mean.dtype)
        rhs = rhs.astype(prior.mean.dtype)
        lam = prior.lam + np.matmul(matrix.T, matrix)
        mean = np.matmul(
            linalg.pinv(lam),
            np.matmul(prior.lam, prior.mean) + np.matmul(matrix.T, rhs),
        )
        # The bracket telescopes over batches: twice the accumulated scale is
        # the pooled residual sum of squares at the updated mean
        incremental = float(
            rhs @ rhs + prior.mean @ (prior.lam @ prior.mean) - mean @ (lam @ mean)
        )
        return cls(
            mean=mean,
            lam=lam,
            ig=InverseGamma(
                shape=prior.ig.shape + matrix.shape[0] / 2.0,
                scale=prior.ig.scale + max(incremental, 0.0) / 2.0,
            ),
        )

    def update(self, matrix: np.ndarray, rhs: np.ndarray) -> "NormalInverseGamma":
        """
        Posterior after folding in one more batch; `self` is the prior.
        """
        return NormalInverseGamma.bayes_linear_regression(
            matrix=matrix, rhs=rhs, prior=self
        )

    @property
    def _information_inverse(self) -> np.ndarray:
        if self._cov_scale is None:
            self._cov_scale = linalg.pinv(self.lam)
        return self._cov_scale

    def _sigma2_draw(self, rng: np.random.Generator) -> float:
        """Draws one variance from the Inverse-Gamma posterior."""
        # `b / G` with `G ~ Gamma(shape, rate=1)` has density InvGamma(shape, b)
        if self.ig.scale <= 0.0:
            return 0.0
        return self.ig.scale / float(rng.gamma(self.ig.shape, 1.0))


class LearningRateSchedule(abc.ABC):
    """
    This class updates the learning rate based on the episode,
    step or both - using a given `schedule` function.
    """

    def __init__(self, initial_lr: float):
        super().__init__()
        self.initial_lr = initial_lr

    @abc.abstractmethod
    def schedule(self, episode: int | None = None, step: int | None = None):
        pass

    def __call__(self, episode: int, step: int):
        return self.schedule(episode, step)


class ConstantLRSchedule(LearningRateSchedule):
    def schedule(self, episode=None, step=None):
        del episode
        del step
        return self.initial_lr


class StreamingMean:
    """
    Calculates a mean value in streaming mode.
    """

    def __init__(self):
        self.mean: float | None = None
        self.count: int = 0

    def add(self, value: float) -> float:
        """
        Update's the mean with a new value
        """
        self.count += 1
        if self.mean is None:
            self.mean = value
        else:
            delta = value - self.mean
            self.mean = self.mean + delta / self.count
        return self.mean


def psd_factor(cov: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Returns `factor` with `factor @ factor.T` the nearest PSD matrix to `cov`.
    """
    # symmetrize to guard against round-off asymmetry, then project
    # onto the PSD cone by clipping eigenvalues below epsilon
    cov = 0.5 * (cov + cov.T)
    eig_values, eig_vectors = np.linalg.eigh(cov)
    eig_values = np.clip(eig_values, epsilon, None)
    return eig_vectors @ np.diag(np.sqrt(eig_values))  # type: ignore[no-any-return]


def delay_reward_data(buffer, delay: int, sample_size: int):
    if delay < 2:
        raise ValueError(f"`delay` must be greater than one. Got {delay}")
    if sample_size < 1:
        raise ValueError(f"`sample_size` must be greater than zero. Got {sample_size}")

    action = np.stack([example[1] for example in buffer])
    reward = np.stack([example[3] for example in buffer])
    states = np.concatenate(
        [
            np.stack([example[0] for example in buffer]),
            np.stack([example[2] for example in buffer]),
        ],
        axis=1,
    )

    # repr: (m1,a1)(m2,a1)..
    obs_dim = states.shape[1] // 2
    mdim = obs_dim * len(np.unique(action)) + obs_dim

    # build samples
    mask = np.random.choice(states.shape[0], (sample_size, delay))
    delayed_obs = states[mask]  # batch x delay x dim
    delayed_act = action[mask]  # batch x delay
    delayed_rew = np.sum(reward[mask], axis=1)  # batch x delay -> batch

    rhat_matrix = np.zeros(shape=(sample_size, mdim))

    # Vectorized operations for building rhat_matrix
    # Create indices for the action-based offsets
    action_offsets = delayed_act * obs_dim
    batch_indices = np.arange(sample_size)[:, None]

    # For each timestep in delay sequence
    for j in range(delay):
        # Split current states into obs and next_obs
        obs = delayed_obs[:, j, :obs_dim]
        next_obs = delayed_obs[:, j, obs_dim:]

        # Add obs to action-specific columns
        col_indices = action_offsets[:, j, None] + np.arange(obs_dim)
        rhat_matrix[batch_indices, col_indices] += obs

        # Add next_obs to final columns
        rhat_matrix[:, -obs_dim:] += next_obs
    return rhat_matrix, delayed_rew


def proj_obs_to_rwest_vec(buffer, sample_size: int):
    if sample_size < 1:
        raise ValueError(f"`sample_size` must be greater than zero. Got {sample_size}")

    action = np.stack([example[1] for example in buffer])
    reward = np.stack([example[3] for example in buffer])
    states = np.concatenate(
        [
            np.stack([example[0] for example in buffer]),
            np.stack([example[2] for example in buffer]),
        ],
        axis=1,
    )

    # repr: (m1,a1)(m2,a1)..
    obs_dim = states.shape[1] // 2
    mdim = obs_dim * len(np.unique(action)) + obs_dim

    # build samples
    mask = np.random.choice(states.shape[0], sample_size)
    delayed_obs = states[mask]  # batch x dim
    delayed_act = action[mask]
    delayed_rew = reward[mask]  # batch x delay -> batch

    rhat_matrix = np.zeros(shape=(len(delayed_obs), mdim))

    # Vectorized operations for building rhat_matrix
    # Create indices for the action-based offsets
    action_offsets = delayed_act * obs_dim
    batch_indices = np.arange(len(delayed_obs))[:, None]

    # Add obs to action-specific columns
    col_indices = action_offsets[:, None] + np.arange(obs_dim)
    rhat_matrix[batch_indices, col_indices] += delayed_obs[:, :obs_dim]

    # Add next_obs to final columns
    rhat_matrix[:, -obs_dim:] += delayed_obs[:, obs_dim:]
    return rhat_matrix, delayed_rew


def solve_least_squares(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solves a least-squares problem Ax = b.
    `A` is matrix.
    `b` is rhs (right-hand side).
    """
    try:
        solution, _, _, _ = np.linalg.lstsq(a=matrix, b=rhs, rcond=None)
        return solution  # type: ignore
    except np.linalg.LinAlgError as err:
        # the computation failed, likely due to the matix being unsuitable (no solution).
        raise ValueError("Failed to solve linear system") from err


def solve_rwe(env: gym.Env, num_steps: int, sample_size: int, delay: int):
    """
    Estimates rewards by collecting trajectory windows,
    and running least squares.
    """
    buffer = dataproc.collection_traj_data(env, steps=num_steps)
    matrix, rhs = delay_reward_data(buffer, delay=delay, sample_size=sample_size)
    return buffer, solve_least_squares(matrix=matrix, rhs=rhs)


def solve_convex_least_squares(
    matrix: np.ndarray,
    rhs: np.ndarray,
    constraint_fn: Callable[[cp.Variable], Sequence],
    warm_start_initial_guess: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solves Least Squares with convex optimisation,
    with added constraints.
    """
    solution = cp.Variable(matrix.shape[-1], value=warm_start_initial_guess)
    objective = cp.Minimize(cp.sum_squares(matrix @ solution - rhs))
    prob = cp.Problem(objective, constraint_fn(solution))

    try:
        prob.solve(warm_start=True)
        if prob.status == cp.OPTIMAL:
            return solution.value  # type: ignore
        raise ValueError(f"No Solution. Status: {prob.status}; Sol: {solution.value}")
    except cvxpy.error.SolverError as err:
        raise ValueError(f"No Solution. Status {prob.status}") from err


def matrix_factors_rank(matrix: np.ndarray) -> int:
    """
    Calculates factors present in columns
    """
    factors_sum = np.sum(matrix != 0, axis=0)
    factors_present: int = np.sum(factors_sum > 0).astype(np.int64).item()
    return factors_present


def matrix_numerical_rank(matrix: np.ndarray) -> int:
    """
    SVD-based numerical rank of a matrix.
    """
    return int(np.linalg.matrix_rank(matrix))
