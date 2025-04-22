import abc
import dataclasses
from typing import Optional

import gymnasium as gym
import numpy as np
from scipy import linalg

from drmdp import dataproc


@dataclasses.dataclass
class MultivariateNormal:
    """
    A multivariate normal distribution.
    """

    mean: np.ndarray
    cov: np.ndarray

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
    ) -> "MultivariateNormal":
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
            cov = cls.perturb_covariance_matrix(cov)
        except linalg.LinAlgError as err:
            if "Singular matrix" in err.args[0]:
                return None
            raise err
        return MultivariateNormal(coeff, cov)

    @classmethod
    def bayes_linear_regression(
        cls, matrix, rhs, prior: "MultivariateNormal"
    ) -> "MultivariateNormal":
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
            mean = np.matmul(
                cov, (np.matmul(inv_prior_sigma, prior.mean) + np.matmul(matrix.T, rhs))
            )
            return MultivariateNormal(mean, cov)
        except linalg.LinAlgError as err:
            raise ValueError("Failed Bayesian estimation") from err


class LearningRateSchedule(abc.ABC):
    """
    This class updates the learning rate based on the episode,
    step or both - using a given `schedule` function.
    """

    def __init__(self, initial_lr: float):
        super().__init__()
        self.initial_lr = initial_lr

    @abc.abstractmethod
    def schedule(self, episode: Optional[int] = None, step: Optional[int] = None):
        pass

    def __call__(self, episode: int, step: int):
        return self.schedule(episode, step)


class ConstantLRSchedule(LearningRateSchedule):
    def schedule(self, episode=None, step=None):
        del episode
        del step
        return self.initial_lr


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
    try:
        solution, _, _, _ = np.linalg.lstsq(a=matrix, b=rhs, rcond=None)
        return solution  # type: ignore
    except np.linalg.LinAlgError as err:
        # the computation failed, likely due to the matix being unsuitable (no solution).
        raise ValueError("Failed to solve linear system") from err


def solve_rwe(env: gym.Env, num_steps: int, sample_size: int, delay: int):
    buffer = dataproc.collection_traj_data(env, steps=num_steps)
    Xd, yd = delay_reward_data(buffer, delay=delay, sample_size=sample_size)
    return buffer, solve_least_squares(Xd, yd)
