import abc
from typing import Optional

import gymnasium as gym
import numpy as np

from drmdp import dataproc


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


class FixedLRSchedule(LearningRateSchedule):
    def __init__(self, initial_lr: float):
        super().__init__(initial_lr)

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
