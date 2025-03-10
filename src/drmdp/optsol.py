import math

import gymnasium as gym
import numpy as np

from drmdp import data


def delay_reward_data(buffer, delay: int, sample_size: int):
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
    obs_dim =states.shape[1] // 2
    mdim = obs_dim * len(np.unique(action)) + obs_dim

    # build samples
    mask = np.random.choice(states.shape[0], (sample_size, delay))
    delayed_obs = states[mask]  # batch x delay x dim
    delayed_act = action[mask] # batch x delay
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


def rmse(v_pred: np.ndarray, v_true: np.ndarray, axis: int):
    if np.shape(v_pred) != np.shape(v_true):
        raise ValueError(
            f"Tensors have different shapes: {np.shape(v_pred)} != {np.shape(v_true)}"
        )
    delta = v_pred - v_true
    sq = delta * delta  # np.power(delta, 2)
    sqsum = np.sum(sq, axis=axis) / np.shape(v_pred)[axis]
    sqsqrt = np.sqrt(sqsum)
    return sqsqrt


def solve_rwe(env: gym.Env, num_steps: int, sample_size: int, delay: int):
    buffer = data.collection_traj_data(env, steps=num_steps)
    Xd, yd = delay_reward_data(buffer, delay=delay, sample_size=sample_size)
    return buffer, solve_least_squares(Xd, yd)
