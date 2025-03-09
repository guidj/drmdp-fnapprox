import math

import gym
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
    obs_dim = math.floor(states.shape[1] / 2)
    mdim = obs_dim * len(np.unique(action)) + obs_dim

    # build samples
    mask = np.random.choice(states.shape[0], (sample_size, delay))
    delayed_obs = states[mask]  # batch x delay x dim
    delayed_act = action[mask]
    delayed_rew = np.sum(reward[mask], axis=1)  # batch x delay -> batch

    rhat_matrix = np.zeros(shape=(sample_size, mdim))

    # TODO: use pandas to speed this up
    for i, (states, action) in enumerate(zip(delayed_obs, delayed_act)):
        for j in range(delay):
            obs, next_obs = states[j][:obs_dim], states[j][obs_dim:]
            c = obs_dim * action[j]
            rhat_matrix[i, c : c + obs_dim] += obs
            rhat_matrix[i, -obs_dim:] += next_obs

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
    obs_dim = math.floor(states.shape[1] / 2)
    mdim = obs_dim * len(np.unique(action)) + obs_dim

    # build samples
    mask = np.random.choice(states.shape[0], sample_size)
    delayed_obs = states[mask]  # batch x dim
    delayed_act = action[mask]
    delayed_rew = reward[mask]  # batch x delay -> batch

    rhat_matrix = np.zeros(shape=(len(delayed_obs), mdim))

    # TODO: use pandas to speed this up
    for i, (states, action) in enumerate(zip(delayed_obs, delayed_act)):
        obs, next_obs = states[:obs_dim], states[obs_dim:]
        c = obs_dim * action
        rhat_matrix[i, c : c + obs_dim] += obs
        rhat_matrix[i, -obs_dim:] += next_obs

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
