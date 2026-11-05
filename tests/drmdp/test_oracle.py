"""Tests for oracle — the task oracles of the estimation campaigns."""

import numpy as np
import pytest

from drmdp import envs, oracle
from drmdp.workflows import controlexps

CLIFF_REWARD = -100.0
OPEN_GRID = ["ooog", "oxxo", "sooo"]
ACTION_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))


class TestGridOracle:
    def test_following_the_oracle_realises_the_optimal_return(self):
        env = make_env("GridWorld-MINES")
        grid_oracle = oracle.task_oracle("GridWorld-MINES", env)
        observation, _ = env.reset(seed=7)
        optimal_return = grid_oracle.value(observation)
        realised = rollout(env, grid_oracle, observation)
        np.testing.assert_allclose(realised.returns, optimal_return, atol=1e-9)
        assert realised.terminated

    def test_takes_the_shortest_path_and_never_falls_into_a_cliff(self):
        env = make_env("GridWorld-MINES")
        grid_oracle = oracle.task_oracle("GridWorld-MINES", env)
        observation, _ = env.reset(seed=3)
        start = tuple(int(coordinate) for coordinate in observation)
        realised = rollout(env, grid_oracle, observation)
        assert realised.terminated
        assert realised.steps == shortest_path_length(OPEN_GRID, start)
        assert CLIFF_REWARD not in realised.rewards

    def test_reaches_the_exit_from_every_sampled_start(self):
        env = make_env("GridWorld-MINES", grid=controlexps.MINES_GW_GRID)
        grid_oracle = oracle.task_oracle("GridWorld-MINES", env)
        for seed in range(5):
            observation, _ = env.reset(seed=seed)
            realised = rollout(env, grid_oracle, observation, max_steps=200)
            assert realised.terminated
            assert CLIFF_REWARD not in realised.rewards


class TestMountainCarOracle:
    @pytest.mark.parametrize(
        "observation, expected",
        [
            ((-0.5, -0.01), oracle.MOUNTAIN_CAR_LEFT),
            ((-0.5, 0.0), oracle.MOUNTAIN_CAR_RIGHT),
            ((-0.5, 0.02), oracle.MOUNTAIN_CAR_RIGHT),
        ],
    )
    def test_pushes_in_the_direction_of_the_velocity(self, observation, expected):
        mountain_car_oracle = oracle.MountainCarOracle()
        assert mountain_car_oracle.action(np.array(observation)) == expected

    def test_reaches_the_flag_from_seeded_starts(self):
        env = make_env("MountainCar-v0")
        mountain_car_oracle = oracle.task_oracle("MountainCar-v0", env)
        for seed in range(5):
            observation, _ = env.reset(seed=seed)
            realised = rollout(env, mountain_car_oracle, observation, max_steps=2500)
            assert realised.terminated


class TestAcrobotOracle:
    @pytest.mark.parametrize(
        "joint_velocity, expected",
        [
            (1.0, oracle.ACROBOT_POSITIVE_TORQUE),
            (-1.0, oracle.ACROBOT_NEGATIVE_TORQUE),
        ],
    )
    def test_pumps_energy_in_the_direction_of_the_joint_velocity(
        self, joint_velocity, expected
    ):
        # hanging straight down: cos(theta_1) = cos(theta_2) = 1, so the
        # system energy sits below the upright configuration's energy
        hanging = np.array([1.0, 0.0, 1.0, 0.0, 0.0, joint_velocity])
        acrobot_oracle = oracle.AcrobotOracle(make_env("Acrobot-v1"))
        assert acrobot_oracle.action(hanging) == expected

    def test_reaches_the_goal_height_on_most_seeded_starts(self):
        env = make_env("Acrobot-v1")
        acrobot_oracle = oracle.task_oracle("Acrobot-v1", env)
        terminations = 0
        for seed in range(10):
            observation, _ = env.reset(seed=seed)
            realised = rollout(env, acrobot_oracle, observation, max_steps=500)
            terminations += realised.terminated
        assert terminations >= 8


class TestTaskOracle:
    @pytest.mark.parametrize(
        "env_name, expected_type",
        [
            ("GridWorld-MINES", oracle.GridOracle),
            ("MountainCar-v0", oracle.MountainCarOracle),
            ("Acrobot-v1", oracle.AcrobotOracle),
        ],
    )
    def test_dispatch_selects_the_environment_oracle(self, env_name, expected_type):
        env = make_env(env_name)
        assert isinstance(oracle.task_oracle(env_name, env), expected_type)

    def test_unknown_environment_raises(self):
        with pytest.raises(ValueError, match="no task oracle"):
            oracle.task_oracle("IceWorld-v0", object())

    def test_grid_without_a_transition_model_raises(self):
        with pytest.raises(ValueError, match="no transition model"):
            oracle.task_oracle("GridWorld-MINES", NoModelEnv())


class Rollout:
    """Outcome of acting with an oracle until termination or a step cap."""

    def __init__(self, returns, steps, rewards, terminated):
        self.returns = returns
        self.steps = steps
        self.rewards = rewards
        self.terminated = terminated


class NoModelEnv:
    """Environment stand-in that declares no transition model."""


def make_env(env_name: str, grid: list[str] | None = None):
    """Environment of a campaign arm, with the test grid for GridWorld."""
    if env_name.startswith("GridWorld-"):
        return envs.make(env_name, grid=grid or OPEN_GRID, max_episode_steps=200)
    if env_name == "MountainCar-v0":
        return envs.make(env_name, max_episode_steps=2500)
    return envs.make(env_name, max_episode_steps=500)


def shortest_path_length(grid, start) -> int:
    """Breadth-first distance from `start` to the exit, cliffs blocked."""
    cells = {
        (row, col): char
        for row, text in enumerate(grid)
        for col, char in enumerate(text)
    }
    exit_cell = next(cell for cell, char in cells.items() if char == "g")
    distances = {start: 0}
    frontier = [start]
    while frontier:
        following = []
        for cell in frontier:
            for delta_row, delta_col in ACTION_DELTAS:
                neighbour = (cell[0] + delta_row, cell[1] + delta_col)
                if neighbour in distances or neighbour not in cells:
                    continue
                if cells[neighbour] == "x":
                    continue
                distances[neighbour] = distances[cell] + 1
                following.append(neighbour)
        frontier = following
    return distances[exit_cell]


def rollout(env, task_oracle, observation, max_steps: int = 50) -> Rollout:
    """Acts with `task_oracle` from `observation` until termination or the cap."""
    rewards: list[float] = []
    terminated = False
    truncated = False
    while not (terminated or truncated) and len(rewards) < max_steps:
        observation, reward, terminated, truncated, _ = env.step(
            task_oracle.action(observation)
        )
        rewards.append(float(reward))
    return Rollout(float(np.sum(rewards)), len(rewards), rewards, bool(terminated))
