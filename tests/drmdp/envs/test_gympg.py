import time

import numpy as np
import pytest

from drmdp import constants
from drmdp.envs import gympg
from drmdp.workflows.controlexps import MINES_GW_GRID

# Start 's' at (0, 0); cliffs (1, 0)..(1, 4); exit 'g' at (2, 5).
# BFS distances to the exit over the 12 start-eligible cells have
# median 3.5; cells at or above it are the six expected candidates.
CANDIDATES_GRID = ["sooooo", "xxxxxo", "ooooog"]
CANDIDATES_GRID_STARTS = ((0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1))
# Exit at (0, 4); the open cells in rows 2-3 cannot reach it. Median
# distance over the four eligible top-row cells is 2.5.
POCKET_GRID = ["sooog", "xxxxx", "oooxx", "oooxx"]
POCKET_GRID_STARTS = ((0, 0), (0, 1))
# No cell can reach the exit at (0, 3).
UNSOLVABLE_GRID = ["sxxg"]


class TestMakeAcrobot:
    def test_obs_shape_and_action_count(self):
        env = gympg.make("Acrobot-v1")
        assert env.observation_space.shape == (6,)
        assert env.action_space.n == 3
        env.close()

    def test_reset_returns_valid_obs(self):
        env = gympg.make("Acrobot-v1", max_episode_steps=500)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (6,)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_step_returns_negative_reward(self):
        env = gympg.make("Acrobot-v1", max_episode_steps=500)
        env.reset(seed=0)
        _, rew, _, _, _ = env.step(0)
        assert rew == -1.0
        env.close()


class TestMakeMountainCar:
    def test_obs_shape_and_action_count(self):
        env = gympg.make("MountainCar-v0")
        assert env.observation_space.shape == (2,)
        assert env.action_space.n == 3
        env.close()

    def test_step_returns_negative_reward(self):
        env = gympg.make("MountainCar-v0", max_episode_steps=200)
        env.reset(seed=0)
        _, rew, _, _, _ = env.step(0)
        assert rew == -1.0
        env.close()


class TestMakeGridWorld:
    def test_obs_is_grid_coordinates(self):
        grid = ["sog"]
        env = gympg.make("GridWorld-test", grid=grid)
        obs, _ = env.reset()
        assert obs.shape == (2,)
        assert obs[0] >= 0 and obs[1] >= 0
        env.close()

    def test_step_produces_valid_transition(self):
        grid = ["sog"]
        env = gympg.make("GridWorld-test", grid=grid, max_episode_steps=50)
        env.reset(seed=0)
        obs, rew, _, _, _ = env.step(0)
        assert obs.shape == (2,)
        assert isinstance(rew, (int, float))
        env.close()

    def test_uses_random_start_env(self):
        env = gympg.make("GridWorld-test", grid=CANDIDATES_GRID)
        assert isinstance(env.unwrapped, gympg.RandomStartGridWorld)
        env.close()


class TestRandomStartGridWorld:
    def test_start_candidates_are_furthest_half(self):
        env = gympg.make("GridWorld-test", grid=CANDIDATES_GRID)
        assert env.unwrapped.start_candidates == CANDIDATES_GRID_STARTS
        env.close()

    def test_reset_draws_from_candidates_only(self):
        env = gympg.make("GridWorld-test", grid=CANDIDATES_GRID, max_episode_steps=50)
        allowed = set(CANDIDATES_GRID_STARTS)
        for seed in range(300):
            obs, _ = env.reset(seed=seed)
            assert (int(obs[0]), int(obs[1])) in allowed
        env.close()

    def test_reset_covers_all_candidates(self):
        env = gympg.make("GridWorld-test", grid=CANDIDATES_GRID, max_episode_steps=50)
        seen = {
            (int(obs[0]), int(obs[1]))
            for obs, _ in (env.reset(seed=seed) for seed in range(300))
        }
        assert seen == set(CANDIDATES_GRID_STARTS)
        env.close()

    def test_reset_seed_is_reproducible_across_envs(self):
        env_a = gympg.make("GridWorld-test", grid=CANDIDATES_GRID, max_episode_steps=50)
        env_b = gympg.make("GridWorld-test", grid=CANDIDATES_GRID, max_episode_steps=50)
        obs_a, _ = env_a.reset(seed=7)
        obs_b, _ = env_b.reset(seed=7)
        np.testing.assert_array_equal(obs_a, obs_b)
        obs_a2, _ = env_a.reset(seed=7)
        np.testing.assert_array_equal(obs_a, obs_a2)
        env_a.close()
        env_b.close()

    def test_cliff_fall_returns_to_sampled_start(self):
        env = gympg.make("GridWorld-test", grid=CANDIDATES_GRID, max_episode_steps=50)
        # (0, 3) is a candidate; moving down enters the cliff at (1, 3)
        seed = next(
            seed
            for seed in range(300)
            if tuple(env.reset(seed=seed)[0].tolist()) == (0, 3)
        )
        env.reset(seed=seed)
        next_obs, reward, terminated, truncated, _ = env.step(3)
        assert tuple(next_obs.tolist()) == (0, 3)
        assert reward == -100.0
        assert not terminated and not truncated
        env.close()

    def test_transition_table_matches_episode_start(self):
        env = gympg.make("GridWorld-test", grid=CANDIDATES_GRID, max_episode_steps=50)
        seed = next(
            seed
            for seed in range(300)
            if tuple(env.reset(seed=seed)[0].tolist()) == (0, 3)
        )
        obs, _ = env.reset(seed=seed)
        start_pos = (int(obs[0]), int(obs[1]))
        state_id = env.get_wrapper_attr("states_mapping")[start_pos]
        transition = env.get_wrapper_attr("transition")
        cliff_targets = {
            next_state
            for actions in transition.values()
            for entries in actions.values()
            for prob, next_state, reward, _ in entries
            if prob == 1.0 and reward == -100.0
        }
        assert cliff_targets == {state_id}
        env.close()

    def test_excludes_cells_that_cannot_reach_exit(self):
        env = gympg.make("GridWorld-test", grid=POCKET_GRID, max_episode_steps=50)
        assert env.unwrapped.start_candidates == POCKET_GRID_STARTS
        for seed in range(200):
            obs, _ = env.reset(seed=seed)
            assert (int(obs[0]), int(obs[1])) in set(POCKET_GRID_STARTS)
        env.close()

    def test_unsolvable_grid_falls_back_to_textual_start(self):
        env = gympg.make("GridWorld-test", grid=UNSOLVABLE_GRID, max_episode_steps=50)
        assert env.unwrapped.start_candidates == ((0, 0),)
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(obs, (0, 0))
        env.close()

    def test_create_and_reset_worst_case_under_one_second(self):
        start_time = time.perf_counter()
        env = gympg.make("GridWorld-MINES", grid=MINES_GW_GRID, max_episode_steps=200)
        candidates = set(env.unwrapped.start_candidates)
        seen: set[tuple[int, int]] = set()
        seed = 0
        while seen != candidates and seed < 2000:
            obs, _ = env.reset(seed=seed)
            seen.add((int(obs[0]), int(obs[1])))
            # reading the table after each reset forces one build per
            # distinct start, the worst case for this env
            assert len(env.get_wrapper_attr("transition")) == env.get_wrapper_attr(
                "num_states"
            )
            seed += 1
        elapsed = time.perf_counter() - start_time
        assert seen == candidates
        assert elapsed < 1.0
        env.close()


class TestMakeRedGreen:
    def test_step_and_reset(self):
        env = gympg.make("RedGreen-v0", max_episode_steps=50)
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] > 0
        obs2, _, _, _, _ = env.step(0)
        assert obs2.shape == obs.shape
        env.close()


class TestMakeWithWrapper:
    def test_scale_wrapper_normalises_obs(self):
        env = gympg.make(
            "MountainCar-v0", wrapper=constants.SCALE, max_episode_steps=200
        )
        obs, _ = env.reset(seed=0)
        assert np.all(obs >= 0) and np.all(obs <= 1)
        env.step(0)
        obs2, _, _, _, _ = env.step(2)
        assert np.all(obs2 >= 0) and np.all(obs2 <= 1)
        env.close()


class TestUnknownEnv:
    def test_raises_value_error(self):
        with pytest.raises(ValueError):
            gympg.make("NonexistentEnv-v0")
