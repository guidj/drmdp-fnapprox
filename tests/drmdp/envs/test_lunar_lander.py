"""Tests for the LunarLander reward-linear observation."""

import gymnasium as gym
import numpy as np
import pytest

from drmdp.envs import lunar_lander


class TestRewardBasis:
    def test_returns_the_five_potential_terms(self):
        state = np.array([3.0, 4.0, 6.0, 8.0, -0.5, 0.25, 1.0, 0.0])
        np.testing.assert_allclose(
            lunar_lander.reward_basis(state),
            np.array([5.0, 10.0, 0.5, 1.0, 0.0]),
        )

    def test_rejects_a_mislabelled_observation(self):
        with pytest.raises(ValueError, match="raw observation"):
            lunar_lander.reward_basis(np.zeros(lunar_lander.OBSERVATION_SIZE))


class TestObservationSpace:
    def test_bounds_cover_every_emitted_term(self):
        raw = _raw_observation_space()
        space = lunar_lander.observation_space(raw)
        assert space.shape == (lunar_lander.OBSERVATION_SIZE,)
        # Basis bounds follow from the raw box rather than from the environment.
        np.testing.assert_allclose(
            space.low[: lunar_lander.REWARD_BLOCK_SIZE],
            np.zeros(lunar_lander.REWARD_BLOCK_SIZE),
        )
        np.testing.assert_allclose(
            space.high[: lunar_lander.REWARD_BLOCK_SIZE],
            np.array(
                [
                    np.linalg.norm(raw.high[0:2]),
                    np.linalg.norm(raw.high[2:4]),
                    raw.high[4],
                    raw.high[6],
                    raw.high[7],
                ]
            ),
        )
        # The raw block keeps the environment's own bounds.
        np.testing.assert_array_equal(
            space.low[lunar_lander.RAW_OBSERVATION_OFFSET :], raw.low
        )
        np.testing.assert_array_equal(
            space.high[lunar_lander.RAW_OBSERVATION_OFFSET :], raw.high
        )

    def test_every_emitted_value_lies_inside_the_declared_box(self):
        env = gym.make("LunarLander-v3")
        wrapped = lunar_lander.LunarLanderRewardObs(env)
        try:
            rng = np.random.default_rng(0)
            observation, _ = wrapped.reset(seed=0)
            terminated = truncated = False
            while not (terminated or truncated):
                assert wrapped.observation_space.contains(observation)
                observation, _, terminated, truncated, _ = wrapped.step(
                    int(rng.integers(0, wrapped.action_space.n))
                )
        finally:
            wrapped.close()


class TestAwakeInfoWrapper:
    def test_step_info_carries_the_awake_bit(self):
        env = gym.make("LunarLander-v3")
        wrapped = lunar_lander.LunarLanderAwakeInfoWrapper(env)
        try:
            wrapped.reset(seed=0)
            _, _, _, _, info = wrapped.step(0)
            awake = env.unwrapped.lander.awake
            assert info[lunar_lander.AWAKE_INFO_KEY] == bool(awake)
            assert isinstance(info[lunar_lander.AWAKE_INFO_KEY], bool)
        finally:
            wrapped.close()

    def test_terminal_awake_bit_matches_the_reward_branch(self):
        # The environment pays +100 exactly on `not lander.awake`, so on a
        # terminal step the bit must equal the branch the reward selected.
        env = gym.make("LunarLander-v3")
        wrapped = lunar_lander.LunarLanderAwakeInfoWrapper(env)
        try:
            wrapped.reset(seed=11)
            rng = np.random.default_rng(11)
            terminated = truncated = False
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = wrapped.step(
                    int(rng.integers(0, wrapped.action_space.n))
                )
            if not truncated:
                assert info[lunar_lander.AWAKE_INFO_KEY] == bool(reward < 0)
        finally:
            wrapped.close()


class TestObservationWrapper:
    def test_emits_the_basis_followed_by_the_raw_observation(self):
        env = gym.make("LunarLander-v3")
        wrapped = lunar_lander.LunarLanderRewardObs(env)
        try:
            raw, _ = env.reset(seed=3)
            observation, _ = wrapped.reset(seed=3)
            np.testing.assert_allclose(
                observation[: lunar_lander.REWARD_BLOCK_SIZE],
                lunar_lander.reward_basis(raw),
            )
            np.testing.assert_allclose(
                observation[lunar_lander.RAW_OBSERVATION_OFFSET :], raw
            )
        finally:
            wrapped.close()

    def test_no_term_is_carried_across_episodes(self):
        env = gym.make("LunarLander-v3")
        wrapped = lunar_lander.LunarLanderRewardObs(env)
        try:
            first, _ = wrapped.reset(seed=0)
            second, _ = wrapped.reset(seed=0)
            # A stateless map returns the same observation for the same state.
            np.testing.assert_allclose(first, second)
            synthetic = np.zeros(lunar_lander.RAW_OBSERVATION_SIZE)
            np.testing.assert_allclose(
                wrapped.observation(synthetic),
                wrapped.observation(synthetic),
            )
        finally:
            wrapped.close()

    def test_rejects_a_non_box_observation(self):
        with pytest.raises(TypeError, match="Box"):
            lunar_lander.LunarLanderRewardObs(_DiscreteObservationEnv())


class _DiscreteObservationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return 0, {}


def _raw_observation_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.array(
            [-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10.0, 0.0, 0.0], dtype=np.float32
        ),
        high=np.array(
            [2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10.0, 1.0, 1.0], dtype=np.float32
        ),
    )
