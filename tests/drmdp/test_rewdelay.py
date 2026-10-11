import dataclasses

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from drmdp import optsol, rewdelay, transform


class TestDelayedRewardWrapper:
    """Tests for DelayedRewardWrapper — window boundaries, aggregation, and episode-boundary flushes."""

    def test_delayed_reward_wrapper_with_fixed_delay_init(self):
        env = DummyEnv()
        wrapped = rewdelay.DelayedRewardWrapper(
            env, reward_delay=rewdelay.FixedDelay(2)
        )

        assert isinstance(wrapped.reward_delay, rewdelay.FixedDelay)
        assert wrapped.reward_delay.delay == 2
        assert wrapped.observation_space == env.observation_space
        assert wrapped.action_space == env.action_space
        assert wrapped.segment is None
        assert wrapped.segment_step is None
        assert wrapped.delay is None
        assert wrapped.op(range(10)) == sum(range(10))

    def test_delayed_reward_wrapper_with_fixed_delay_step(self):
        env = DummyEnv(term_steps=4)
        wrapped = rewdelay.DelayedRewardWrapper(
            env, reward_delay=rewdelay.FixedDelay(2)
        )

        obs, info = wrapped.reset()
        np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
        assert info == {"delay": 2, "segment": 0, "segment_step": -1, "next_delay": 2}

        # Seg 1, Step 1
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
        assert (reward, term, trunc) == (None, False, False)
        assert info == {"delay": 2, "segment": 0, "segment_step": 0, "next_delay": None}

        # Ep 1, Seg 1, Step 2
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([2, 2, 2]))
        assert (reward, term, trunc) == (2.0, False, False)
        assert info == {"delay": 2, "segment": 0, "segment_step": 1, "next_delay": 2}

        # Ep 1, Seg 2, Step 1
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([3, 3, 3]))
        assert (reward, term, trunc) == (None, False, False)
        assert info == {"delay": 2, "segment": 1, "segment_step": 0, "next_delay": None}

        # Ep 1, Seg 2, Step 2
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([4, 4, 4]))
        assert (reward, term, trunc) == (2.0, True, False)
        assert info == {"delay": 2, "segment": 1, "segment_step": 1, "next_delay": 2}

        # Reset after termination
        obs, info = wrapped.reset()
        np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
        assert info == {"delay": 2, "segment": 0, "segment_step": -1, "next_delay": 2}

        # Ep 2, Seg 1, Step 1
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
        assert (reward, term, trunc) == (None, False, False)
        assert info == {"delay": 2, "segment": 0, "segment_step": 0, "next_delay": None}

    def test_delayed_reward_wrapper_with_fixed_delay_reset(self):
        env = DummyEnv()
        wrapped = rewdelay.DelayedRewardWrapper(
            env, reward_delay=rewdelay.FixedDelay(2)
        )

        wrapped.reset()
        # Seg 1, Step 1
        wrapped.step(0)
        assert wrapped.segment == 0
        assert wrapped.segment_step == 0
        assert wrapped.rewards == [1]

        # Seg 1, Step 2
        # sement info has been reset
        # for the next step
        wrapped.step(0)
        assert wrapped.segment == 1
        assert wrapped.segment_step == -1
        assert not wrapped.rewards

        # Seg 2, Step 1 — env still terminated, partial flush clears rewards
        wrapped.step(0)
        assert wrapped.segment == 1
        assert wrapped.segment_step == 0
        assert not wrapped.rewards

        # Reset
        wrapped.reset()
        assert wrapped.segment == 0
        assert wrapped.segment_step == -1
        assert not wrapped.rewards

    def test_delayed_reward_wrapper_with_poisson_delay_init(self):
        env = DummyEnv()
        wrapped = rewdelay.DelayedRewardWrapper(
            env, reward_delay=rewdelay.ClippedPoissonDelay(2, min_delay=2, max_delay=5)
        )

        assert isinstance(wrapped.reward_delay, rewdelay.ClippedPoissonDelay)
        assert wrapped.reward_delay.lam == 2
        assert wrapped.observation_space == env.observation_space
        assert wrapped.action_space == env.action_space
        assert wrapped.segment is None
        assert wrapped.segment_step is None
        assert wrapped.delay is None
        assert wrapped.op(range(10)) == sum(range(10))

    def test_delayed_reward_wrapper_with_poisson_delay_step(self, monkeypatch):
        class MockPoissonRng:
            def __init__(self, return_value: int):
                self.return_value = return_value

            def poisson(self, lam: int):
                del lam
                return self.return_value

        env = DummyEnv(term_steps=4)
        reward_delay = rewdelay.ClippedPoissonDelay(2, min_delay=2, max_delay=5)
        wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)

        # Delay of 3
        monkeypatch.setattr(reward_delay, "rng", MockPoissonRng(3))

        wrapped.reset()
        # Ep 1, Seg 1, Step 1
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
        assert (reward, term, trunc) == (None, False, False)
        assert info == {"delay": 3, "segment": 0, "segment_step": 0, "next_delay": None}

        # # Ep 1, Seg 1, Step 2
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([2, 2, 2]))
        assert (reward, term, trunc) == (None, False, False)
        assert info == {"delay": 3, "segment": 0, "segment_step": 1, "next_delay": None}

        # Override the sampler for the coming step
        monkeypatch.setattr(reward_delay, "rng", MockPoissonRng(2))

        # Ep 1, Seg 1, Step 3
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([3, 3, 3]))
        assert (reward, term, trunc) == (3, False, False)
        assert info == {"delay": 3, "segment": 0, "segment_step": 2, "next_delay": 2}

        # Ep 1, Seg 2, Step 1
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([4, 4, 4]))
        assert (reward, term, trunc) == (1.0, True, False)
        assert info == {"delay": 2, "segment": 1, "segment_step": 0, "next_delay": None}

        # Reset after termination
        wrapped.reset()

        # Ep 2, Seg 1, Step 1
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([1, 1, 1]))
        assert (reward, term, trunc) == (None, False, False)
        assert info == {"delay": 2, "segment": 0, "segment_step": 0, "next_delay": None}

        # Ep 2, Seg 1, Step 2
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([2, 2, 2]))
        assert (reward, term, trunc) == (2.0, False, False)
        assert info == {"delay": 2, "segment": 0, "segment_step": 1, "next_delay": 2}

        # Ep 2, Seg 2, Step 1
        obs, reward, term, trunc, info = wrapped.step(0)
        np.testing.assert_array_equal(obs, np.array([3, 3, 3]))
        assert (reward, term, trunc) == (None, False, False)
        assert info == {"delay": 2, "segment": 1, "segment_step": 0, "next_delay": None}

    def test_delayed_reward_wrapper_no_flush_on_truncation(self):
        env = DummyTruncEnv(trunc_steps=3)
        reward_delay = rewdelay.FixedDelay(2)
        wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)

        wrapped.reset()
        # Step 1: segment 0, segment_step=0
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (None, False, False)

        # Step 2: segment 0, segment_step=1 (boundary)
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (2.0, False, False)

        # Step 3: segment 1, segment_step=0 — truncated, NOT terminated
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (None, False, True)

    def test_delayed_reward_wrapper_flush_on_term_mid_window(self):
        env = DummyEnv(term_steps=5)
        reward_delay = rewdelay.FixedDelay(3)
        wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)

        wrapped.reset()
        # Steps 1-3: segment 0 (boundary at step 3)
        wrapped.step(0)
        wrapped.step(0)
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (3.0, False, False)

        # Step 4: segment 1, segment_step=0
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (None, False, False)

        # Step 5: segment 1, segment_step=1 — terminates mid-window
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (2.0, True, False)

    def test_delayed_reward_wrapper_term_at_segment_step_zero(self):
        env = DummyEnv(term_steps=3)
        reward_delay = rewdelay.FixedDelay(2)
        wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)

        wrapped.reset()
        # Steps 1-2: segment 0 (boundary)
        wrapped.step(0)
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (2.0, False, False)

        # Step 3: segment 1, segment_step=0 — terminates
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (1.0, True, False)

    def test_delayed_reward_wrapper_multi_episode_mid_window_term(self):
        env = DummyEnv(term_steps=3)
        reward_delay = rewdelay.FixedDelay(2)
        wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)

        # Episode 1
        wrapped.reset()
        wrapped.step(0)
        _, reward, _, _, _ = wrapped.step(0)
        assert reward == 2.0

        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (1.0, True, False)

        # Episode 2 — no state leakage
        wrapped.reset()
        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (None, False, False)

        _, reward, term, trunc, _ = wrapped.step(0)
        assert (reward, term, trunc) == (2.0, False, False)

    def test_delayed_reward_wrapper_delay_one(self):
        env = DummyEnv(term_steps=3)
        reward_delay = rewdelay.FixedDelay(1)
        wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)

        wrapped.reset()
        for step_idx in range(3):
            _, reward, term, trunc, _ = wrapped.step(0)
            assert reward == 1.0
            if step_idx < 2:
                assert (term, trunc) == (False, False)
            else:
                assert (term, trunc) == (True, False)


class TestDataBuffer:
    """Tests for DataBuffer — accounting modes for capacity and size-byte limits."""

    def test_data_buffer(self):
        buffer = rewdelay.DataBuffer()
        assert buffer.max_capacity is None
        assert buffer.max_size_bytes is None
        assert buffer.size() == 0
        assert buffer.size_bytes() == 0

        buffer.add(1)
        assert buffer.buffer == [1]
        assert buffer.size() == 1
        assert buffer.size_bytes() == 32

        buffer.add(1)
        buffer.add(3)
        buffer.add(5)
        buffer.add(7)
        assert buffer.buffer == [1, 1, 3, 5, 7]
        assert buffer.size() == 5
        assert buffer.size_bytes() == 64

        buffer.clear()
        assert buffer.size() == 0
        assert buffer.size_bytes() == 0

        buffer.add(1)
        assert buffer.buffer == [1]
        assert buffer.size() == 1
        assert buffer.size_bytes() == 32

    def test_data_buffer_max_capacity_with_latest_acc_mode(self):
        buffer = rewdelay.DataBuffer(
            max_capacity=10,
        )

        for value in range(100):
            buffer.add(value)
            assert buffer.size() <= 10
            # latest
            assert buffer.buffer == list(range(max(0, value - 10 + 1), value + 1))

        buffer.clear()
        assert buffer.size() == 0
        assert buffer.size_bytes() == 0

        buffer.add(1)
        assert buffer.buffer == [1]
        assert buffer.size() == 1
        assert buffer.size_bytes() == 32

    def test_data_buffer_max_capacity_with_first_acc_mode(self):
        buffer = rewdelay.DataBuffer(max_capacity=10, acc_mode="FIRST")

        for value in range(100):
            buffer.add(value)
            assert buffer.size() <= 10
            # first
            assert buffer.buffer == list(range(min(value + 1, 10)))

        buffer.clear()
        assert buffer.size() == 0
        assert buffer.size_bytes() == 0

        buffer.add(1)
        assert buffer.buffer == [1]
        assert buffer.size() == 1
        assert buffer.size_bytes() == 32

    def test_data_buffer_max_size_bytes_with_latest_acc_mode(self):
        buffer = rewdelay.DataBuffer(max_size_bytes=128)

        for value in range(100):
            buffer.add(value)
            assert buffer.size() > 0
            assert buffer.size_bytes() <= 128
            # latest
            assert buffer.buffer[-1] == value

        buffer.clear()
        assert buffer.size() == 0
        assert buffer.size_bytes() == 0

        buffer.add(1)
        assert buffer.buffer == [1]
        assert buffer.size() == 1
        assert buffer.size_bytes() == 32

    def test_data_buffer_max_size_bytes_with_first_acc_mode(self):
        buffer = rewdelay.DataBuffer(max_size_bytes=128, acc_mode="FIRST")

        for value in range(100):
            buffer.add(value)
            assert buffer.size() > 0
            assert buffer.size_bytes() <= 128
            # latest
            assert buffer.buffer[0] == 0

        buffer.clear()
        assert buffer.size() == 0
        assert buffer.size_bytes() == 0

        buffer.add(1)
        assert buffer.buffer == [1]
        assert buffer.size() == 1
        assert buffer.size_bytes() == 32

    def test_data_buffer_max_capacity_max_size_bytes_with_latest_acc_mode(self):
        buffer = rewdelay.DataBuffer(
            max_capacity=2,
            max_size_bytes=128,
        )

        for value in range(100):
            buffer.add(value)
            assert 0 < buffer.size() <= 2
            assert buffer.size_bytes() <= 128
            # latest
            assert buffer.buffer[-1] == value

        buffer.clear()
        assert buffer.size() == 0
        assert buffer.size_bytes() == 0

        buffer.add(1)
        assert buffer.buffer == [1]
        assert buffer.size() == 1
        assert buffer.size_bytes() == 32


class TestLeastLfaGenerativeRewardWrapper:
    """Tests for LeastLfaGenerativeRewardWrapper — stepping, estimation, episode-boundary buffering, segment feature accumulation, and per-step reward prediction."""

    def _make_wrapper(self, term_steps, delay):
        env = DummyEnv(term_steps=term_steps)
        ft_op = DummyFTOp(env)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=delay))
        wrapper = rewdelay.LeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=ft_op,
            attempt_estimation_episode=1,
            use_bias=False,
            check_factors=False,
        )
        return wrapper

    def _force_estimate(self, wrapper):
        """Run enough episodes to buffer data, then force estimation."""
        mdim = wrapper.mdim
        while wrapper.est_buffer.size() < mdim:
            wrapper.reset()
            done = False
            while not done:
                _, _, term, trunc, _ = wrapper.step(0)
                done = term or trunc
        wrapper.estimate_rewards()
        assert wrapper.weights is not None

    def test_least_lfa_generative_reward_wrapper_init(self):
        env = DummyEnv()
        ft_op = DummyFTOp(env)
        wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
            env, ft_op=ft_op, attempt_estimation_episode=5
        )

        assert wrapped
        assert wrapped.mdim == 4
        assert wrapped.weights is None
        assert wrapped.est_buffer.size() == 0

    def test_least_lfa_generative_reward_wrapper_step(self):
        env = DummyEnv()
        wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
            rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2)),
            ft_op=DummyFTOp(env),
            attempt_estimation_episode=2,
            use_bias=False,
        )

        obs, info = wrapped.reset()

        np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
        assert info == {"delay": 2, "segment": 0, "segment_step": -1, "next_delay": 2}

        # Ep 1, Ep Seg 1, Total Seg 1
        _, rew1, term, trunc, _ = wrapped.step(0)  # First step gets zero reward
        assert (rew1, term, trunc) == (0.0, False, False)
        _, rew2, term, trunc, _ = wrapped.step(1)  # Second step gets aggregated reward
        assert (rew2, term, trunc) == (2.0, True, False)

        wrapped.reset()

        # Ep 2, Ep Seg 1, Total Seg 2
        _, rew3, term, trunc, _ = wrapped.step(0)
        assert (rew3, term, trunc) == (0.0, False, False)
        _, rew4, term, trunc, _ = wrapped.step(2)
        assert (rew4, term, trunc) == (2.0, True, False)

        # After `attempt_estimation_episode` segments, should estimate rewards
        # but matrix isn't tall yet, so we force it after
        buffer = [
            ([1.0, -1.0, 1.0, -1.0], 2.0),
            ([1.0, -1.0, 1.0, -1.0], 2.0),
        ]
        assert wrapped.weights is None
        wrapped.estimate_rewards()
        assert wrapped.weights is not None
        np.testing.assert_equal(wrapped.est_buffer.buffer, buffer)

    def test_least_lfa_generative_reward_wrapper_invalid_spaces(self):
        env = DummyEnv()

        # Test invalid observation space
        with pytest.raises(TypeError):
            ft_op = DummyFTOp(env)
            ft_op._output_space = dataclasses.replace(
                ft_op._output_space, observation_space=spaces.Discrete(5)
            )
            rewdelay.LeastLfaGenerativeRewardWrapper(
                env,
                ft_op=ft_op,
                attempt_estimation_episode=5,
            )

        # Test invalid action space
        with pytest.raises(TypeError):
            ft_op = DummyFTOp(env)
            ft_op._output_space = dataclasses.replace(
                ft_op._output_space, action_space=spaces.Box(low=-1, high=1, shape=(1,))
            )
            rewdelay.LeastLfaGenerativeRewardWrapper(
                env,
                ft_op=ft_op,
                attempt_estimation_episode=5,
            )

    def test_least_lfa_generative_reward_wrapper_with_next_state(self):
        """Test that use_next_state=True doubles mdim and concatenates features."""

        class StateDependentFTOp(transform.FTOp):
            """FTOp that returns different vectors based on observation value."""

            def __init__(self, env: gym.Env):
                super().__init__(
                    transform.ExampleSpace(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                    )
                )
                if not isinstance(env.action_space, gym.spaces.Discrete):
                    raise TypeError(
                        f"Action space must be Discrete. Got {env.action_space}"
                    )
                self._output_space = transform.ExampleSpace(
                    observation_space=spaces.Box(low=-10, high=10, shape=(2,)),
                    action_space=env.action_space,
                )

            def apply(self, example: transform.Example) -> transform.Example:
                # Use the mean of observation vector as a unique identifier
                obs_id = np.mean(example.observation)
                # Return different features based on observation
                return transform.Example(
                    observation=np.array([obs_id, obs_id * 2]),
                    action=example.action,
                )

            @property
            def output_space(self):
                return self._output_space

        env = DummyEnv()
        ft_op = StateDependentFTOp(env)

        # Create wrapper with use_next_state=True
        wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
            rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2)),
            ft_op=ft_op,
            attempt_estimation_episode=2,
            use_bias=False,
            use_next_state=True,
        )

        # Verify mdim is doubled (2 * 2 = 4)
        assert wrapped.mdim == 4, f"Expected mdim=4, got {wrapped.mdim}"

        obs, info = wrapped.reset()
        np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
        assert info == {"delay": 2, "segment": 0, "segment_step": -1, "next_delay": 2}
        # Initial obs is [-1, -1, -1], mean = -1, features = [-1, -2]

        # Ep 1, Seg 1
        # Step 1: current state = [-1,-1,-1] (mean=-1), next state = [1,1,1] (mean=1)
        # Features should be: [-1, -2, 1, 2] (current + next)
        _, rew1, term, trunc, _ = wrapped.step(0)
        assert (rew1, term, trunc) == (0.0, False, False)

        # Step 2: current state = [1,1,1] (mean=1), next state = [2,2,2] (mean=2) - terminal
        # Features should be: [1, 2, 2, 4] (current + next)
        _, rew2, term, trunc, _ = wrapped.step(1)
        assert (rew2, term, trunc) == (2.0, True, False)

        wrapped.reset()

        # Ep 2, Seg 1
        _, rew3, term, trunc, _ = wrapped.step(0)
        assert (rew3, term, trunc) == (0.0, False, False)
        _, rew4, term, trunc, _ = wrapped.step(2)
        assert (rew4, term, trunc) == (2.0, True, False)

        # Force estimation
        wrapped.estimate_rewards()
        assert wrapped.weights is not None

        # Verify buffered features have doubled size (4 dimensions) and correct concatenation
        assert len(wrapped.est_buffer.buffer) == 2

        # First segment from Ep 1: accumulated features from two steps
        # Step 1: [-1, -2, 1, 2]
        # Step 2: [1, 2, 2, 4]
        # Accumulated (additive): [0, 0, 3, 6]
        features1, reward1 = wrapped.est_buffer.buffer[0]
        assert len(features1) == 4, f"Expected feature size=4, got {len(features1)}"
        np.testing.assert_array_almost_equal(features1, [0.0, 0.0, 3.0, 6.0])
        assert reward1 == 2.0

        # Second segment from Ep 2 should be identical
        features2, reward2 = wrapped.est_buffer.buffer[1]
        np.testing.assert_array_almost_equal(features2, [0.0, 0.0, 3.0, 6.0])
        assert reward2 == 2.0

    def test_least_lfa_partial_segment_buffered_on_termination(self):
        env = DummyEnv(term_steps=3)
        delay_env = rewdelay.DelayedRewardWrapper(
            env, reward_delay=rewdelay.FixedDelay(2)
        )
        ft_op = DummyFTOp(env)
        wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
            delay_env,
            ft_op=ft_op,
            impute_value=0.0,
            attempt_estimation_episode=100,
        )

        wrapped.reset()
        # Steps 1-2: full segment (boundary), features buffered
        wrapped.step(0)
        wrapped.step(0)
        assert wrapped.est_buffer.size() == 1

        # Step 3: terminates mid-window — partial segment buffered
        wrapped.step(0)
        assert wrapped.est_buffer.size() == 2
        _, partial_reward = wrapped.est_buffer.buffer[-1]
        assert partial_reward == 1.0
        np.testing.assert_array_equal(
            wrapped._segment_features, wrapped._initialize_segment_features()
        )

    def test_least_lfa_no_partial_buffer_on_truncation(self):
        env = DummyTruncEnv(trunc_steps=3)
        delay_env = rewdelay.DelayedRewardWrapper(
            env, reward_delay=rewdelay.FixedDelay(2)
        )
        ft_op = DummyFTOp(env)
        wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
            delay_env,
            ft_op=ft_op,
            impute_value=0.0,
            attempt_estimation_episode=100,
        )

        wrapped.reset()
        wrapped.step(0)
        wrapped.step(0)
        assert wrapped.est_buffer.size() == 1

        # Step 3: truncates mid-window — partial segment NOT buffered
        _, reward, term, trunc, _ = wrapped.step(0)
        assert wrapped.est_buffer.size() == 1
        assert (term, trunc) == (False, True)
        assert reward == 0.0

    def test_segment_features_reset_before_estimation(self):
        """Before estimation, _segment_features resets at segment boundaries."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        wrapper.reset()

        # DummyFTOp returns [0.5, -0.5, 0.5, -0.5] for every step
        per_step = np.array([0.5, -0.5, 0.5, -0.5])

        # Step 1 of segment 0: accumulate
        wrapper.step(0)
        np.testing.assert_allclose(wrapper._segment_features, per_step)

        # Step 2 of segment 0 (segment end): buffer and reset
        wrapper.step(0)
        np.testing.assert_allclose(wrapper._segment_features, np.zeros(4), atol=1e-10)

        # Step 1 of segment 1: fresh accumulation
        wrapper.step(0)
        np.testing.assert_allclose(wrapper._segment_features, per_step)

    def test_segment_features_stop_accumulating_after_estimation(self):
        """For one-shot estimators, _segment_features stops accumulating
        after estimation since the result is never buffered."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        self._force_estimate(wrapper)

        wrapper.reset()
        for _ in range(8):
            wrapper.step(0)
            np.testing.assert_allclose(
                wrapper._segment_features,
                np.zeros(4),
                atol=1e-10,
            )

    def test_predicted_reward_is_per_step_after_estimation(self):
        """After estimation, each step returns the per-step reward
        (not a cumulative sum)."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        self._force_estimate(wrapper)
        weights = wrapper.weights

        per_step = np.array([0.5, -0.5, 0.5, -0.5])
        per_step_reward = float(np.dot(per_step, weights))

        wrapper.reset()
        rewards = []
        for _ in range(8):
            _, reward, _, _, _ = wrapper.step(0)
            rewards.append(reward)

        for step_idx, reward in enumerate(rewards):
            np.testing.assert_allclose(
                reward,
                per_step_reward,
                atol=1e-6,
                err_msg=f"step {step_idx}: reward should be per-step, not cumulative",
            )

    def test_predicted_reward_constant_across_segments(self):
        """After estimation, reward is the same per-step value
        across segment boundaries."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        self._force_estimate(wrapper)
        weights = wrapper.weights
        per_step = np.array([0.5, -0.5, 0.5, -0.5])
        per_step_reward = float(np.dot(per_step, weights))

        wrapper.reset()
        # Segment 0: steps 1-2
        _, rew_seg0_step1, _, _, _ = wrapper.step(0)
        _, rew_seg0_step2, _, _, _ = wrapper.step(0)
        # Segment 1: steps 3-4
        _, rew_seg1_step1, _, _, _ = wrapper.step(0)
        _, rew_seg1_step2, _, _, _ = wrapper.step(0)

        np.testing.assert_allclose(rew_seg0_step1, per_step_reward, atol=1e-6)
        np.testing.assert_allclose(rew_seg0_step2, per_step_reward, atol=1e-6)
        np.testing.assert_allclose(rew_seg1_step1, per_step_reward, atol=1e-6)
        np.testing.assert_allclose(rew_seg1_step2, per_step_reward, atol=1e-6)

    def test_least_lfa_records_single_event(self):
        env = DummyEnv(term_steps=2)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
        seen_weights = []
        wrapper = rewdelay.LeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=DummyFTOp(env),
            attempt_estimation_episode=2,
            check_factors=False,
            reward_error_fn=stub_reward_error_fn(seen_weights),
        )
        # 2-step episodes at delay 2 buffer one segment each;
        # estimation needs 4 segments
        drive_episodes(wrapper, num_episodes=6, steps_per_episode=2)

        assert wrapper.weights is not None
        errors = wrapper.estimation_meta["reward_errors"]
        assert len(errors) == 1
        np.testing.assert_allclose(errors[0]["rmse"], 0.5)
        assert errors[0]["num_samples"] == 7
        assert errors[0]["update_index"] == 1
        # one segment per 2-step episode: the buffer reaches
        # mdim (4) at the end of the 4th episode
        assert errors[0]["episode"] == 4
        np.testing.assert_allclose(seen_weights[0], wrapper.weights)

    def test_least_lfa_without_fn_records_no_events(self):
        env = DummyEnv(term_steps=2)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
        wrapper = rewdelay.LeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=DummyFTOp(env),
            attempt_estimation_episode=2,
            check_factors=False,
        )
        drive_episodes(wrapper, num_episodes=6, steps_per_episode=2)

        assert wrapper.weights is not None
        assert wrapper.estimation_meta["reward_errors"] == []

    def test_failing_error_fn_does_not_break_estimation(self):
        """
        A raising evaluator is skipped: the measurement must not fail
        the estimation it instruments.
        """
        env = DummyEnv(term_steps=2)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))

        def failing_error_fn(weights):
            del weights
            raise ArithmeticError("measurement failure")

        wrapper = rewdelay.LeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=DummyFTOp(env),
            attempt_estimation_episode=2,
            check_factors=False,
            reward_error_fn=failing_error_fn,
        )
        drive_episodes(wrapper, num_episodes=6, steps_per_episode=2)

        assert wrapper.weights is not None
        assert wrapper.estimation_meta["reward_errors"] == []


class TestDiscretisedLeastLfaGenerativeRewardWrapper:
    """Tests per-step reward prediction in DiscretisedLeastLfaGenerativeRewardWrapper."""

    def _make_wrapper(self, term_steps, delay, num_features=4):
        env = DummyEnv(term_steps=term_steps)
        ft_op = DiscreteDummyFTOp(env, num_features=num_features)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=delay))
        wrapper = rewdelay.DiscretisedLeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=ft_op,
            attempt_estimation_episode=1,
            use_bias=False,
            check_factors=False,
        )
        return wrapper

    def _force_estimate(self, wrapper):
        mdim = wrapper.mdim
        while wrapper.est_buffer.size() < mdim:
            wrapper.reset()
            done = False
            while not done:
                _, _, term, trunc, _ = wrapper.step(0)
                done = term or trunc
        wrapper.estimate_rewards()
        assert wrapper.weights is not None

    def test_predicted_reward_is_per_step_after_estimation(self):
        """After estimation, each step returns a per-step reward (not cumulative)."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        self._force_estimate(wrapper)
        weights = wrapper.weights

        wrapper.reset()
        rewards = []
        for _ in range(8):
            _, reward, _, _, _ = wrapper.step(0)
            rewards.append(reward)

        for step_idx, reward in enumerate(rewards):
            np.testing.assert_allclose(
                reward,
                weights[step_idx % len(weights)],
                atol=1e-6,
                err_msg=f"step {step_idx}: reward should be per-step",
            )

    def test_predicted_reward_constant_across_segments(self):
        """Reward values do not grow across segment boundaries."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        self._force_estimate(wrapper)

        wrapper.reset()
        rewards = []
        for _ in range(8):
            _, reward, _, _, _ = wrapper.step(0)
            rewards.append(reward)

        # Rewards should not grow — each should be a single weight value
        for reward in rewards:
            assert abs(reward) <= np.max(np.abs(wrapper.weights)) + 1e-6


class TestBayesLeastLfaGenerativeRewardWrapper:
    """Tests for BayesLeastLfaGenerativeRewardWrapper — episode-boundary buffering, segment accumulation for posterior updates, and sampled-weights control."""

    def _make_wrapper(
        self,
        term_steps,
        delay,
        sample_weights=False,
        use_bias=False,
        schedule_rho=rewdelay.WindowedTaskSchedule.DEFAULT_RHO,
    ):
        base = DummyEnv(term_steps=term_steps)
        delayed = rewdelay.DelayedRewardWrapper(base, rewdelay.FixedDelay(delay=delay))
        ft_op = DummyFTOp(base)
        wrapper = rewdelay.BayesLeastLfaGenerativeRewardWrapper(
            env=delayed,
            ft_op=ft_op,
            init_attempt_estimation_episode=1,
            check_factors=False,
            schedule_rho=schedule_rho,
            sample_weights=sample_weights,
            use_bias=use_bias,
        )
        return wrapper

    def test_default_schedule_is_exponential(self):
        """The wrapper defaults to the exponential schedule (rho 1.05)."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        assert wrapper.mode == rewdelay.WindowedTaskSchedule.EXPONENTIAL
        assert (
            wrapper.windowed_task_schedule.rho
            == rewdelay.WindowedTaskSchedule.DEFAULT_RHO
        )

    def test_schedule_rho_passthrough(self):
        wrapper = self._make_wrapper(term_steps=8, delay=2, schedule_rho=1.2)
        assert wrapper.windowed_task_schedule.rho == 1.2

    def test_schedule_rho_changes_update_episodes(self):
        """
        The growth factor sets the episodes at which posterior
        updates fire: rho=1.05 grows windows 10, 11, 12, ...;
        rho=2 gives the doubling windows 10, 20, 40.
        """
        episodes_by_rho = {}
        for rho in (1.05, 2.0):
            env = DummyEnv(term_steps=2)
            delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
            wrapper = rewdelay.BayesLeastLfaGenerativeRewardWrapper(
                delayed,
                ft_op=DummyFTOp(env),
                init_attempt_estimation_episode=10,
                check_factors=False,
                schedule_rho=rho,
                reward_error_fn=stub_reward_error_fn([]),
            )
            drive_episodes(wrapper, num_episodes=85, steps_per_episode=2)
            episodes_by_rho[rho] = [
                event["episode"]
                for event in wrapper.estimation_meta["reward_errors"]
                if event["update_index"] > 1
            ]
        assert episodes_by_rho[1.05] == [20, 31, 43, 56, 70, 85]
        assert episodes_by_rho[2.0] == [20, 40, 80]

    def _force_estimate(self, wrapper):
        mdim = wrapper.mdim
        while wrapper.est_buffer.size() < mdim:
            wrapper.reset()
            done = False
            while not done:
                _, _, term, trunc, _ = wrapper.step(0)
                done = term or trunc
        wrapper.estimate_rewards()
        assert wrapper._has_estimate()

    def test_bayes_partial_segment_buffered_on_termination(self):
        env = DummyEnv(term_steps=3)
        delay_env = rewdelay.DelayedRewardWrapper(
            env, reward_delay=rewdelay.FixedDelay(2)
        )
        ft_op = DummyFTOp(env)
        wrapped = rewdelay.BayesLeastLfaGenerativeRewardWrapper(
            delay_env,
            ft_op=ft_op,
            impute_value=0.0,
            init_attempt_estimation_episode=100,
        )

        wrapped.reset()
        wrapped.step(0)
        wrapped.step(0)
        assert wrapped.est_buffer.size() == 1

        wrapped.step(0)
        assert wrapped.est_buffer.size() == 2
        _, partial_reward = wrapped.est_buffer.buffer[-1]
        assert partial_reward == 1.0
        np.testing.assert_array_equal(
            wrapped._segment_features, wrapped._initialize_segment_features()
        )

    def test_segment_features_accumulate_and_reset_after_estimation(self):
        """After estimation, _segment_features still accumulates within
        segments and resets at segment boundaries for continual learning."""
        wrapper = self._make_wrapper(term_steps=8, delay=2)
        self._force_estimate(wrapper)

        per_step = np.array([0.5, -0.5, 0.5, -0.5])

        wrapper.reset()
        # Step 1 of segment 0
        wrapper.step(0)
        np.testing.assert_allclose(wrapper._segment_features, per_step)

        # Step 2 of segment 0 (segment end) — resets
        wrapper.step(0)
        np.testing.assert_allclose(wrapper._segment_features, np.zeros(4), atol=1e-10)

        # Step 1 of segment 1 — fresh accumulation
        wrapper.step(0)
        np.testing.assert_allclose(wrapper._segment_features, per_step)

    def test_mean_mode_is_deterministic(self):
        wrapper = self._make_wrapper(term_steps=8, delay=2, sample_weights=False)
        self._force_estimate(wrapper)
        feats = np.array([0.5, -0.5, 0.5, -0.5])
        feats_with_est = wrapper._get_estimation_inputs(feats)
        rewards = [wrapper._get_estimated_reward(feats_with_est) for _ in range(20)]
        assert all(r == rewards[0] for r in rewards)
        # and the value is the posterior mean's dot product
        expected = float(np.dot(feats_with_est, wrapper.mv_normal_rewards.mean))
        assert rewards[0] == expected

    def test_sample_mode_stochastic_across_episodes(self):
        # one weight draw per episode: rewards are constant within an
        # episode (weights held) and vary across episodes (posterior
        # sampling, PSRL-style)
        wrapper = self._make_wrapper(term_steps=8, delay=2, sample_weights=True)
        self._force_estimate(wrapper)
        feats = np.array([0.5, -0.5, 0.5, -0.5])
        feats_with_est = wrapper._get_estimation_inputs(feats)
        within_episode = [
            wrapper._get_estimated_reward(feats_with_est) for _ in range(50)
        ]
        assert all(reward == within_episode[0] for reward in within_episode)

        across_episodes = {within_episode[0]}
        for _ in range(20):
            wrapper.reset()
            across_episodes.add(wrapper._get_estimated_reward(feats_with_est))
        # every reset redraws: 21 draws from a continuous distribution
        # are pairwise distinct with probability 1 - ~1e-14, so equality
        # is strictly stronger than `> 1` at no flakiness cost
        assert len(across_episodes) == 21

    def test_weights_redrawn_on_posterior_update(self):
        # continual updates replace the posterior object; the held weights
        # must be redrawn from the new posterior even without an episode
        # reset in between
        wrapper = self._make_wrapper(term_steps=8, delay=2, sample_weights=True)
        self._force_estimate(wrapper)
        feats = np.array([0.5, 0.5, 0.5, 0.5])
        feats_with_est = wrapper._get_estimation_inputs(feats)
        first = wrapper._get_estimated_reward(feats_with_est)

        posterior = wrapper.mv_normal_rewards
        wrapper.mv_normal_rewards = optsol.MultivariateNormal(
            mean=posterior.mean + 1e6, cov=posterior.cov
        )
        second = wrapper._get_estimated_reward(feats_with_est)
        third = wrapper._get_estimated_reward(feats_with_est)
        # the mean shift is along the all-ones direction:
        # dot(feats, 1e6 * ones) = 2e6, sampling noise is orders below
        assert abs((second - first) - 2e6) < 1e4
        # redrawn weights are held for the rest of the episode
        assert second == third

    def test_weights_held_on_estimation_failure(self, monkeypatch):
        # a failed continual update must leave the posterior object (and
        # thus the held weights) untouched: rewards continue from the
        # last draw instead of reverting or disappearing
        wrapper = self._make_wrapper(term_steps=8, delay=2, sample_weights=True)
        self._force_estimate(wrapper)
        feats = np.array([0.5, -0.5, 0.5, -0.5])
        feats_with_est = wrapper._get_estimation_inputs(feats)
        held = wrapper._get_estimated_reward(feats_with_est)
        posterior = wrapper.mv_normal_rewards

        # re-buffer one episode so a continual update is attempted
        done = False
        while not done:
            _, _, term, trunc, _ = wrapper.step(0)
            done = term or trunc

        def _raise_estimation_error(*args, **kwargs):
            raise ValueError("estimation failure")

        monkeypatch.setattr(
            optsol.MultivariateNormal,
            "bayes_linear_regression",
            _raise_estimation_error,
        )
        assert wrapper.estimate_rewards() is False
        assert wrapper.mv_normal_rewards is posterior
        assert wrapper._get_estimated_reward(feats_with_est) == held

    def test_bias_mode_sampled_rewards_dimension_consistent(self):
        # with use_bias=True the posterior gains a bias dimension and the
        # estimation inputs append 1.0: sampled weights must match that
        # dimension and remain held within the episode
        wrapper = self._make_wrapper(
            term_steps=8, delay=2, sample_weights=True, use_bias=True
        )
        self._force_estimate(wrapper)
        assert wrapper.mv_normal_rewards.mean.shape[0] == wrapper.mdim + 1
        feats = np.array([0.5, -0.5, 0.5, -0.5])
        feats_with_bias = wrapper._get_estimation_inputs(feats)
        assert feats_with_bias.shape[0] == wrapper.mdim + 1
        first = wrapper._get_estimated_reward(feats_with_bias)
        second = wrapper._get_estimated_reward(feats_with_bias)
        assert isinstance(first, float)
        assert first == second

    def test_step_rewards_stable_within_episode(self):
        # within an episode, equal features (constant DummyFTOp output)
        # map to equal step rewards: the sampled weights are held
        wrapper = self._make_wrapper(term_steps=8, delay=2, sample_weights=True)
        self._force_estimate(wrapper)
        wrapper.reset()
        _, reward_one, _, _, _ = wrapper.step(0)
        _, reward_two, _, _, _ = wrapper.step(0)
        assert reward_one is not None
        assert reward_one == reward_two

    def test_estimator_info_includes_sample_weights(self):
        wrapper = self._make_wrapper(term_steps=8, delay=2, sample_weights=True)
        info = wrapper._get_estimator_info()
        assert info["sample_weights"] is True

        wrapper_off = self._make_wrapper(term_steps=8, delay=2, sample_weights=False)
        info_off = wrapper_off._get_estimator_info()
        assert info_off["sample_weights"] is False

    def test_bayes_least_lfa_records_sequential_events(self):
        env = DummyEnv(term_steps=2)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
        seen_weights = []
        wrapper = rewdelay.BayesLeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=DummyFTOp(env),
            init_attempt_estimation_episode=2,
            check_factors=False,
            reward_error_fn=stub_reward_error_fn(seen_weights),
        )
        # prior on first full buffer, then updates on each exponential window
        drive_episodes(wrapper, num_episodes=36, steps_per_episode=2)

        errors = wrapper.estimation_meta["reward_errors"]
        assert len(errors) >= 2
        assert [event["update_index"] for event in errors] == list(
            range(1, len(errors) + 1)
        )
        episodes = [event["episode"] for event in errors]
        assert episodes == sorted(episodes)
        # events are measured against the posterior mean,
        # which is the final recorded weights
        np.testing.assert_allclose(seen_weights[-1], wrapper.mv_normal_rewards.mean)
        for event in errors:
            np.testing.assert_allclose(event["rmse"], 0.5)
            assert event["num_samples"] == 7

    def test_bayes_least_lfa_without_fn_records_no_events(self):
        env = DummyEnv(term_steps=2)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
        wrapper = rewdelay.BayesLeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=DummyFTOp(env),
            init_attempt_estimation_episode=2,
            check_factors=False,
        )
        drive_episodes(wrapper, num_episodes=10, steps_per_episode=2)

        assert wrapper._has_estimate()
        assert wrapper.estimation_meta["reward_errors"] == []

    def test_bayes_least_lfa_measures_posterior_mean_not_samples(self):
        """
        With sample_weights enabled, events record the posterior mean
        (the estimate), not the sampled weights used for control.
        """
        env = DummyEnv(term_steps=2)
        delayed = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
        seen_weights = []
        wrapper = rewdelay.BayesLeastLfaGenerativeRewardWrapper(
            delayed,
            ft_op=DummyFTOp(env),
            init_attempt_estimation_episode=2,
            check_factors=False,
            sample_weights=True,
            reward_error_fn=stub_reward_error_fn(seen_weights),
        )
        drive_episodes(wrapper, num_episodes=10, steps_per_episode=2)

        assert len(seen_weights) >= 1
        for weights in seen_weights:
            np.testing.assert_allclose(weights, wrapper.mv_normal_rewards.mean)


class TestConvexSolverGenerativeRewardWrapper:
    """Tests for ConvexSolverGenerativeRewardWrapper — initialisation, stepping, and estimation."""

    def test_convex_solver_generative_reward_wrapper_init(self):
        env = DummyEnv()
        ft_op = DummyFTOp(env)
        wrapped = rewdelay.ConvexSolverGenerativeRewardWrapper(
            env, ft_op=ft_op, attempt_estimation_episode=5
        )

        assert wrapped.mdim == 4
        assert wrapped.weights is None
        assert wrapped.est_buffer.size() == 0

    def test_convex_solver_generative_reward_wrapper_step(self):
        env = DummyEnv()
        ft_op = DummyFTOp(env)
        delay_wrapper = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
        wrapped = rewdelay.ConvexSolverGenerativeRewardWrapper(
            delay_wrapper, ft_op=ft_op, attempt_estimation_episode=2, use_bias=False
        )

        obs, info = wrapped.reset()

        np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
        assert info == {"delay": 2, "segment": 0, "segment_step": -1, "next_delay": 2}

        # Ep 1, Ep Seg 1, Total Seg 1
        _, rew1, term, trunc, _ = wrapped.step(0)  # First step gets zero reward
        assert (rew1, term, trunc) == (0.0, False, False)
        _, rew2, term, trunc, _ = wrapped.step(1)  # Second step gets aggregated reward
        assert (rew2, term, trunc) == (2.0, True, False)

        wrapped.reset()

        # Ep 2, Ep Seg 1, Total Seg 2
        _, rew3, term, trunc, _ = wrapped.step(0)
        assert (rew3, term, trunc) == (0.0, False, False)
        _, rew4, term, trunc, _ = wrapped.step(2)
        assert (rew4, term, trunc) == (2.0, True, False)

        # After `attempt_estimation_episode` segments, should estimate rewards
        # but matrix isn't tall yet, so we force it later
        buffer = [
            ([1.0, -1.0, 1.0, -1.0], 2.0),
            ([1.0, -1.0, 1.0, -1.0], 2.0),
        ]
        assert wrapped.weights is None
        wrapped.estimate_rewards()
        assert wrapped.weights is not None
        np.testing.assert_equal(wrapped.est_buffer.buffer, buffer)

    def test_convex_solver_generative_reward_wrapper_invalid_spaces(self):
        env = DummyEnv()

        # Test invalid observation space
        with pytest.raises(TypeError):
            ft_op = DummyFTOp(env)
            ft_op._output_space = dataclasses.replace(
                ft_op._output_space, observation_space=spaces.Discrete(5)
            )
            rewdelay.ConvexSolverGenerativeRewardWrapper(
                env,
                ft_op=ft_op,
                attempt_estimation_episode=5,
            )

        # Test invalid action space
        with pytest.raises(TypeError):
            ft_op = DummyFTOp(env)
            ft_op._output_space = dataclasses.replace(
                ft_op._output_space, action_space=spaces.Box(low=-1, high=1, shape=(1,))
            )
            rewdelay.ConvexSolverGenerativeRewardWrapper(
                env,
                ft_op=ft_op,
                attempt_estimation_episode=5,
            )


class TestWindowedTaskSchedule:
    """Windowed update schedules: when updates fire."""

    def _boundaries(self, schedule, max_episode):
        """Episodes at which the schedule opens a new window."""
        boundaries = []
        for episode in range(1, max_episode + 1):
            if episode == schedule.next_update_ep:
                boundaries.append(episode)
                schedule.step(episode)
        return boundaries

    def test_fixed_schedule_boundaries(self):
        schedule = rewdelay.WindowedTaskSchedule(
            mode=rewdelay.WindowedTaskSchedule.FIXED, init_update_ep=10
        )
        assert self._boundaries(schedule, 100) == [
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ]

    def test_double_schedule_boundaries(self):
        schedule = rewdelay.WindowedTaskSchedule(
            mode=rewdelay.WindowedTaskSchedule.DOUBLE, init_update_ep=10
        )
        assert self._boundaries(schedule, 500) == [20, 40, 80, 160, 320]

    def test_exponential_schedule_boundaries(self):
        """
        Windows 10, 11, 12, 13, ... — each 5% larger, rounded up to
        whole episodes.
        """
        schedule = rewdelay.WindowedTaskSchedule(
            mode=rewdelay.WindowedTaskSchedule.EXPONENTIAL,
            init_update_ep=10,
            rho=1.05,
        )
        assert self._boundaries(schedule, 101) == [20, 31, 43, 56, 70, 85, 101]

    def test_default_is_exponential(self):
        schedule = rewdelay.WindowedTaskSchedule(init_update_ep=10)
        assert schedule.mode == rewdelay.WindowedTaskSchedule.EXPONENTIAL
        assert schedule.rho == rewdelay.WindowedTaskSchedule.DEFAULT_RHO
        assert self._boundaries(schedule, 101) == [20, 31, 43, 56, 70, 85, 101]

    @pytest.mark.parametrize(
        "mode",
        (
            rewdelay.WindowedTaskSchedule.FIXED,
            rewdelay.WindowedTaskSchedule.EXPONENTIAL,
            rewdelay.WindowedTaskSchedule.DOUBLE,
        ),
    )
    def test_first_update_at_twice_init_for_all_modes(self, mode):
        """The first window spans `init_update_ep` episodes on every mode."""
        schedule = rewdelay.WindowedTaskSchedule(mode=mode, init_update_ep=7)
        assert schedule.next_update_ep == 14

    @pytest.mark.parametrize("rho", (1.0, 0.5, -1.0, float("nan"), float("inf")))
    def test_exponential_rejects_non_growth_rho(self, rho):
        with pytest.raises(ValueError, match="finite rho > 1"):
            rewdelay.WindowedTaskSchedule(
                mode=rewdelay.WindowedTaskSchedule.EXPONENTIAL,
                init_update_ep=10,
                rho=rho,
            )

    def test_init_update_ep_must_be_positive(self):
        with pytest.raises(ValueError, match="positive episode count"):
            rewdelay.WindowedTaskSchedule(
                mode=rewdelay.WindowedTaskSchedule.FIXED, init_update_ep=0
            )

    @pytest.mark.parametrize(
        "mode",
        (
            rewdelay.WindowedTaskSchedule.FIXED,
            rewdelay.WindowedTaskSchedule.DOUBLE,
        ),
    )
    def test_fixed_and_double_ignore_rho(self, mode):
        """Only the exponential mode reads rho."""
        default = rewdelay.WindowedTaskSchedule(mode=mode, init_update_ep=10)
        other = rewdelay.WindowedTaskSchedule(mode=mode, init_update_ep=10, rho=3.0)
        assert self._boundaries(default, 100) == self._boundaries(other, 100)

    def test_step_at_non_boundary_is_noop(self):
        """
        Only an exact match with `next_update_ep` opens a window; the
        wrappers rely on this exact-equality semantics.
        """
        schedule = rewdelay.WindowedTaskSchedule(
            mode=rewdelay.WindowedTaskSchedule.EXPONENTIAL,
            init_update_ep=10,
        )
        schedule.set_state(True)
        before = (
            schedule.curr_update_ep,
            schedule.next_update_ep,
            schedule._window,
            schedule.current_window_done,
        )
        for episode in (1, 5, 19, 21, 30):
            schedule.step(episode)
        after = (
            schedule.curr_update_ep,
            schedule.next_update_ep,
            schedule._window,
            schedule.current_window_done,
        )
        assert before == after

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            rewdelay.WindowedTaskSchedule(mode="quarterly", init_update_ep=10)

    def test_new_window_resets_state(self):
        schedule = rewdelay.WindowedTaskSchedule(
            mode=rewdelay.WindowedTaskSchedule.FIXED, init_update_ep=2
        )
        schedule.set_state(True)
        assert schedule.current_window_done
        schedule.step(schedule.next_update_ep)
        assert not schedule.current_window_done


def stub_reward_error_fn(seen_weights):
    """Evaluator stub recording the weights it measured."""

    def error_fn(weights):
        seen_weights.append(np.array(weights, copy=True))
        return {"rmse": 0.5, "num_samples": 7}

    return error_fn


def drive_episodes(wrapper, num_episodes, steps_per_episode):
    """Steps the wrapper through whole episodes, resetting after each."""
    wrapper.reset()
    for _ in range(num_episodes):
        for _ in range(steps_per_episode):
            wrapper.step(0)
        wrapper.reset()


class DummyFTOp(transform.FTOp):
    """
    Returns a constant vector regardless
    of the observation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(
            transform.ExampleSpace(
                observation_space=env.observation_space, action_space=env.action_space
            )
        )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError(f"Action space must be Discrete. Got {env.action_space}")
        self._output_space = transform.ExampleSpace(
            observation_space=spaces.Box(low=-1, high=1, shape=(4,)),
            action_space=env.action_space,
        )

    def apply(self, example: transform.Example) -> transform.Example:
        return transform.Example(
            # place according to action
            observation=np.array([0.5, -0.5, 0.5, -0.5]),
            action=example.action,
        )

    @property
    def output_space(self):
        return self._output_space


class DummyEnv(gym.Env):
    """
    Terminates on `term_steps`.
    The observation is a vector with the
    step count, with possible values {-1, 1, 2, ... term_steps}
    """

    def __init__(self, term_steps: int = 2):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = spaces.Discrete(3)
        self.step_count = 0
        self.term_steps = term_steps

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.ones(3) * self.step_count
        reward = 1.0

        terminated = self.step_count >= self.term_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        del seed
        del options
        self.step_count = 0
        return np.ones(3) * -1, {}


class DummyTruncEnv(gym.Env):
    """Truncates on `trunc_steps`, never terminates."""

    def __init__(self, trunc_steps: int = 3):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = spaces.Discrete(3)
        self.step_count = 0
        self.trunc_steps = trunc_steps

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.ones(3) * self.step_count
        reward = 1.0
        terminated = False
        truncated = self.step_count >= self.trunc_steps
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        del seed
        del options
        self.step_count = 0
        return np.ones(3) * -1, {}


class DiscreteDummyFTOp(transform.FTOp):
    """Returns a discrete observation index (cycles through 0..n-1)."""

    def __init__(self, env: gym.Env, num_features: int = 4):
        super().__init__(
            transform.ExampleSpace(
                observation_space=env.observation_space, action_space=env.action_space
            )
        )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError(f"Action space must be Discrete. Got {env.action_space}")
        self.num_features = num_features
        self._output_space = transform.ExampleSpace(
            observation_space=spaces.Discrete(num_features),
            action_space=env.action_space,
        )
        self._call_count = 0

    def apply(self, example: transform.Example) -> transform.Example:
        idx = self._call_count % self.num_features
        self._call_count += 1
        return transform.Example(observation=idx, action=example.action)

    @property
    def output_space(self):
        return self._output_space

    def reset_counter(self):
        self._call_count = 0
