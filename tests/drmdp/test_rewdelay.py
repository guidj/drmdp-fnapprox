import dataclasses

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from drmdp import rewdelay, transform


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
            raise ValueError(f"Action space must be Discrete. Got {env.action_space}")
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


def test_least_lfa_generative_reward_wrapper_init():
    env = DummyEnv()
    ft_op = DummyFTOp(env)
    wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
        env, ft_op=ft_op, attempt_estimation_episode=5
    )

    assert wrapped
    assert wrapped.mdim == 4
    assert wrapped.weights is None
    assert wrapped.est_buffer.size() == 0


def test_least_lfa_generative_reward_wrapper_step():
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


def test_least_lfa_generative_reward_wrapper_invalid_spaces():
    env = DummyEnv()

    # Test invalid observation space
    with pytest.raises(ValueError):
        ft_op = DummyFTOp(env)
        setattr(
            ft_op,
            "_output_space",
            dataclasses.replace(
                getattr(ft_op, "_output_space"), observation_space=spaces.Discrete(5)
            ),
        )
        rewdelay.LeastLfaGenerativeRewardWrapper(
            env,
            ft_op=ft_op,
            attempt_estimation_episode=5,
        )

    # Test invalid action space
    with pytest.raises(ValueError):
        ft_op = DummyFTOp(env)
        setattr(
            ft_op,
            "_output_space",
            dataclasses.replace(
                getattr(ft_op, "_output_space"),
                action_space=spaces.Box(low=-1, high=1, shape=(1,)),
            ),
        )
        rewdelay.LeastLfaGenerativeRewardWrapper(
            env,
            ft_op=ft_op,
            attempt_estimation_episode=5,
        )


def test_least_lfa_generative_reward_wrapper_with_next_state():
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
                raise ValueError(
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


def test_convex_solver_generative_reward_wrapper_init():
    env = DummyEnv()
    ft_op = DummyFTOp(env)
    wrapped = rewdelay.ConvexSolverGenerativeRewardWrapper(
        env, ft_op=ft_op, attempt_estimation_episode=5
    )

    assert wrapped.mdim == 4
    assert wrapped.weights is None
    assert wrapped.est_buffer.size() == 0


def test_convex_solver_generative_reward_wrapper_step():
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


def test_convex_solver_generative_reward_wrapper_invalid_spaces():
    env = DummyEnv()

    # Test invalid observation space
    with pytest.raises(ValueError):
        ft_op = DummyFTOp(env)
        setattr(
            ft_op,
            "_output_space",
            dataclasses.replace(
                getattr(ft_op, "_output_space"), observation_space=spaces.Discrete(5)
            ),
        )
        rewdelay.ConvexSolverGenerativeRewardWrapper(
            env,
            ft_op=ft_op,
            attempt_estimation_episode=5,
        )

    # Test invalid action space
    with pytest.raises(ValueError):
        ft_op = DummyFTOp(env)
        setattr(
            ft_op,
            "_output_space",
            dataclasses.replace(
                getattr(ft_op, "_output_space"),
                action_space=spaces.Box(low=-1, high=1, shape=(1,)),
            ),
        )
        rewdelay.ConvexSolverGenerativeRewardWrapper(
            env,
            ft_op=ft_op,
            attempt_estimation_episode=5,
        )


def test_delayed_reward_wrapper_with_fixed_delay_init():
    env = DummyEnv()
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

    assert isinstance(wrapped.reward_delay, rewdelay.FixedDelay)
    assert wrapped.reward_delay.delay == 2
    assert wrapped.observation_space == env.observation_space
    assert wrapped.action_space == env.action_space
    assert wrapped.segment is None
    assert wrapped.segment_step is None
    assert wrapped.delay is None
    assert wrapped.op(range(10)) == sum(range(10))


def test_delayed_reward_wrapper_with_fixed_delay_step():
    env = DummyEnv(term_steps=4)
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

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


def test_delayed_reward_wrapper_with_fixed_delay_reset():
    env = DummyEnv()
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

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

    # Seg 2, Step 1
    wrapped.step(0)
    assert wrapped.segment == 1
    assert wrapped.segment_step == 0
    assert wrapped.rewards == [1]

    # Reset
    wrapped.reset()
    assert wrapped.segment == 0
    assert wrapped.segment_step == -1
    assert not wrapped.rewards


def test_delayed_reward_wrapper_with_poisson_delay_init():
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


def test_delayed_reward_wrapper_with_poisson_delay_step(monkeypatch):
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
    assert (reward, term, trunc) == (None, True, False)
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


def test_data_buffer():
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


def test_data_buffer_max_capacity_with_latest_acc_mode():
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


def test_data_buffer_max_capacity_with_first_acc_mode():
    buffer = rewdelay.DataBuffer(max_capacity=10, acc_mode="FIRST")

    for value in range(100):
        buffer.add(value)
        assert buffer.size() <= 10
        # first
        assert buffer.buffer == list(range(0, min(value + 1, 10)))

    buffer.clear()
    assert buffer.size() == 0
    assert buffer.size_bytes() == 0

    buffer.add(1)
    assert buffer.buffer == [1]
    assert buffer.size() == 1
    assert buffer.size_bytes() == 32


def test_data_buffer_max_size_bytes_with_latest_acc_mode():
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


def test_data_buffer_max_size_bytes_with_first_acc_mode():
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


def test_data_buffer_max_capacity_max_size_bytes_with_latest_acc_mode():
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
