import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from drmdp import rewdelay


class DummyObsWrapper(gym.ObservationWrapper):
    """
    Returns a constant vector regardless of the
    observation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = spaces.Discrete(3)

    def observation(self, obs):
        del obs
        return np.array([0.5, -0.5])


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
        self.step_count = 0
        return np.ones(3) * -1, {}


def test_least_lfa_generative_reward_wrapper_init():
    env = DummyEnv()
    obs_wrapper = DummyObsWrapper(env)
    wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
        env, obs_wrapper, attempt_estimation_episode=5
    )

    assert wrapped.obs_dim == 2
    assert wrapped.mdim == 8  # 2 * 3 + 2
    assert wrapped.weights is None
    assert len(wrapped.obs_buffer) == 0
    assert len(wrapped.rew_buffer) == 0


def test_least_lfa_generative_reward_wrapper_step():
    env = DummyEnv()
    obs_wrapper = DummyObsWrapper(env)
    delay_wrapper = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
    wrapped = rewdelay.LeastLfaGenerativeRewardWrapper(
        delay_wrapper, obs_wrapper, attempt_estimation_episode=2, use_bias=False
    )

    obs, info = wrapped.reset()

    np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
    assert info == {"delay": 2, "segment": 0, "segment_step": -1}

    # First segment
    _, rew1, term, trunc, _ = wrapped.step(0)  # First step gets zero reward
    assert (rew1, term, trunc) == (0.0, False, False)
    _, rew2, term, trunc, _ = wrapped.step(1)  # Second step gets aggregated reward
    assert (rew2, term, trunc) == (2.0, True, False)

    wrapped.reset()

    # Second segment
    _, rew3, term, trunc, _ = wrapped.step(0)
    assert (rew3, term, trunc) == (0.0, False, False)
    _, rew4, term, trunc, _ = wrapped.step(2)
    assert (rew4, term, trunc) == (2.0, True, False)

    # After estimation_sample_size segments, should estimate rewards
    obs_buffer = np.array(
        [
            [0.5, -0.5, 0.5, -0.5, 0.0, 0.0, 1.0, -1.0],
            [0.5, -0.5, 0.0, 0.0, 0.5, -0.5, 1.0, -1.0],
        ]
    )
    rew_buffer = np.array([2.0, 2.0])
    assert wrapped.weights is not None
    np.testing.assert_array_equal(wrapped.obs_buffer, obs_buffer)
    np.testing.assert_array_equal(wrapped.rew_buffer, rew_buffer)


def test_least_lfa_generative_reward_wrapper_invalid_spaces():
    env = DummyEnv()

    # Test invalid observation space
    class InvalidObsSpace(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Discrete(5)
            self.action_space = spaces.Discrete(3)

        def observation(self, obs):
            del obs
            return 0

    with pytest.raises(ValueError):
        rewdelay.LeastLfaGenerativeRewardWrapper(
            env, InvalidObsSpace(env), attempt_estimation_episode=5
        )

    # Test invalid action space
    class InvalidActSpace(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

        def observation(self, obs):
            del obs
            return np.zeros(2)

    with pytest.raises(ValueError):
        rewdelay.LeastLfaGenerativeRewardWrapper(
            env, InvalidActSpace(env), attempt_estimation_episode=5
        )


def test_convex_solver_generative_reward_wrapper_init():
    env = DummyEnv()
    obs_wrapper = DummyObsWrapper(env)
    wrapped = rewdelay.ConvexSolverGenerativeRewardWrapper(
        env, obs_wrapper, attempt_estimation_episode=5
    )

    assert wrapped.obs_dim == 2
    assert wrapped.mdim == 8  # 2 * 3 + 2
    assert wrapped.weights is None
    assert len(wrapped.obs_buffer) == 0
    assert len(wrapped.rew_buffer) == 0


def test_convex_solver_generative_reward_wrapper_step():
    env = DummyEnv()
    obs_wrapper = DummyObsWrapper(env)
    delay_wrapper = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(delay=2))
    wrapped = rewdelay.ConvexSolverGenerativeRewardWrapper(
        delay_wrapper, obs_wrapper, attempt_estimation_episode=2, use_bias=False
    )

    obs, info = wrapped.reset()

    np.testing.assert_array_equal(obs, np.array([-1, -1, -1]))
    assert info == {"delay": 2, "segment": 0, "segment_step": -1}

    # First segment
    _, rew1, term, trunc, _ = wrapped.step(0)  # First step gets zero reward
    assert (rew1, term, trunc) == (0.0, False, False)
    _, rew2, term, trunc, _ = wrapped.step(1)  # Second step gets aggregated reward
    assert (rew2, term, trunc) == (2.0, True, False)

    wrapped.reset()

    # Second segment
    _, rew3, term, trunc, _ = wrapped.step(0)
    assert (rew3, term, trunc) == (0.0, False, False)
    _, rew4, term, trunc, _ = wrapped.step(2)
    assert (rew4, term, trunc) == (2.0, True, False)

    # After estimation_sample_size segments, should estimate rewards
    obs_buffer = np.array(
        [
            [0.5, -0.5, 0.5, -0.5, 0.0, 0.0, 1.0, -1.0],
            [0.5, -0.5, 0.0, 0.0, 0.5, -0.5, 1.0, -1.0],
        ]
    )
    rew_buffer = np.array([2.0, 2.0])
    assert wrapped.weights is not None
    np.testing.assert_array_equal(wrapped.obs_buffer, obs_buffer)
    np.testing.assert_array_equal(wrapped.rew_buffer, rew_buffer)


def test_convex_solver_generative_reward_wrapper_invalid_spaces():
    env = DummyEnv()

    # Test invalid observation space
    class InvalidObsSpace(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Discrete(5)
            self.action_space = spaces.Discrete(3)

        def observation(self, obs):
            del obs
            return 0

    with pytest.raises(ValueError):
        rewdelay.ConvexSolverGenerativeRewardWrapper(
            env, InvalidObsSpace(env), attempt_estimation_episode=5
        )

    # Test invalid action space
    class InvalidActSpace(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

        def observation(self, obs):
            del obs
            return np.zeros(2)

    with pytest.raises(ValueError):
        rewdelay.ConvexSolverGenerativeRewardWrapper(
            env, InvalidActSpace(env), attempt_estimation_episode=5
        )


def test_delayed_reward_wrapper_init():
    env = DummyObsWrapper(DummyEnv())
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

    assert isinstance(wrapped.reward_delay, rewdelay.FixedDelay)
    assert wrapped.reward_delay.delay == 2
    assert wrapped.observation_space == env.observation_space
    assert wrapped.action_space == env.action_space
    assert wrapped.segment is None
    assert wrapped.segment_step is None
    assert wrapped.delay is None
    assert wrapped.op(range(10)) == sum(range(10))


def test_delayed_reward_wrapper_step():
    env = DummyObsWrapper(DummyEnv(term_steps=4))
    wrapped = rewdelay.DelayedRewardWrapper(env, reward_delay=rewdelay.FixedDelay(2))

    wrapped.reset()
    # Seg 1, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([0.5, -0.5]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 0}

    # Ep 1, Seg 1, Step 2
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([0.5, -0.5]))
    assert (reward, term, trunc) == (2.0, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 1}

    # Ep 1, Seg 2, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([0.5, -0.5]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 1, "segment_step": 0}

    # Ep 1, Seg 2, Step 2
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([0.5, -0.5]))
    assert (reward, term, trunc) == (2.0, True, False)
    assert info == {"delay": 2, "segment": 1, "segment_step": 1}

    # Reset after termination
    wrapped.reset()

    # Ep 2, Seg 1, Step 1
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_array_equal(obs, np.array([0.5, -0.5]))
    assert (reward, term, trunc) == (None, False, False)
    assert info == {"delay": 2, "segment": 0, "segment_step": 0}


def test_delayed_reward_wrapper_reset():
    env = DummyObsWrapper(DummyEnv())
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
