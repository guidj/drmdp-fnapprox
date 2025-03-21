import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from drmdp.rewdelay import LeastLfaMissingWrapper, DelayedRewardWrapper, FixedDelay


class DummyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = spaces.Discrete(3)
        
    def observation(self, obs):
        return np.array([0.5, -0.5])


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = spaces.Discrete(3)
        self.step_count = 0
        
    def step(self, action):
        self.step_count += 1
        obs = np.ones(3) * self.step_count
        reward = 1.0

        terminated = self.step_count > self.observation_space.high
        truncated = False
        return obs, reward, terminated, truncated, {}
        
    def reset(self, seed=None, options=None):
        self.step_count = 0
        return np.ones(3) * -1, {}


def test_least_lfa_missing_wrapper_init():
    env = DummyEnv()
    obs_wrapper = DummyObsWrapper(env)
    wrapped = LeastLfaMissingWrapper(env, obs_wrapper, estimation_sample_size=5)
    
    assert wrapped.obs_dim == 2
    assert wrapped.mdim == 8  # 2 * 3 + 2
    assert wrapped.weights is None
    assert len(wrapped.obs_buffer) == 0
    assert len(wrapped.rew_buffer) == 0


def test_least_lfa_missing_wrapper_step():
    env = DummyEnv()
    obs_wrapper = DummyObsWrapper(env)
    delay_wrapper = DelayedRewardWrapper(env, FixedDelay(delay=2))
    wrapped = LeastLfaMissingWrapper(delay_wrapper, obs_wrapper, estimation_sample_size=2)
    
    obs, info = wrapped.reset()
    
    # First segment
    _, rew1, _, _, _ = wrapped.step(0)  # First step gets zero reward
    assert rew1 == 0.0
    _, rew2, _, _, _ = wrapped.step(1)  # Second step gets aggregated reward
    assert rew2 == 2.0
    
    # Second segment  
    _, rew3, _, _, _ = wrapped.step(0)
    assert rew3 == 0.0
    _, rew4, _, _, _ = wrapped.step(2)
    assert rew4 == 2.0
    
    # After estimation_sample_size segments, should estimate rewards
    obs_buffer = np.array([[ 0.5, -0.5,  0.5, -0.5,  0. ,  0. ,  1. , -1. ], [ 0.5, -0.5,  0. ,  0. ,  0.5, -0.5,  1. , -1. ]])
    rew_buffer = np.array([2.0,2.0])
    assert wrapped.weights is not None
    np.testing.assert_array_equal(
        wrapped.obs_buffer, obs_buffer
    )
    np.testing.assert_array_equal(
        wrapped.rew_buffer, rew_buffer
    )


def test_least_lfa_missing_wrapper_invalid_spaces():
    env = DummyEnv()
    
    # Test invalid observation space
    class InvalidObsSpace(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Discrete(5)
            self.action_space = spaces.Discrete(3)
        def observation(self, obs):
            return 0
            
    with pytest.raises(ValueError):
        LeastLfaMissingWrapper(env, InvalidObsSpace(env), estimation_sample_size=5)
    
    # Test invalid action space  
    class InvalidActSpace(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        def observation(self, obs):
            return np.zeros(2)
            
    with pytest.raises(ValueError):
        LeastLfaMissingWrapper(env, InvalidActSpace(env), estimation_sample_size=5)
