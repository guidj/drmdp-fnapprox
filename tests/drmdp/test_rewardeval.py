"""Tests for rewardeval — true-reward error measurement of estimated rewards."""

import json
import math
import os

import gymnasium as gym
import numpy as np
import pytest

from drmdp import core, optsol, rewardeval, rewardfeat, task, transform
from drmdp.envs import gympg, models
from drmdp.workflows import controlexps
from tests.drmdp.hookhelpers import FTOpHook

CRAFTED_GRID = ["soooo", "ooooo", "oooxg"]


class TestModelProviderEnumeration:
    def test_enumerates_declared_model_in_state_and_action_order(self):
        env = _ModelEnv()
        features, rewards = rewardeval.discrete_state_action_rewards(
            env, _SuccessorAndTerminationFeatures()
        )
        np.testing.assert_allclose(
            features, [[0.25, 0.0], [1.5, 0.0], [3.75, 1.0], [0.25, 0.0]], atol=1e-6
        )
        np.testing.assert_allclose(rewards, [0.25, 1.5, 3.75, 0.25], atol=1e-6)

    def test_enumeration_preserves_observation_dtype(self):
        features, _ = rewardeval.discrete_state_action_rewards(
            _ModelEnv(), _ObservationDtypeFeatures()
        )
        np.testing.assert_allclose(features[:, 0], [0.25, 1.5, 3.75, 0.25], atol=1e-6)

    def test_measure_uses_model_without_stepping_and_caches_features(self):
        env = _ModelEnv()
        hook = _CountingSuccessorFeatures()
        measure = rewardeval.true_reward_error_fn(env, hook, num_samples=17)
        first = measure(np.ones(1))
        second = measure(np.zeros(1))
        np.testing.assert_allclose(first["rmse"], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            second["rmse"],
            np.sqrt((0.25**2 + 1.5**2 + 3.75**2 + 0.25**2) / 4),
            atol=1e-6,
        )
        assert first["num_samples"] == second["num_samples"] == 4
        assert hook.calls == 4
        assert env.steps == 0

    def test_time_limits_enumerate_the_underlying_mdp(self):
        env = gym.wrappers.TimeLimit(
            gym.wrappers.TimeLimit(_ModelEnv(), max_episode_steps=2),
            max_episode_steps=1,
        )
        report = rewardeval.true_reward_error_fn(env, _SuccessorValueFeatures())(
            np.ones(1)
        )
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
        assert report["num_samples"] == 4
        assert env.unwrapped.steps == 0

    @pytest.mark.parametrize("wrapper_name", ["reward", "time-limit"])
    def test_wrappers_cannot_expose_an_unmodified_inner_model(self, wrapper_name):
        wrapper_type = {
            "reward": _RewardShiftWrapper,
            "time-limit": _RewardShiftTimeLimit,
        }[wrapper_name]
        env = wrapper_type(_ModelEnv())
        assert not rewardeval.supports_discrete_enumeration(env)
        report = rewardeval.true_reward_error_fn(
            env, _ConstantAndSuccessorFeatures(), num_samples=17, seed=42
        )(np.array([5.0, 1.0]))
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
        assert report["num_samples"] == 17

    def test_subclass_cannot_inherit_a_model_without_redeclaring_it(self):
        env = _RewardShiftModelEnv()
        assert not rewardeval.supports_discrete_enumeration(env)
        report = rewardeval.true_reward_error_fn(
            env, _ConstantAndSuccessorFeatures(), num_samples=17, seed=42
        )(np.array([5.0, 1.0]))
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
        assert report["num_samples"] == 17

    def test_provider_can_decline_enumeration(self):
        env = _DecliningModelEnv()
        assert not rewardeval.supports_discrete_enumeration(env)
        report = rewardeval.true_reward_error_fn(
            env, _SuccessorValueFeatures(), num_samples=17, seed=42
        )(np.ones(1))
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
        assert report["num_samples"] == 17

    def test_stochastic_declared_models_use_realized_steps(self):
        env = _StochasticModelEnv()
        assert not rewardeval.supports_discrete_enumeration(env)
        with pytest.raises(ValueError, match="deterministic"):
            rewardeval.discrete_state_action_rewards(env, _SuccessorValueFeatures())
        report = rewardeval.true_reward_error_fn(
            env, _SuccessorValueFeatures(), num_samples=128, seed=42
        )(np.ones(1))
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
        assert report["num_samples"] == 128

    def test_no_nonterminal_pairs_raises(self):
        env = _ModelEnv()
        env.terminal_state_ids = frozenset(env.transition)
        with pytest.raises(ValueError, match="no nonterminal state-action pairs"):
            rewardeval.discrete_state_action_rewards(env, _SuccessorValueFeatures())


class TestDiscreteStateActionRewards:
    """Discrete enumeration over the transition table."""

    def test_enumerates_live_state_action_pairs(self):
        """Every live (non-absorbing) state's pairs, absorbing excluded."""
        proxy, ft_op = self._mines_setup()
        features, true_rewards = rewardeval.discrete_state_action_rewards(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op)
        )
        transition = proxy.get_wrapper_attr("transition")
        live_states = [
            state_id
            for state_id in sorted(transition)
            if state_id not in proxy.get_wrapper_attr("terminal_state_ids")
        ]
        expected_rows = len(live_states) * proxy.action_space.n
        assert features.shape[0] == expected_rows
        assert true_rewards.shape == (expected_rows,)
        # the exit cell is the grid's only absorbing state
        assert len(live_states) == len(transition) - 1
        proxy.close()

    def test_rewards_agree_with_random_policy_steps(self):
        """
        Every reward in the enumeration matches the env's actual reward
        for the same (position, action).
        """
        proxy, ft_op = self._mines_setup()
        _, true_rewards = rewardeval.discrete_state_action_rewards(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op)
        )
        states_mapping = proxy.get_wrapper_attr("states_mapping")
        transition = proxy.get_wrapper_attr("transition")
        live_states = [
            state_id
            for state_id in sorted(transition)
            if state_id not in proxy.get_wrapper_attr("terminal_state_ids")
        ]
        row_of_state = {
            state_id: row_idx * proxy.action_space.n
            for row_idx, state_id in enumerate(live_states)
        }

        def reward_for(position, action):
            state_id = states_mapping[tuple(position)]
            return true_rewards[row_of_state[state_id] + action]

        proxy.action_space.seed(0)
        obs, _ = proxy.reset(seed=0)
        checked = 0
        while checked < 200:
            action = proxy.action_space.sample()
            next_obs, reward, term, trunc, _ = proxy.step(action)
            assert reward_for(obs, action) == reward
            checked += 1
            obs = next_obs
            if term or trunc:
                obs, _ = proxy.reset()
        proxy.close()

    def test_exact_recovery_by_least_squares(self):
        """
        The enumeration is linearly representable in its own features:
        solving least squares on it yields zero error.
        """
        proxy, ft_op = self._mines_setup()
        features, true_rewards = rewardeval.discrete_state_action_rewards(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op)
        )
        weights = optsol.solve_least_squares(matrix=features, rhs=true_rewards)
        measure = rewardeval.true_reward_error_fn(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op)
        )
        report = measure(weights)
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-10)
        proxy.close()

    def _mines_setup(self):
        result = task.create_env(
            "GridWorld-MINES",
            {"grid": controlexps.MINES_GW_GRID, "max_episode_steps": 200},
        )
        proxy = result.proxy
        ft_op = task.create_ft_ops(
            proxy,
            feats_spec=[{"name": "flat-grid-observation-action-ft", "args": {}}],
        )
        return proxy, ft_op


class TestTrueRewardErrorFn:
    """The returned callable measures weights against true rewards."""

    def test_zero_weights_rmse(self):
        """
        With zero weights every prediction is 0, so the rmse equals
        the root mean square of the true rewards.
        """
        proxy, ft_op = self._crafted_setup()
        _, true_rewards = rewardeval.discrete_state_action_rewards(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op)
        )
        expected = float(np.sqrt(np.mean(np.square(true_rewards))))
        measure = rewardeval.true_reward_error_fn(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op)
        )
        report = measure(np.zeros(weights_dim(ft_op)))
        np.testing.assert_allclose(report["rmse"], expected, atol=1e-10)
        assert report["num_samples"] == len(true_rewards)
        proxy.close()

    def test_repeated_calls_are_deterministic(self):
        proxy, ft_op = self._crafted_setup()
        measure = rewardeval.true_reward_error_fn(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op)
        )
        weights = np.ones(weights_dim(ft_op))
        first = measure(weights)
        second = measure(weights)
        assert first == second
        proxy.close()

    def test_continuous_sampler_seeded_determinism(self):
        result = task.create_env("MountainCar-v0", {"max_episode_steps": 200})
        proxy = result.proxy
        ft_op = task.create_ft_ops(
            proxy,
            feats_spec=[
                {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
            ],
        )
        measure = rewardeval.true_reward_error_fn(
            proxy_env=proxy,
            feats_fn=FTOpHook(ft_op),
            num_samples=100,
            seed=0,
        )
        weights = np.zeros(weights_dim(ft_op))
        first = measure(weights)
        second = measure(weights)
        assert first == second
        assert first["num_samples"] == 100
        proxy.close()

    def test_continuous_rewards_are_env_rewards(self):
        """
        Unseeded sampling on the proxy yields the env's true per-step
        rewards; MountainCar's are -1 except at the goal.
        """
        result = task.create_env("MountainCar-v0", {"max_episode_steps": 200})
        proxy = result.proxy
        ft_op = task.create_ft_ops(
            proxy,
            feats_spec=[
                {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
            ],
        )
        _, true_rewards = rewardeval.random_policy_rewards(
            proxy_env=proxy, feats_fn=FTOpHook(ft_op), num_samples=500
        )
        assert len(true_rewards) == 500
        # dense -1 rewards with at most a handful of goal transitions (0)
        assert np.count_nonzero(true_rewards == -1.0) >= 490
        assert set(true_rewards.tolist()).issubset({-1.0, 0.0})
        proxy.close()

    def _crafted_setup(self):
        result = task.create_env(
            "GridWorld-CRAFTED", {"grid": CRAFTED_GRID, "max_episode_steps": 100}
        )
        proxy = result.proxy
        dead_ohe = controlexps.grid_dead_ohe_indices(CRAFTED_GRID)
        ft_op = task.create_ft_ops(
            proxy,
            feats_spec=[
                {"name": "flat-grid-observation-action-ft", "args": {}},
                {
                    "name": "drop-observation-dims-ft",
                    "args": {"axis_dims": {0: dead_ohe}},
                },
            ],
        )
        return proxy, ft_op


class TestEvaluationDomain:
    def test_time_limited_gridworld_retains_table_enumeration(self):
        proxy = task.create_env(
            "GridWorld-test", {"grid": CRAFTED_GRID, "max_episode_steps": 100}
        ).proxy
        try:
            assert rewardeval.supports_discrete_enumeration(proxy)
        finally:
            proxy.close()

    def test_shaped_gridworld_scores_proxy_rewards_by_sampling(self):
        costs = np.array([0.0, 0.5, 1.0, 1.5])
        proxy = task.create_env(
            "GridWorld-test",
            {
                "grid": ["soo", "ooo", "oog"],
                "max_episode_steps": 100,
                "reward_shaping": {
                    "name": "action-cost",
                    "args": {"action_costs": costs},
                },
            },
        ).proxy
        try:
            assert not rewardeval.supports_discrete_enumeration(proxy)
            hook = rewardfeat.ActionTerminationFeatures(4)
            weights = np.concatenate([costs - 1.0, -np.ones(4)])
            report = rewardeval.true_reward_error_fn(
                proxy, hook, num_samples=128, seed=42
            )(weights)
            np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
            assert report["num_samples"] == 128
        finally:
            proxy.close()

    def test_stochastic_successor_and_reward_are_scored_together(self):
        env = _StochasticTableEnv()
        report = rewardeval.true_reward_error_fn(
            env, _SuccessorValueFeatures(), num_samples=128, seed=42
        )(np.ones(1))
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
        assert report["num_samples"] == 128

    def test_nonterminal_self_loops_remain_in_the_evaluation_domain(self):
        env = _NonterminalSelfLoopEnv()
        report = rewardeval.true_reward_error_fn(
            env, rewardfeat.NonterminalActionFeatures(1), num_samples=17, seed=42
        )(np.array([2.0]))
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
        assert report["num_samples"] == 17

    def test_gridworld_subclasses_with_unverified_rewards_are_sampled(self):
        env = gympg.GridWorldObsAsVectorWrapper(
            _RewardShiftGridWorld(size=(2, 2), cliffs=(), exits=((0, 1),), start=(1, 0))
        )
        try:
            hook = rewardfeat.ConstantAndSuccessorBasisFeatures(
                lambda obs: np.zeros(1), 1
            )
            report = rewardeval.true_reward_error_fn(
                env, hook, num_samples=17, seed=42
            )(np.array([4.0, 0.0]))
            np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-6)
            assert report["num_samples"] == 17
        finally:
            env.close()

    def test_stochastic_table_on_gridworld_is_not_enumerated(self):
        env = gympg.make("GridWorld-test", grid=["soo", "oog"])
        try:
            table = env.get_wrapper_attr("transition")
            table[0][0] = [(0.75, 0, -1.0, False), (0.25, 1, -1.0, False)]
            assert not rewardeval.supports_discrete_enumeration(env)
            with pytest.raises(ValueError, match="deterministic"):
                rewardeval.discrete_state_action_rewards(
                    env, rewardfeat.ActionTerminationFeatures(4)
                )
        finally:
            env.close()

    @pytest.mark.parametrize("num_samples", [0, -1])
    def test_sampling_requires_a_positive_sample_count(self, num_samples):
        with pytest.raises(ValueError, match="num_samples must be positive"):
            rewardeval.random_policy_rewards(
                _StochasticTableEnv(),
                _SuccessorValueFeatures(),
                num_samples=num_samples,
            )

    def test_direct_enumeration_rejects_an_unrecognized_model(self):
        with pytest.raises(ValueError, match="declared transition model"):
            rewardeval.discrete_state_action_rewards(
                _StochasticTableEnv(), _SuccessorValueFeatures()
            )


class TestStepFeatures:
    """Per-step features mirror the estimator's fitted basis."""

    def test_bias_appends_constant_column(self):
        result = task.create_env("MountainCar-v0", {"max_episode_steps": 200})
        proxy = result.proxy
        ft_op = task.create_ft_ops(
            proxy,
            feats_spec=[
                {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
            ],
        )
        obs = np.array([0.0, 0.0])
        feats = rewardeval.step_features(
            feats_fn=FTOpHook(ft_op), obs=obs, action=1, next_obs=obs, intercept=True
        )
        assert feats.shape[0] == weights_dim(ft_op) + 1
        assert feats[-1] == 1.0
        proxy.close()

    def test_without_bias_no_constant_is_appended(self):
        result = task.create_env("MountainCar-v0", {"max_episode_steps": 200})
        proxy = result.proxy
        ft_op = task.create_ft_ops(
            proxy,
            feats_spec=[
                {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
            ],
        )
        obs = np.array([0.0, 0.0])
        feats = rewardeval.step_features(
            feats_fn=FTOpHook(ft_op), obs=obs, action=1, next_obs=obs
        )
        assert feats.shape[0] == weights_dim(ft_op)
        proxy.close()


class TestPolicyControlRewardErrorLogging:
    """End-to-end: events land in the written experiment logs."""

    def test_least_lfa_single_event_near_zero_rmse(self, tmp_path):
        # exact RMSE requires full state-action coverage before LEAST's
        # single fit: minimum-norm least squares assigns 0 to unvisited
        # columns, so an uncovered live pair would break rmse ~ 0.
        # 80 episodes on this grid cover all 13 live states' pairs; the
        # exit state is absorbing and excluded from the enumeration.
        events = self._run_control("least-lfa", tmp_path, episodes=80)
        assert len(events) == 1
        event = events[0]
        assert event["update_index"] == 1
        assert event["episode"] <= 80
        assert event["num_samples"] == 52  # 13 live states x 4 actions
        np.testing.assert_allclose(event["rmse"], 0.0, atol=1e-6)

    def test_bayes_least_lfa_sequential_events(self, tmp_path):
        events = self._run_control("bayes-least-lfa", tmp_path, episodes=80)
        assert len(events) >= 2
        assert [event["update_index"] for event in events] == list(
            range(1, len(events) + 1)
        )
        episodes = [event["episode"] for event in events]
        assert episodes == sorted(episodes)
        assert all(event["rmse"] >= 0 for event in events)
        np.testing.assert_allclose(events[-1]["rmse"], 0.0, atol=1e-6)

    def _run_control(self, mapper_name, tmp_path, episodes):
        run_dir = os.path.join(str(tmp_path), mapper_name)
        dead_ohe = controlexps.grid_dead_ohe_indices(CRAFTED_GRID)
        attempt_arg = (
            {"attempt_estimation_episode": 10}
            if mapper_name == "least-lfa"
            else {"init_attempt_estimation_episode": 10}
        )
        exp_instance = core.ExperimentInstance(
            exp_id=f"test-{mapper_name}",
            instance_id=0,
            experiment=core.Experiment(
                env_spec=core.EnvSpec(
                    name="GridWorld-CRAFTED",
                    args={"grid": CRAFTED_GRID, "max_episode_steps": 100},
                    feats_spec=[
                        {
                            "name": "tile-observation-action-ft",
                            "args": {"tiling_dim": 5},
                        }
                    ],
                ),
                problem_spec=core.ProblemSpec(
                    policy_type="markovian",
                    reward_mapper={
                        "name": mapper_name,
                        "args": {
                            **attempt_arg,
                            "transition_feats_spec": {
                                "name": "current-features",
                                "args": {
                                    "current_spec": [
                                        {
                                            "name": "flat-grid-observation-action-ft",
                                            "args": {},
                                        },
                                        {
                                            "name": "drop-observation-dims-ft",
                                            "args": {"axis_dims": {0: dead_ohe}},
                                        },
                                    ]
                                },
                            },
                            "impute_value": 0,
                            "estimation_buffer_mult": 25,
                        },
                    },
                    delay_config={
                        "name": "clipped-poisson",
                        "args": {"lam": 2, "min_delay": 2, "max_delay": 4},
                    },
                    epsilon=0.2,
                    gamma=0.99,
                    learning_rate_config={
                        "name": "constant",
                        "args": {"initial_lr": 0.01},
                    },
                ),
                epochs=1,
            ),
            run_config=core.RunConfig(
                num_runs=1,
                episodes_per_run=episodes,
                log_episode_frequency=5,
                use_seed=True,
                output_dir=run_dir,
            ),
            context={"dummy": 0},
            export_model=False,
        )
        task.policy_control(exp_instance)
        log_file = os.path.join(run_dir, "experiment-logs.jsonl")
        return read_reward_errors(log_file)


class _ModelEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 4, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.observations = {
            41: np.array([3.75], dtype=np.float32),
            19: np.array([1.5], dtype=np.float32),
            7: np.array([0.25], dtype=np.float32),
        }
        self.transition = {
            41: {0: [(1.0, 41, 99.0, False)], 1: [(1.0, 41, 99.0, False)]},
            19: {1: [(1.0, 7, 0.25, False)], 0: [(1.0, 41, 3.75, True)]},
            7: {
                1: [(1.0, 19, 1.5, False)],
                0: [(0.0, 41, 99.0, True), (1.0, 7, 0.25, False)],
            },
        }
        self.terminal_state_ids = frozenset({41})
        self.state = 7
        self.steps = 0

    def discrete_transition_model(self) -> models.DiscreteTransitionModel | None:
        return models.DiscreteTransitionModel(
            observations=self.observations,
            transition=self.transition,
            terminal_state_ids=self.terminal_state_ids,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 7
        return self.observations[self.state].copy(), {}

    def step(self, action):
        entries = self.transition[self.state][action]
        index = int(
            self.np_random.choice(len(entries), p=[entry[0] for entry in entries])
        )
        _, self.state, reward, terminated = entries[index]
        self.steps += 1
        return self.observations[self.state].copy(), reward, terminated, False, {}


class _RewardShiftModelEnv(_ModelEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward + 5.0, terminated, truncated, info


class _DecliningModelEnv(_ModelEnv):
    def discrete_transition_model(self) -> None:
        return None


class _StochasticModelEnv(_ModelEnv):
    def __init__(self):
        super().__init__()
        self.transition[7][0] = [(0.75, 19, 1.5, False), (0.25, 41, 3.75, True)]

    def discrete_transition_model(self) -> models.DiscreteTransitionModel | None:
        return super().discrete_transition_model()


class _RewardShiftWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward + 5.0


class _RewardShiftTimeLimit(gym.wrappers.TimeLimit):
    def __init__(self, env):
        super().__init__(env, max_episode_steps=2)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward + 5.0, terminated, truncated, info


class _SuccessorAndTerminationFeatures:
    feature_dim = 2

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        return np.array([transition.next_observation[0], float(transition.terminated)])


class _ObservationDtypeFeatures:
    feature_dim = 1

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        assert transition.observation.dtype == np.float32
        assert transition.next_observation.dtype == np.float32
        return transition.next_observation


class _CountingSuccessorFeatures:
    def __init__(self):
        self.feature_dim = 1
        self.calls = 0

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        self.calls += 1
        return transition.next_observation


class _ConstantAndSuccessorFeatures:
    feature_dim = 2

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        return np.array([1.0, transition.next_observation[0]])


class _RewardShiftGridWorld(gympg.RandomStartGridWorld):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward + 5.0, terminated, truncated, info


class _StochasticTableEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 2, shape=(1,), dtype=np.int64)
        self.action_space = gym.spaces.Discrete(1)
        self.states_mapping = {(0,): 0, (1,): 1, (2,): 2}
        self.transition = {
            0: {0: [(0.75, 1, 1.0, False), (0.25, 2, 2.0, False)]},
            1: {0: [(1.0, 0, 0.0, False)]},
            2: {0: [(1.0, 0, 0.0, False)]},
        }
        self.state = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return np.array([0]), {}

    def step(self, action):
        entries = self.transition[self.state][action]
        index = int(
            self.np_random.choice(len(entries), p=[entry[0] for entry in entries])
        )
        _, self.state, reward, terminated = entries[index]
        return np.array([self.state]), reward, terminated, False, {}


class _NonterminalSelfLoopEnv(_StochasticTableEnv):
    def __init__(self):
        super().__init__()
        self.transition = {0: {0: [(1.0, 0, 2.0, False)]}}
        self.states_mapping = {(0,): 0}


class _SuccessorValueFeatures:
    feature_dim = 1

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        return np.asarray(transition.next_observation, dtype=np.float64)


def read_reward_errors(log_file: str) -> list[dict]:
    """Unique estimation events from an experiment log, by update index."""
    events: dict[int, dict] = {}
    with open(log_file) as readable:
        for line in readable:
            entry = json.loads(line)
            for event in entry.get("info", {}).get("reward_errors", []):
                events[event["update_index"]] = event
    return [events[idx] for idx in sorted(events)]


def weights_dim(ft_op: transform.FTOp) -> int:
    """Dimension of the estimator weights for a Box feature space."""
    box_space: gym.spaces.Box = ft_op.output_space.observation_space  # type: ignore[assignment]
    return math.prod(box_space.shape)
