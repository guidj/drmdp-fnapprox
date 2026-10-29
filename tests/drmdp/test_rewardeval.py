"""Tests for rewardeval — true-reward error measurement of estimated rewards."""

import json
import math
import os

import gymnasium as gym
import numpy as np

from drmdp import core, optsol, rewardeval, task, transform
from drmdp.workflows import controlexps
from tests.drmdp.hookhelpers import FTOpHook

CRAFTED_GRID = ["soooo", "ooooo", "oooxg"]


class TestDiscreteStateActionRewards:
    """Discrete enumeration over the transition table."""

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
            if not rewardeval._is_absorbing_state(transition[state_id], state_id)
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
            if not rewardeval._is_absorbing_state(transition[state_id], state_id)
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


class TestIsAbsorbingState:
    """Self-loop detection over the dense transition table."""

    def _mines_proxy(self):
        result = task.create_env(
            "GridWorld-MINES",
            {"grid": controlexps.MINES_GW_GRID, "max_episode_steps": 200},
        )
        return result.proxy

    def test_exit_state_is_absorbing(self):
        proxy = self._mines_proxy()
        transition = proxy.get_wrapper_attr("transition")
        states_mapping = proxy.get_wrapper_attr("states_mapping")
        goal_id = states_mapping[(5, 11)]
        assert rewardeval._is_absorbing_state(transition[goal_id], goal_id)
        proxy.close()

    def test_live_states_are_not_absorbing(self):
        proxy = self._mines_proxy()
        transition = proxy.get_wrapper_attr("transition")
        states_mapping = proxy.get_wrapper_attr("states_mapping")
        start_id = states_mapping[(5, 0)]
        # a wall move stays, but at least one action leaves the cell
        assert not rewardeval._is_absorbing_state(transition[start_id], start_id)
        proxy.close()


class TestTrueRewardErrorFn:
    """The returned callable measures weights against true rewards."""

    def _crafted_setup(self):
        result = task.create_env(
            "GridWorld-CRAFTED", {"grid": CRAFTED_GRID, "max_episode_steps": 100}
        )
        proxy = result.proxy
        dead_ohe = controlexps._grid_dead_ohe_indices(CRAFTED_GRID)
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
        dead_ohe = controlexps._grid_dead_ohe_indices(CRAFTED_GRID)
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
