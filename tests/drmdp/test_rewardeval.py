"""Tests for rewardeval — true-reward error measurement of estimated rewards."""

import json
import os

import numpy as np

from drmdp import core, optsol, rewardeval, task, transform
from drmdp.workflows import controlexps

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

    def test_enumerates_every_state_action_pair(self):
        proxy, ft_op = self._mines_setup()
        features, true_rewards = rewardeval.discrete_state_action_rewards(
            proxy_env=proxy, ft_op=ft_op
        )
        states_mapping = proxy.get_wrapper_attr("states_mapping")
        expected_rows = len(states_mapping) * proxy.action_space.n
        assert features.shape[0] == expected_rows
        assert true_rewards.shape == (expected_rows,)
        proxy.close()

    def test_rewards_agree_with_random_policy_steps(self):
        """
        Every reward in the enumeration matches the env's actual reward
        for the same (position, action).
        """
        proxy, ft_op = self._mines_setup()
        _, true_rewards = rewardeval.discrete_state_action_rewards(
            proxy_env=proxy, ft_op=ft_op
        )
        states_mapping = proxy.get_wrapper_attr("states_mapping")
        nactions = proxy.action_space.n

        def reward_for(position, action):
            state_id = states_mapping[tuple(position)]
            return true_rewards[state_id * nactions + action]

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
            proxy_env=proxy, ft_op=ft_op
        )
        weights = optsol.solve_least_squares(matrix=features, rhs=true_rewards)
        measure = rewardeval.true_reward_error_fn(proxy_env=proxy, ft_op=ft_op)
        report = measure(weights)
        np.testing.assert_allclose(report["rmse"], 0.0, atol=1e-10)
        proxy.close()

    def test_use_next_state_enumeration_on_random_start_env(self):
        """
        The table is start-independent, so next-state features are
        enumerable on a random-start grid: cliff falls target the
        construction start, and enumerated rows match the env's
        runtime next states.
        """
        proxy, ft_op = self._mines_setup()
        features, true_rewards = rewardeval.discrete_state_action_rewards(
            proxy_env=proxy, ft_op=ft_op, use_next_state=True
        )
        assert features.shape[0] == len(true_rewards)
        assert features.shape[1] == 2 * weights_dim(ft_op)

        states_mapping = proxy.get_wrapper_attr("states_mapping")
        num_actions = proxy.action_space.n
        proxy.action_space.seed(0)
        obs, _ = proxy.reset(seed=0)
        checked = 0
        while checked < 100:
            action = proxy.action_space.sample()
            next_obs, _, term, trunc, _ = proxy.step(action)
            row = features[states_mapping[tuple(obs)] * num_actions + action]
            expected = rewardeval.step_features(
                ft_op=ft_op,
                obs=obs,
                action=action,
                next_obs=next_obs,
                use_next_state=True,
            )
            np.testing.assert_array_equal(row, expected)
            checked += 1
            obs = next_obs
            if term or trunc:
                obs, _ = proxy.reset()
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
            proxy_env=proxy, ft_op=ft_op
        )
        expected = float(np.sqrt(np.mean(np.square(true_rewards))))
        measure = rewardeval.true_reward_error_fn(proxy_env=proxy, ft_op=ft_op)
        report = measure(np.zeros(weights_dim(ft_op)))
        np.testing.assert_allclose(report["rmse"], expected, atol=1e-10)
        assert report["num_samples"] == len(true_rewards)
        proxy.close()

    def test_repeated_calls_are_deterministic(self):
        proxy, ft_op = self._crafted_setup()
        measure = rewardeval.true_reward_error_fn(
            proxy_env=proxy, ft_op=ft_op, use_bias=True
        )
        weights = np.ones(weights_dim(ft_op) + 1)
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
            ft_op=ft_op,
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
            proxy_env=proxy, ft_op=ft_op, num_samples=500
        )
        assert len(true_rewards) == 500
        # dense -1 rewards with at most a handful of goal transitions (0)
        assert np.count_nonzero(true_rewards == -1.0) >= 490
        assert set(true_rewards.tolist()).issubset({-1.0, 0.0})
        proxy.close()


class TestStepFeatures:
    """Feature computation mirrors the reward wrappers'."""

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
            ft_op=ft_op, obs=obs, action=1, next_obs=obs, use_bias=True
        )
        assert feats.shape[0] == weights_dim(ft_op) + 1
        assert feats[-1] == 1.0
        proxy.close()

    def test_use_next_state_concats_next_features(self):
        result = task.create_env("MountainCar-v0", {"max_episode_steps": 200})
        proxy = result.proxy
        ft_op = task.create_ft_ops(
            proxy,
            feats_spec=[
                {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
            ],
        )
        obs = np.array([0.0, 0.0])
        next_obs = np.array([0.5, 0.5])
        feats = rewardeval.step_features(
            ft_op=ft_op,
            obs=obs,
            action=1,
            next_obs=next_obs,
            use_next_state=True,
        )
        expected_first = ft_op(transform.Example(obs, 1)).observation
        expected_next = ft_op(transform.Example(next_obs, 0)).observation
        assert feats.shape[0] == weights_dim(ft_op) * 2
        np.testing.assert_array_equal(feats[: len(expected_first)], expected_first)
        np.testing.assert_array_equal(feats[len(expected_first) :], expected_next)
        proxy.close()


class TestPolicyControlRewardErrorLogging:
    """End-to-end: events land in the written experiment logs."""

    def test_least_lfa_single_event_near_zero_rmse(self, tmp_path):
        # exact RMSE requires full state-action coverage before LEAST's
        # single fit: minimum-norm least squares assigns 0 to unvisited
        # columns, so an uncovered live pair would break rmse ~ 0.
        # 80 episodes on this 14-state grid cover all pairs.
        events = self._run_control("least-lfa", tmp_path, episodes=80)
        assert len(events) == 1
        event = events[0]
        assert event["update_index"] == 1
        assert event["episode"] <= 80
        assert event["num_samples"] == 56  # 14 states x 4 actions
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
                            "feats_spec": [
                                {
                                    "name": "flat-grid-observation-action-ft",
                                    "args": {},
                                },
                                {
                                    "name": "drop-observation-dims-ft",
                                    "args": {"axis_dims": {0: dead_ohe}},
                                },
                            ],
                            "use_bias": False,
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
    space = ft_op.output_space.observation_space
    return int(np.prod(space.shape))
