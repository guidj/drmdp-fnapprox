"""Tests for rewardfeat — reward features computed from transitions."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.envs.box2d import lunar_lander as gym_lunar_lander

from drmdp import rewardfeat, rewdelay, task, transform
from drmdp.envs import lunar_lander


class TestActionAndTerminationFeatures:
    """Per-action indicators with a separate terminal block."""

    def test_nonterminal_row_marks_its_action(self):
        hook = self._hook()
        assert hook.feature_dim == 6
        np.testing.assert_array_equal(
            hook(_transition(action=1, terminated=False)),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def test_terminal_row_marks_the_terminal_block(self):
        hook = self._hook()
        np.testing.assert_array_equal(
            hook(_transition(action=1, terminated=True)),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )

    def _hook(self):
        return rewardfeat.transition_feature_fn(
            _stub_env(num_actions=3), {"name": "action-and-termination-features"}
        )


class TestNonterminalActionFeatures:
    """Action indicators that are zero at termination."""

    def test_terminal_rows_are_zero(self):
        hook = rewardfeat.transition_feature_fn(
            _stub_env(num_actions=3), {"name": "nonterminal-action-features"}
        )
        assert hook.feature_dim == 3
        np.testing.assert_array_equal(
            hook(_transition(action=2, terminated=True)), np.zeros(3)
        )

    def test_nonterminal_row_marks_its_action(self):
        hook = rewardfeat.transition_feature_fn(
            _stub_env(num_actions=3), {"name": "nonterminal-action-features"}
        )
        np.testing.assert_array_equal(
            hook(_transition(action=2, terminated=False)),
            np.array([0.0, 0.0, 1.0]),
        )


class TestFTOpFeatures:
    def test_discrete_output_is_encoded_as_a_vector(self):
        env = _stub_env()
        env.observation_space = spaces.Discrete(3)
        pipe = transform.transform_pipeline(
            env,
            [{"name": "flat-grid-observation-action-ft", "args": {"ohe": False}}],
        )
        features = rewardfeat.FTOpFeatures(pipe)
        np.testing.assert_array_equal(
            features(_transition(obs=np.array(1), action=1)),
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )


class TestCurrentFeatures:
    """The current state-action block alone."""

    def test_matches_the_declared_pipeline(self):
        env = _stub_env()
        spec = [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}]
        hook = rewardfeat.transition_feature_fn(
            env, rewardfeat.current_features_spec(spec)
        )
        pipe = transform.transform_pipeline(env, spec)
        expected = np.asarray(
            pipe(transform.Example(np.array([0.25]), 1)).observation, dtype=np.float64
        )
        np.testing.assert_array_equal(
            hook(_transition(obs=np.array([0.25]), action=1)), expected
        )


class TestCurrentAndSuccessorFeatures:
    """A current block followed by a state-only successor block."""

    def test_successor_block_ignores_the_action(self):
        hook = self._hook(_stub_env(observation_dim=2))
        next_obs = np.array([0.2, -0.4])
        first = hook(_transition(action=0, next_obs=next_obs))
        second = hook(_transition(action=2, next_obs=next_obs))
        successor_dim = hook.successor.feature_dim
        current_dim = hook.current.feature_dim
        np.testing.assert_array_equal(first[-successor_dim:], second[-successor_dim:])
        assert not np.array_equal(first[:current_dim], second[:current_dim])

    def test_successor_block_ignores_the_current_observation(self):
        hook = self._hook(_stub_env(observation_dim=2))
        next_obs = np.array([0.2, -0.4])
        first = hook(_transition(obs=np.array([0.1, 0.1]), next_obs=next_obs))
        second = hook(_transition(obs=np.array([-0.9, 0.9]), next_obs=next_obs))
        successor_dim = hook.successor.feature_dim
        np.testing.assert_array_equal(first[-successor_dim:], second[-successor_dim:])

    def test_mountaincar_width_is_current_plus_state_only_successor(self):
        env = gym.make("MountainCar-v0")
        hook = self._hook(env)
        assert hook.feature_dim == 288
        env.close()

    def test_discrete_blocks_use_current_action_and_successor_state(self):
        env = _stub_env(num_actions=2)
        env.observation_space = spaces.Discrete(3)
        spec = [{"name": "flat-grid-observation-action-ft", "args": {"ohe": False}}]
        hook = rewardfeat.transition_feature_fn(
            env, rewardfeat.current_and_successor_features_spec(spec, spec)
        )
        np.testing.assert_array_equal(
            hook(_transition(obs=np.array(1), action=1, next_obs=np.array(2))),
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        assert hook.feature_dim == 9

    def test_composes_transition_feature_protocols(self):
        hook = rewardfeat.CurrentAndSuccessorFeatures(
            rewardfeat.ActionTerminationFeatures(2),
            rewardfeat.ConstantAndEventFeatures("event"),
        )
        np.testing.assert_array_equal(
            hook(_transition(action=1, info={"event": True})),
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        )

    def test_successor_hash_dim_is_rejected(self):
        env = _stub_env()
        spec = [
            {
                "name": "tile-observation-action-ft",
                "args": {"tiling_dim": 2, "hash_dim": 16},
            }
        ]
        with pytest.raises(ValueError, match="hash_dim"):
            rewardfeat.transition_feature_fn(
                env,
                {
                    "name": "current-and-successor-features",
                    "args": {"current_spec": spec, "successor_spec": spec},
                },
            )

    def test_terminal_gate_zeroes_the_successor_block_on_termination(self):
        env = _stub_env()
        drop = {"name": "drop-observation-dims-ft", "args": {"axis_dims": {0: [0]}}}
        spec = [drop, {"name": "constant-ft", "args": {}}]
        hook = rewardfeat.transition_feature_fn(
            env,
            rewardfeat.current_and_successor_features_spec(
                spec, spec, terminal_gate=True
            ),
        )
        assert hook.feature_dim == 2
        np.testing.assert_array_equal(
            hook(_transition(terminated=True)),
            [1.0, 0.0],
        )
        np.testing.assert_array_equal(
            hook(_transition()),
            [1.0, 1.0],
        )

    def _hook(self, env: gym.Env):
        spec = [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}]
        return rewardfeat.transition_feature_fn(
            env,
            {
                "name": "current-and-successor-features",
                "args": {"current_spec": spec, "successor_spec": spec},
            },
        )


class TestNonterminalConstantFeatures:
    """A unit on nonterminal steps, zero on the terminal one."""

    def test_nonterminal_row_is_one_and_terminal_row_is_zero(self):
        hook = rewardfeat.transition_feature_fn(
            _stub_env(), {"name": "nonterminal-constant-features"}
        )
        assert hook.feature_dim == 1
        np.testing.assert_array_equal(hook(_transition(terminated=False)), [1.0])
        np.testing.assert_array_equal(hook(_transition(terminated=True)), [0.0])


class TestConcatenatedFeatures:
    """The concatenation of named transition-feature blocks."""

    def test_blocks_are_concatenated_in_order(self):
        drop = {"name": "drop-observation-dims-ft", "args": {"axis_dims": {0: [0]}}}
        constant = {"name": "constant-ft", "args": {}}
        hook = rewardfeat.transition_feature_fn(
            _stub_env(),
            rewardfeat.concatenated_features_spec(
                [
                    {"name": "nonterminal-constant-features", "args": {}},
                    {
                        "name": "current-and-successor-features",
                        "args": {
                            "current_spec": [drop, constant],
                            "successor_spec": [drop, constant],
                            "terminal_gate": True,
                        },
                    },
                ]
            ),
        )
        assert hook.feature_dim == 3
        np.testing.assert_array_equal(
            hook(_transition(terminated=True)),
            [0.0, 1.0, 0.0],
        )
        np.testing.assert_array_equal(
            hook(_transition()),
            [1.0, 1.0, 1.0],
        )

    def test_empty_blocks_are_rejected(self):
        with pytest.raises(ValueError, match="at least one block"):
            rewardfeat.transition_feature_fn(
                _stub_env(), rewardfeat.concatenated_features_spec([])
            )


class TestConstantAndSuccessorBasisFeatures:
    """A constant plus a gated successor basis expansion."""

    def test_sine_basis_evaluates_the_successor_position(self):
        hook = self._hook({"name": "sine", "args": {"frequency": 3.0, "coordinate": 0}})
        assert hook.feature_dim == 2
        np.testing.assert_allclose(
            hook(_transition(next_obs=np.array([0.5]))),
            np.array([1.0, np.sin(1.5)]),
            atol=1e-12,
        )

    def test_terminal_step_gates_the_basis(self):
        hook = self._hook({"name": "sine", "args": {"frequency": 3.0, "coordinate": 0}})
        np.testing.assert_array_equal(
            hook(_transition(next_obs=np.array([0.5]), terminated=True)),
            np.array([1.0, 0.0]),
        )

    def test_uniform_bins_place_the_value_in_one_bin(self):
        hook = self._hook(
            {
                "name": "uniform-bins",
                "args": {"count": 4, "low": -1.2, "high": 0.6, "coordinate": 0},
            }
        )
        assert hook.feature_dim == 5
        features = hook(_transition(next_obs=np.array([-1.2])))
        np.testing.assert_array_equal(features, np.array([1.0, 1.0, 0.0, 0.0, 0.0]))

    def test_uniform_bins_clip_outside_the_declared_interval(self):
        hook = self._hook(
            {
                "name": "uniform-bins",
                "args": {"count": 4, "low": -1.2, "high": 0.6, "coordinate": 0},
            }
        )
        features = hook(_transition(next_obs=np.array([5.0])))
        np.testing.assert_array_equal(features, np.array([1.0, 0.0, 0.0, 0.0, 1.0]))

    def _hook(self, basis_spec):
        return rewardfeat.transition_feature_fn(
            _stub_env(),
            {
                "name": "constant-and-successor-basis-features",
                "args": {"basis_spec": basis_spec},
            },
        )


class TestConstantAndEventFeatures:
    """A constant plus an env-emitted event indicator."""

    def test_event_reads_from_info(self):
        hook = rewardfeat.transition_feature_fn(
            _stub_env(),
            {
                "name": "constant-and-event-features",
                "args": {"event_key": "cliff_entry"},
            },
        )
        assert hook.feature_dim == 2
        np.testing.assert_array_equal(
            hook(_transition(info={"cliff_entry": True})), np.array([1.0, 1.0])
        )
        np.testing.assert_array_equal(
            hook(_transition(info={"cliff_entry": False})), np.array([1.0, 0.0])
        )

    def test_event_key_selects_any_info_entry(self):
        hook = rewardfeat.transition_feature_fn(
            _stub_env(),
            {
                "name": "constant-and-event-features",
                "args": {"event_key": "goal_reached"},
            },
        )
        np.testing.assert_array_equal(
            hook(_transition(info={"goal_reached": True})), np.array([1.0, 1.0])
        )

    def test_spec_requires_event_key(self):
        with pytest.raises(TypeError):
            rewardfeat.transition_feature_fn(
                _stub_env(), {"name": "constant-and-event-features"}
            )

    def test_missing_event_key_raises_instead_of_scoring_zero(self):
        hook = rewardfeat.transition_feature_fn(
            _stub_env(),
            {
                "name": "constant-and-event-features",
                "args": {"event_key": "cliff_entry"},
            },
        )
        with pytest.raises(KeyError, match="cliff_entry"):
            hook(_transition())


class TestStateBinsActionFeatures:
    """A joint-cell partition of the observation box per action."""

    def test_marks_the_joint_cell_of_its_action_block(self):
        hook = self._hook((4, 2))
        assert hook.feature_dim == 24
        # obs (0.9, -0.9) on [-1, 1]^2 with bins (4, 2): cell indices
        # (3, 0), joint row-major index 3 * 2 + 0 = 6, action block 2
        features = hook(_transition(obs=np.array([0.9, -0.9]), action=2))
        expected = np.zeros(24)
        expected[2 * 8 + 6] = 1.0
        np.testing.assert_array_equal(features, expected)

    def test_observation_at_the_upper_bound_lands_in_the_last_cell(self):
        hook = self._hook((4, 2))
        features = hook(_transition(obs=np.array([1.0, 1.0]), action=0))
        # clamped cell indices (3, 1) -> joint 7
        assert features[7] == 1.0
        assert features.sum() == 1.0

    def test_bins_must_match_the_observation_dimensionality(self):
        with pytest.raises(ValueError, match="match the observation dimensionality"):
            self._hook((4,))

    def test_bin_counts_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            self._hook((4, 0))

    def test_requires_a_box_observation_space(self):
        env = _stub_env()
        env.observation_space = spaces.Discrete(4)
        with pytest.raises(TypeError, match="Box observation space"):
            rewardfeat.transition_feature_fn(
                env, {"name": "state-bins-action-features", "args": {"bins": [2]}}
            )

    def _hook(self, bins, observation_dim=2, num_actions=3):
        return rewardfeat.transition_feature_fn(
            _stub_env(observation_dim=observation_dim, num_actions=num_actions),
            {"name": "state-bins-action-features", "args": {"bins": list(bins)}},
        )


class TestGridWorldNominalTransitionAnomalyFeatures:
    """A constant and a GridWorld nominal-move anomaly indicator."""

    def test_plain_move_is_not_an_anomaly(self):
        hook = self._hook()
        np.testing.assert_array_equal(
            hook(
                _transition(obs=np.array([1, 1]), action=1, next_obs=np.array([1, 2]))
            ),
            np.array([1.0, 0.0]),
        )

    def test_wall_stay_is_not_an_anomaly(self):
        hook = self._hook()
        # LEFT at the left edge: the nominal move clamps to the cell
        # itself and the env stays
        np.testing.assert_array_equal(
            hook(
                _transition(obs=np.array([1, 0]), action=0, next_obs=np.array([1, 0]))
            ),
            np.array([1.0, 0.0]),
        )

    def test_teleport_is_an_anomaly(self):
        hook = self._hook()
        # RIGHT would land on (1, 2); a cliff resets the agent to the
        # start cell instead
        np.testing.assert_array_equal(
            hook(
                _transition(obs=np.array([1, 1]), action=1, next_obs=np.array([0, 0]))
            ),
            np.array([1.0, 1.0]),
        )

    def test_non_two_dimensional_box_is_rejected(self):
        env = _grid_stub_env()
        env.observation_space = spaces.Box(
            low=np.zeros(3, dtype=np.int64), high=np.array([2, 2, 2])
        )
        with pytest.raises(ValueError, match="moves are 2-dimensional"):
            rewardfeat.transition_feature_fn(
                env,
                {
                    "name": "gridworld-nominal-transition-anomaly-features",
                    "args": {"low": [0, 0], "high": [2, 2]},
                },
            )

    def test_noninteger_bounds_are_rejected(self):
        env = _stub_env(observation_dim=2)
        env.observation_space = spaces.Box(
            low=np.array([0.25, 0.0]), high=np.array([2.0, 2.0])
        )
        with pytest.raises(ValueError, match="whole-number observation bounds"):
            rewardfeat.transition_feature_fn(
                env,
                {
                    "name": "gridworld-nominal-transition-anomaly-features",
                    "args": {"low": [0, 0], "high": [2, 2]},
                },
            )

    def _hook(self, num_actions=4):
        return rewardfeat.transition_feature_fn(
            _grid_stub_env(num_actions=num_actions),
            {
                "name": "gridworld-nominal-transition-anomaly-features",
                "args": {"low": [0, 0], "high": [2, 2]},
            },
        )


class TestGridWorldNominalNextCellFeatures:
    """A constant and a one-hot over the nominal next cell."""

    def test_one_hot_at_the_nominal_cell(self):
        hook = self._hook()
        assert hook.feature_dim == 1 + 9
        # RIGHT from (1, 1) targets (1, 2): row-major cell 1 * 3 + 2 = 5
        features = hook(_transition(obs=np.array([1, 1]), action=1))
        expected = np.zeros(10)
        expected[0] = 1.0
        expected[1 + 5] = 1.0
        np.testing.assert_array_equal(features, expected)

    def test_boundary_move_clamps_to_the_last_cell(self):
        hook = self._hook()
        # RIGHT at the right edge clamps to the cell itself
        features = hook(_transition(obs=np.array([2, 2]), action=1))
        assert features[1 + 8] == 1.0
        assert features.sum() == 2.0

    def test_ignores_the_successor_observation(self):
        hook = self._hook()
        first = hook(
            _transition(obs=np.array([1, 1]), action=1, next_obs=np.array([0, 0]))
        )
        second = hook(
            _transition(obs=np.array([1, 1]), action=1, next_obs=np.array([2, 2]))
        )
        np.testing.assert_array_equal(first, second)

    def _hook(self, num_actions=4):
        return rewardfeat.transition_feature_fn(
            _grid_stub_env(num_actions=num_actions),
            {
                "name": "gridworld-nominal-next-cell-features",
                "args": {"low": [0, 0], "high": [2, 2]},
            },
        )


class TestGridWorldBounds:
    @pytest.mark.parametrize(
        "name",
        [
            "gridworld-nominal-transition-anomaly-features",
            "gridworld-nominal-next-cell-features",
        ],
    )
    def test_overrides_do_not_bypass_environment_dimensionality(self, name):
        env = _stub_env(observation_dim=3, num_actions=4)
        with pytest.raises(ValueError, match="moves are 2-dimensional"):
            rewardfeat.transition_feature_fn(
                env, {"name": name, "args": {"low": [0, 0], "high": [2, 2]}}
            )

    @pytest.mark.parametrize(
        "low, high",
        [([0], [2, 2]), ([0, 0], [2]), ([0, 0], [2.5, 2]), ([2, 0], [1, 2])],
    )
    def test_malformed_bounds_are_rejected_at_construction(self, low, high):
        with pytest.raises(ValueError, match="bounds"):
            rewardfeat.transition_feature_fn(
                _grid_stub_env(),
                {
                    "name": "gridworld-nominal-next-cell-features",
                    "args": {"low": low, "high": high},
                },
            )

    @pytest.mark.parametrize("num_actions", [3, 5])
    def test_action_count_must_match_gridworld_moves(self, num_actions):
        with pytest.raises(ValueError, match="four actions"):
            rewardfeat.transition_feature_fn(
                _grid_stub_env(num_actions=num_actions),
                {"name": "gridworld-nominal-transition-anomaly-features"},
            )

    def test_gridworld_defaults_use_reachable_coordinates(self):
        env = task.create_env("GridWorld-test", {"grid": ["soo", "ooo", "oog"]}).proxy
        try:
            anomaly = rewardfeat.transition_feature_fn(
                env, {"name": "gridworld-nominal-transition-anomaly-features"}
            )
            next_cell = rewardfeat.transition_feature_fn(
                env, {"name": "gridworld-nominal-next-cell-features"}
            )
            step = _transition(
                obs=np.array([1, 2]), action=1, next_obs=np.array([1, 2])
            )
            np.testing.assert_array_equal(anomaly(step), [1.0, 0.0])
            expected = np.zeros(10)
            expected[[0, 6]] = 1.0
            np.testing.assert_array_equal(next_cell(step), expected)
        finally:
            env.close()


class TestTerminationSemantics:
    @pytest.mark.parametrize(
        "hook, observation, action, expected",
        [
            (rewardfeat.ActionTerminationFeatures(2), [0.5], 1, [0.0, 1.0, 0.0, 0.0]),
            (rewardfeat.NonterminalActionFeatures(2), [0.5], 1, [0.0, 1.0]),
            (
                rewardfeat.ConstantAndSuccessorBasisFeatures(lambda obs: obs, 1),
                [0.5],
                0,
                [1.0, 0.5],
            ),
        ],
    )
    def test_truncation_preserves_nonterminal_features(
        self, hook, observation, action, expected
    ):
        np.testing.assert_allclose(
            hook(
                _transition(
                    next_obs=np.array(observation), action=action, truncated=True
                )
            ),
            expected,
            atol=1e-6,
        )


class TestFourierBasis:
    """The Fourier successor basis of the constant-and-successor builder."""

    def test_harmonics_of_the_scaled_coordinate(self):
        hook = self._hook(order=2)
        assert hook.feature_dim == 1 + 4
        # x-hat = (0.5 - (-1)) / 2 = 0.75
        scaled = 0.75
        expected = np.array(
            [
                np.cos(np.pi * scaled),
                np.sin(np.pi * scaled),
                np.cos(2 * np.pi * scaled),
                np.sin(2 * np.pi * scaled),
            ]
        )
        np.testing.assert_allclose(
            hook(_transition(next_obs=np.array([0.5])))[1:], expected, atol=1e-12
        )

    def test_terminal_step_gates_the_harmonics(self):
        hook = self._hook(order=1)
        np.testing.assert_array_equal(
            hook(_transition(next_obs=np.array([0.5]), terminated=True)),
            np.array([1.0, 0.0, 0.0]),
        )

    def test_out_of_interval_values_clamp_to_unit_scale(self):
        hook = self._hook(order=1, low=0.0, high=1.0)
        # 5.0 clamps to 1.0: cos(pi) = -1, sin(pi) ~ 0
        features = hook(_transition(next_obs=np.array([5.0])))
        np.testing.assert_allclose(features[1], -1.0, atol=1e-12)
        np.testing.assert_allclose(features[2], 0.0, atol=1e-12)

    def test_nonpositive_order_raises(self):
        with pytest.raises(ValueError, match="order must be positive"):
            self._hook(order=0)

    def test_degenerate_interval_raises(self):
        with pytest.raises(ValueError, match="high > low"):
            self._hook(order=1, low=1.0, high=1.0)

    def _hook(self, order, low=-1.0, high=1.0):
        return rewardfeat.transition_feature_fn(
            _stub_env(),
            {
                "name": "constant-and-successor-basis-features",
                "args": {
                    "basis_spec": {
                        "name": "fourier",
                        "args": {
                            "order": order,
                            "coordinate": 0,
                            "low": low,
                            "high": high,
                        },
                    }
                },
            },
        )


class TestTransitionFeatureFn:
    """Dispatch and specification validation."""

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown transition feature"):
            rewardfeat.transition_feature_fn(_stub_env(), {"name": "nope"})


class TestWrapperContract:
    """The estimator consumes the hook and nothing else."""

    def test_feature_width_mismatch_raises(self):
        env = rewdelay.DelayedRewardWrapper(
            _stepping_env(), reward_delay=rewdelay.FixedDelay(delay=2)
        )
        wrapper = rewdelay.LeastLfaGenerativeRewardWrapper(
            env=env, feats_fn=_WrongWidthFeatures(), attempt_estimation_episode=1
        )
        wrapper.reset(seed=0)
        with pytest.raises(ValueError, match="returned shape"):
            wrapper.step(0)
        wrapper.close()

    def test_estimator_accumulates_the_hook_output(self):
        env = rewdelay.DelayedRewardWrapper(
            _stepping_env(), reward_delay=rewdelay.FixedDelay(delay=2)
        )
        hook = _CountingFeatures()
        wrapper = rewdelay.LeastLfaGenerativeRewardWrapper(
            env=env, feats_fn=hook, attempt_estimation_episode=100
        )
        wrapper.reset(seed=0)
        wrapper.step(0)
        wrapper.step(1)
        assert hook.calls == 2
        features, reward = wrapper.est_buffer.buffer[0]
        np.testing.assert_array_equal(features, [2.0, 0.0])
        assert reward == 2.0
        wrapper.close()

    def test_task_rejects_feats_spec_on_an_estimator(self):
        env = _stepping_env()
        proxy = _stepping_env()
        with pytest.raises(ValueError, match="no longer accept feats_spec"):
            task.reward_mapper(
                env,
                proxy_env=proxy,
                mapping_spec={
                    "name": "least-lfa",
                    "args": {
                        "attempt_estimation_episode": 1,
                        "feats_spec": [{"name": "flat-grid-observation-action-ft"}],
                    },
                },
            )

    def test_task_requires_a_transition_feature_specification(self):
        env = _stepping_env()
        proxy = _stepping_env()
        with pytest.raises(ValueError, match="require transition_feats_spec"):
            task.reward_mapper(
                env,
                proxy_env=proxy,
                mapping_spec={
                    "name": "least-lfa",
                    "args": {"attempt_estimation_episode": 1},
                },
            )


class TestLunarLanderRewardFeatures:
    """The row that makes the LunarLander reward linear in the observation."""

    def test_row_is_the_gated_basis_pairs_fuel_and_terminal_columns(self):
        hook = self._hook()
        current = _lunar_state(
            x=3.0, y=4.0, vx=6.0, vy=8.0, angle=-0.5, angular=0.1, legs=(1.0, 0.0)
        )
        successor = _lunar_state(
            x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.25, angular=0.2, legs=(1.0, 1.0)
        )
        row = hook(_lunar_transition(current, action=2, successor=successor))
        np.testing.assert_allclose(
            row,
            np.array(
                [
                    0.0,
                    0.0,
                    0.25,
                    1.0,
                    1.0,
                    -5.0,
                    -10.0,
                    -0.5,
                    -1.0,
                    -0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )

    def test_terminal_rows_zero_the_potential_and_fuel_blocks(self):
        hook = self._hook()
        current = _lunar_state(
            x=1.0, y=1.0, vx=1.0, vy=1.0, angle=0.1, angular=0.1, legs=(0.0, 0.0)
        )
        row = hook(
            _lunar_transition(
                current, action=2, successor=current, terminated=True, awake=True
            )
        )
        np.testing.assert_allclose(row[:14], np.zeros(14))

    def test_landing_and_crash_columns_are_exclusive(self):
        hook = self._hook()
        landed = _lunar_state(
            x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.01, angular=0.0, legs=(1.0, 1.0)
        )
        crashed = _lunar_state(
            x=0.0, y=0.0, vx=2.0, vy=-1.0, angle=0.9, angular=0.4, legs=(0.0, 0.0)
        )
        landing_row = hook(
            _lunar_transition(
                landed, action=0, successor=landed, terminated=True, awake=False
            )
        )
        crash_row = hook(
            _lunar_transition(
                crashed, action=0, successor=crashed, terminated=True, awake=True
            )
        )
        np.testing.assert_allclose(landing_row[-2:], np.array([1.0, 0.0]))
        np.testing.assert_allclose(crash_row[-2:], np.array([0.0, 1.0]))

    def test_landing_label_reads_the_awake_bit_from_info(self):
        hook = self._hook()
        moving = _lunar_state(
            x=0.0, y=0.0, vx=2.0, vy=-1.0, angle=0.9, angular=0.4, legs=(0.0, 0.0)
        )
        at_rest = _lunar_state(
            x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.01, angular=0.0, legs=(1.0, 1.0)
        )
        # The bit decides the terminal columns in both directions: a moving
        # successor cannot pass the rest rule, an at-rest successor does.
        landing_row = hook(
            _lunar_transition(
                moving, action=0, successor=moving, terminated=True, awake=False
            )
        )
        crash_row = hook(
            _lunar_transition(
                at_rest, action=0, successor=at_rest, terminated=True, awake=True
            )
        )
        np.testing.assert_allclose(landing_row[-2:], np.array([1.0, 0.0]))
        np.testing.assert_allclose(crash_row[-2:], np.array([0.0, 1.0]))

    def test_awake_bit_is_ignored_on_a_nonterminal_transition(self):
        hook = self._hook()
        state = _lunar_state(
            x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.01, angular=0.0, legs=(1.0, 1.0)
        )
        row = hook(_lunar_transition(state, action=0, successor=state, awake=False))
        np.testing.assert_allclose(row[-2:], np.zeros(2))

    def test_no_terminal_column_is_set_on_a_nonterminal_transition(self):
        hook = self._hook()
        state = _lunar_state(
            x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.0, angular=0.0, legs=(0.0, 0.0)
        )
        row = hook(_lunar_transition(state, action=0, successor=state))
        np.testing.assert_allclose(row[-2:], np.zeros(2))

    def test_least_squares_recovers_every_nonterminal_reward(self):
        env = gym.make("LunarLander-v3")
        wrapped = lunar_lander.LunarLanderRewardObs(env)
        try:
            hook = rewardfeat.transition_feature_fn(
                wrapped, {"name": "lunar-lander-reward-features", "args": {}}
            )
            matrix, rewards = _random_episode_rows(hook, wrapped, seed=0)
            weights, *_ = np.linalg.lstsq(matrix, rewards, rcond=None)
            residuals = matrix @ weights - rewards
            assert residuals.shape == (len(rewards),)
            assert np.max(np.abs(residuals)) < 1e-4
        finally:
            wrapped.close()

    def test_terminal_columns_match_the_branch_on_real_terminals(self):
        hook = self._hook()
        landings = np.asarray(_terminal_rows(hook, heuristic=True, episodes=4))
        crashes = np.asarray(_terminal_rows(hook, heuristic=False, episodes=60))
        assert len(landings) >= 4
        assert len(crashes) >= 40
        np.testing.assert_allclose(landings[:, -2], np.ones(len(landings)))
        np.testing.assert_allclose(landings[:, -1], np.zeros(len(landings)))
        np.testing.assert_allclose(crashes[:, -2], np.zeros(len(crashes)))
        np.testing.assert_allclose(crashes[:, -1], np.ones(len(crashes)))

    def test_missing_awake_bit_raises_a_terminal_transition(self):
        hook = self._hook()
        state = _lunar_state(
            x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.0, angular=0.0, legs=(0.0, 0.0)
        )
        transition = rewardfeat.Transition(
            observation=state,
            action=0,
            next_observation=state,
            terminated=True,
            truncated=False,
            info={},
        )
        with pytest.raises(KeyError, match=lunar_lander.AWAKE_INFO_KEY):
            hook(transition)

    def test_rejects_an_unwrapped_observation_width(self):
        with pytest.raises(ValueError, match="observation space"):
            rewardfeat.transition_feature_fn(
                _stub_env(observation_dim=2), {"name": "lunar-lander-reward-features"}
            )

    def _hook(self) -> rewardfeat.TransitionFeatureFn:
        env = gym.make("LunarLander-v3")
        wrapped = lunar_lander.LunarLanderRewardObs(env)
        try:
            return rewardfeat.transition_feature_fn(
                wrapped, {"name": "lunar-lander-reward-features", "args": {}}
            )
        finally:
            wrapped.close()


class _StubEnv(gym.Env):
    """Spaces only; used to build feature maps."""

    def __init__(self, observation_dim: int = 1, num_actions: int = 3):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(observation_dim,), dtype=np.float64
        )
        self.action_space = spaces.Discrete(num_actions)

    def reset(self, *, seed=None, options=None):
        del seed, options
        return np.zeros(self.observation_space.shape), {}

    def step(self, action):
        del action
        return np.zeros(self.observation_space.shape), 0.0, False, False, {}


class _SteppingEnv(_StubEnv):
    """Terminates after a fixed number of steps."""

    def __init__(self, term_steps: int = 4):
        super().__init__(observation_dim=1, num_actions=2)
        self.term_steps = term_steps
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        del action
        self.step_count += 1
        terminated = self.step_count >= self.term_steps
        return np.ones(1) * self.step_count, 1.0, terminated, False, {}


class _GridStubEnv(gym.Env):
    """Integer grid-coordinate observation space."""

    def __init__(self, num_actions: int = 4):
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([2, 2]), dtype=np.int64
        )
        self.action_space = spaces.Discrete(num_actions)

    def reset(self, *, seed=None, options=None):
        del seed, options
        return np.zeros(2, dtype=np.int64), {}

    def step(self, action):
        del action
        return np.zeros(2, dtype=np.int64), 0.0, False, False, {}


class _CountingFeatures:
    """Two features that record their calls."""

    def __init__(self):
        self.calls = 0

    @property
    def feature_dim(self) -> int:
        return 2

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, float(transition.terminated)])


class _WrongWidthFeatures:
    """Declares width 2 but emits 3 coordinates."""

    @property
    def feature_dim(self) -> int:
        return 2

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        del transition
        return np.zeros(3)


def _grid_stub_env(num_actions: int = 4) -> _GridStubEnv:
    return _GridStubEnv(num_actions=num_actions)


def _stub_env(observation_dim: int = 1, num_actions: int = 3) -> _StubEnv:
    return _StubEnv(observation_dim=observation_dim, num_actions=num_actions)


def _stepping_env() -> _SteppingEnv:
    return _SteppingEnv()


def _transition(
    obs: np.ndarray | None = None,
    action: int = 0,
    next_obs: np.ndarray | None = None,
    terminated: bool = False,
    info: dict | None = None,
    truncated: bool = False,
) -> rewardfeat.Transition:
    return rewardfeat.Transition(
        observation=np.zeros(1) if obs is None else obs,
        action=action,
        next_observation=np.zeros(1) if next_obs is None else next_obs,
        terminated=terminated,
        truncated=truncated,
        info={} if info is None else info,
    )


def _lunar_state(
    x: float,
    y: float,
    vx: float,
    vy: float,
    angle: float,
    angular: float,
    legs: tuple[float, float],
) -> np.ndarray:
    """A wrapped LunarLander observation around the given raw state."""
    raw = np.array([x, y, vx, vy, angle, angular, legs[0], legs[1]], dtype=np.float32)
    return np.concatenate([lunar_lander.reward_basis(raw), raw]).astype(np.float32)


def _lunar_transition(
    observation: np.ndarray,
    action: int,
    successor: np.ndarray,
    terminated: bool = False,
    awake: bool = True,
) -> rewardfeat.Transition:
    info = {lunar_lander.AWAKE_INFO_KEY: awake}
    return rewardfeat.Transition(
        observation=observation,
        action=action,
        next_observation=successor,
        terminated=terminated,
        truncated=False,
        info=info,
    )


def _random_episode_rows(
    hook: rewardfeat.TransitionFeatureFn,
    env: gym.Env,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows and true rewards of one random-policy episode's interior steps."""
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    rewards: list[float] = []
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(0, env.action_space.n))
        successor, reward, terminated, truncated, _ = env.step(action)
        if not terminated:
            rows.append(hook(_lunar_transition(observation, action, successor)))
            rewards.append(float(reward))
        observation = successor
    return np.asarray(rows), np.asarray(rewards)


def _terminal_rows(
    hook: rewardfeat.TransitionFeatureFn,
    heuristic: bool,
    episodes: int,
) -> list[np.ndarray]:
    """Feature rows of terminal transitions, landed or crashed."""
    rows: list[np.ndarray] = []
    rng = np.random.default_rng(11)
    env = lunar_lander.LunarLanderAwakeInfoWrapper(gym.make("LunarLander-v3"))
    try:
        for episode in range(episodes):
            raw, _ = env.reset(seed=episode)
            observation = _reward_observation(raw)
            terminated = truncated = False
            while not (terminated or truncated):
                if heuristic:
                    action = _heuristic_action(env, raw)
                else:
                    action = int(rng.integers(0, env.action_space.n))
                raw, _, terminated, truncated, info = env.step(action)
                successor = _reward_observation(raw)
                if terminated:
                    rows.append(
                        hook(
                            _lunar_transition(
                                observation,
                                action,
                                successor,
                                terminated=True,
                                awake=bool(info[lunar_lander.AWAKE_INFO_KEY]),
                            )
                        )
                    )
                observation = successor
    finally:
        env.close()
    return rows


def _reward_observation(raw: np.ndarray) -> np.ndarray:
    """The wrapped observation the encoding reads, built from a raw state."""
    return np.concatenate([lunar_lander.reward_basis(raw), raw]).astype(np.float32)


def _heuristic_action(env: gym.Env, raw: np.ndarray) -> int:
    """The action the environment's bundled landing heuristic selects."""
    return int(gym_lunar_lander.heuristic(env.unwrapped, raw))
