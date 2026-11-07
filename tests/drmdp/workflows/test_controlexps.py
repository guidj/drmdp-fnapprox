from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from drmdp import constants, task
from drmdp.workflows import controlexps, estimationencodings, estimationexps


class TestGridDeadOheIndices:
    def test_mines_gw_grid(self):
        indices = controlexps.grid_dead_ohe_indices(controlexps.MINES_GW_GRID)
        nrows = len(controlexps.MINES_GW_GRID)
        ncols = len(controlexps.MINES_GW_GRID[0])
        nstates = nrows * ncols
        assert ncols == 12
        cliff_cells = [
            0 * 12 + 4,
            1 * 12 + 9,
            2 * 12 + 1,
            2 * 12 + 7,
            3 * 12 + 5,
            4 * 12 + 2,
            4 * 12 + 9,
        ] + [5 * 12 + col for col in range(1, 11)]
        goal_cells = [5 * 12 + 11]
        dead_cells = cliff_cells + goal_cells
        for state_idx in dead_cells:
            for action in range(4):
                assert action * nstates + state_idx in indices
        assert len(indices) == len(dead_cells) * 4

    def test_simple_grid(self):
        grid = ["sox", "xog"]
        indices = controlexps.grid_dead_ohe_indices(grid)
        nstates = 2 * 3
        cliff_states = [2, 3]
        goal_states = [5]
        dead_states = cliff_states + goal_states
        expected = sorted(
            action * nstates + state for state in dead_states for action in range(4)
        )
        assert sorted(indices) == expected

    def test_no_cliffs(self):
        grid = ["sog"]
        indices = controlexps.grid_dead_ohe_indices(grid)
        nstates = 1 * 3
        goal_states = [2]
        expected = sorted(
            action * nstates + state for state in goal_states for action in range(4)
        )
        assert sorted(indices) == expected


class TestDefaultDelayConfig:
    def test_uniform_ranges(self):
        configs = controlexps.default_delay_config()
        assert [
            (config["args"]["min_delay"], config["args"]["max_delay"])
            for config in configs
        ] == [(2, 4), (6, 8), (10, 12)]

    def test_configs_are_uniform(self):
        for config in controlexps.default_delay_config():
            assert config["name"] == "uniform"

    def test_bounds_are_integers(self):
        # the delay wrapper samples with random.randint, which requires
        # integer bounds
        for config in controlexps.default_delay_config():
            assert isinstance(config["args"]["min_delay"], int)
            assert isinstance(config["args"]["max_delay"], int)

    def test_only_first_range_within_option_budget(self):
        # option specs are derived for delay configs whose minimum delay
        # is within MAX_OPTIONS_DELAY; only [2, 4] qualifies
        within_budget = [
            controlexps._delay_min(config) <= controlexps.MAX_OPTIONS_DELAY
            for config in controlexps.default_delay_config()
        ]
        assert within_budget == [True, False, False]


class TestDiscountFactorGrid:
    """The discount sweep, propagated from the shared default to the specs."""

    def test_family_specs_carry_the_default_discount_sweep(self):
        # the family passes no `discounts` argument, so the shared default
        # alone sets its arm count
        gammas = {
            problem_spec["gamma"]
            for spec in controlexps.gaussian_illustration_experiment_specs()
            for problem_spec in spec["problem_specs"]
        }
        assert gammas == set(controlexps.DEFAULT_DISCOUNT_FACTORS)
        assert gammas == {1.0, 0.99, 0.98}

    def test_common_specs_one_no_delay_arm_per_discount(self):
        # every discount contributes exactly one no-delay FR spec; a missing
        # discount would also remove that method's reference for the gamma
        no_delay = [
            spec
            for spec in controlexps.common_problem_specs(include_options=False)
            if spec["delay_config"] is None
        ]
        assert [spec["gamma"] for spec in no_delay] == list(
            controlexps.DEFAULT_DISCOUNT_FACTORS
        )


class TestGaussianIllustrationNoiseSpecs:
    def test_relative_noise_with_unit_magnitude(self):
        specs = controlexps.gaussian_illustration_experiment_specs()
        for exp in specs:
            noise = exp["args"]["reward_noise"]
            assert noise["name"] == "relative-gaussian"
            # unit = modal per-step reward magnitude, 1 for all envs
            assert noise["args"]["reference_std"] == 1.0
            # sigma = std_multiplier * reference_std, and the multiplier
            # is the sweep value itself
            assert noise["args"]["std_multiplier"] >= 0.0

    def test_std_multipliers_cover_the_sweep(self):
        specs = controlexps.gaussian_illustration_experiment_specs()
        multipliers = sorted(
            {exp["args"]["reward_noise"]["args"]["std_multiplier"] for exp in specs}
        )
        assert multipliers == [0.0, 0.25, 0.5, 1.0, 2.0]

    def test_metadata_carries_no_noise_scale(self):
        for exp in controlexps.gaussian_illustration_experiment_specs():
            assert "noise_scale" not in (exp.get("metadata") or {})


class TestBladeTdRewardSampling:
    def test_gaussian_illustration_consumes_posterior_mean(self):
        # the family compares reward estimators; a per-episode posterior
        # weight draw is a third stochastic source, so BLADE-TD consumes
        # the fixed posterior mean
        specs = controlexps.gaussian_illustration_experiment_specs()
        bayes_args = [
            problem_spec["reward_mapper"]["args"]
            for spec in specs
            for problem_spec in spec["problem_specs"]
            if problem_spec["reward_mapper"]["name"] == "bayes-least-lfa"
        ]
        assert bayes_args
        for args in bayes_args:
            assert args["sample_weights"] is False


class TestEstimatorEncoding:
    """The estimator encoding per environment is a campaign encoding."""

    def test_encoding_is_the_selected_campaign_encoding(self):
        expected = {
            "MountainCar-v0": estimationencodings.mc_structural_sine(),
            "Acrobot-v1": estimationencodings.action_termination(),
            "GridWorld-MINES": estimationencodings.gridworld_transition_anomaly(),
        }
        for spec in controlexps.gaussian_illustration_experiment_specs():
            for mapper_name in ("least-lfa", "bayes-least-lfa"):
                for args in _estimator_args(spec, mapper_name):
                    assert args["transition_feats_spec"] == expected[spec["name"]]

    def test_encoding_is_a_campaign_cell(self):
        # the selected encodings are campaign encodings, so a control run
        # compares against a campaign cell rather than a third recipe
        campaign: dict[str, list[Mapping[str, Any]]] = {}
        for spec in estimationexps.estimation_experiment_specs():
            for args in _estimator_args(spec, "least-lfa"):
                campaign.setdefault(spec["name"], []).append(
                    args["transition_feats_spec"]
                )
        for spec in controlexps.gaussian_illustration_experiment_specs():
            for args in _estimator_args(spec, "least-lfa"):
                assert args["transition_feats_spec"] in campaign[spec["name"]]

    def test_feature_map_dimensions(self):
        # constant and terminal-gated sine: 2; action-gated termination: 6;
        # constant and anomaly indicator: 2
        expected = {"MountainCar-v0": 2, "Acrobot-v1": 6, "GridWorld-MINES": 2}
        measured: dict[str, int] = {}
        for spec in controlexps.gaussian_illustration_experiment_specs():
            if spec["name"] in measured:
                continue
            proxied = task.create_env(spec["name"], args=dict(spec["args"]), gamma=1.0)
            try:
                dims = {
                    task.estimator_feats_fn(
                        proxied.proxy, "least-lfa", dict(args)
                    ).feature_dim
                    for args in _estimator_args(spec, "least-lfa")
                }
                # every delay and check mode of a cell shares one encoding
                assert len(dims) == 1
                measured[spec["name"]] = dims.pop()
            finally:
                proxied.env.close()
                proxied.proxy.close()
        assert measured == expected


class TestBayesLeastSpecsMapperName:
    """The mapper name carries the error-variance mode."""

    _FEATS: tuple[Mapping[str, Any], ...] = ({"name": "current-features", "args": {}},)

    def test_unit_variance_keeps_the_blade_td_name(self):
        specs = controlexps.bayes_least_specs(
            init_attempt_estimation_episodes=(10,), transition_feats_specs=self._FEATS
        )
        assert specs
        for spec in specs:
            assert spec["reward_mapper"]["name"] == "bayes-least-lfa"
            assert spec["reward_mapper"]["args"]["error_variance"] == "unit"

    def test_nig_mode_selects_the_nig_mapper_name(self):
        # aggregation labels a run by its mapper name alone, so the NIG
        # mode needs a name of its own to be separated from unit variance
        specs = controlexps.bayes_least_specs(
            init_attempt_estimation_episodes=(10,),
            transition_feats_specs=self._FEATS,
            error_variance="nig",
        )
        assert specs
        for spec in specs:
            assert spec["reward_mapper"]["name"] == "blade-td-nig"
            assert spec["reward_mapper"]["args"]["error_variance"] == "nig"

    def test_unknown_error_variance_is_rejected(self):
        with pytest.raises(ValueError, match="error_variance"):
            controlexps.bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=self._FEATS,
                error_variance="inverse-gamma",
            )


class TestEstimatorConfiguration:
    """The estimator configurations are unchanged by the encoding selection."""

    def test_least_keeps_the_factors_gate(self):
        for spec in controlexps.gaussian_illustration_experiment_specs():
            for args in _estimator_args(spec, "least-lfa"):
                assert args["estimation_check"] == "factors"
                assert args["attempt_estimation_episode"] == 10
                assert args["estimation_buffer_mult"] == (
                    constants.DEFAULT_ESTIMATION_BUFFER_MULT
                )

    def test_bayes_keeps_the_absent_gate(self):
        # no estimation_check key: the wrapper's default is no gate
        for spec in controlexps.gaussian_illustration_experiment_specs():
            for args in _estimator_args(spec, "bayes-least-lfa"):
                assert "estimation_check" not in args
                assert args["init_attempt_estimation_episode"] == 10
                assert args["estimation_buffer_mult"] == (
                    constants.DEFAULT_ESTIMATION_BUFFER_MULT
                )


def _estimator_args(
    spec: Mapping[str, Any], mapper_name: str
) -> Sequence[Mapping[str, Any]]:
    return [
        problem_spec["reward_mapper"]["args"]
        for problem_spec in spec["problem_specs"]
        if problem_spec["reward_mapper"]["name"] == mapper_name
    ]
