from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from drmdp import constants, task
from drmdp.workflows import encodings
from drmdp.workflows.control import exps
from drmdp.workflows.reward_estimation.encodings import exps as encodings_exps


class TestDefaultDelayConfig:
    def test_uniform_ranges(self):
        configs = exps.default_delay_config()
        assert [
            (config["args"]["min_delay"], config["args"]["max_delay"])
            for config in configs
        ] == [(2, 4), (6, 8), (10, 12)]

    def test_configs_are_uniform(self):
        for config in exps.default_delay_config():
            assert config["name"] == "uniform"

    def test_bounds_are_integers(self):
        # the delay wrapper samples with random.randint, which requires
        # integer bounds
        for config in exps.default_delay_config():
            assert isinstance(config["args"]["min_delay"], int)
            assert isinstance(config["args"]["max_delay"], int)

    def test_only_first_range_within_option_budget(self):
        # option specs are derived for delay configs whose minimum delay
        # is within MAX_OPTIONS_DELAY; only [2, 4] qualifies
        within_budget = [
            exps._delay_min(config) <= exps.MAX_OPTIONS_DELAY
            for config in exps.default_delay_config()
        ]
        assert within_budget == [True, False, False]


class TestDiscountFactorGrid:
    """The discount sweep, propagated from the shared default to the specs."""

    def test_family_specs_carry_the_default_discount_sweep(self):
        # the family passes no `discounts` argument, so the shared default
        # alone sets its arm count
        gammas = {
            problem_spec["gamma"]
            for spec in exps.gaussian_illustration_experiment_specs()
            for problem_spec in spec["problem_specs"]
        }
        assert gammas == set(exps.DEFAULT_DISCOUNT_FACTORS)
        assert gammas == {1.0, 0.99}

    def test_common_specs_one_no_delay_arm_per_discount(self):
        # every discount contributes exactly one no-delay FR spec; a missing
        # discount would also remove that method's reference for the gamma
        no_delay = [
            spec
            for spec in exps.common_problem_specs(include_options=False)
            if spec["delay_config"] is None
        ]
        assert [spec["gamma"] for spec in no_delay] == list(
            exps.DEFAULT_DISCOUNT_FACTORS
        )


class TestGaussianIllustrationNoiseSpecs:
    def test_relative_noise_with_unit_magnitude(self):
        specs = exps.gaussian_illustration_experiment_specs()
        for exp in specs:
            noise = exp["args"]["reward_noise"]
            assert noise["name"] == "relative-gaussian"
            # unit = modal per-step reward magnitude, 1 for all envs
            assert noise["args"]["reference_std"] == 1.0
            # sigma = std_multiplier * reference_std, and the multiplier
            # is the sweep value itself
            assert noise["args"]["std_multiplier"] >= 0.0

    def test_std_multipliers_cover_the_sweep(self):
        specs = exps.gaussian_illustration_experiment_specs()
        multipliers = sorted(
            {exp["args"]["reward_noise"]["args"]["std_multiplier"] for exp in specs}
        )
        assert multipliers == [0.0, 0.25, 0.5, 1.0, 2.0]

    def test_metadata_carries_no_noise_scale(self):
        for exp in exps.gaussian_illustration_experiment_specs():
            assert "noise_scale" not in (exp.get("metadata") or {})


class TestGaussianIllustrationShapingSpecs:
    """Every family env pays a potential-based reward and exposes its potential."""

    _SHAPING: Mapping[str, str] = {
        "MountainCar-v0": "mountain-car-height-pbrs",
        "Acrobot-v1": "acrobot-tip-height-pbrs",
        "GridWorld-MINES": "gridworld-distance-pbrs",
    }

    def test_shaping_and_potential_wrapper_per_env(self):
        specs = exps.gaussian_illustration_experiment_specs()
        unshaped = {"LunarLander-v3"}
        assert {spec["name"] for spec in specs} == set(self._SHAPING) | unshaped
        for spec in specs:
            if spec["name"] in unshaped:
                # the environment's own reward, with no shaping wrapper
                assert "reward_shaping" not in spec["args"]
                assert "reward_obs" not in spec["args"]
                continue
            shaping = self._SHAPING[spec["name"]]
            assert spec["args"]["reward_shaping"]["name"] == shaping
            assert spec["args"]["reward_obs"]["name"] == f"{shaping}-potential"

    def test_discount_is_not_pinned_in_the_shaping_args(self):
        # task.create_env injects the problem spec's gamma, so the shaping
        # discount follows each arm
        for spec in exps.gaussian_illustration_experiment_specs():
            if "reward_shaping" not in spec["args"]:
                continue
            args = spec["args"]["reward_shaping"].get("args") or {}
            assert "gamma" not in args

    def test_gridworld_pins_the_quadratic_exponent(self):
        # exponent 1 pays a constant reward on every step of a shortest
        # path, which removes the signal the family estimates
        for spec in exps.gaussian_illustration_experiment_specs():
            if spec["name"] != "GridWorld-MINES":
                continue
            args = spec["args"]["reward_shaping"]["args"]
            assert args["exponent"] == 2.0

    def test_every_env_runs_at_least_two_steps(self):
        configured = set()
        for spec in exps.gaussian_illustration_experiment_specs():
            if spec["name"] in configured:
                continue
            configured.add(spec["name"])
            env_args = dict(spec["args"])
            # the reward without the noise the sweep adds
            env_args.pop("reward_noise", None)
            proxied = task.create_env(spec["name"], args=env_args, gamma=1.0)
            try:
                rewards = _rollout_rewards(proxied.env, seed=0, steps=100)
            finally:
                proxied.env.close()
                proxied.proxy.close()
            assert len(rewards) >= 2, spec["name"]


class TestBladeTdRewardSampling:
    def test_gaussian_illustration_consumes_posterior_mean(self):
        # the family compares reward estimators; a per-episode posterior
        # weight draw is a third stochastic source, so BLADE-TD consumes
        # the fixed posterior mean
        specs = exps.gaussian_illustration_experiment_specs()
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
            "MountainCar-v0": encodings.mc_potential_curr_succ(),
            "Acrobot-v1": encodings.acrobot_potential_curr_succ(),
            "GridWorld-MINES": encodings.gw_potential_curr_succ(),
            "LunarLander-v3": encodings.lunar_lander_reward(),
        }
        for spec in exps.gaussian_illustration_experiment_specs():
            for mapper_name in ("least-lfa", "bayes-least-lfa"):
                for args in _estimator_args(spec, mapper_name):
                    assert args["transition_feats_spec"] == expected[spec["name"]]

    def test_encoding_is_a_campaign_cell(self):
        # the selected encodings are campaign encodings, so a control run
        # compares against a campaign cell rather than a third recipe
        campaign: dict[str, list[Mapping[str, Any]]] = {}
        for spec in encodings_exps.experiment_specs():
            for args in _estimator_args(spec, "least-lfa"):
                campaign.setdefault(spec["name"], []).append(
                    args["transition_feats_spec"]
                )
        for spec in exps.gaussian_illustration_experiment_specs():
            # LunarLander is not an environment of the estimation campaign, so
            # its encoding has no campaign cell to match.
            if spec["name"] not in campaign:
                continue
            for args in _estimator_args(spec, "least-lfa"):
                assert args["transition_feats_spec"] in campaign[spec["name"]]

    def test_feature_map_dimensions(self):
        # current potential entry and constant, gated successor entry, plus
        # the per-env gated term: MountainCar 3, Acrobot 3, GridWorld 4.
        # LunarLander carries the basis on both sides of the transition plus a
        # fuel column per action and the two terminal columns.
        expected = {
            "MountainCar-v0": 3,
            "Acrobot-v1": 3,
            "GridWorld-MINES": 4,
            "LunarLander-v3": 16,
        }
        measured: dict[str, int] = {}
        for spec in exps.gaussian_illustration_experiment_specs():
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
        specs = exps.bayes_least_specs(
            init_attempt_estimation_episodes=(10,), transition_feats_specs=self._FEATS
        )
        assert specs
        for spec in specs:
            assert spec["reward_mapper"]["name"] == "bayes-least-lfa"
            assert spec["reward_mapper"]["args"]["error_variance"] == "unit"

    def test_nig_mode_selects_the_nig_mapper_name(self):
        # aggregation labels a run by its mapper name alone, so the NIG
        # mode needs a name of its own to be separated from unit variance
        specs = exps.bayes_least_specs(
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
            exps.bayes_least_specs(
                init_attempt_estimation_episodes=(10,),
                transition_feats_specs=self._FEATS,
                error_variance="inverse-gamma",
            )


class TestEstimatorConfiguration:
    """The estimator configurations are unchanged by the encoding selection."""

    def test_least_keeps_the_factors_gate(self):
        for spec in exps.gaussian_illustration_experiment_specs():
            for args in _estimator_args(spec, "least-lfa"):
                assert args["estimation_check"] == "factors"
                assert args["attempt_estimation_episode"] == 10
                assert args["estimation_buffer_mult"] == (
                    constants.DEFAULT_ESTIMATION_BUFFER_MULT
                )

    def test_bayes_keeps_the_absent_gate(self):
        # no estimation_check key: the wrapper's default is no gate
        for spec in exps.gaussian_illustration_experiment_specs():
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


def _rollout_rewards(env, seed: int, steps: int) -> set[float]:
    """Distinct rewards a seeded random policy collects."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    rewards = set()
    for _ in range(steps):
        action = int(rng.integers(0, env.action_space.n))
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.add(round(float(reward), 12))
        if terminated or truncated:
            env.reset(seed=seed)
    return rewards


class TestLearningRateRules:
    def test_dimension_rule_divides_by_the_observation_width(self):
        assert exps.learning_rate_from_dimension(0.1, 13) == pytest.approx(0.1 / 13)
        assert exps.learning_rate_from_dimension(0.1, 26) == pytest.approx(
            0.5 * exps.learning_rate_from_dimension(0.1, 13)
        )

    def test_active_feature_rule_divides_by_the_tiling_count(self):
        assert exps.learning_rate_from_active_features(0.1, 32) == pytest.approx(
            0.003125
        )
        assert exps.learning_rate_from_active_features(0.1, 64) == pytest.approx(
            0.5 * exps.learning_rate_from_active_features(0.1, 32)
        )

    def test_fixed_rule_ignores_the_dimension(self):
        first = exps.learning_rate_fixed(0.1, 13)
        second = exps.learning_rate_fixed(0.1, 4096)
        assert first == second == exps.LEARNING_RATE_SPEC["args"]["initial_lr"]

    def test_every_rule_rejects_a_nonpositive_dimension(self):
        with pytest.raises(ValueError, match="positive"):
            exps.learning_rate_from_dimension(0.1, 0)
        with pytest.raises(ValueError, match="positive"):
            exps.learning_rate_from_active_features(0.1, 0)


class TestDerivedLearningRateConfig:
    def test_dim_rule_resolves_against_the_wrapped_observation_width(self):
        config, provenance = exps.derived_learning_rate_config(
            env_name="LunarLander-v3",
            env_args=_LUNAR_BASE_ARGS,
            feats_specs=_LUNAR_FEATS_SPECS,
            rule="dim",
        )
        assert provenance["env_name"] == "LunarLander-v3"
        assert provenance["learning_rate_rule"] == "dim"
        assert provenance["base_step_size"] == 0.1
        assert provenance["observation_dim"] == 13
        assert provenance["dimension_used"] == 13
        assert provenance["num_tilings"] is None
        assert config["args"]["initial_lr"] == pytest.approx(0.1 / 13)
        assert provenance["learning_rate"] == pytest.approx(0.1 / 13)

    def test_actives_rule_resolves_against_the_tiling_count(self):
        config, provenance = exps.derived_learning_rate_config(
            env_name="LunarLander-v3",
            env_args=_LUNAR_BASE_ARGS,
            feats_specs=_LUNAR_FEATS_SPECS,
            rule="actives",
        )
        assert provenance["observation_dim"] == 13
        assert provenance["dimension_used"] == 32
        assert provenance["num_tilings"] == 32
        assert config["args"]["initial_lr"] == pytest.approx(0.1 / 32)

    def test_dim_rule_counts_the_wrapped_observation_not_the_tile_input(self):
        config, provenance = exps.derived_learning_rate_config(
            env_name="MountainCar-v0",
            env_args=_MC_BASE_ARGS,
            feats_specs=_MC_FEATS_SPECS,
            rule="dim",
        )
        # the tile op receives 2 dims (the potential column is dropped); the
        # wrapped observation carries 3 components
        assert provenance["observation_dim"] == 3
        assert config["args"]["initial_lr"] == pytest.approx(0.1 / 3)

    def test_actives_rule_reads_the_default_tiling_count_through_the_pipeline(self):
        config, provenance = exps.derived_learning_rate_config(
            env_name="MountainCar-v0",
            env_args=_MC_BASE_ARGS,
            feats_specs=_MC_FEATS_SPECS,
            rule="actives",
        )
        assert provenance["num_tilings"] == 8
        assert config["args"]["initial_lr"] == pytest.approx(0.1 / 8)

    def test_fixed_rule_resolves_without_constructing_an_env(self, monkeypatch):
        def fail_create_env(*args, **kwargs):
            raise AssertionError("the fixed rule must not construct an env")

        monkeypatch.setattr(exps.task, "create_env", fail_create_env)
        config, provenance = exps.derived_learning_rate_config(
            env_name="MountainCar-v0",
            env_args={},
            feats_specs=[],
            rule="fixed",
        )
        assert (
            config["args"]["initial_lr"]
            == (exps.LEARNING_RATE_SPEC["args"]["initial_lr"])
        )
        assert provenance["learning_rate_rule"] == "fixed"
        assert provenance["observation_dim"] is None
        assert provenance["dimension_used"] is None
        assert provenance["num_tilings"] is None

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown learning-rate rule"):
            exps.derived_learning_rate_config(
                env_name="MountainCar-v0",
                env_args={},
                feats_specs=[],
                rule="nope",
            )


class TestGaussianIllustrationIncludesLunarLander:
    def test_lunar_lander_is_a_peer_env_of_the_illustration_family(self):
        specs = exps.gaussian_illustration_experiment_specs(
            noise_std_multipliers=(0.0,)
        )
        names = [spec["name"] for spec in specs]
        assert "LunarLander-v3" in names
        # One spec per (env, multiplier): the family, not a set of its own.
        assert len(names) == len(set(names))

    def test_lunar_spec_carries_no_shaping_and_the_reward_observation(self):
        spec = _lunar_spec()
        assert "reward_shaping" not in spec["args"]
        assert "reward_obs" not in spec["args"]
        assert spec["args"]["wrapper"] == constants.LUNAR_LANDER_REWARD_OBS
        assert spec["args"]["max_episode_steps"] == 1000

    def test_lunar_spec_uses_the_derived_rate_for_every_arm(self):
        spec = _lunar_spec()
        np.testing.assert_allclose(sorted(_rates(spec["problem_specs"])), [0.1 / 13])
        assert spec["metadata"]["learning_rate_rule"] == "dim"
        assert spec["metadata"]["learning_rate"] == pytest.approx(0.1 / 13)

    def test_lunar_spec_selects_a_different_rule_without_touching_the_env(self):
        family_spec = _lunar_spec()
        actives_template = exps._lunar_lander_template(rule="actives")
        np.testing.assert_allclose(
            sorted(_rates(actives_template["problem_specs_fn"]())), [0.1 / 32]
        )
        assert (
            actives_template["base_args"]["wrapper"] == family_spec["args"]["wrapper"]
        )
        assert actives_template["feats_specs"] == family_spec["feats_specs"]

    def test_lunar_spec_runs_ten_thousand_episodes_per_run(self):
        # 1000 episodes per epoch, the job script's --num-episodes.
        assert _lunar_spec()["epochs"] == 10

    def test_lunar_estimator_arms_use_the_reward_encoding(self):
        spec = _lunar_spec()
        mappers = [problem["reward_mapper"] for problem in spec["problem_specs"]]
        estimator_arms = [
            mapper for mapper in mappers if mapper["name"] in _ESTIMATOR_MAPPERS
        ]
        assert estimator_arms
        expected = encodings.lunar_lander_reward()
        for mapper in estimator_arms:
            assert mapper["args"]["transition_feats_spec"] == expected

    def test_other_envs_keep_the_control_rate(self):
        specs = exps.gaussian_illustration_experiment_specs(
            noise_std_multipliers=(0.0,)
        )
        others = [spec for spec in specs if spec["name"] != "LunarLander-v3"]
        assert others
        for spec in others:
            for problem in spec["problem_specs"]:
                assert (
                    problem["learning_rate_config"]["args"]["initial_lr"]
                    == exps.LEARNING_RATE_SPEC["args"]["initial_lr"]
                )

    def test_every_env_spec_records_its_learning_rate_provenance(self):
        specs = exps.gaussian_illustration_experiment_specs(
            noise_std_multipliers=(0.0,)
        )
        for spec in specs:
            metadata = spec["metadata"]
            assert metadata["env_name"] == spec["name"]
            for problem in spec["problem_specs"]:
                assert (
                    problem["learning_rate_config"]["args"]["initial_lr"]
                    == metadata["learning_rate"]
                )
            if spec["name"] == "LunarLander-v3":
                assert metadata["learning_rate_rule"] == "dim"
                assert metadata["learning_rate"] == pytest.approx(0.1 / 13)
            else:
                assert metadata["learning_rate_rule"] == "fixed"
                assert (
                    metadata["learning_rate"]
                    == (exps.LEARNING_RATE_SPEC["args"]["initial_lr"])
                )

    def test_custom_rate_reaches_every_helper_spec(self):
        config = {"name": "constant", "args": {"initial_lr": 0.123}}
        specs = (
            exps.common_problem_specs(learning_rate_config=config)
            + exps.least_specs(
                attempt_estimation_episodes=(1,),
                transition_feats_specs=[{"name": "current-features", "args": {}}],
                learning_rate_config=config,
            )
            + exps.bayes_least_specs(
                init_attempt_estimation_episodes=(1,),
                transition_feats_specs=[{"name": "current-features", "args": {}}],
                learning_rate_config=config,
            )
        )
        assert specs
        for problem in specs:
            assert problem["learning_rate_config"] is config


_ESTIMATOR_MAPPERS = ("least-lfa", "bayes-least-lfa")


def _rates(problem_specs: Sequence[Mapping[str, Any]]) -> list[float]:
    """The learning rates a set of problem specs carries, one per distinct value."""
    return sorted(
        {
            problem["learning_rate_config"]["args"]["initial_lr"]
            for problem in problem_specs
        }
    )


def _lunar_spec() -> Mapping[str, Any]:
    """The illustration family's single LunarLander-v3 spec."""
    specs = exps.gaussian_illustration_experiment_specs(noise_std_multipliers=(0.0,))
    return next(spec for spec in specs if spec["name"] == "LunarLander-v3")


_LUNAR_BASE_ARGS = {
    "max_episode_steps": 1000,
    "wrapper": constants.LUNAR_LANDER_REWARD_OBS,
}
_LUNAR_FEATS_SPECS = [
    [
        {
            "name": "splice-tile-observation-action-ft",
            "args": {
                "tiling_dim": 8,
                "num_tilings": 32,
                "hash_dim": 2048,
            },
        }
    ]
]
_MC_BASE_ARGS = {
    "max_episode_steps": 2500,
    "reward_shaping": {
        "name": "mountain-car-height-pbrs",
        "args": {"scale": 1.0},
    },
    "reward_obs": {
        "name": "mountain-car-height-pbrs-potential",
        "args": {},
    },
}
_MC_FEATS_SPECS = [
    [
        encodings.keep_raw_dims(raw_dim=2, observation_dim=3),
        {
            "name": "tile-observation-action-ft",
            "args": {"tiling_dim": 4},
        },
    ]
]
