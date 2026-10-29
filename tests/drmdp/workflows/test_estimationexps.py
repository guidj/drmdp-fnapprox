"""Tests for estimationexps — the estimation-only encoding campaign specs."""

import pytest

from drmdp import task
from drmdp.workflows import estimationexps

EXPECTED_LABELS = {
    "MountainCar-v0": {
        "structural-sine",
        "fourier-succ-o5",
        "succ-bins-54",
        "sa-bins-4x2",
        "sa-bins-18x4",
        "current-tiles-td3",
        "currsucc-tiles-td3",
    },
    "Acrobot-v1": {"action-termination", "nonterminal-action", "sa-bins-2x6"},
    "GridWorld-MINES": {
        "transition-anomaly",
        "next-cell-ohe",
        "sa-ohe",
        "tiles-td3",
    },
}


class TestEstimationExperimentSpecs:
    """The campaign grid: encodings, methods, delays, noise levels."""

    def test_one_spec_per_env_encoding_noise_combination(self):
        specs = estimationexps.estimation_experiment_specs()
        assert len(specs) == 28  # 14 encodings x 2 noise multipliers
        combos = {(spec["name"], spec["metadata"]["encoding"]) for spec in specs}
        for env_name, labels in EXPECTED_LABELS.items():
            for label in labels:
                assert (env_name, label) in combos
        multipliers = {spec["metadata"]["noise_std_multiplier"] for spec in specs}
        assert multipliers == {0.0, 1.0}

    def test_problem_specs_hold_the_estimator_mode_and_delay_axes(self):
        specs = estimationexps.estimation_experiment_specs()
        for spec in specs:
            # 2 estimators x 3 check modes x 2 delays = 12
            assert len(spec["problem_specs"]) == 12
            least_specs = [
                problem
                for problem in spec["problem_specs"]
                if problem["reward_mapper"]["name"] == "least-lfa"
            ]
            assert len(least_specs) == 6  # 3 modes x 2 delays
            least_modes = {
                problem["reward_mapper"]["args"]["estimation_check"]
                for problem in least_specs
            }
            assert least_modes == {"none", "factors", "full-rank"}
            bayes_specs = [
                problem
                for problem in spec["problem_specs"]
                if problem["reward_mapper"]["name"] == "bayes-least-lfa"
            ]
            assert len(bayes_specs) == 6  # 3 modes x 2 delays
            bayes_modes = {
                problem["reward_mapper"]["args"]["estimation_check"]
                for problem in bayes_specs
            }
            assert bayes_modes == {"none", "factors", "full-rank"}
            delays = {str(problem["delay_config"]) for problem in spec["problem_specs"]}
            assert len(delays) == 2
            assert spec["feats_specs"] == [[]]

    def test_expected_run_count_matches_the_grid(self):
        # 14 encodings x 6 methods x 2 delays x 2 noise = 336 per run
        assert estimationexps.expected_run_count(num_runs=1) == 336
        assert estimationexps.expected_run_count(num_runs=5) == 1680

    def test_episode_cap(self):
        assert estimationexps.episode_cap("MountainCar-v0") == 500
        assert estimationexps.episode_cap("Acrobot-v1") == 500
        assert estimationexps.episode_cap("GridWorld-MINES") == 500
        with pytest.raises(ValueError, match="no episode cap"):
            estimationexps.episode_cap("UnknownEnv-v0")


class TestSpecsBuildOnTheirEnvironments:
    """Every encoding resolves to a transition-feature map of the right form."""

    def test_every_encoding_builds(self):
        specs = estimationexps.estimation_experiment_specs()
        seen = set()
        for spec in specs:
            env_name = spec["name"]
            encoding = spec["metadata"]["encoding"]
            if (env_name, encoding) in seen:
                continue
            seen.add((env_name, encoding))
            proxied = task.create_env(
                env_name, args=spec["args"], gamma=estimationexps.GAMMA
            )
            mapper_args = dict(spec["problem_specs"][0]["reward_mapper"]["args"])
            feats_fn = task.estimator_feats_fn(proxied.proxy, "least-lfa", mapper_args)
            assert feats_fn.feature_dim >= 2
            proxied.env.close()
            proxied.proxy.close()
