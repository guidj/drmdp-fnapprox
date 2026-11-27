"""Tests for the bias-variance campaign spec builders."""

from drmdp import rewdelay
from drmdp.workflows.reward_estimation.bias_variance import exps


class TestLeastLfaSpecs:
    def test_cartesian_product_of_episodes_and_feats(self):
        feats = [
            [{"name": "a", "args": None}],
            [{"name": "b", "args": None}],
        ]
        result = exps.least_lfa_specs(
            attempt_estimation_episodes=(10, 50),
            feat_specs=feats,
        )
        assert len(result) == 4
        episodes_seen = {spec["args"]["attempt_estimation_episode"] for spec in result}
        assert episodes_seen == {10, 50}

    def test_feats_wired_into_spec_args(self):
        feats = [[{"name": "custom-ft", "args": {"dim": 3}}]]
        result = exps.least_lfa_specs(
            attempt_estimation_episodes=(10,),
            feat_specs=feats,
        )
        assert (
            result[0]["args"]["transition_feats_spec"]["args"]["current_spec"]
            == feats[0]
        )


class TestBayesLeastLfaSpecs:
    def test_mode_is_exponential(self):
        """
        The recorded mode must be a schedule the library accepts, and
        must be the default mode.
        """
        feats = [[{"name": "scale-observation-ft", "args": None}]]
        result = exps.bayes_least_lfa_specs(
            init_attempt_estimation_episodes=(10,),
            feat_specs=feats,
        )
        args = result[0]["args"]
        schedule = rewdelay.WindowedTaskSchedule(
            mode=args["mode"],
            init_update_ep=args["init_attempt_estimation_episode"],
        )
        assert schedule.mode == rewdelay.WindowedTaskSchedule.EXPONENTIAL
