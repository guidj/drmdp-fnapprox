"""Tests for bayesleastjob.py — spec builders."""

from drmdp.workflows import bayesleastjob


class TestBayesLeastSpecs:
    def test_covers_all_delay_values(self):
        result = bayesleastjob.bayes_least_specs(
            feats_spec={"name": "scale", "args": None}
        )
        delays = {
            spec["delay_config"]["args"]["delay"]
            for spec in result
            if spec["delay_config"]
        }
        assert delays == {2, 4, 6, 8}

    def test_covers_all_discount_factors(self):
        result = bayesleastjob.bayes_least_specs(
            feats_spec={"name": "scale", "args": None}
        )
        gammas = {spec["gamma"] for spec in result}
        assert gammas == {1.0, 0.99, 0.995}

    def test_feats_spec_wired_into_reward_mapper(self):
        feats = {"name": "custom-ft", "args": {"dim": 7}}
        result = bayesleastjob.bayes_least_specs(feats_spec=feats)
        for spec in result:
            mapper_feats = spec["reward_mapper"]["args"]["transition_feats_spec"]
            assert mapper_feats["name"] == "current-features"
            assert mapper_feats["args"]["current_spec"] == feats
