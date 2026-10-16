"""Tests for learning-time evaluation learningtime."""

import numpy as np
import pytest
from scipy import stats as sstats

from drmdp import learningtime


class TestCalcThreshold:
    """Threshold derivation for positive and negative returns."""

    def test_positive_r_max_matches_paper_formula(self):
        # the paper's tasks: max return 100 -> threshold 90
        assert learningtime.calc_threshold(100.0) == pytest.approx(90.0)

    def test_negative_r_max_allows_shortfall_on_magnitude(self):
        # a multiplicative 0.9 factor would give -188.1, above the best
        # observed return; the shortfall lands below the maximum
        assert learningtime.calc_threshold(-209.0) == pytest.approx(-229.9)

    def test_negative_threshold_is_never_above_r_max(self):
        assert learningtime.calc_threshold(-0.5) < -0.5

    def test_custom_shortfall(self):
        assert learningtime.calc_threshold(100.0, shortfall=0.2) == (
            pytest.approx(80.0)
        )

    def test_zero_r_max_is_zero(self):
        assert learningtime.calc_threshold(0.0) == pytest.approx(0.0)


class TestCalcCheckpointEvalMeans:
    """Exact recovery of per-checkpoint means from cumulative means."""

    def test_recovers_known_sequence(self):
        eval_means = np.array([[10.0, 20.0, -30.0, 50.0]])
        cumulative = np.cumsum(eval_means, axis=1) / np.arange(1, 5)
        recovered = learningtime.calc_checkpoint_eval_means(cumulative)
        np.testing.assert_allclose(recovered, eval_means, atol=1e-6)

    def test_recovers_random_curves_for_many_runs(self):
        rng = np.random.default_rng(seed=42)
        eval_means = rng.normal(-50.0, 20.0, size=(7, 40))
        cumulative = np.cumsum(eval_means, axis=1) / np.arange(1, 41)
        recovered = learningtime.calc_checkpoint_eval_means(cumulative)
        np.testing.assert_allclose(recovered, eval_means, atol=1e-6)

    def test_single_checkpoint_returns_itself(self):
        # one evaluation: cumulative mean equals the evaluation mean
        assert learningtime.calc_checkpoint_eval_means(
            np.array([[-13.2]])
        ) == pytest.approx(-13.2)

    def test_one_dimensional_input_treated_as_single_run(self):
        cumulative = np.array([-5.0, -5.0, -5.0])
        recovered = learningtime.calc_checkpoint_eval_means(cumulative)
        np.testing.assert_allclose(recovered, [-5.0, -5.0, -5.0], atol=1e-6)


class TestCalcLearningTimeByEval:
    """First-crossing logic of the evaluation criterion."""

    def test_first_crossing_checkpoint_episode(self):
        eval_means = np.array([[-100.0, -95.0, -89.0, -70.0, -70.0]])
        episodes = np.array([0, 25, 50, 75, 100])
        threshold = -90.0
        times, solved = learningtime.calc_learning_time_by_eval(
            eval_means, episodes, threshold
        )
        assert solved[0]
        assert times[0] == 50

    def test_never_reaching_threshold_is_censored(self):
        eval_means = np.array([[-100.0, -99.0, -98.0]])
        times, solved = learningtime.calc_learning_time_by_eval(
            eval_means, np.array([0, 25, 50]), -90.0
        )
        assert not solved[0]
        assert np.isinf(times[0])

    def test_uses_first_not_later_crossing(self):
        # dips below the threshold after crossing: the first crossing
        # counts, matching "episodes required to solve the task"
        eval_means = np.array([[-100.0, -85.0, -95.0, -80.0]])
        times, solved = learningtime.calc_learning_time_by_eval(
            eval_means, np.array([0, 25, 50, 75]), -90.0
        )
        assert solved[0]
        assert times[0] == 25

    def test_runs_are_independent(self):
        eval_means = np.array(
            [
                [-100.0, -85.0, -80.0],
                [-100.0, -100.0, -100.0],
                [-80.0, -80.0, -80.0],
            ]
        )
        times, solved = learningtime.calc_learning_time_by_eval(
            eval_means, np.array([0, 25, 50]), -90.0
        )
        np.testing.assert_array_equal(solved, [True, False, True])
        assert times[0] == 25
        assert np.isinf(times[1])
        # the run starts above the threshold: solved before training
        assert times[2] == 0


class TestCalcLearningTimeSummary:
    """Median and quantile band, with and without censoring."""

    def test_uncensored_uses_empirical_quantiles(self):
        times = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        summary = learningtime.calc_learning_time_summary(times, cap=1000.0)
        assert summary["median"] == pytest.approx(300.0)
        assert summary["n_censored"] == 0
        assert summary["sigma_fit"] is None
        assert summary["band"][0.4] == pytest.approx(np.quantile(times, 0.4))

    def test_censored_lognormal_fit_recovers_distribution(self):
        rng = np.random.default_rng(seed=42)
        true_median, true_sigma = 500.0, 0.6
        cap = 1500.0
        sample = rng.lognormal(np.log(true_median), true_sigma, size=400)
        observed = np.where(sample > cap, np.inf, sample)
        summary = learningtime.calc_learning_time_summary(observed, cap)
        assert summary["n_censored"] > 0
        # mu is fixed to the finished-run median; the fitted sigma
        # recovers the true spread within sampling error
        assert summary["median"] == pytest.approx(true_median, rel=0.1)
        assert summary["sigma_fit"] == pytest.approx(true_sigma, rel=0.2)
        # the fitted band brackets the empirical uncensored band
        expected_q40 = true_median * np.exp(true_sigma * sstats.norm.ppf(0.4))
        assert summary["band"][0.4] == pytest.approx(expected_q40, rel=0.2)

    def test_median_from_finished_runs_only(self):
        times = np.array([100.0, 200.0, 300.0, np.inf, np.inf])
        summary = learningtime.calc_learning_time_summary(times, cap=2500.0)
        assert summary["median"] == pytest.approx(200.0)
        assert summary["n_finished"] == 3
        assert summary["n_censored"] == 2

    def test_robustness_flag_follows_paper_condition(self):
        assert learningtime.calc_learning_time_summary(
            np.array([1.0, 2.0, 3.0, np.inf, np.inf]), cap=10.0
        )["median_robust"]
        assert not learningtime.calc_learning_time_summary(
            np.array([1.0, 2.0, np.inf, np.inf, np.inf]), cap=10.0
        )["median_robust"]

    def test_no_finished_runs_raises(self):
        with pytest.raises(ValueError, match="no finished runs"):
            learningtime.calc_learning_time_summary(
                np.array([np.inf, np.inf]), cap=10.0
            )


class TestCalcWilcoxonSignedRank:
    """Paired comparison protocol, censoring, and degenerate cases."""

    def test_shifted_pairs_are_significant(self):
        rng = np.random.default_rng(seed=42)
        reference = rng.lognormal(np.log(500.0), 0.5, size=40)
        candidate = rng.lognormal(np.log(5000.0), 0.5, size=40)
        result = learningtime.calc_wilcoxon_signed_rank(
            candidate, reference, cap=10_000.0
        )
        assert result["pvalue"] < 0.01
        assert result["n_pairs"] == 40
        assert result["n_pairs_both_censored"] == 0

    def test_self_comparison_reports_null_pvalue(self):
        times = np.linspace(100.0, 500.0, num=40)
        result = learningtime.calc_wilcoxon_signed_rank(times, times, cap=1000.0)
        assert result["pvalue"] == pytest.approx(0.5)
        assert np.isnan(result["statistic"])

    def test_censored_runs_substituted_with_cap(self):
        reference = np.full(10, 100.0)
        candidate = np.full(10, 200.0)
        candidate[0] = np.inf
        result = learningtime.calc_wilcoxon_signed_rank(
            candidate, reference, cap=1000.0
        )
        # the censored pair contributes a cap-vs-100 difference
        assert result["n_pairs_both_censored"] == 0
        assert result["pvalue"] < 0.01

    def test_both_censored_pairs_counted(self):
        candidate = np.array([100.0, np.inf, np.inf])
        reference = np.array([50.0, np.inf, np.inf])
        result = learningtime.calc_wilcoxon_signed_rank(
            candidate, reference, cap=1000.0
        )
        assert result["n_pairs_both_censored"] == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            learningtime.calc_wilcoxon_signed_rank(
                np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), cap=10.0
            )
