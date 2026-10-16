"""Learning-time evaluation metrics for delayed-reward experiments.

Operationalizes the evaluation protocol of the RUDDER paper (Return
Decomposition for Delayed Rewards, NeurIPS 2020): learning time is the
number of training episodes until a method's greedy-policy evaluation
return reaches a threshold derived from the maximal expected return.
The definitions are extracted in papers/summary/rudder.md and illustrated
on simulated data in notebooks/learning-time-metrics.ipynb.

Experiment data comes from task.policy_control, which evaluates the
greedy policy on the un-noised proxy environment at every
`log_episode_frequency` training episodes and logs the cumulative mean
of the evaluation means. The per-checkpoint evaluation mean is recovered
exactly by differencing (see `calc_checkpoint_eval_means`).

Aggregation across runs follows the paper: median learning time with a
quantile band, where runs that never reach the threshold are
right-censored at the training cap and the band comes from a Log-normal
fit on the censored data. Method comparisons use a paired Wilcoxon
signed-rank test.
"""

from typing import Any

import numpy as np
from scipy import optimize, stats


def calc_threshold(r_max: float, shortfall: float = 0.1) -> float:
    """Learning-time threshold: `r_max` reduced by a shortfall fraction.

    For non-negative `r_max` the result equals the paper's
    `(1 - shortfall) * r_max` (90% of the maximal expected return at the
    default shortfall of 0.1). For negative returns — the control
    environments return between roughly -1 and -300 — a multiplicative
    factor would place the threshold above the maximum, i.e. better than
    the best observed return, so the shortfall is taken on the magnitude
    instead: `r_max - shortfall * |r_max|`.

    Args:
        r_max: maximal expected return; the empirical stand-in is the
            best evaluation return observed in the experiment family.
        shortfall: allowed fraction of the magnitude below `r_max`.

    Returns:
        The return threshold a run must reach.
    """
    return float(r_max - shortfall * abs(r_max))


def calc_checkpoint_eval_means(cumulative_means: np.ndarray) -> np.ndarray:
    """Per-checkpoint evaluation means from logged cumulative means.

    `task.policy_control` logs the mean over all evaluations so far.
    With `C[j]` the logged cumulative mean after `j + 1` evaluations,
    the `(j + 1)`-th evaluation mean is
    `e[j] = (j + 1) * C[j] - j * C[j - 1]`, so the per-checkpoint
    means are recovered exactly by differencing — no approximation and
    no re-run of the experiments.

    Args:
        cumulative_means: (n_runs, n_checkpoints) logged returns, where
            column `j` holds the cumulative mean after `j + 1`
            evaluations. A 1-D input is treated as a single run.

    Returns:
        (n_runs, n_checkpoints) per-checkpoint evaluation means; the
        first column equals the first cumulative mean.
    """
    is_1d = np.ndim(cumulative_means) == 1
    cumulative = np.atleast_2d(np.asarray(cumulative_means, dtype=float))
    checkpoint_idx = np.arange(cumulative.shape[1], dtype=float)
    # C[j-1] shifted right by one column; C[-1] does not exist, so the
    # first column of the shift is zero
    shifted = np.concatenate(
        [np.zeros((cumulative.shape[0], 1)), cumulative[:, :-1]], axis=1
    )
    eval_means = (checkpoint_idx + 1) * cumulative - checkpoint_idx * shifted
    # indexing an ndarray is typed Any; asarray keeps the row a view
    return np.asarray(eval_means[0]) if is_1d else eval_means


def calc_learning_time_by_eval(
    eval_means: np.ndarray,
    eval_episodes: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Learning time under the greedy-policy evaluation criterion
    (RUDDER appendix A4.1.1): training episodes at the first checkpoint
    whose evaluation mean reaches `threshold`.

    Args:
        eval_means: (n_runs, n_checkpoints) per-checkpoint evaluation
            means (see `calc_checkpoint_eval_means`).
        eval_episodes: (n_checkpoints,) training episodes at the
            checkpoints, ascending.
        threshold: return threshold (see `calc_threshold`).

    Returns:
        times: (n_runs,) training episodes at the first crossing;
            np.inf for runs that never reach the threshold.
        solved: (n_runs,) mask of runs that reach the threshold.
    """
    eval_means = np.atleast_2d(np.asarray(eval_means, dtype=float))
    eval_episodes = np.asarray(eval_episodes)
    crossing = eval_means >= threshold
    solved = crossing.any(axis=1)
    # argmax returns 0 for rows without a crossing; masked by `solved`
    first_crossing = np.argmax(crossing, axis=1)
    times = eval_episodes[first_crossing]
    return np.where(solved, times, np.inf), solved


def calc_learning_time_summary(
    times: np.ndarray,
    cap: float,
    quantiles: tuple[float, ...] = (0.4, 0.6),
) -> dict[str, Any]:
    """Median learning time and quantile band (RUDDER appendix A4.1.1).

    Finished runs have finite `times`; censored runs — training stopped
    at `cap` without reaching the threshold — have np.inf. The median is
    the empirical median of the finished runs, reported by the paper as
    robust when more than half of the runs have finished. Without
    censoring the band uses empirical quantiles; with censoring a
    Log-normal distribution is fitted to the right-censored data
    (finished runs contribute the density, censored runs the survival
    function at the cap) with mu fixed to log(median of finished runs)
    and sigma from maximum likelihood, the paper's estimation procedure.

    Args:
        times: (n_runs,) learning times; np.inf for censored runs.
        cap: training cap at which censored runs stopped.
        quantiles: band quantiles of the summary.

    Returns:
        Mapping with `median`, `band` (per-quantile values),
        `n_runs`, `n_finished`, `n_censored`, `median_robust`, and
        `sigma_fit` (None when the fit is not used).

    Raises:
        ValueError: no finished runs, so the median is not estimable;
            the paper omits such configurations from its learning-time
            plots.
    """
    times = np.asarray(times, dtype=float)
    finished = times[np.isfinite(times)]
    n_runs = len(times)
    n_censored = n_runs - len(finished)
    if finished.size == 0:
        raise ValueError("no finished runs: the median learning time is not estimable")
    median = float(np.median(finished))
    result: dict[str, Any] = {
        "median": median,
        "band": {},
        "n_runs": int(n_runs),
        "n_finished": int(finished.size),
        "n_censored": int(n_censored),
        "median_robust": bool(finished.size > n_runs / 2),
        "sigma_fit": None,
    }
    if n_censored == 0:
        result["band"] = {q: float(np.quantile(finished, q)) for q in quantiles}
        return result
    mu = np.log(median)
    fit = optimize.minimize(
        _calc_censored_neg_log_likelihood,
        x0=_calc_sigma_init(finished),
        args=(finished, cap, mu, n_censored),
        method="L-BFGS-B",
        bounds=[(1e-6, 10.0)],
    )
    sigma_fit = float(fit.x[0])
    result["sigma_fit"] = sigma_fit
    result["band"] = {
        q: float(stats.lognorm.ppf(q, s=sigma_fit, scale=np.exp(mu))) for q in quantiles
    }
    return result


def calc_wilcoxon_signed_rank(
    times_candidate: np.ndarray,
    times_reference: np.ndarray,
    cap: float,
) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired learning times
    (RUDDER appendix A4.1.2 and A4.1.3).

    Runs are paired by seed index; both arrays must have equal length.
    Censored runs (np.inf) are substituted with `cap`, a lower bound on
    their true learning time; the paper does not state the handling. An
    all-zero difference vector (a self-comparison) reports the null
    p-value 0.5, the value shown in the paper's sanity row.

    Args:
        times_candidate: (n_runs,) candidate learning times; np.inf when
            censored.
        times_reference: (n_runs,) reference learning times, paired by
            index.
        cap: training cap substituted for censored runs.

    Returns:
        Mapping with `statistic`, `pvalue`, `n_pairs`, and
        `n_pairs_both_censored`.

    Raises:
        ValueError: the arrays have different lengths.
    """
    if len(times_candidate) != len(times_reference):
        raise ValueError("paired Wilcoxon signed-rank requires equal-length samples")
    candidate = np.where(np.isfinite(times_candidate), times_candidate, cap)
    reference = np.where(np.isfinite(times_reference), times_reference, cap)
    differences = candidate - reference
    n_both_censored = int(
        np.sum(~np.isfinite(times_candidate) & ~np.isfinite(times_reference))
    )
    result: dict[str, Any] = {
        "n_pairs": len(differences),
        "n_pairs_both_censored": n_both_censored,
    }
    if np.all(differences == 0):
        # the convention of the paper's Table A1 sanity row
        return {"statistic": np.nan, "pvalue": 0.5, **result}
    statistic, pvalue = stats.wilcoxon(candidate, reference, alternative="two-sided")
    return {"statistic": float(statistic), "pvalue": float(pvalue), **result}


def _calc_sigma_init(finished: np.ndarray) -> float:
    """Solves Var = (exp(sigma^2) - 1) * median^2 * exp(sigma^2) for
    sigma with mu = log(median): the variance-to-sigma transform that
    initializes the paper's maximum-likelihood estimation."""
    ratio = np.var(finished) / np.median(finished) ** 2
    root = (1.0 + np.sqrt(1.0 + 4.0 * ratio)) / 2.0
    return float(np.sqrt(np.log(root)))


def _calc_censored_neg_log_likelihood(
    sigma: np.ndarray,
    finished: np.ndarray,
    cap: float,
    mu: float,
    n_censored: int,
) -> float:
    """Negative log-likelihood of the learning times under a
    Log-normal with fixed mu: finished runs contribute the density,
    censored runs the survival function at the cap."""
    log_density = stats.lognorm.logpdf(finished, s=sigma, scale=np.exp(mu))
    log_survival = stats.lognorm.logsf(cap, s=sigma, scale=np.exp(mu))
    # `sigma` arrives from the optimizer as a length-1 array, so the
    # survival term is a length-1 array; item() extracts the scalar
    return float(np.asarray(-(log_density.sum() + n_censored * log_survival)).item())
