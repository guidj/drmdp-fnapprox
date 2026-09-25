"""
Validation script: LEAST vs BLADE-TD vs IMR across grid sizes and tuning parameters.

Compares reward estimation methods on GridWorld environments of varying size,
measuring learned vs true reward accuracy, state-action coverage, and returns.
"""

import argparse
import collections
import itertools
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from drmdp import mathutils
from drmdp.workflows import controlexps
from drmdp.workflows.grids import rewest_runner

DELAY_LAMBDAS = (2, 5, 7)
SEEDS = (0, 1, 2)
GAMMAS = (0.99, 1.0)
EPSILON = 0.2
GRIDS_PER_SIZE = 2

EPSILON_SWEEP_VALUES = (0.1, 0.2, 0.4)
EPSILON_SWEEP_DELAY_LAM = 5
EPSILON_SWEEP_NUM_EPISODES = 500

METHOD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "identity": {},
    "least-lfa": {
        "attempt_estimation_episode": 10,
        "use_bias": False,
        "impute_value": 0,
        "estimation_buffer_mult": 25,
        "use_next_state": False,
        "check_factors": True,
    },
    "bayes-least-lfa": {
        "init_attempt_estimation_episode": 10,
        "use_bias": False,
        "impute_value": 0,
        "estimation_buffer_mult": 25,
        "use_next_state": False,
    },
    "impute-missing": {
        "impute_value": 0,
    },
}

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "identity": "FR",
    "least-lfa": "LEAST",
    "bayes-least-lfa": "BLADE-TD",
    "impute-missing": "IMR",
}


def build_delay_configs() -> List[Dict[str, Any]]:
    delay_configs = []
    for lam in DELAY_LAMBDAS:
        lower, upper = mathutils.poisson_exact_confidence_interval(observed_value=lam)
        delay_configs.append(
            {
                "name": "clipped-poisson",
                "args": {"lam": lam, "min_delay": max(2, lower), "max_delay": upper},
            }
        )
    return delay_configs


def load_grids(
    grid_files: Sequence[str],
    min_passes: int = 4,
    grids_per_size: int = GRIDS_PER_SIZE,
) -> List[Mapping[str, Any]]:
    all_grids: List[Mapping[str, Any]] = []
    for path in grid_files:
        grids = controlexps.load_solvable_grids(path=path, min_passes=min_passes)
        all_grids.extend(grids)

    specs_by_size: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for grid in all_grids:
        size_key = f"{grid['size'][0]}x{grid['size'][1]}"
        specs_by_size[size_key].append(grid)

    selected: List[Mapping[str, Any]] = []
    for size_key in sorted(specs_by_size.keys()):
        grids_for_size = specs_by_size[size_key][:grids_per_size]
        selected.extend(grids_for_size)
    return selected


def build_run_configs(
    grids: Sequence[Mapping[str, Any]],
    max_steps_values: Sequence[int],
    num_episodes_values: Sequence[int],
) -> List[Dict[str, Any]]:
    delay_configs = build_delay_configs()
    run_configs: List[Dict[str, Any]] = []

    delayed_methods = [name for name in METHOD_CONFIGS if name != "identity"]
    for (
        grid_spec,
        method,
        gamma,
        delay_config,
        seed,
        max_steps,
        num_episodes,
    ) in itertools.product(
        grids,
        delayed_methods,
        GAMMAS,
        delay_configs,
        SEEDS,
        max_steps_values,
        num_episodes_values,
    ):
        grid_with_steps = dict(grid_spec)
        grid_with_steps["max_episode_steps"] = max_steps

        run_configs.append(
            {
                "grid_spec": grid_with_steps,
                "method": method,
                "method_args": METHOD_CONFIGS[method],
                "delay_config": delay_config,
                "gamma": gamma,
                "num_episodes": num_episodes,
                "seed": seed,
                "epsilon": EPSILON,
            }
        )

    # "identity" (FR) has no delay to sweep over -- one run per remaining axis.
    for grid_spec, gamma, seed, max_steps, num_episodes in itertools.product(
        grids, GAMMAS, SEEDS, max_steps_values, num_episodes_values
    ):
        grid_with_steps = dict(grid_spec)
        grid_with_steps["max_episode_steps"] = max_steps

        run_configs.append(
            {
                "grid_spec": grid_with_steps,
                "method": "identity",
                "method_args": METHOD_CONFIGS["identity"],
                "delay_config": None,
                "gamma": gamma,
                "num_episodes": num_episodes,
                "seed": seed,
                "epsilon": EPSILON,
            }
        )

    return run_configs


def build_epsilon_sweep_configs(
    grids: Sequence[Mapping[str, Any]],
    max_steps_values: Sequence[int] = (200, 500),
    epsilon_values: Sequence[float] = EPSILON_SWEEP_VALUES,
) -> List[Dict[str, Any]]:
    """LEAST-only sweep isolating exploration rate from step-count/episode-count.

    One grid per size, a single delay level (median lambda), fixed
    num_episodes -- only epsilon and max_steps vary, to test whether low
    exploration (not buffer size or episode budget) is what blocks LEAST
    from reaching a reward estimate on GridWorld (`check_factors` coverage
    gate, see `agents/analyses/2026-09-04-least-blade-imr-reward-estimation-analysis.md`).
    """
    lower, upper = mathutils.poisson_exact_confidence_interval(
        observed_value=EPSILON_SWEEP_DELAY_LAM
    )
    delay_config = {
        "name": "clipped-poisson",
        "args": {
            "lam": EPSILON_SWEEP_DELAY_LAM,
            "min_delay": max(2, lower),
            "max_delay": upper,
        },
    }

    one_grid_per_size: List[Mapping[str, Any]] = []
    seen_sizes = set()
    for grid_spec in grids:
        size_key = f"{grid_spec['size'][0]}x{grid_spec['size'][1]}"
        if size_key not in seen_sizes:
            seen_sizes.add(size_key)
            one_grid_per_size.append(grid_spec)

    run_configs: List[Dict[str, Any]] = []
    for grid_spec, max_steps, epsilon, seed in itertools.product(
        one_grid_per_size, max_steps_values, epsilon_values, SEEDS
    ):
        grid_with_steps = dict(grid_spec)
        grid_with_steps["max_episode_steps"] = max_steps

        run_configs.append(
            {
                "grid_spec": grid_with_steps,
                "method": "least-lfa",
                "method_args": METHOD_CONFIGS["least-lfa"],
                "delay_config": delay_config,
                "gamma": 1.0,
                "num_episodes": EPSILON_SWEEP_NUM_EPISODES,
                "seed": seed,
                "epsilon": epsilon,
            }
        )
    return run_configs


def summarize_results(
    results: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for result in results:
        delay_config = result["delay_config"]
        summaries.append(
            {
                "env_name": result["env_name"],
                "method": METHOD_DISPLAY_NAMES.get(result["method"], result["method"]),
                "gamma": result["gamma"],
                "grid_size": f"{result['grid_size'][0]}x{result['grid_size'][1]}",
                "max_episode_steps": result["max_episode_steps"],
                "num_episodes": result["num_episodes"],
                "epsilon": result["epsilon"],
                "delay_lam": delay_config["args"].get("lam") if delay_config else None,
                "seed": result["seed"],
                "all_rmse": float(result["all_rmse"]),
                "post_est_rmse": float(result["post_est_rmse"])
                if not np.isnan(result["post_est_rmse"])
                else None,
                "solution_step": result["solution_step"],
                "mean_return": result["mean_return"],
                "return_variance": result["return_variance"],
                "coverage_pct": result["coverage_pct"],
                "factors_coverage_pct": result["factors_coverage_pct"],
            }
        )
    return summaries


def generate_report(
    summaries: Sequence[Dict[str, Any]],
    epsilon_sweep_summaries: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    lines: List[str] = []
    lines.append("# LEAST vs BLADE-TD vs IMR Validation Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Total runs: {len(summaries)}\n")

    import pandas as pd

    df = pd.DataFrame(summaries)

    lines.append("\n## Table 1: Returns by Method x Gamma x Grid Size\n")
    table1 = (
        df.groupby(["method", "gamma", "grid_size"])
        .agg(
            mean_return=("mean_return", "mean"),
            std_return=("mean_return", "std"),
            mean_variance=("return_variance", "mean"),
        )
        .round(2)
    )
    lines.append(table1.to_markdown())
    lines.append("")

    lines.append("\n## Table 2: Coverage % by Grid Size x Max Steps x Num Episodes\n")
    table2 = (
        df.groupby(["method", "grid_size", "max_episode_steps", "num_episodes"])
        .agg(
            mean_coverage=("coverage_pct", "mean"),
            mean_factors_coverage=("factors_coverage_pct", "mean"),
        )
        .round(3)
    )
    lines.append(table2.to_markdown())
    lines.append("")

    lines.append("\n## Table 3: RMSE by Method x Gamma x Grid Size\n")
    table3 = (
        df.groupby(["method", "gamma", "grid_size"])
        .agg(
            mean_all_rmse=("all_rmse", "mean"),
            std_all_rmse=("all_rmse", "std"),
            mean_post_rmse=("post_est_rmse", "mean"),
        )
        .round(4)
    )
    lines.append(table3.to_markdown())
    lines.append("")

    lines.append(
        "\n## Table 4: LEAST Solution Rate by Max Steps x Num Episodes x Grid Size\n"
    )
    df_least = df[df["method"] == "LEAST"]
    if not df_least.empty:
        df_least = df_least.copy()
        df_least["solved"] = df_least["solution_step"].notna().astype(int)
        table4 = (
            df_least.groupby(["grid_size", "max_episode_steps", "num_episodes"])
            .agg(
                solution_rate=("solved", "mean"),
                mean_solution_step=("solution_step", "mean"),
            )
            .round(2)
        )
        lines.append(table4.to_markdown())
    else:
        lines.append("No LEAST results found.")
    lines.append("")

    lines.append(
        "\n## Table 5: LEAST Coverage / Solution Rate by Epsilon x Max Steps x Grid Size\n"
    )
    lines.append(
        "Isolates exploration rate from step-count/episode-count "
        "(see Q1, `agents/analyses/2026-09-04-least-blade-imr-reward-estimation-analysis.md`).\n"
    )
    if epsilon_sweep_summaries:
        df_eps = pd.DataFrame(epsilon_sweep_summaries)
        df_eps["solved"] = df_eps["solution_step"].notna().astype(int)
        table5 = (
            df_eps.groupby(["grid_size", "max_episode_steps", "epsilon"])
            .agg(
                mean_coverage=("coverage_pct", "mean"),
                mean_factors_coverage=("factors_coverage_pct", "mean"),
                solution_rate=("solved", "mean"),
            )
            .round(3)
        )
        lines.append(table5.to_markdown())
    else:
        lines.append("No epsilon-sweep results provided.")
    lines.append("")

    lines.append("\n## Table 6: IMR vs FR (identity) by Gamma x Delay\n")
    lines.append(
        "Quantifies the gamma=1 near-exact-match and gamma<1 gap for IMR (see Q2).\n"
    )
    df_imr = df[df["method"] == "IMR"].copy()
    df_fr = df[df["method"] == "FR"].copy()
    join_keys = [
        "env_name",
        "grid_size",
        "gamma",
        "seed",
        "max_episode_steps",
        "num_episodes",
    ]
    if not df_imr.empty and not df_fr.empty:
        df_fr_slim = df_fr[join_keys + ["mean_return", "all_rmse"]].rename(
            columns={"mean_return": "fr_mean_return", "all_rmse": "fr_all_rmse"}
        )
        df_joined = df_imr.merge(df_fr_slim, on=join_keys, how="inner")
        df_joined["return_gap"] = df_joined["fr_mean_return"] - df_joined["mean_return"]
        table6 = (
            df_joined.groupby(["gamma", "delay_lam"])
            .agg(
                mean_imr_return=("mean_return", "mean"),
                mean_fr_return=("fr_mean_return", "mean"),
                mean_return_gap=("return_gap", "mean"),
                mean_imr_rmse=("all_rmse", "mean"),
            )
            .round(2)
        )
        lines.append(table6.to_markdown())
    else:
        lines.append("Missing IMR or FR (identity) results for comparison.")
    lines.append("")

    lines.append(
        "\n## Table 7: Paired LEAST vs BLADE-TD, conditioned on LEAST reaching a solution\n"
    )
    lines.append(
        "Filtered to configs where LEAST's `solution_step` is not null, joined "
        "on matching (grid, gamma, delay, seed, max_steps, num_episodes) (see Q3). "
        "Caveat: coverage (nonzero-column count) is necessary but not sufficient "
        "for a well-conditioned fit -- see Q1's note on repetitive trajectories.\n"
    )
    df_least_solved = df[(df["method"] == "LEAST") & (df["solution_step"].notna())]
    df_blade = df[df["method"] == "BLADE-TD"]
    pair_keys = [
        "env_name",
        "grid_size",
        "gamma",
        "delay_lam",
        "seed",
        "max_episode_steps",
        "num_episodes",
    ]
    if not df_least_solved.empty and not df_blade.empty:
        df_blade_slim = df_blade[pair_keys + ["mean_return", "all_rmse"]].rename(
            columns={"mean_return": "blade_mean_return", "all_rmse": "blade_all_rmse"}
        )
        df_paired = df_least_solved.merge(df_blade_slim, on=pair_keys, how="inner")
        df_paired["return_diff"] = (
            df_paired["mean_return"] - df_paired["blade_mean_return"]
        )
        df_paired["rmse_diff"] = df_paired["all_rmse"] - df_paired["blade_all_rmse"]
        lines.append(
            f"Matched pairs: {len(df_paired)}. "
            f"LEAST wins on return (higher): {(df_paired['return_diff'] > 0).sum()}, "
            f"ties: {(df_paired['return_diff'] == 0).sum()}, "
            f"loses: {(df_paired['return_diff'] < 0).sum()}."
        )
        table7 = (
            df_paired.groupby(["grid_size", "gamma"])
            .agg(
                mean_return_diff=("return_diff", "mean"),
                mean_rmse_diff=("rmse_diff", "mean"),
                num_pairs=("return_diff", "count"),
            )
            .round(2)
        )
        lines.append(table7.to_markdown())
    else:
        lines.append("No matched LEAST(solved)/BLADE-TD pairs found.")
    lines.append("")

    return "\n".join(lines)


def run_configs_with_executor(
    run_configs: Sequence[Mapping[str, Any]],
    executor: str,
    cluster_uri: Optional[str],
) -> List[Mapping[str, Any]]:
    if executor == "sequential":
        return rewest_runner.run_sequential(run_configs)
    if executor == "joblib":
        return rewest_runner.run_parallel(run_configs)
    if executor == "ray":
        return rewest_runner.run_ray(run_configs, cluster_uri=cluster_uri)
    raise ValueError(f"Unknown executor: {executor}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-files",
        nargs="+",
        required=True,
        help="Paths to solvable grid JSON files",
    )
    parser.add_argument(
        "--max-steps",
        nargs="+",
        type=int,
        default=[200, 500],
        help="Max episode steps values to test",
    )
    parser.add_argument(
        "--num-episodes",
        nargs="+",
        type=int,
        default=[250, 500, 1000],
        help="Number of training episodes to test",
    )
    parser.add_argument(
        "--skip-epsilon-sweep",
        action="store_true",
        help="Skip the LEAST-only epsilon sweep (Table 5)",
    )
    parser.add_argument(
        "--executor",
        choices=("sequential", "joblib", "ray"),
        default="ray",
        help="Execution backend for running experiments",
    )
    parser.add_argument(
        "--cluster-uri",
        default=None,
        help="Ray cluster address (only used with --executor ray); "
        "omit for a local Ray instance",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.dirname(__file__),
        help="Output directory for results -- accepts a gs:// URI or local path",
    )
    parser.add_argument(
        "--local-output-dir",
        default=None,
        help="If set, copy the two output files here after writing to "
        "--output-path (useful when --output-path is a gs:// URI)",
    )
    args = parser.parse_args()

    grids = load_grids(args.grid_files)
    size_labels = sorted(
        set("{}x{}".format(grid["size"][0], grid["size"][1]) for grid in grids)
    )
    print(f"Loaded {len(grids)} grids across sizes: {size_labels}")

    run_configs = build_run_configs(
        grids=grids,
        max_steps_values=args.max_steps,
        num_episodes_values=args.num_episodes,
    )
    print(f"Total experiment runs: {len(run_configs)}")

    ts_start = time.time()
    results = run_configs_with_executor(
        run_configs, executor=args.executor, cluster_uri=args.cluster_uri
    )
    elapsed = time.time() - ts_start
    print(f"Completed main matrix in {elapsed:.0f}s")

    summaries = summarize_results(results)

    epsilon_sweep_summaries: Optional[List[Dict[str, Any]]] = None
    if not args.skip_epsilon_sweep:
        epsilon_configs = build_epsilon_sweep_configs(
            grids=grids, max_steps_values=args.max_steps
        )
        print(f"Total epsilon-sweep runs: {len(epsilon_configs)}")
        ts_start = time.time()
        epsilon_results = run_configs_with_executor(
            epsilon_configs, executor=args.executor, cluster_uri=args.cluster_uri
        )
        elapsed = time.time() - ts_start
        print(f"Completed epsilon sweep in {elapsed:.0f}s")
        epsilon_sweep_summaries = summarize_results(epsilon_results)

    results_path = os.path.join(args.output_path, "validation_results.json")
    with tf.io.gfile.GFile(results_path, "w") as fh:
        json.dump(
            {
                "main_matrix": summaries,
                "epsilon_sweep": epsilon_sweep_summaries,
            },
            fh,
            indent=2,
        )
    print(f"Results saved to {results_path}")

    report = generate_report(summaries, epsilon_sweep_summaries)
    report_path = os.path.join(args.output_path, "validation_report.md")
    with tf.io.gfile.GFile(report_path, "w") as fh:
        fh.write(report)
    print(f"Report saved to {report_path}")

    if args.local_output_dir:
        if not tf.io.gfile.exists(args.local_output_dir):
            tf.io.gfile.makedirs(args.local_output_dir)
        for filename in ("validation_results.json", "validation_report.md"):
            src = os.path.join(args.output_path, filename)
            dst = os.path.join(args.local_output_dir, filename)
            tf.io.gfile.copy(src, dst, overwrite=True)
        print(f"Copied results to {args.local_output_dir}")


if __name__ == "__main__":
    main()
