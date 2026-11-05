"""Offline pre-aggregation of estimation-only campaign logs.

Reads ``experiment-params.json`` and the last line of each
``experiment-logs.jsonl`` from the run directories, and writes three
Parquet files: ``runs.parquet``, ``estimations.parquet``,
``deferrals.parquet``.

The last log line carries the full accumulated ``reward_errors`` and
``deferred_attempts`` lists, so earlier lines (which re-emit the growing
lists at each log interval) are skipped.

Runs as a Ray job for GCS-resident logs. Each run directory is
processed by a Ray task; the coordinator collects results and writes
Parquet.
"""

import argparse
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import ray
import tensorflow as tf

from drmdp.workflows import raytasks

logger = logging.getLogger(__name__)


def run_config_row(params: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the run configuration from experiment-params.json."""
    experiment = params["experiment"]
    env_spec = experiment["env_spec"]
    problem = experiment["problem_spec"]
    mapper = problem["reward_mapper"]
    mapper_args = mapper.get("args") or {}
    env_args = env_spec.get("args") or {}
    noise_args = (env_args.get("reward_noise") or {}).get("args") or {}
    delay_args = (problem.get("delay_config") or {}).get("args") or {}
    metadata = env_spec.get("metadata") or {}
    return {
        "exp_id": params["exp_id"],
        "instance_id": params["instance_id"],
        "env": env_spec["name"],
        "encoding": metadata.get("encoding"),
        "knowledge": metadata.get("knowledge"),
        "method": mapper["name"],
        "estimation_check": mapper_args.get("estimation_check", "none"),
        "min_delay": delay_args.get("min_delay"),
        "max_delay": delay_args.get("max_delay"),
        "noise_std": noise_args.get("std_multiplier"),
        "episodes_cap": params["run_config"]["episodes_per_run"],
    }


def read_last_line(path: str) -> str:
    """Read only the last non-empty line of a file via tf.io.gfile."""
    with tf.io.gfile.GFile(path, "r") as readable:
        content = readable.read()
    lines = content.strip().split("\n")
    return lines[-1] if lines else ""


def process_run_dir(run_dir: str) -> dict[str, Any]:
    """Process one run directory, returning config + events + deferrals."""
    params_path = os.path.join(run_dir, "experiment-params.json")
    logs_path = os.path.join(run_dir, "experiment-logs.jsonl")

    with tf.io.gfile.GFile(params_path, "r") as readable:
        params = json.load(readable)
    config = run_config_row(params)
    run_id = f"{config['exp_id']}/{config['instance_id']}"
    config["run"] = run_id

    last_line = read_last_line(logs_path)
    if not last_line:
        return {
            "run": {**config, "final_episode": None, "final_returns": None},
            "events": [],
            "deferrals": [],
        }

    last_entry = json.loads(last_line)
    run_row = {
        **config,
        "final_episode": last_entry.get("episode"),
        "final_returns": last_entry.get("returns"),
    }

    info = last_entry.get("info") or {}
    events = [
        {**config, "run": run_id, **event} for event in info.get("reward_errors", [])
    ]
    deferrals = [
        {**config, "run": run_id, **attempt}
        for attempt in info.get("deferred_attempts", [])
    ]
    return {"run": run_row, "events": events, "deferrals": deferrals}


@ray.remote
def process_run_dirs_batch(run_dirs: Sequence[str]) -> dict[str, Any]:
    """Process a batch of run directories."""
    runs: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    deferrals: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            result = process_run_dir(run_dir)
            runs.append(result["run"])
            events.extend(result["events"])
            deferrals.extend(result["deferrals"])
        except Exception as err:  # noqa: BLE001 - one bad run must not abort the batch
            logger.error("Failed to process %s: %s", run_dir, err)
    return {"runs": runs, "events": events, "deferrals": deferrals}


def discover_run_dirs(input_dir: str) -> list[str]:
    """Find all run directories containing experiment logs."""
    run_dirs = []
    for root, _, files in tf.io.gfile.walk(input_dir):
        if "experiment-logs.jsonl" in files:
            run_dirs.append(root)
    return run_dirs


def chunks(seq: Sequence[str], chunk_size: int) -> list[list[str]]:
    return [list(seq[idx : idx + chunk_size]) for idx in range(0, len(seq), chunk_size)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Aggregate estimation campaign logs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-uri", default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    logger.info("Discovering run directories in %s", args.input_dir)
    run_dirs = discover_run_dirs(args.input_dir)
    logger.info("Found %d run directories", len(run_dirs))

    with ray.init(args.cluster_uri) as context:
        logger.info("Ray Context: %s", context)
        batches = chunks(run_dirs, args.batch_size)
        logger.info("Submitting %d batches", len(batches))

        refs = [process_run_dirs_batch.remote(batch) for batch in batches]
        raytasks.wait_till_completion(refs, name="aggregation-batch")

        all_runs: list[dict[str, Any]] = []
        all_events: list[dict[str, Any]] = []
        all_deferrals: list[dict[str, Any]] = []
        for ref in refs:
            result = ray.get(ref)
            all_runs.extend(result["runs"])
            all_events.extend(result["events"])
            all_deferrals.extend(result["deferrals"])

    logger.info(
        "Collected: %d runs, %d events, %d deferrals",
        len(all_runs),
        len(all_events),
        len(all_deferrals),
    )

    tf.io.gfile.makedirs(args.output_dir)

    runs_path = os.path.join(args.output_dir, "runs.parquet")
    pd.DataFrame(all_runs).to_parquet(runs_path, index=False)
    logger.info("Wrote %s: %d rows", runs_path, len(all_runs))

    events_path = os.path.join(args.output_dir, "estimations.parquet")
    df_events = pd.DataFrame(all_events) if all_events else pd.DataFrame()
    df_events.to_parquet(events_path, index=False)
    logger.info("Wrote %s: %d rows", events_path, len(df_events))

    deferrals_path = os.path.join(args.output_dir, "deferrals.parquet")
    df_deferrals = pd.DataFrame(all_deferrals) if all_deferrals else pd.DataFrame()
    df_deferrals.to_parquet(deferrals_path, index=False)
    logger.info("Wrote %s: %d rows", deferrals_path, len(df_deferrals))


if __name__ == "__main__":
    main()
