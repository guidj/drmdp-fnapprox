"""
Log aggregation for the control campaigns.

Reads `experiment-params.json` and `experiment-logs.jsonl` from every
run directory under the input root, joins the records, and writes one
Parquet table per log key for the analysis notebooks.
"""

import argparse
import dataclasses
import json
import logging
import math
import os.path
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import ray
import ray.data
import tensorflow as tf

from drmdp import dataproc
from drmdp.workflows import raytasks

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PipelineArgs:
    """
    Pipeline arguments.
    """

    input_dir: str
    output_dir: str


class StepSnapshotAggregator(ray.data.aggregate.AggregateFn):
    """
    Aggregates returns.
    """

    AggType = dict[str, list[dict[str, Any]]]
    Row = Mapping[str, Any]

    def __init__(self, name: str = "StepSnapshotAggregator()"):
        super().__init__(
            init=self._init,
            merge=self._merge,
            accumulate_row=self._accumulate_row,
            finalize=self._finalize,
            name=name,
        )

    def _init(self, key: Any) -> AggType:
        del key
        return {}

    def _accumulate_row(self, acc: AggType, row: Row) -> AggType:
        """
        Add a single row to aggregation.
        """
        records = acc.get(row["exp_id"])
        if records is None:
            records = acc[row["exp_id"]] = []
        records.append(
            {
                "instance_id": row["instance_id"],
                "returns": row["returns"],
                "steps": row["steps"],
            }
        )
        return acc

    def _merge(self, acc_left: AggType, acc_right: AggType) -> AggType:
        """
        Combine two accumulators.
        """
        # Arrow stores a partial accumulator as a struct with one field
        # per exp_id of the block; an exp_id that logged no run at an
        # episode of that block comes back as a null field ({exp_id: None}).
        # A null entry means "no records here", not data, so it merges
        # as absent on either side.
        acc = dict(acc_left)
        for exp_id, records in acc_right.items():
            if records is None:
                continue
            entry = acc.get(exp_id)
            if entry is None:
                acc[exp_id] = records
            else:
                entry.extend(records)
        return acc

    def _finalize(self, acc: AggType) -> Any:
        """
        Project final output.
        """
        # Records are sorted by run so array position k is run k of the
        # same experiment for every row of the output; the arrival-order
        # arrays were an arbitrary per-row permutation, which made
        # index pairing across rows invalid
        projected = {}
        for exp_id, records in acc.items():
            # exp_id logged no run at this episode in any block (see
            # _merge on null struct fields)
            if records is None:
                continue
            records = sorted(records, key=lambda record: record["instance_id"])
            projected[exp_id] = {
                "returns": [record["returns"] for record in records],
                "steps": [record["steps"] for record in records],
                "instance_ids": [record["instance_id"] for record in records],
            }
        return projected


def main():
    """
    Entry point
    """
    args = parse_args()
    # The trailing slash at the end is for gcs paths
    paths = tf.io.gfile.glob(os.path.join(args.input_dir, "**/**/**/**/"))
    ray_env = {}

    logger.info("Running with args: %s", vars(args))
    logger.info("Found a total of %d paths", len(paths))
    logger.info("Ray environment: %s", ray_env)

    with ray.init(runtime_env=ray_env) as context:
        logger.info("Ray Context: %s", context)
        logger.info("Ray Nodes: %s", ray.nodes())

        metadata = _normalize_metadata(parse_experiment_metadata(paths))
        ds_logs = parse_experiment_logs(paths)

        logger.info("Metadata # keys: %d", len(metadata))
        logger.info("Datalogs size: %fMB", ds_logs.size_bytes() / 1024 / 1024)
        fields = derive_exp_fields(metadata)
        exp_meta = derive_exp_meta(metadata)
        ds_logs_and_fields = join_logs_and_fields(ds_logs, fields)

        datasets = pipeline(ds_logs_and_fields, exp_meta)
        for key in list(datasets):
            write_ref = write_dataset.remote(datasets.pop(key), key, args.output_dir)
            raytasks.wait_till_completion([write_ref])


def parse_experiment_metadata(paths: Sequence[str]) -> Mapping[str, Any]:
    """
    Parses logs for an experiment.
    """
    metadata_files = [os.path.join(path, "experiment-params.json") for path in paths]
    results = {}
    read_remote = ray.remote(parse_experiments_metadata_files)
    refs = [read_remote.remote(seq) for seq in chunks(metadata_files, chunk_size=100)]
    while True:
        completed_refs, refs = ray.wait(refs)
        for completed_ref in completed_refs:
            entries = ray.get(completed_ref)
            for entry in entries:
                path = parse_path_from_filename(entry["path"])
                results[path] = entry
        if not refs:
            break
    return results


def parse_experiments_metadata_files(paths: Sequence[str]):
    results = []
    for path in paths:
        entry = read_experiment_metadata(path)
        results.append(entry)
    return results


def read_experiment_metadata(path: str):
    """
    Reads metadata file.
    """
    with tf.io.gfile.GFile(path, "r") as readable:
        entry = json.load(readable)
        entry["path"] = path
    return entry


def parse_experiment_logs(paths: Sequence[str]) -> ray.data.Dataset:
    """
    Parses logs for an experiment.
    """
    logs_files = [os.path.join(path, "experiment-logs.jsonl") for path in paths]
    ds_logs = ray.data.read_json(
        logs_files, include_paths=True, file_extensions=["jsonl"]
    )
    return ds_logs


def _normalize_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Experiment-params entries with empty dicts mapped to None, recursively.

    An empty dict reaches Arrow as a fieldless struct, which pyarrow cannot
    write to Parquet; the empty mapping appears wherever a spec declares no
    options, e.g. `EnvSpec.metadata` or the `args` of a reward-observation
    wrapper. Readers of the normalized fields fall back to an empty
    mapping.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized = {key: normalize(item) for key, item in value.items()}
            return None if not normalized else normalized
        return value

    return {path: normalize(entry) for path, entry in metadata.items()}


EXP_FIELDS_COLUMNS: tuple[str, ...] = (
    "exp_id",
    "instance_id",
    "env_name",
    "gamma",
    "delay_config",
    "method",
    "estimation_check",
    "feats_spec",
    "noise_scale",
    "episodes_per_run",
    "log_episode_frequency",
)


def derive_exp_fields(
    metadata: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """
    Scalar experiment fields keyed by run path.

    The fields are the per-row constants the pipeline consumes;
    deriving them once per run cell replaces attaching the full
    metadata entry to every log row.
    """
    fields = {}
    for path, entry in metadata.items():
        experiment = entry["experiment"]
        env_spec = experiment["env_spec"]
        problem_spec = experiment["problem_spec"]
        reward_mapper = problem_spec["reward_mapper"]
        mapper_args = dict(reward_mapper.get("args") or {})
        fields[path] = {
            "exp_id": entry["exp_id"],
            "instance_id": entry["instance_id"],
            "env_name": env_spec["name"],
            "gamma": problem_spec["gamma"],
            "delay_config": problem_spec["delay_config"],
            "method": dataproc.MAPPERS_NAMES[reward_mapper["name"]],
            "estimation_check": _resolve_check_mode(mapper_args),
            "feats_spec": json.dumps(
                mapper_args.get("transition_feats_spec") or {}, sort_keys=True
            ),
            "noise_scale": _noise_std_multiplier(env_spec),
            "episodes_per_run": entry["run_config"]["episodes_per_run"],
            "log_episode_frequency": entry["run_config"]["log_episode_frequency"],
        }
    return fields


def derive_exp_meta(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    One metadata entry per experiment, keyed by exp_id, `path` removed.

    The entry with the smallest instance id represents its experiment,
    a deterministic pick where the per-row stream previously carried an
    arbitrary run's entry.
    """
    by_exp: dict[str, dict[str, Any]] = {}
    for path in sorted(metadata, key=lambda p: metadata[p]["instance_id"]):
        entry = metadata[path]
        by_exp.setdefault(
            entry["exp_id"],
            {key: value for key, value in entry.items() if key != "path"},
        )
    return by_exp


def add_exp_fields(
    df_logs: pd.DataFrame, fields: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    """
    Adds the derived field columns to one block of log rows.
    """
    paths = df_logs["path"].apply(parse_path_from_filename)
    derived = pd.DataFrame([fields[path] for path in paths], columns=EXP_FIELDS_COLUMNS)
    derived.index = df_logs.index
    return pd.concat([df_logs.drop(columns=["path"]), derived], axis=1)


def join_logs_and_fields(
    ds_logs: ray.data.Dataset, fields: Mapping[str, Mapping[str, Any]]
) -> ray.data.Dataset:
    """
    Joins the derived fields onto the log rows of a dataset.
    """
    return ds_logs.map_batches(
        lambda df: add_exp_fields(df, fields), batch_format="pandas"
    )


def convert_metadata_to_mapping(df_metadata: pd.DataFrame) -> Mapping[str, Any]:
    """
    Converts the metadata into a mapping.
    """
    mapping = {}
    for row in df_metadata.to_dict(orient="records"):
        path = parse_path_from_filename(row["path"])
        mapping[path] = row
    return mapping


def parse_path_from_filename(file_name: str) -> str:
    """
    Returns the path.
    """
    dir_name, _ = os.path.split(file_name)
    # Ray IO readers include_paths strip the
    # prefix of cloud buckets
    # so we do the same for consistency.
    matches = re.match("gs://(.*?)/(.*)", dir_name)
    return os.path.join(*matches.groups()) if matches else dir_name


class EstimationEventDedup(ray.data.aggregate.AggregateFn):
    """
    Keeps the first row per (exp_id, instance_id, update_index).

    Events are attached to every log entry of their run, so the same
    event reaches the aggregation many times.
    """

    AggType = Mapping[str, Any] | None
    Row = Mapping[str, Any]

    def __init__(self, name: str = "estimation_event_dedup"):
        super().__init__(
            init=self._init,
            merge=self._merge,
            accumulate_row=self._accumulate_row,
            finalize=self._finalize,
            name=name,
        )

    def _init(self, key: Any) -> AggType:
        del key
        return None

    def _accumulate_row(self, acc: AggType, row: Row) -> AggType:
        if acc is not None:
            return acc
        return dict(row)

    def _merge(self, acc_left: AggType, acc_right: AggType) -> AggType:
        return acc_left if acc_left is not None else acc_right

    def _finalize(self, acc: AggType) -> Any:
        return acc


def pipeline(
    ds_logs: ray.data.Dataset, exp_meta: Mapping[str, Any]
) -> Mapping[str, ray.data.Dataset]:
    """
    This pipeline aggregates the output of
    multiple runs from each experiment.
    """
    # Estimation events are attached to every log entry of their run;
    # keep one row per event and update.
    ds_estimations = (
        ds_logs.filter(has_estimation_events)
        .flat_map(estimation_event_rows)
        .groupby(["exp_id", "instance_id", "update_index"])
        .aggregate(EstimationEventDedup())
        .flat_map(lambda row: [row["estimation_event_dedup"]])
    )
    # Deferred attempts share the log-attachment pattern; one row per
    # (run, episode), the episode on which the attempt was scheduled.
    ds_deferrals = (
        ds_logs.filter(has_deferred_attempts)
        .flat_map(deferred_attempt_rows)
        .groupby(["exp_id", "instance_id", "episode"])
        .aggregate(EstimationEventDedup(name="deferred_attempt_dedup"))
        .flat_map(lambda row: [row["deferred_attempt_dedup"]])
    )
    # Per-run rows are projected from the joined stream before `ds_logs`
    # is rebound to the episode-grouped view, which drops per-run rows
    ds_runs = ds_logs.flat_map(run_rows)
    ds_logs = (
        ds_logs.groupby("episode")
        .aggregate(StepSnapshotAggregator(name="step_snapshot_agg"))
        .flat_map(lambda row: step_snapshot_rows(row, exp_meta))
    )
    ds_metrics = calculate_metrics(ds_logs)
    return {
        "logs": ds_logs,
        "metrics": ds_metrics,
        "estimations": ds_estimations,
        "deferrals": ds_deferrals,
        "runs": ds_runs,
    }


@ray.remote(max_calls=1)
def write_dataset(ds: ray.data.Dataset, key: str, output_dir: str) -> None:
    """
    Writes one derived dataset to Parquet.

    The worker process is discarded after the task returns
    (`max_calls=1`), so the state one dataset's count and write retain
    does not carry into the next dataset's execution.
    """
    if ds.count() == 0:
        logger.info("Skipping empty %s", key)
        return
    output_path = os.path.join(output_dir, key)
    logger.info("Writing %s to %s", key, output_path)
    ds.write_parquet(output_path)


def has_estimation_events(row: Mapping[str, Any]) -> bool:
    """
    Whether a log record carries reward-estimation events.
    """
    return len(reward_error_events(row)) > 0


def has_deferred_attempts(row: Mapping[str, Any]) -> bool:
    """
    Whether a log record carries deferred-attempt records.
    """
    return len(deferred_attempt_events(row)) > 0


def reward_error_events(row: Mapping[str, Any]) -> Sequence[Any]:
    """
    Estimation events attached to a log record.

    Ray reads the events as a numpy object array of dicts; log entries
    from before the first estimation carry an empty `info`.
    """
    info = row.get("info")
    if not info:
        return ()
    events = info.get("reward_errors")
    if events is None:
        return ()
    return list(events)


def deferred_attempt_events(row: Mapping[str, Any]) -> Sequence[Any]:
    """
    Deferred-attempt records attached to a log record.

    One record per scheduled estimation attempt that a gate deferred;
    the list is attached to every log entry of the run, like the
    estimation events.
    """
    info = row.get("info")
    if not info:
        return ()
    attempts = info.get("deferred_attempts")
    if attempts is None:
        return ()
    return list(attempts)


def _resolve_check_mode(mapper_args: Mapping[str, Any]) -> str:
    """Resolve estimation check mode from mapper args."""
    return str(mapper_args.get("estimation_check", "none"))


def estimation_event_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Per estimation-event rows from one experiment log record.

    Each row carries the run identity, the event's episode and update
    index, the error report, and the experiment configuration the
    event belongs to.
    """
    events = reward_error_events(row)
    if not events:
        return []
    return [
        {
            "exp_id": row["exp_id"],
            "instance_id": row["instance_id"],
            "episode": event["episode"],
            "steps": event.get("steps"),
            "elapsed_s": event.get("elapsed_s"),
            "update_index": event["update_index"],
            "rmse": event["rmse"],
            "num_samples": event["num_samples"],
            "method": row["method"],
            "estimation_check": row["estimation_check"],
            "feats_spec": row["feats_spec"],
            "env_name": row["env_name"],
            "gamma": row["gamma"],
            "delay_config": row["delay_config"],
        }
        for event in events
    ]


def deferred_attempt_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Per deferred-attempt rows from one experiment log record.

    Each row carries the run identity, the attempt's episode and buffer
    state, and the experiment configuration, so a run that never
    estimates states why: the nonzero-column count against the design
    width.
    """
    attempts = deferred_attempt_events(row)
    if not attempts:
        return []
    return [
        {
            "exp_id": row["exp_id"],
            "instance_id": row["instance_id"],
            "episode": attempt["episode"],
            "steps": attempt.get("steps"),
            "buffer_rows": attempt["buffer_rows"],
            "nonzero_columns": attempt["nonzero_columns"],
            "matrix_columns": attempt["matrix_columns"],
            "numerical_rank": attempt.get("numerical_rank"),
            "method": row["method"],
            "estimation_check": row["estimation_check"],
            "feats_spec": row["feats_spec"],
            "env_name": row["env_name"],
            "gamma": row["gamma"],
            "delay_config": row["delay_config"],
        }
        for attempt in attempts
    ]


def run_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    One per-run row from one experiment log record.

    Each row carries the run identity, the logged point of the
    cumulative evaluation-mean curve, and the experiment configuration,
    so per-run analyses such as learning time need no re-read of the raw
    logs. `policy_eval_mean` is None for runs logged before the
    per-checkpoint evaluation mean was recorded; those runs are
    recovered by differencing the cumulative mean.
    """
    info = row.get("info") or {}
    policy_eval = info.get("policy_eval") or {}
    return [
        {
            "exp_id": row["exp_id"],
            "instance_id": row["instance_id"],
            "episode": row["episode"],
            "returns": row["returns"],
            "steps": row["steps"],
            "policy_eval_mean": policy_eval.get("mean"),
            "method": row["method"],
            "estimation_check": row["estimation_check"],
            "env_name": row["env_name"],
            "gamma": row["gamma"],
            "delay_config": row["delay_config"],
            "noise_scale": row["noise_scale"],
            "episodes_per_run": row["episodes_per_run"],
            "log_episode_frequency": row["log_episode_frequency"],
        }
    ]


def step_snapshot_rows(
    row: Mapping[str, Any], meta_by_exp: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """
    One row per experiment of an episode's snapshot aggregation.

    A null-valued entry (see StepSnapshotAggregator._merge on null
    struct fields) is an experiment that logged no run at the episode;
    it produces no row.
    """
    return [
        {
            "episode": row["episode"],
            "exp_id": exp_id,
            "meta": meta_by_exp[exp_id],
            **data,
        }
        for exp_id, data in row["step_snapshot_agg"].items()
        if data is not None
    ]


def calculate_metrics(ds: ray.data.Dataset) -> ray.data.Dataset:
    """
    Calculates metrics across runs for each experiment
    entry.
    """

    def apply(row):
        return {
            **row,
            "metrics": {
                "returns": {
                    "mean": np.mean(row["returns"]),
                    "std": np.std(row["returns"]),
                },
                "steps": {"mean": np.mean(row["steps"]), "std": np.std(row["steps"])},
            },
        }

    return ds.map(apply)


def chunks(seq: Sequence[Any], chunk_size: int):
    """Yield successive chunk_size-sized chunks from seq."""
    for idx in range(0, len(seq), chunk_size):
        yield seq[idx : idx + chunk_size]


def parse_args() -> PipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input-dir", type=str)
    arg_parser.add_argument("--output-dir", type=str)
    known_args, unknown_args = arg_parser.parse_known_args()
    logger.info("Unknown args: %s", unknown_args)
    return PipelineArgs(**vars(known_args))


def _noise_std_multiplier(env_spec: Mapping[str, Any]) -> float | None:
    """Noise standard-deviation multiplier of an experiment spec.

    Reads the `std_multiplier` arg of the reward-noise wrapper, falling back
    to the retired `scale` (a variance multiplier) and then to the retired
    metadata key, so that both old and new record directories produce the
    same column.
    """
    noise_args = ((env_spec.get("args") or {}).get("reward_noise") or {}).get(
        "args"
    ) or {}
    if "std_multiplier" in noise_args:
        return float(noise_args["std_multiplier"])
    if "scale" in noise_args:
        return math.sqrt(float(noise_args["scale"]))
    metadata = env_spec.get("metadata") or {}
    if "std_multiplier" in metadata:
        return float(metadata["std_multiplier"])
    legacy = metadata.get("noise_scale")
    return None if legacy is None else float(legacy)


if __name__ == "__main__":
    main()
