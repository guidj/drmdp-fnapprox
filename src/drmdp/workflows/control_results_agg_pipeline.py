"""
Data aggregation pipeline to compute.
Aggregates results from multiple
instances from each experiment.
"""

import argparse
import copy
import dataclasses
import json
import logging
import os.path
import re
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import ray
import ray.data
import ray.data.datasource
import tensorflow as tf
from ray.data import aggregate
from ray.util import multiprocessing as ray_mp


@dataclasses.dataclass(frozen=True)
class PipelineArgs:
    """
    Pipeline arguments.
    """

    input_dir: str
    output_dir: str


class StepSnapshotAggretator(aggregate.AggregateFn):
    """
    Aggregates returns.
    """

    AggType = Mapping[Tuple[str, int], Any]
    Row = Mapping[str, Any]

    def __init__(self, name: str = "StepSnapshotAggretator()"):
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
        new_acc = dict(**copy.deepcopy(acc))
        if row["exp_id"] not in acc:
            meta = copy.deepcopy(row["meta"])
            # Path is a random pick
            # from one of the runs
            del meta["path"]
            new_acc[row["exp_id"]] = {"meta": meta, "returns": [], "steps": []}
        new_acc[row["exp_id"]]["returns"].append(row["returns"])
        new_acc[row["exp_id"]]["steps"].append(row["steps"])
        return new_acc

    def _merge(self, acc_left: AggType, acc_right: AggType) -> AggType:
        """
        Combine two accumulators.
        """
        acc = dict(**copy.deepcopy(acc_left))
        for key, value in acc_right.items():
            if key not in acc:
                # copy meta and values
                acc[key] = copy.deepcopy(value)
            else:
                acc[key]["returns"].extend(value["returns"])
                acc[key]["steps"].extend(value["steps"])
        return acc

    def _finalize(self, acc: AggType) -> Any:
        """
        Project final output.
        """
        return acc


def main():
    """
    Entry point
    """
    args = parse_args()
    # The trailing slash at the end is for gcs paths
    paths = tf.io.gfile.glob(os.path.join(args.input_dir, "**/**/**/**/"))
    ray_env = {}

    logging.info("Running with args: %s", vars(args))
    logging.info("Found a total of %d paths", len(paths))
    logging.info("Ray environment: %s", ray_env)

    with ray.init(runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

        metadata = parse_experiment_metadata(paths)
        ds_logs = parse_experiment_logs(paths)

        logging.info("Metadata # keys: %d", len(metadata))
        logging.info("Datalogs size: %fMB", ds_logs.size_bytes() / 1024 / 1024)
        ds_logs_and_metadata = join_logs_and_metadata(ds_logs, metadata)

        results: Mapping[str, ray.data.Dataset] = ray.get(
            pipeline.remote(ds_logs_and_metadata)
        )
        for key, ds in results.items():
            output_path = os.path.join(args.output_dir, key)
            logging.info("Writing %s to %s", key, output_path)
            ds.write_parquet(output_path)


def parse_experiment_metadata(paths: Sequence[str]) -> Mapping[str, Any]:
    """
    Parses logs for an experiment.
    """
    metadata_files = [os.path.join(path, "experiment-params.json") for path in paths]
    results = {}
    with ray_mp.Pool(ray_address="auto") as pool:
        for entry in pool.map(read_experiment_metadata, metadata_files):
            path = parse_path_from_filename(entry["path"])
            results[path] = entry
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


def join_logs_and_metadata(
    ds_logs: ray.data.Dataset, metadata: Mapping[str, Any]
) -> ray.data.Dataset:
    """
    Parses logs for an experiment.
    """

    def get_exp_id(df: pd.DataFrame) -> pd.Series:
        return df["meta"].apply(lambda meta: meta["exp_id"])  # type: ignore

    def get_instance_id(df: pd.DataFrame) -> pd.Series:
        return df["meta"].apply(lambda meta: meta["instance_id"])  # type: ignore

    def get_metadata(df: pd.DataFrame) -> pd.Series:
        paths = df["path"].apply(parse_path_from_filename)
        return paths.apply(lambda path: metadata[path])  # type: ignore

    return (
        ds_logs.add_column("meta", get_metadata)
        .add_column("exp_id", get_exp_id)
        .add_column("instance_id", get_instance_id)
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


@ray.remote
def pipeline(ds_logs: ray.data.Dataset) -> Mapping[str, ray.data.Dataset]:
    """
    This pipeline aggregates the output of
    multiple runs from each experiment.
    """
    ds_logs = (
        ds_logs.groupby("episode")
        .aggregate(StepSnapshotAggretator(name="step_snapshot_agg"))
        .flat_map(
            lambda row: [
                {"episode": row["episode"], "exp_id": exp_id, **data}
                for exp_id, data in row["step_snapshot_agg"].items()
            ]
        )
    )
    ds_metrics = calculate_metrics(ds_logs)
    return {"logs": ds_logs, "metrics": ds_metrics}


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


def parse_args() -> PipelineArgs:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input-dir", type=str)
    arg_parser.add_argument("--output-dir", type=str)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return PipelineArgs(**vars(known_args))


if __name__ == "__main__":
    main()
