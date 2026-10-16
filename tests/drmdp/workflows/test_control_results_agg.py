"""Tests for control_results_agg_pipeline.py — pure utility functions."""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from drmdp.workflows import control_results_agg_pipeline as agg


class TestParsePathFromFilename:
    def test_local_path(self):
        result = agg.parse_path_from_filename("/some/dir/file.json")
        assert result == "/some/dir"

    def test_gcs_path(self):
        result = agg.parse_path_from_filename("gs://bucket/path/to/file.json")
        assert result == os.path.join("bucket", "path/to")


class TestConvertMetadataToMapping:
    def test_creates_mapping(self):
        df = pd.DataFrame(
            {"path": ["/a/b/file.json", "/c/d/file.json"], "value": [1, 2]}
        )
        result = agg.convert_metadata_to_mapping(df)
        assert "/a/b" in result
        assert "/c/d" in result


class TestChunks:
    def test_even_chunks(self):
        result = list(agg.chunks([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_remainder(self):
        result = list(agg.chunks([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]


class TestReadExperimentMetadata:
    def test_reads_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "params.json")
            with open(path, "w") as fh:
                json.dump({"exp_id": "test", "gamma": 0.99}, fh)
            result = agg.read_experiment_metadata(path)
            assert result["exp_id"] == "test"
            assert result["path"] == path


class TestParseExperimentsMetadataFiles:
    def test_reads_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for idx in range(3):
                path = os.path.join(tmpdir, f"params_{idx}.json")
                with open(path, "w") as fh:
                    json.dump({"idx": idx}, fh)
                paths.append(path)
            result = agg.parse_experiments_metadata_files(paths)
            assert len(result) == 3


class TestHasEstimationEvents:
    def _row(self, info):
        return {"episode": 1, "steps": 10, "returns": -1.0, "info": info}

    def test_events_present(self):
        row = self._row({"reward_errors": [{"update_index": 1}]})
        assert agg.has_estimation_events(row)

    def test_events_as_numpy_array(self):
        events = np.array([{"update_index": 1}], dtype=object)
        row = self._row({"reward_errors": events})
        assert agg.has_estimation_events(row)

    def test_empty_info(self):
        assert not agg.has_estimation_events(self._row({}))

    def test_none_info(self):
        assert not agg.has_estimation_events(self._row(None))

    def test_events_missing(self):
        assert not agg.has_estimation_events(self._row({"other": 1}))


class TestEstimationEventRows:
    def _log_row(self, info, instance_id=3):
        return {
            "episode": 20,
            "steps": 100,
            "returns": -1.0,
            "info": info,
            "meta": {
                "exp_id": "exp",
                "instance_id": instance_id,
                "experiment": {
                    "env_spec": {"name": "GridWorld-MINES"},
                    "problem_spec": {
                        "reward_mapper": {"name": "bayes-least-lfa"},
                        "gamma": 0.99,
                        "delay_config": {"name": "clipped-poisson", "args": {}},
                    },
                },
            },
            "exp_id": "exp",
            "instance_id": instance_id,
        }

    def _event(self, update_index, episode):
        return {
            "update_index": update_index,
            "episode": episode,
            "rmse": 0.25 * update_index,
            "num_samples": 220,
        }

    def test_extracts_one_row_per_event(self):
        row = self._log_row({"reward_errors": [self._event(1, 4), self._event(2, 20)]})
        rows = agg.estimation_event_rows(row)
        assert len(rows) == 2
        first, second = rows
        assert first["exp_id"] == "exp"
        assert first["instance_id"] == 3
        assert first["method"] == "BLADE-TD"
        assert first["env_name"] == "GridWorld-MINES"
        assert first["gamma"] == 0.99
        assert first["delay_config"] == {"name": "clipped-poisson", "args": {}}
        assert first["update_index"] == 1
        assert first["episode"] == 4
        assert first["rmse"] == 0.25
        assert first["num_samples"] == 220
        assert second["update_index"] == 2
        assert second["episode"] == 20

    def test_events_as_numpy_array(self):
        events = np.array([self._event(1, 4)], dtype=object)
        rows = agg.estimation_event_rows(self._log_row({"reward_errors": events}))
        assert len(rows) == 1
        assert rows[0]["update_index"] == 1

    def test_empty_info_yields_no_rows(self):
        assert agg.estimation_event_rows(self._log_row({})) == []

    def test_none_info_yields_no_rows(self):
        assert agg.estimation_event_rows(self._log_row(None)) == []


class TestEstimationEventDedup:
    """Keep-first aggregation: one row per (exp_id, instance_id, update)."""

    def test_keeps_first_row(self):
        dedup = agg.EstimationEventDedup()
        acc = dedup._init(key=None)
        acc = dedup._accumulate_row(acc, {"update_index": 1, "rmse": 0.5})
        acc = dedup._accumulate_row(acc, {"update_index": 1, "rmse": 0.9})
        assert acc == {"update_index": 1, "rmse": 0.5}
        assert dedup._finalize(acc) == acc

    def test_merge_keeps_present_side(self):
        dedup = agg.EstimationEventDedup()
        acc = {"update_index": 1, "rmse": 0.5}
        assert dedup._merge(acc, None) is acc
        assert dedup._merge(None, acc) is acc
        assert dedup._merge(None, None) is None

    def test_init_is_empty(self):
        dedup = agg.EstimationEventDedup()
        assert dedup._init(key=None) is None


class TestStepSnapshotAggretator:
    """Records carry run identity; finalize sorts into parallel arrays."""

    def _row(self, exp_id, instance_id, returns, steps=100):
        return {
            "episode": 25,
            "exp_id": exp_id,
            "instance_id": instance_id,
            "returns": returns,
            "steps": steps,
            "meta": {"exp_id": exp_id, "path": "/some/run"},
        }

    def _aggregator(self):
        return agg.StepSnapshotAggretator()

    def test_accumulate_keeps_run_identity(self):
        aggregator = self._aggregator()
        acc = aggregator._init(key=None)
        acc = aggregator._accumulate_row(acc, self._row("exp", 0, -10.0))
        acc = aggregator._accumulate_row(acc, self._row("exp", 1, -20.0))
        records = acc["exp"]["records"]
        assert [record["instance_id"] for record in records] == [0, 1]
        assert [record["returns"] for record in records] == [-10.0, -20.0]
        assert [record["steps"] for record in records] == [100, 100]

    def test_accumulate_separates_experiments(self):
        aggregator = self._aggregator()
        acc = aggregator._init(key=None)
        acc = aggregator._accumulate_row(acc, self._row("exp-a", 0, -10.0))
        acc = aggregator._accumulate_row(acc, self._row("exp-b", 0, -30.0))
        assert [record["returns"] for record in acc["exp-a"]["records"]] == [-10.0]
        assert [record["returns"] for record in acc["exp-b"]["records"]] == [-30.0]

    def test_merge_extends_records(self):
        aggregator = self._aggregator()
        acc_left = aggregator._init(key=None)
        acc_left = aggregator._accumulate_row(acc_left, self._row("exp", 0, -10.0))
        acc_right = aggregator._init(key=None)
        acc_right = aggregator._accumulate_row(acc_right, self._row("exp", 1, -20.0))
        merged = aggregator._merge(acc_left, acc_right)
        records = merged["exp"]["records"]
        assert [record["instance_id"] for record in records] == [0, 1]

    def test_merge_keeps_disjoint_experiments(self):
        aggregator = self._aggregator()
        acc_left = aggregator._init(key=None)
        acc_left = aggregator._accumulate_row(acc_left, self._row("exp-a", 0, -10.0))
        acc_right = aggregator._init(key=None)
        acc_right = aggregator._accumulate_row(acc_right, self._row("exp-b", 1, -20.0))
        merged = aggregator._merge(acc_left, acc_right)
        assert set(merged) == {"exp-a", "exp-b"}

    def test_finalize_sorts_records_into_parallel_arrays(self):
        aggregator = self._aggregator()
        acc = aggregator._init(key=None)
        # arrival order differs from run order
        acc = aggregator._accumulate_row(acc, self._row("exp", 2, -30.0, steps=9))
        acc = aggregator._accumulate_row(acc, self._row("exp", 0, -10.0, steps=200))
        acc = aggregator._accumulate_row(acc, self._row("exp", 1, -20.0, steps=54))
        result = aggregator._finalize(acc)["exp"]
        assert result["instance_ids"] == [0, 1, 2]
        # position k of each array is run k
        assert result["returns"] == [-10.0, -20.0, -30.0]
        assert result["steps"] == [200, 54, 9]

    def test_finalize_drops_path_from_meta(self):
        aggregator = self._aggregator()
        acc = aggregator._init(key=None)
        acc = aggregator._accumulate_row(acc, self._row("exp", 0, -10.0))
        result = aggregator._finalize(acc)["exp"]
        assert "path" not in result["meta"]
        assert result["meta"]["exp_id"] == "exp"


class TestRunRows:
    """Per-run projection of one log record."""

    def _log_row(self, info, instance_id=3):
        return {
            "episode": 25,
            "steps": 100,
            "returns": -41.0,
            "info": info,
            "meta": {
                "exp_id": "exp",
                "instance_id": instance_id,
                "experiment": {
                    "env_spec": {
                        "name": "GridWorld-MINES",
                        "metadata": {"noise_scale": 0.5},
                    },
                    "problem_spec": {
                        "reward_mapper": {"name": "least-lfa"},
                        "gamma": 0.99,
                        "delay_config": {
                            "name": "clipped-poisson",
                            "args": {"lam": 5},
                        },
                    },
                },
                "run_config": {
                    "episodes_per_run": 2500,
                    "log_episode_frequency": 25,
                },
            },
            "exp_id": "exp",
            "instance_id": instance_id,
        }

    def test_projects_run_and_config(self):
        rows = agg.run_rows(self._log_row(info={}))
        assert len(rows) == 1
        row = rows[0]
        assert row["exp_id"] == "exp"
        assert row["instance_id"] == 3
        assert row["episode"] == 25
        assert row["returns"] == -41.0
        assert row["steps"] == 100
        assert row["policy_eval_mean"] is None
        assert row["method"] == "LEAST"
        assert row["env_name"] == "GridWorld-MINES"
        assert row["gamma"] == 0.99
        assert row["delay_config"] == {
            "name": "clipped-poisson",
            "args": {"lam": 5},
        }
        assert row["noise_scale"] == 0.5
        assert row["episodes_per_run"] == 2500
        assert row["log_episode_frequency"] == 25

    def test_reads_policy_eval_mean(self):
        info = {"policy_eval": {"mean": -12.5, "num_episodes": 10}}
        rows = agg.run_rows(self._log_row(info=info))
        assert rows[0]["policy_eval_mean"] == -12.5

    def test_empty_info_yields_none_mean(self):
        assert agg.run_rows(self._log_row(info={}))[0]["policy_eval_mean"] is None

    def test_none_info_yields_none_mean(self):
        # Ray reads an empty `info` as None for some readers
        assert agg.run_rows(self._log_row(info=None))[0]["policy_eval_mean"] is None

    def test_missing_policy_eval_yields_none_mean(self):
        info = {"reward_errors": []}
        assert agg.run_rows(self._log_row(info=info))[0]["policy_eval_mean"] is None

    def test_delay_config_none_is_kept(self):
        row = self._log_row(info={})
        row["meta"]["experiment"]["problem_spec"]["delay_config"] = None
        assert agg.run_rows(row)[0]["delay_config"] is None

    def test_missing_noise_metadata_yields_none(self):
        row = self._log_row(info={})
        del row["meta"]["experiment"]["env_spec"]["metadata"]
        assert agg.run_rows(row)[0]["noise_scale"] is None
