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
