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


class TestNormalizeMetadata:
    """Empty env-spec metadata maps to None so Arrow writes a null field."""

    def _entry(self, metadata):
        return {
            "experiment": {
                "env_spec": {"name": "MountainCar-v0", "metadata": metadata}
            },
            "run_config": {"episodes_per_run": 100},
        }

    def test_empty_metadata_becomes_none(self):
        result = agg._normalize_metadata({"/run": self._entry({})})
        assert result["/run"]["experiment"]["env_spec"]["metadata"] is None

    def test_none_metadata_stays_none(self):
        result = agg._normalize_metadata({"/run": self._entry(None)})
        assert result["/run"]["experiment"]["env_spec"]["metadata"] is None

    def test_populated_metadata_is_preserved(self):
        result = agg._normalize_metadata({"/run": self._entry({"size": [10, 10]})})
        assert result["/run"]["experiment"]["env_spec"]["metadata"] == {
            "size": [10, 10]
        }

    def test_missing_metadata_key_is_untouched(self):
        entry = self._entry(None)
        del entry["experiment"]["env_spec"]["metadata"]
        result = agg._normalize_metadata({"/run": entry})
        assert "metadata" not in result["/run"]["experiment"]["env_spec"]

    def test_empty_dict_fields_outside_env_spec_are_untouched(self):
        entry = self._entry({})
        entry["run_config"] = {}
        result = agg._normalize_metadata({"/run": entry})
        assert result["/run"]["run_config"] == {}


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
                        "reward_mapper": {
                            "name": "bayes-least-lfa",
                            "args": {
                                "check_factors": False,
                                "transition_feats_spec": {
                                    "name": "action-and-termination-features",
                                    "args": {},
                                },
                            },
                        },
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
            "steps": 100 * update_index,
            "elapsed_s": 0.5 * update_index,
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
        assert first["steps"] == 100
        assert first["elapsed_s"] == 0.5
        assert first["rmse"] == 0.25
        assert first["num_samples"] == 220
        assert first["check_factors"] is False
        assert first["feats_spec"] == json.dumps(
            {"name": "action-and-termination-features", "args": {}}, sort_keys=True
        )
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


class TestDeferredAttemptRows:
    """Per-deferral rows and the routing predicate."""

    def _log_row(self, attempts):
        row = {
            "episode": 20,
            "steps": 100,
            "returns": -1.0,
            "info": {"deferred_attempts": attempts},
            "meta": {
                "exp_id": "exp",
                "instance_id": 0,
                "experiment": {
                    "env_spec": {"name": "MountainCar-v0"},
                    "problem_spec": {
                        "reward_mapper": {
                            "name": "least-lfa",
                            "args": {
                                "check_factors": True,
                                "transition_feats_spec": {
                                    "name": "state-bins-action-features",
                                    "args": {"bins": [4, 2]},
                                },
                            },
                        },
                        "gamma": 0.99,
                        "delay_config": {"name": "clipped-poisson", "args": {}},
                    },
                },
            },
            "exp_id": "exp",
            "instance_id": 0,
        }
        return row

    def test_extracts_one_row_per_attempt(self):
        attempts = [
            {
                "episode": 10,
                "steps": 2500,
                "buffer_rows": 60,
                "nonzero_columns": 20,
                "matrix_columns": 24,
            },
            {
                "episode": 11,
                "steps": 5000,
                "buffer_rows": 84,
                "nonzero_columns": 24,
                "matrix_columns": 24,
            },
        ]
        rows = agg.deferred_attempt_rows(self._log_row(attempts))
        assert len(rows) == 2
        first = rows[0]
        assert first["episode"] == 10
        assert first["steps"] == 2500
        assert first["buffer_rows"] == 60
        assert first["nonzero_columns"] == 20
        assert first["matrix_columns"] == 24
        assert first["method"] == "LEAST"
        assert first["check_factors"] is True
        assert first["env_name"] == "MountainCar-v0"

    def test_numpy_array_attempts(self):
        attempts = np.array(
            [
                {
                    "episode": 10,
                    "steps": 2500,
                    "buffer_rows": 60,
                    "nonzero_columns": 20,
                    "matrix_columns": 24,
                }
            ],
            dtype=object,
        )
        rows = agg.deferred_attempt_rows(self._log_row(attempts))
        assert len(rows) == 1
        assert rows[0]["episode"] == 10

    def test_no_attempts_yields_no_rows(self):
        assert agg.deferred_attempt_rows(self._log_row([])) == []
        assert agg.deferred_attempt_rows({"info": None}) == []

    def test_routing_predicates(self):
        attempts = [{"episode": 10}]
        assert agg.has_deferred_attempts(self._log_row(attempts))
        assert not agg.has_deferred_attempts(self._log_row([]))
        assert not agg.has_estimation_events(self._log_row(attempts))


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


class TestStepSnapshotRows:
    """The aggregation output projects to one row per experiment."""

    def _snapshot_row(self, snapshot):
        return {"episode": 25, "step_snapshot_agg": snapshot}

    def _data(self, exp_id):
        return {
            "meta": {"exp_id": exp_id},
            "returns": [-10.0],
            "steps": [100],
            "instance_ids": [0],
        }

    def test_rows_carry_episode_and_exp_identity(self):
        rows = agg.step_snapshot_rows(self._snapshot_row({"exp": self._data("exp")}))
        assert len(rows) == 1
        assert rows[0]["episode"] == 25
        assert rows[0]["exp_id"] == "exp"
        assert rows[0]["instance_ids"] == [0]
        assert rows[0]["meta"] == {"exp_id": "exp"}

    def test_rows_skip_null_entries(self):
        # an exp_id that logged no run at the episode arrives as a null
        # struct field (see StepSnapshotAggregator._merge)
        rows = agg.step_snapshot_rows(
            self._snapshot_row({"exp": self._data("exp"), "exp-b": None, "exp-c": None})
        )
        assert [row["exp_id"] for row in rows] == ["exp"]


class TestStepSnapshotAggregator:
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
        return agg.StepSnapshotAggregator()

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

    def test_merge_treats_null_right_entry_as_absent(self):
        # Arrow represents an accumulator from another block as a struct
        # whose fields are the block's exp_ids; an exp_id with no records
        # at the episode arrives as a null field
        aggregator = self._aggregator()
        acc_left = aggregator._init(key=None)
        acc_left = aggregator._accumulate_row(acc_left, self._row("exp", 0, -10.0))
        acc_right = {"exp": None, "exp-b": None}
        merged = aggregator._merge(acc_left, acc_right)
        records = merged["exp"]["records"]
        assert [record["instance_id"] for record in records] == [0]

    def test_merge_replaces_null_left_entry_with_records(self):
        aggregator = self._aggregator()
        acc_left = aggregator._init(key=None)
        acc_left = aggregator._accumulate_row(acc_left, self._row("exp", 0, -10.0))
        acc_left["exp-b"] = None
        acc_right = aggregator._init(key=None)
        acc_right = aggregator._accumulate_row(acc_right, self._row("exp-b", 1, -20.0))
        merged = aggregator._merge(acc_left, acc_right)
        records = merged["exp-b"]["records"]
        assert [record["instance_id"] for record in records] == [1]

    def test_merge_null_on_both_sides_keeps_no_records(self):
        aggregator = self._aggregator()
        acc_left = aggregator._init(key=None)
        acc_left = aggregator._accumulate_row(acc_left, self._row("exp", 0, -10.0))
        acc_left["exp-b"] = None
        merged = aggregator._merge(acc_left, {"exp-b": None})
        assert merged["exp"]["records"][0]["returns"] == -10.0
        # a null on both sides carries no records; finalize drops it
        assert merged["exp-b"] is None

    def test_finalize_skips_null_entries(self):
        aggregator = self._aggregator()
        acc = aggregator._init(key=None)
        acc = aggregator._accumulate_row(acc, self._row("exp", 0, -10.0))
        acc["exp-b"] = None
        result = aggregator._finalize(acc)
        assert set(result) == {"exp"}

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
