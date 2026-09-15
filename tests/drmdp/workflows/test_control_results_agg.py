"""Tests for control_results_agg_pipeline.py — pure utility functions."""

import json
import os
import tempfile

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
