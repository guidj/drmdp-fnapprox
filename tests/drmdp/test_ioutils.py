"""Tests for ioutils.py — JSON record read/write."""

import os
import tempfile

from drmdp import ioutils


class TestWriteAndReadRecordsJson:
    def test_gzip_roundtrip(self):
        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            ioutils.write_records_json(path, records, gzip_compression=True)
            loaded = ioutils.read_records_json(path + ".gzip", gzip_compression=True)
            assert loaded == records

    def test_plain_json_roundtrip(self):
        records = [{"x": 42}, {"x": 99}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            ioutils.write_records_json(path, records, gzip_compression=False)
            loaded = ioutils.read_records_json(path, gzip_compression=False)
            assert loaded == records

    def test_single_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "single.jsonl")
            ioutils.write_records_json(path, [{"only": True}], gzip_compression=False)
            loaded = ioutils.read_records_json(path, gzip_compression=False)
            assert loaded == [{"only": True}]

    def test_gzip_extension_appended(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            ioutils.write_records_json(path, [{"a": 1}], gzip_compression=True)
            assert os.path.exists(path + ".gzip")

    def test_gzip_extension_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl.gzip")
            ioutils.write_records_json(path, [{"a": 1}], gzip_compression=True)
            assert os.path.exists(path)
