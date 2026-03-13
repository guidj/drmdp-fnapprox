import gzip
import json
import logging
import sys
from typing import Any, Mapping, Sequence

import tensorflow as tf


def write_records_json(
    output_path: str,
    records: Sequence[Mapping[str, Any]],
    gzip_compression: bool = True,
) -> None:
    """
    Exports records into JSON files.
    """
    bytes_size = sys.getsizeof(records)
    if gzip_compression and not output_path.endswith(".gzip"):
        output_path = ".".join([output_path, "gzip"])

    logging.debug(
        "Writing partition of %fMB to %s",
        bytes_size / 1024 / 1024,
        output_path,
    )

    if gzip_compression:
        with tf.io.gfile.GFile(output_path, "wb") as writable:
            with gzip.GzipFile(fileobj=writable, mode="wb") as writer:
                for record in records:
                    content = "".join([json.dumps(record), "\n"])
                    writer.write(content.encode("UTF-8"))
    else:
        with tf.io.gfile.GFile(output_path, "w") as writable:
            for record in records:
                json.dump(record, fp=writable)
                writable.write("\n")


def read_records_json(
    input_path: str, gzip_compression: bool = True
) -> Sequence[Mapping[str, Any]]:
    """
    Read records from JSON.
    """
    logging.debug(
        "Reading file %s",
        input_path,
    )

    records = []
    if gzip_compression:
        with tf.io.gfile.GFile(input_path, "rb") as readable:
            with gzip.GzipFile(fileobj=readable, mode="rb") as reader:
                for line in reader:
                    records.append(json.loads(line.decode("UTF-8")))
    else:
        with tf.io.gfile.GFile(input_path, "r") as readable:
            for line in readable:
                records.append(json.loads(line))
    return records
