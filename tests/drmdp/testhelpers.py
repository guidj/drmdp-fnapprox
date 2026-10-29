"""Shared test helpers for drmdp tests."""

from collections.abc import Sequence

import numpy as np

from drmdp import algorithms


def assert_training_snapshots(
    snapshots: Sequence[algorithms.PolicyControlSnapshot],
    expected_count: int,
) -> None:
    """Verify invariant properties that every training run must satisfy."""
    assert len(snapshots) == expected_count
    for snapshot in snapshots:
        assert isinstance(snapshot, algorithms.PolicyControlSnapshot)
        assert snapshot.steps > 0
        assert isinstance(snapshot.weights, np.ndarray)
        assert np.isfinite(snapshot.returns)

    shapes = [snap.weights.shape for snap in snapshots]
    assert len(set(shapes)) == 1, f"Inconsistent weight shapes: {set(shapes)}"

    assert not np.all(snapshots[-1].weights == 0)
