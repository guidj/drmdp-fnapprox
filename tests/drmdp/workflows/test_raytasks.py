"""Tests for raytasks.py — completion tracking against a local Ray cluster."""

import pytest
import ray

from drmdp.workflows import raytasks


class TestWaitTillCompletion:
    def test_returns_refs_when_all_tasks_succeed(self, local_cluster):
        task_refs = [succeeding_task.remote() for _ in range(3)]

        results = raytasks.wait_till_completion(task_refs)

        assert list(results) == task_refs

    def test_returns_values_when_fetching(self, local_cluster):
        task_refs = [succeeding_task.remote() for _ in range(3)]

        results = raytasks.wait_till_completion(task_refs, fetch=True)

        assert list(results) == ["ok", "ok", "ok"]

    @pytest.mark.parametrize(
        ("num_failures", "total"),
        [(1, 3), (2, 5)],
    )
    def test_raises_when_tasks_fail(self, local_cluster, num_failures, total):
        task_refs = [
            failing_task.remote(f"experiment {idx} crashed")
            for idx in range(num_failures)
        ]
        task_refs.extend(succeeding_task.remote() for _ in range(total - num_failures))

        with pytest.raises(RuntimeError) as excinfo:
            raytasks.wait_till_completion(task_refs, name="experiment")

        assert f"{num_failures} of {total} experiment task(s) failed" in str(
            excinfo.value
        )
        assert isinstance(excinfo.value.__cause__, ray.exceptions.RayTaskError)
        assert "crashed" in str(excinfo.value.__cause__)

    def test_raises_unlabeled_summary_without_name(self, local_cluster):
        task_refs = [failing_task.remote("boom")]

        with pytest.raises(RuntimeError, match="1 of 1 task\\(s\\) failed"):
            raytasks.wait_till_completion(task_refs)

    def test_returns_for_empty_task_set(self):
        assert list(raytasks.wait_till_completion([])) == []
        assert list(raytasks.wait_till_completion([], fetch=True)) == []


@pytest.fixture(scope="module")
def local_cluster():
    """
    Starts a local Ray cluster so completion tracking runs against real
    task semantics; reuses an existing cluster when one is running.
    """
    already_running = ray.is_initialized()
    if not already_running:
        ray.init(num_cpus=1, include_dashboard=False, logging_level="ERROR")
    yield
    if not already_running:
        ray.shutdown()


@ray.remote
def succeeding_task() -> str:
    """Remote task that completes normally."""
    return "ok"


@ray.remote
def failing_task(message: str) -> None:
    """Remote task that fails with an application-level exception."""
    raise RuntimeError(message)
