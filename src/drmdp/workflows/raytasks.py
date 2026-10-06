"""
Scheduling and awaiting helpers for distributed experiment tasks.
"""

import logging
from collections.abc import Sequence
from typing import Any

import ray

logger = logging.getLogger(__name__)


def wait_till_completion(
    task_refs: Sequence[ray.ObjectRef],
    name: str | None = None,
    fetch: bool = False,
) -> Sequence[Any]:
    """
    Waits for every ray task to complete, raising if any failed.

    Task failures surface only when results are fetched via ray.get;
    ray.wait alone reports failed tasks as ready. Raising at the end
    keeps the driver's exit code - and thus the job's status -
    consistent with task outcomes, without aborting independent
    tasks on the first failure.

    Returns fetched task results, in completion order, when `fetch`
    is True; otherwise the input refs.
    """
    task_label = f"{name} task(s)" if name else "task(s)"
    unfinished_tasks = list(task_refs)
    results: list[Any] = []
    failures: list[Exception] = []
    while unfinished_tasks:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        for task_ref in finished_tasks:
            try:
                result = ray.get(task_ref)
            except ray.exceptions.RayError as err:
                # app-level task failures arrive wrapped in RayError
                # subclasses; anything else is a driver bug and must abort
                failures.append(err)
                logger.error("%s failed: %s", task_label, err)
                continue
            if fetch:
                results.append(result)
        logger.info(
            "Finished %d %s, %d failure(s) so far. %d left out of %d.",
            len(finished_tasks),
            task_label,
            len(failures),
            len(unfinished_tasks),
            len(task_refs),
        )
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(task_refs)} {task_label} failed"
        ) from failures[0]
    return results if fetch else task_refs
