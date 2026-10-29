"""Shared utilities for experiment specification parsing."""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from drmdp import core


def parse_experiment_specs(
    specs: Sequence[Mapping[str, Any]],
) -> Sequence[core.Experiment]:
    """Convert experiment spec dicts into typed Experiment objects.

    Each spec dict is expected to contain:
    - ``name``: environment name
    - ``args``: environment construction args
    - ``feats_specs``: sequence of feature transform specs
    - ``problem_specs``: sequence of problem spec dicts
    - ``epochs``: number of training epochs
    - ``metadata`` (optional): experiment metadata
    """
    experiment_specs: list[core.Experiment] = []
    for spec in specs:
        for feat_tfx_spec, problem_spec in itertools.product(
            spec["feats_specs"], spec["problem_specs"]
        ):
            experiment_specs.append(
                core.Experiment(
                    env_spec=core.EnvSpec(
                        name=spec["name"],
                        args=spec["args"],
                        feats_spec=feat_tfx_spec,
                        metadata=spec.get("metadata"),
                    ),
                    problem_spec=core.ProblemSpec(**problem_spec),
                    epochs=spec["epochs"],
                )
            )
    return experiment_specs
