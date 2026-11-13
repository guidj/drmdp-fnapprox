"""The estimation buffer multiple is one constant across the experiment families.

`constants.DEFAULT_ESTIMATION_BUFFER_MULT` sets the capacity of the buffer that
feeds reward estimation, as a multiple of the estimator's feature dimension.
Every family that derives estimator arguments must emit it, so estimation error
is compared at one sample budget.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from drmdp import constants, task
from drmdp.workflows import biasvarjob, controlexps, estimationexps, rewest
from drmdp.workflows.grids import validate_least_coverage

CONTROL_FEATS: Sequence[Mapping[str, Any]] = [
    {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
]
ESTIMATOR_MAPPERS = ("least-lfa", "bayes-least-lfa")


class TestSpecFamilies:
    """Every family that derives estimator arguments emits the shared multiple."""

    def test_control_family(self):
        for spec in controlexps.gaussian_illustration_experiment_specs():
            for mapper_name in ESTIMATOR_MAPPERS:
                for args in _estimator_args(spec, mapper_name):
                    assert args["estimation_buffer_mult"] == (
                        constants.DEFAULT_ESTIMATION_BUFFER_MULT
                    )

    def test_estimation_campaign(self):
        for spec in estimationexps.estimation_experiment_specs():
            for problem_spec in spec["problem_specs"]:
                args = problem_spec["reward_mapper"]["args"]
                assert args["estimation_buffer_mult"] == (
                    constants.DEFAULT_ESTIMATION_BUFFER_MULT
                )

    def test_rewest_factories(self):
        specs = (
            list(rewest.discrete_least_specs((10,), [CONTROL_FEATS]))
            + list(rewest.least_specs((10,), [CONTROL_FEATS]))
            + list(rewest.bayes_least_specs((10,), [CONTROL_FEATS]))
            + list(rewest.cvlps_specs((10,), [CONTROL_FEATS]))
            + list(rewest.recurring_cvlps((10,), [CONTROL_FEATS]))
        )
        assert specs
        for spec in specs:
            assert spec["args"]["estimation_buffer_mult"] == (
                constants.DEFAULT_ESTIMATION_BUFFER_MULT
            )

    def test_biasvarjob_factories(self):
        specs = list(biasvarjob.least_lfa_specs((10,), [CONTROL_FEATS])) + list(
            biasvarjob.bayes_least_lfa_specs((10,), [CONTROL_FEATS])
        )
        assert specs
        for spec in specs:
            assert spec["args"]["estimation_buffer_mult"] == (
                constants.DEFAULT_ESTIMATION_BUFFER_MULT
            )

    def test_grid_validation_method_configs(self):
        for mapper_name in ESTIMATOR_MAPPERS:
            config = validate_least_coverage.METHOD_CONFIGS[mapper_name]
            assert config["estimation_buffer_mult"] == (
                constants.DEFAULT_ESTIMATION_BUFFER_MULT
            )


class TestBufferCapacity:
    """The multiple reaches the buffer as capacity = multiple x feature dim."""

    def test_capacity_is_the_multiple_times_the_feature_dim(self):
        seen: set[str] = set()
        for spec in controlexps.gaussian_illustration_experiment_specs():
            if spec["name"] in seen:
                continue
            seen.add(spec["name"])
            mapper_args = _estimator_args(spec, "least-lfa")[0]
            proxied = task.create_env(spec["name"], args=dict(spec["args"]), gamma=1.0)
            try:
                feats_fn = task.estimator_feats_fn(
                    proxied.proxy, "least-lfa", dict(mapper_args)
                )
                wrapper = task.reward_mapper(
                    proxied.env,
                    proxy_env=proxied.proxy,
                    mapping_spec={"name": "least-lfa", "args": dict(mapper_args)},
                )
                assert wrapper.est_buffer.max_capacity == (
                    constants.DEFAULT_ESTIMATION_BUFFER_MULT * feats_fn.feature_dim
                )
                wrapper.close()
            finally:
                proxied.env.close()
                proxied.proxy.close()
        assert seen == {"MountainCar-v0", "Acrobot-v1", "GridWorld-MINES"}


def _estimator_args(
    spec: Mapping[str, Any], mapper_name: str
) -> Sequence[Mapping[str, Any]]:
    return [
        problem_spec["reward_mapper"]["args"]
        for problem_spec in spec["problem_specs"]
        if problem_spec["reward_mapper"]["name"] == mapper_name
    ]
