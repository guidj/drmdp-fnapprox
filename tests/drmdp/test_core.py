import dataclasses
import json

from drmdp import core


def _make_env_spec(**kwargs):
    defaults = {"name": "GridWorld-v0", "args": {"grid": "test"}, "feats_spec": []}
    defaults.update(kwargs)
    return core.EnvSpec(**defaults)


def _make_experiment_instance(env_spec):
    return core.ExperimentInstance(
        exp_id="test-exp",
        instance_id=0,
        experiment=core.Experiment(
            env_spec=env_spec,
            problem_spec=core.ProblemSpec(
                policy_type="sarsa",
                reward_mapper={"name": "identity", "args": {}},
                delay_config=None,
                epsilon=0.1,
                gamma=0.99,
                learning_rate_config={"name": "constant", "args": {"value": 0.01}},
            ),
            epochs=1,
        ),
        run_config=core.RunConfig(
            num_runs=1,
            episodes_per_run=10,
            log_episode_frequency=1,
            use_seed=True,
            output_dir="/tmp/test",
        ),
        context=None,
        export_model=False,
    )


class TestEnvSpecMetadata:
    def test_without_metadata(self):
        spec = _make_env_spec()
        assert spec.metadata is None
        d = dataclasses.asdict(spec)
        raw = json.loads(json.dumps(d))
        assert raw["metadata"] is None

    def test_with_metadata(self):
        meta = {"grid_size": [6, 12], "distance": 5}
        spec = _make_env_spec(metadata=meta)
        d = dataclasses.asdict(spec)
        raw = json.loads(json.dumps(d))
        assert raw["metadata"] == meta

    def test_deserialize_missing_metadata(self):
        raw = {"name": "GridWorld-v0", "args": {}, "feats_spec": []}
        spec = core.EnvSpec(**raw)
        assert spec.metadata is None

    def test_experiment_instance_roundtrip(self):
        meta = {"num_cliffs": 10, "distance": 42}
        env_spec = _make_env_spec(metadata=meta)
        instance = _make_experiment_instance(env_spec)
        raw = json.loads(json.dumps(dataclasses.asdict(instance)))
        assert raw["experiment"]["env_spec"]["metadata"] == meta
