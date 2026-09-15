"""Tests for logger.py — experiment logging and utilities."""

import json
import os
import tempfile

import numpy as np
import pytest

from drmdp import core, logger


class TestExperimentLogger:
    def test_creates_param_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            instance = _make_experiment_instance(log_dir)
            exp_logger = logger.ExperimentLogger(
                log_dir=log_dir, experiment_instance=instance
            )
            assert os.path.exists(exp_logger.param_file)
            with open(exp_logger.param_file) as fh:
                params = json.load(fh)
            assert params["exp_id"] == "test-exp"

    def test_context_manager_writes_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            instance = _make_experiment_instance(log_dir)
            with logger.ExperimentLogger(
                log_dir=log_dir, experiment_instance=instance
            ) as exp_logger:
                exp_logger.log(episode=0, steps=10, returns=-100.0, info={})
                exp_logger.log(episode=1, steps=20, returns=-90.0, info={"x": 1})

            with open(exp_logger.log_file) as fh:
                lines = fh.readlines()
            assert len(lines) == 2
            entry = json.loads(lines[0])
            assert entry["episode"] == 0
            assert entry["steps"] == 10
            assert entry["returns"] == -100.0

    def test_close_without_open_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            instance = _make_experiment_instance(log_dir)
            exp_logger = logger.ExperimentLogger(
                log_dir=log_dir, experiment_instance=instance
            )
            with pytest.raises(RuntimeError, match="not opened"):
                exp_logger.close()

    def test_log_without_open_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            instance = _make_experiment_instance(log_dir)
            exp_logger = logger.ExperimentLogger(
                log_dir=log_dir, experiment_instance=instance
            )
            with pytest.raises(RuntimeError, match="not opened"):
                exp_logger.log(episode=0, steps=10, returns=-100.0)


class TestSaveModel:
    def test_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = np.array([1.0, 2.0, 3.0])
            logger.save_model(weights, name="test_weights", model_dir=tmpdir)
            path = os.path.join(tmpdir, "test_weights.npz")
            assert os.path.exists(path)
            loaded = np.load(path)
            np.testing.assert_array_equal(loaded, weights)

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "subdir", "models")
            weights = np.array([1.0])
            logger.save_model(weights, name="w.npz", model_dir=model_dir)
            assert os.path.exists(os.path.join(model_dir, "w.npz"))


class TestDataclassFromDict:
    def test_creates_run_config(self):
        data = {
            "num_runs": 5,
            "episodes_per_run": 100,
            "log_episode_frequency": 10,
            "use_seed": True,
            "output_dir": "/tmp/test",
        }
        result = logger.dataclass_from_dict(core.RunConfig, data)
        assert isinstance(result, core.RunConfig)
        assert result.num_runs == 5

    def test_non_dataclass_raises(self):
        with pytest.raises(ValueError, match="dataclass"):
            logger.dataclass_from_dict(dict, {"a": 1})


class TestJsonFromDict:
    def test_no_encoding(self):
        data = {"a": {"b": 1}}
        result = logger.json_from_dict(data)
        assert result == {"a": {"b": 1}}

    def test_encode_at_level_0(self):
        data = {"a": {"b": 1}, "c": 2}
        result = logger.json_from_dict(data, dict_encode_level=0)
        assert result["a"] == json.dumps({"b": 1})
        assert result["c"] == 2

    def test_encode_at_level_1(self):
        data = {"a": {"b": {"c": 3}}}
        result = logger.json_from_dict(data, dict_encode_level=1)
        assert isinstance(result["a"], dict)
        assert result["a"]["b"] == json.dumps({"c": 3})


def _make_experiment_instance(output_dir: str) -> core.ExperimentInstance:
    return core.ExperimentInstance(
        exp_id="test-exp",
        instance_id=0,
        experiment=core.Experiment(
            env_spec=core.EnvSpec(
                name="TestEnv",
                args={"max_episode_steps": 100},
                feats_spec=[],
            ),
            problem_spec=core.ProblemSpec(
                policy_type="markovian",
                reward_mapper={"name": "identity", "args": None},
                delay_config=None,
                epsilon=0.1,
                gamma=0.99,
                learning_rate_config={
                    "name": "constant",
                    "args": {"initial_lr": 0.01},
                },
            ),
            epochs=1,
        ),
        run_config=core.RunConfig(
            num_runs=1,
            episodes_per_run=100,
            log_episode_frequency=10,
            use_seed=True,
            output_dir=output_dir,
        ),
        context={},
        export_model=False,
    )
