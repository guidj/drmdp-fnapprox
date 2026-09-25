"""Tests for controljob.py — experiment parsing without Ray."""

from drmdp import core
from drmdp.workflows import controlexps, controljob


class TestParseExperiments:
    def test_illustration_specs(self):
        specs = controlexps.illustration_experiment_specs()
        experiments = controljob.parse_experiments(specs)
        assert len(experiments) > 0
        assert all(isinstance(exp, core.Experiment) for exp in experiments)

    def test_cartesian_product(self):
        specs = [
            {
                "name": "TestEnv",
                "args": {},
                "feats_specs": [
                    [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}],
                    [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}],
                ],
                "problem_specs": [
                    {
                        "policy_type": "markovian",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": None,
                        "epsilon": 0.1,
                        "gamma": 0.99,
                        "learning_rate_config": {
                            "name": "constant",
                            "args": {"initial_lr": 0.01},
                        },
                    }
                ],
                "epochs": 1,
            }
        ]
        experiments = controljob.parse_experiments(specs)
        assert len(experiments) == 2

    def test_preserves_metadata(self):
        specs = [
            {
                "name": "TestEnv",
                "args": {"max_episode_steps": 100},
                "metadata": {"tag": "test"},
                "feats_specs": [
                    [{"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}]
                ],
                "problem_specs": [
                    {
                        "policy_type": "markovian",
                        "reward_mapper": {"name": "identity", "args": None},
                        "delay_config": None,
                        "epsilon": 0.1,
                        "gamma": 0.99,
                        "learning_rate_config": controlexps.LEARNING_RATE_SPEC,
                    }
                ],
                "epochs": 3,
            }
        ]
        experiments = controljob.parse_experiments(specs)
        assert experiments[0].env_spec.metadata == {"tag": "test"}
        assert experiments[0].epochs == 3


class TestCreateTasks:
    def test_illustration_creates_instances(self):
        instances = controljob.create_tasks(
            problem_set="illustration",
            config_args={},
            num_runs=2,
            num_episodes=50,
            output_dir="/tmp/test-controljob",
            task_prefix="test",
            log_episode_frequency=10,
            use_seed=True,
            export_model=False,
        )
        assert len(instances) > 0
        assert all(isinstance(inst, core.ExperimentInstance) for inst in instances)

    def test_instance_count_matches_runs(self):
        specs = controlexps.illustration_experiment_specs()
        experiments = controljob.parse_experiments(specs)
        num_runs = 3
        instances = controljob.create_tasks(
            problem_set="illustration",
            config_args={},
            num_runs=num_runs,
            num_episodes=50,
            output_dir="/tmp/test-controljob",
            task_prefix="test",
            log_episode_frequency=10,
            use_seed=True,
            export_model=False,
        )
        assert len(instances) == len(experiments) * num_runs
