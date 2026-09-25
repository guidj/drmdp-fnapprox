"""Integration tests for algorithm variants — Options SARSA and DropMissing SARSA."""

import json
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from drmdp import algorithms, core, envs, logger, rewdelay, task, transform
from drmdp.workflows import controlexps


class TestOptionsSemigradientSARSA:
    def test_trains_on_gridworld(self):
        env = envs.make(
            "GridWorld-MINES",
            grid=controlexps.MINES_GW_GRID,
            max_episode_steps=200,
        )
        env, mon = task.monitor_wrapper(env)
        delay = rewdelay.FixedDelay(delay=3)
        env = task.delay_wrapper(env, delay)

        ft_op = transform.transform_pipeline(
            env=env,
            specs=[{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}],
        )
        lr = task.learning_rate(name="constant", args={"initial_lr": 0.01})
        algo = task.create_algorithm(
            env=env,
            ft_op=ft_op,
            delay_reward=delay,
            lr=lr,
            gamma=0.99,
            epsilon=0.2,
            policy_type="options",
            base_seed=0,
        )
        assert isinstance(algo, algorithms.OptionsSemigradientSARSAFnApprox)

        snapshots = list(algo.train(env=env, num_episodes=5, monitor=mon))
        _assert_training_snapshots(snapshots, expected_count=5)
        assert mon.step > 0
        env.close()

    def test_single_action_options_trains(self):
        env = envs.make(
            "GridWorld-MINES",
            grid=controlexps.MINES_GW_GRID,
            max_episode_steps=200,
        )
        env, mon = task.monitor_wrapper(env)
        delay = rewdelay.FixedDelay(delay=2)
        env = task.delay_wrapper(env, delay)

        ft_op = transform.transform_pipeline(
            env=env,
            specs=[{"name": "tile-observation-action-ft", "args": {"tiling_dim": 3}}],
        )
        lr = task.learning_rate(name="constant", args={"initial_lr": 0.01})
        algo = task.create_algorithm(
            env=env,
            ft_op=ft_op,
            delay_reward=delay,
            lr=lr,
            gamma=0.99,
            epsilon=0.2,
            policy_type="single-action-options",
            base_seed=0,
        )
        snapshots = list(algo.train(env=env, num_episodes=5, monitor=mon))
        _assert_training_snapshots(snapshots, expected_count=5)
        assert mon.step > 0
        env.close()


class TestDropMissingSemigradientSARSA:
    def test_trains_with_delayed_rewards(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        env, mon = task.monitor_wrapper(env)
        delay = rewdelay.FixedDelay(delay=3)
        env = task.delay_wrapper(env, delay)

        ft_op = transform.transform_pipeline(
            env=env,
            specs=[{"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}],
        )
        lr = task.learning_rate(name="constant", args={"initial_lr": 0.01})
        algo = task.create_algorithm(
            env=env,
            ft_op=ft_op,
            delay_reward=delay,
            lr=lr,
            gamma=0.99,
            epsilon=0.2,
            policy_type="drop-missing",
            base_seed=0,
        )
        assert isinstance(algo, algorithms.DropMissingSemigradientSARSAFnApprox)

        snapshots = list(algo.train(env=env, num_episodes=10, monitor=mon))
        _assert_training_snapshots(snapshots, expected_count=10)
        assert mon.step > 0
        env.close()

    def test_weights_change(self):
        env = envs.make("MountainCar-v0", max_episode_steps=200)
        env, mon = task.monitor_wrapper(env)
        delay = rewdelay.FixedDelay(delay=2)
        env = task.delay_wrapper(env, delay)

        ft_op = transform.transform_pipeline(
            env=env,
            specs=[{"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}],
        )
        lr = task.learning_rate(name="constant", args={"initial_lr": 0.01})
        algo = task.create_algorithm(
            env=env,
            ft_op=ft_op,
            delay_reward=delay,
            lr=lr,
            gamma=0.99,
            epsilon=0.2,
            policy_type="drop-missing",
            base_seed=42,
        )
        snapshots = list(algo.train(env=env, num_episodes=5, monitor=mon))
        _assert_training_snapshots(snapshots, expected_count=5)
        initial_weights = snapshots[0].weights
        final_weights = snapshots[-1].weights
        assert not np.array_equal(initial_weights, final_weights)
        env.close()


class TestPolicyControlEndToEnd:
    def test_log_entries_match_training_schedule(self):
        episodes_per_run = 10
        log_frequency = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "identity-mc")
            instance = _make_experiment_instance(
                output_dir=output_dir,
                episodes_per_run=episodes_per_run,
                log_episode_frequency=log_frequency,
            )
            task.policy_control(instance)

            log_entries = _read_log_entries(output_dir)
            assert len(log_entries) == episodes_per_run // log_frequency
            for entry in log_entries:
                assert entry["episode"] % log_frequency == 0
                assert entry["steps"] > 0
                assert np.isfinite(entry["returns"])

    def test_params_file_records_experiment_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "identity-mc")
            instance = _make_experiment_instance(
                output_dir=output_dir,
                episodes_per_run=5,
                log_episode_frequency=5,
            )
            task.policy_control(instance)

            params = _read_params(output_dir)
            assert params["exp_id"] == "test-mc"
            assert params["instance_id"] == 0
            assert params["experiment"]["env_spec"]["name"] == "MountainCar-v0"
            assert params["experiment"]["problem_spec"]["policy_type"] == "markovian"
            assert params["experiment"]["problem_spec"]["gamma"] == 0.99

    def test_exported_model_weights_are_finite_and_nontrivial(self):
        episodes_per_run = 20
        log_frequency = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "least-mc")
            instance = _make_experiment_instance(
                output_dir=output_dir,
                episodes_per_run=episodes_per_run,
                log_episode_frequency=log_frequency,
                reward_mapper={
                    "name": "least-lfa",
                    "args": {
                        "attempt_estimation_episode": 3,
                        "use_bias": False,
                        "impute_value": 0,
                        "estimation_buffer_mult": 25,
                        "use_next_state": False,
                        "check_factors": True,
                        "feats_spec": [
                            {
                                "name": "tile-observation-action-ft",
                                "args": {"tiling_dim": 2},
                            }
                        ],
                    },
                },
                delay_config={"name": "fixed", "args": {"delay": 3}},
                export_model=True,
            )
            task.policy_control(instance)

            saved_model_dir = os.path.join(output_dir, "saved_model")
            model_files = sorted(os.listdir(saved_model_dir))
            expected_snapshots = episodes_per_run // log_frequency
            assert len(model_files) == expected_snapshots
            for model_file in model_files:
                weights = np.load(os.path.join(saved_model_dir, model_file))
                assert np.all(np.isfinite(weights))
                assert weights.size > 0
                assert not np.all(weights == 0)

    def test_shaped_rewards_differ_from_unshaped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unshaped_dir = os.path.join(tmpdir, "unshaped")
            instance_unshaped = _make_experiment_instance(
                output_dir=unshaped_dir,
                episodes_per_run=10,
                log_episode_frequency=10,
            )
            task.policy_control(instance_unshaped)

            shaped_dir = os.path.join(tmpdir, "shaped")
            instance_shaped = _make_experiment_instance(
                output_dir=shaped_dir,
                episodes_per_run=10,
                log_episode_frequency=10,
                env_args={
                    "max_episode_steps": 200,
                    "reward_shaping": {
                        "name": "mountain-car-height",
                        "args": {"scale": 1.0},
                    },
                },
                exp_id="test-mc-shaped",
            )
            task.policy_control(instance_shaped)

            unshaped_returns = _read_log_entries(unshaped_dir)[0]["returns"]
            shaped_returns = _read_log_entries(shaped_dir)[0]["returns"]
            assert unshaped_returns != shaped_returns


def _assert_training_snapshots(
    snapshots: Sequence[algorithms.PolicyControlSnapshot],
    expected_count: int,
):
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


def _make_experiment_instance(
    output_dir: str,
    episodes_per_run: int,
    log_episode_frequency: int,
    reward_mapper: Optional[Mapping[str, Any]] = None,
    delay_config: Optional[Mapping[str, Any]] = None,
    env_args: Optional[Mapping[str, Any]] = None,
    export_model: bool = False,
    exp_id: str = "test-mc",
) -> core.ExperimentInstance:
    if reward_mapper is None:
        reward_mapper = {"name": "identity", "args": None}
    if env_args is None:
        env_args = {"max_episode_steps": 200}
    return core.ExperimentInstance(
        exp_id=exp_id,
        instance_id=0,
        experiment=core.Experiment(
            env_spec=core.EnvSpec(
                name="MountainCar-v0",
                args=env_args,
                feats_spec=[
                    {
                        "name": "tile-observation-action-ft",
                        "args": {"tiling_dim": 2},
                    }
                ],
            ),
            problem_spec=core.ProblemSpec(
                policy_type="markovian",
                reward_mapper=reward_mapper,
                delay_config=delay_config,
                epsilon=0.2,
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
            episodes_per_run=episodes_per_run,
            log_episode_frequency=log_episode_frequency,
            use_seed=True,
            output_dir=output_dir,
        ),
        context={},
        export_model=export_model,
    )


def _read_log_entries(output_dir: str) -> List[Dict[str, Any]]:
    log_path = os.path.join(output_dir, logger.ExperimentLogger.LOG_FILE_NAME)
    with open(log_path) as readable:
        return [json.loads(line) for line in readable]


def _read_params(output_dir: str) -> Dict[str, Any]:
    param_path = os.path.join(output_dir, logger.ExperimentLogger.PARAM_FILE_NAME)
    with open(param_path) as readable:
        params: Dict[str, Any] = json.load(readable)
    return params
