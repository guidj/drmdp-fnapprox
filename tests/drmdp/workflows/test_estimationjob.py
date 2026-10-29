"""Tests for estimationjob — parsing and the estimation-only loop."""

import json
import os

import numpy as np

from drmdp import core, task
from drmdp.workflows import estimationexps, estimationjob, specutils


class TestParseExperiments:
    def test_experiments_match_the_problem_spec_grid(self):
        specs = estimationexps.estimation_experiment_specs()
        experiments = specutils.parse_experiment_specs(specs)
        assert len(experiments) == 168  # 28 specs x 6 problem specs
        assert all(isinstance(exp, core.Experiment) for exp in experiments)
        assert all(exp.env_spec.feats_spec == [] for exp in experiments)

    def test_encoding_metadata_flows_to_the_env_spec(self):
        specs = estimationexps.estimation_experiment_specs()
        experiments = specutils.parse_experiment_specs(specs)
        metadata = {exp.env_spec.metadata["encoding"] for exp in experiments}
        assert "transition-anomaly" in metadata
        assert "structural-sine" in metadata


class TestCreateTasks:
    def test_instances_match_the_run_grid_with_per_env_episode_caps(self):
        instances = estimationjob.create_tasks(
            num_runs=1,
            output_dir="unused",
            task_prefix="test",
            log_episode_frequency=5,
            use_seed=True,
        )
        assert len(instances) == 168
        caps_by_env = {}
        for instance in instances:
            env_name = instance.experiment.env_spec.name
            cap = instance.run_config.episodes_per_run
            caps_by_env.setdefault(env_name, set()).add(cap)
        assert caps_by_env == {
            "MountainCar-v0": {40},
            "Acrobot-v1": {250},
            "GridWorld-MINES": {400},
        }

    def test_every_cell_has_a_distinct_exp_id(self):
        """The aggregation dedups events by (exp_id, instance_id)."""
        instances = estimationjob.create_tasks(
            num_runs=1,
            output_dir="unused",
            task_prefix="test",
            log_episode_frequency=5,
            use_seed=True,
        )
        identities = [(inst.exp_id, inst.instance_id) for inst in instances]
        assert len(set(identities)) == len(identities)

    def test_cell_slug_separates_gate_variants(self):
        specs = estimationexps.estimation_experiment_specs()
        experiments = specutils.parse_experiment_specs(specs)
        gated = [
            experiment
            for experiment in experiments
            if (experiment.env_spec.metadata or {}).get("encoding") == "structural-sine"
        ]
        slugs = {estimationjob.cell_slug(experiment) for experiment in gated}
        # encoding x 3 methods (gate splits LEAST) x 2 delays x 2 noise
        assert len(slugs) == 12


class TestEstimationOnly:
    """End-to-end: the random-policy loop records events and stops at fires."""

    def _gw_instance(self, tmp_path, encoding, gated, episodes):
        spec = None
        for candidate in estimationexps.estimation_experiment_specs():
            if (
                candidate["name"] == "GridWorld-MINES"
                and candidate["metadata"]["encoding"] == encoding
            ):
                spec = candidate
                break
        problem = next(
            problem
            for problem in spec["problem_specs"]
            if problem["reward_mapper"]["name"] == "least-lfa"
            and problem["reward_mapper"]["args"]["check_factors"] == gated
            and problem["delay_config"]["args"]["min_delay"] == 2
        )
        return core.ExperimentInstance(
            exp_id="test-estimation",
            instance_id=0,
            experiment=core.Experiment(
                env_spec=core.EnvSpec(
                    name=spec["name"],
                    args=spec["args"],
                    feats_spec=[],
                    metadata=spec["metadata"],
                ),
                problem_spec=core.ProblemSpec(**problem),
                epochs=1,
            ),
            run_config=core.RunConfig(
                num_runs=1,
                episodes_per_run=episodes,
                log_episode_frequency=5,
                use_seed=True,
                output_dir=str(tmp_path),
            ),
            context={},
            export_model=False,
        )

    def _read_log(self, tmp_path):
        for root, _, files in os.walk(str(tmp_path)):
            if "experiment-logs.jsonl" in files:
                with open(os.path.join(root, "experiment-logs.jsonl")) as readable:
                    return [json.loads(line) for line in readable]
        raise AssertionError("no experiment log was written")

    def test_one_shot_run_stops_at_first_event(self, tmp_path):
        instance = self._gw_instance(
            tmp_path, encoding="transition-anomaly", gated=False, episodes=30
        )
        task.estimation_only(instance)
        entries = self._read_log(tmp_path)
        events = [
            event
            for entry in entries
            for event in (entry.get("info") or {}).get("reward_errors", [])
        ]
        assert len(events) == 1
        # the anomaly encoding is exact on the live pairs
        np.testing.assert_allclose(events[0]["rmse"], 0.0, atol=1e-9)
        assert events[0]["steps"] > 0
        assert events[0]["elapsed_s"] >= 0.0
        # the run stopped at the fire episode, not the cap
        assert entries[-1]["episode"] < 30

    def test_gated_run_records_deferrals_until_the_gate_opens(self, tmp_path):
        # next-cell-ohe has structurally unreachable columns, so the
        # coverage gate defers; the ungated variant would fire at the
        # first attempt
        instance = self._gw_instance(
            tmp_path, encoding="next-cell-ohe", gated=True, episodes=20
        )
        task.estimation_only(instance)
        entries = self._read_log(tmp_path)
        deferred = [
            attempt
            for entry in entries
            for attempt in (entry.get("info") or {}).get("deferred_attempts", [])
        ]
        assert deferred
        assert all(
            attempt["nonzero_columns"] < attempt["matrix_columns"]
            for attempt in deferred
        )
        assert all(attempt["steps"] > 0 for attempt in deferred)
