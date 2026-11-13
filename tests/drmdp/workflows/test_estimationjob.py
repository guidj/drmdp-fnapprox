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
        assert len(experiments) == 360  # 30 specs x 12 problem specs
        assert all(isinstance(exp, core.Experiment) for exp in experiments)
        assert all(exp.env_spec.feats_spec == [] for exp in experiments)

    def test_encoding_metadata_flows_to_the_env_spec(self):
        specs = estimationexps.estimation_experiment_specs()
        experiments = specutils.parse_experiment_specs(specs)
        metadata = {exp.env_spec.metadata["encoding"] for exp in experiments}
        assert "transition-anomaly" in metadata
        assert "potential-curr-succ" in metadata


class TestCreateTasks:
    def test_instances_match_the_run_grid_with_per_env_episode_caps(self):
        instances = estimationjob.create_tasks(
            num_runs=1,
            output_dir="unused",
            task_prefix="test",
            log_episode_frequency=5,
            use_seed=True,
        )
        assert len(instances) == 360
        caps_by_env = {}
        for instance in instances:
            env_name = instance.experiment.env_spec.name
            cap = instance.run_config.episodes_per_run
            caps_by_env.setdefault(env_name, set()).add(cap)
        assert caps_by_env == {
            "MountainCar-v0": {500},
            "Acrobot-v1": {500},
            "GridWorld-MINES": {500},
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

    def test_cell_key_separates_check_mode_variants(self):
        specs = estimationexps.estimation_experiment_specs()
        experiments = specutils.parse_experiment_specs(specs)
        coords_exps = [
            experiment
            for experiment in experiments
            if (experiment.env_spec.metadata or {}).get("encoding")
            == "potential-curr-succ"
            and experiment.env_spec.name == "MountainCar-v0"
        ]
        keys = {estimationjob.cell_key(experiment) for experiment in coords_exps}
        # 2 estimators x 3 modes x 2 delays x 2 noise = 24
        assert len(keys) == 24


class TestEstimationOnly:
    """End-to-end: the random-policy loop records events and stops at fires."""

    def _gw_instance(self, tmp_path, encoding, estimation_check, episodes):
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
            and problem["reward_mapper"]["args"]["estimation_check"] == estimation_check
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
            tmp_path,
            encoding="potential-curr-succ",
            estimation_check="none",
            episodes=30,
        )
        task.estimation_only(instance)
        entries = self._read_log(tmp_path)
        events = [
            event
            for entry in entries
            for event in (entry.get("info") or {}).get("reward_errors", [])
        ]
        assert len(events) == 1
        # one-shot stop is what this checks; the encoding represents the
        # shaped reward exactly, but a 40-coordinate map needs coverage
        # before the minimum-norm fit reproduces it everywhere
        assert np.isfinite(events[0]["rmse"])
        assert events[0]["rmse"] >= 0.0
        assert events[0]["steps"] > 0
        assert events[0]["elapsed_s"] >= 0.0
        # the run stopped at the fire episode, not the cap
        assert entries[-1]["episode"] < 30

    def test_gated_run_records_deferrals_until_the_gate_opens(self, tmp_path):
        # next-cell-ohe has structurally unreachable columns, so the
        # coverage gate defers; the ungated variant would fire at the
        # first attempt
        instance = self._gw_instance(
            tmp_path,
            encoding="next-cell-ohe",
            estimation_check="factors",
            episodes=20,
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

    def test_seeded_runs_reproduce_their_record(self, tmp_path):
        """Same instance seed, same record: delay and noise draws included."""
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first = self._gw_instance(
            first_dir,
            encoding="transition-anomaly",
            estimation_check="none",
            episodes=30,
        )
        second = self._gw_instance(
            second_dir,
            encoding="transition-anomaly",
            estimation_check="none",
            episodes=30,
        )
        task.estimation_only(first)
        task.estimation_only(second)
        first_entries = _strip_wall_clock(self._read_log(first_dir))
        second_entries = _strip_wall_clock(self._read_log(second_dir))
        assert first_entries == second_entries
        # the record is non-trivial: the run fired within the cap
        assert any(entry["reward_errors"] for entry in first_entries)


def _strip_wall_clock(entries: list[dict]) -> list[dict]:
    """Log entries without wall-clock fields, which no seeded run reproduces."""
    stripped = []
    for entry in entries:
        info = entry.get("info") or {}
        stripped.append(
            {
                "episode": entry["episode"],
                "steps": entry["steps"],
                "returns": entry["returns"],
                "reward_errors": [
                    {key: value for key, value in event.items() if key != "elapsed_s"}
                    for event in info.get("reward_errors", [])
                ],
                "deferred_attempts": info.get("deferred_attempts", []),
            }
        )
    return stripped
