"""Tests for validate_least_coverage.py — grid loading and config generation."""

import os

from drmdp.workflows.grids import validate_least_coverage


class TestBuildRunConfigs:
    def test_identity_has_no_delay(self):
        grids = [
            {
                "env_name": "GridWorld-test",
                "grid": ["sog"],
                "size": [1, 3],
                "max_episode_steps": 100,
                "dead_ohe": [2],
            }
        ]
        configs = validate_least_coverage.build_run_configs(
            grids=grids,
            max_steps_values=(100,),
            num_episodes_values=(50,),
        )
        identity_configs = [cfg for cfg in configs if cfg["method"] == "identity"]
        assert len(identity_configs) > 0
        for cfg in identity_configs:
            assert cfg["delay_config"] is None

    def test_delayed_methods_all_have_delay(self):
        grids = [
            {
                "env_name": "GridWorld-test",
                "grid": ["sog"],
                "size": [1, 3],
                "max_episode_steps": 100,
                "dead_ohe": [2],
            }
        ]
        configs = validate_least_coverage.build_run_configs(
            grids=grids,
            max_steps_values=(100,),
            num_episodes_values=(50,),
        )
        delayed_configs = [cfg for cfg in configs if cfg["method"] != "identity"]
        assert len(delayed_configs) > 0
        for cfg in delayed_configs:
            assert cfg["delay_config"] is not None
            assert cfg["delay_config"]["args"]["min_delay"] >= 2

    def test_max_steps_applied_to_grid_spec(self):
        grids = [
            {
                "env_name": "GridWorld-test",
                "grid": ["sog"],
                "size": [1, 3],
                "max_episode_steps": 100,
                "dead_ohe": [2],
            }
        ]
        configs = validate_least_coverage.build_run_configs(
            grids=grids,
            max_steps_values=(200, 500),
            num_episodes_values=(50,),
        )
        max_steps_seen = {cfg["grid_spec"]["max_episode_steps"] for cfg in configs}
        assert max_steps_seen == {200, 500}


class TestLoadGrids:
    def test_limits_grids_per_size(self):
        grids_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "assets", "grids.json"
        )
        if not os.path.exists(grids_path):
            return
        grids_one = validate_least_coverage.load_grids(
            grid_files=[grids_path],
            min_passes=4,
            grids_per_size=1,
        )
        grids_two = validate_least_coverage.load_grids(
            grid_files=[grids_path],
            min_passes=4,
            grids_per_size=2,
        )
        assert len(grids_two) >= len(grids_one)
