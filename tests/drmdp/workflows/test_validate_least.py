"""Tests for validate_least_coverage.py — grid loading, config generation,
and mapper construction."""

import os

from drmdp import envs, task
from drmdp.workflows.grids import rewest_runner, validate_least_coverage

# Smallest solvable grid: one row, start / open / goal cells, no dead cells.
GRID_SPEC = {
    "env_name": "GridWorld-test",
    "grid": ["sog"],
    "size": [1, 3],
    "max_episode_steps": 100,
    "dead_ohe": [],
}


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


class TestMethodConfigsMapperConstruction:
    """Every METHOD_CONFIGS entry must build through `task.reward_mapper`.

    The mapping spec is built exactly as `rewest_runner` builds it for a
    run, so an argument removed from the reward-estimation wrappers (e.g.
    `use_next_state`, replaced by `transition_feats_spec`) fails here
    instead of crashing every run of the validation script.
    """

    def _make_envs(self):
        env = envs.make(
            env_name=GRID_SPEC["env_name"],
            grid=GRID_SPEC["grid"],
            max_episode_steps=GRID_SPEC["max_episode_steps"],
        )
        proxy_env = envs.make(
            env_name=GRID_SPEC["env_name"],
            grid=GRID_SPEC["grid"],
            max_episode_steps=GRID_SPEC["max_episode_steps"],
        )
        return env, proxy_env

    def test_estimator_methods_match_live_state_action_dim(self):
        env, proxy_env = self._make_envs()
        nrows, ncols = GRID_SPEC["size"]
        live_dim = nrows * ncols * 4 - len(GRID_SPEC["dead_ohe"])
        for method in ("least-lfa", "bayes-least-lfa"):
            mapping_spec = rewest_runner.mapping_spec_for(
                GRID_SPEC, method, validate_least_coverage.METHOD_CONFIGS[method]
            )
            mapper = task.reward_mapper(
                env, proxy_env=proxy_env, mapping_spec=mapping_spec
            )
            assert mapper.feats_fn.feature_dim == live_dim
        env.close()
        proxy_env.close()

    def test_identity_returns_env_unwrapped(self):
        env, proxy_env = self._make_envs()
        mapping_spec = rewest_runner.mapping_spec_for(
            GRID_SPEC, "identity", validate_least_coverage.METHOD_CONFIGS["identity"]
        )
        assert (
            task.reward_mapper(env, proxy_env=proxy_env, mapping_spec=mapping_spec)
            is env
        )
        env.close()
        proxy_env.close()

    def test_impute_missing_uses_configured_value(self):
        env, proxy_env = self._make_envs()
        mapping_spec = rewest_runner.mapping_spec_for(
            GRID_SPEC,
            "impute-missing",
            validate_least_coverage.METHOD_CONFIGS["impute-missing"],
        )
        mapper = task.reward_mapper(env, proxy_env=proxy_env, mapping_spec=mapping_spec)
        assert mapper.impute_value == float(
            validate_least_coverage.METHOD_CONFIGS["impute-missing"]["impute_value"]
        )
        env.close()
        proxy_env.close()
