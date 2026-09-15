"""Tests for dataproc.py — data processing utilities."""

import copy

import gymnasium as gym
import pandas as pd

from drmdp import dataproc


class TestCollectionTrajData:
    def test_collects_trajectory(self):
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        buffer = dataproc.collection_traj_data(env, steps=50, seed=0)
        assert len(buffer) == 50
        obs, action, next_obs, reward = buffer[0]
        assert obs.shape == (2,)
        assert next_obs.shape == (2,)
        env.close()


class TestProcessData:
    def _make_raw_df(self):
        meta = {
            "experiment": {
                "env_spec": {"name": "TestEnv", "args": {}},
                "problem_spec": {
                    "reward_mapper": {"name": "identity"},
                    "policy_type": "markovian",
                },
            }
        }
        return pd.DataFrame({"meta": [copy.deepcopy(meta)], "returns": [[-1.0]]})

    def test_process_adds_method_column(self):
        df = self._make_raw_df()
        result = dataproc.process_data(df)
        assert "method" in result.columns

    def test_process_simplifies_meta(self):
        df = self._make_raw_df()
        result = dataproc.process_data(df)
        meta = result.iloc[0]["meta"]
        assert "reward_mapper" in meta
        assert "experiment" not in meta


class TestWideMetrics:
    def test_explodes_returns(self):
        df = pd.DataFrame(
            {"metrics": ["dummy"], "returns": [[1, 2, 3]], "env": ["test"]}
        )
        result = dataproc.wide_metrics(df)
        assert "metrics" not in result.columns
        assert len(result) == 3


class TestGetDistinctEnvs:
    def test_extracts_envs(self):
        df = pd.DataFrame(
            {
                "meta": [
                    {"env_spec": {"name": "Env1", "args": {"a": 1}}},
                    {"env_spec": {"name": "Env2", "args": {"b": 2}}},
                    {"env_spec": {"name": "Env1", "args": {"a": 1}}},
                ]
            }
        )
        result = dataproc.get_distinct_envs(df)
        assert len(result) == 2
        assert "Env1" in result
        assert "Env2" in result


class TestDropDuplicateSets:
    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": [2, 1, 2], "c": ["x", "y", "z"]})
        result = dataproc.drop_duplicate_sets(df, keys=["a", "b"])
        assert len(result) == 1

    def test_keeps_distinct_sets(self):
        df = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
        result = dataproc.drop_duplicate_sets(df, keys=["a", "b"])
        assert len(result) == 2
