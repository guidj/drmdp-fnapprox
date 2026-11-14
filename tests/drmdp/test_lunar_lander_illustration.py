"""End-to-end test of a shortened LunarLander illustration run."""

import json
import os

import numpy as np

from drmdp import constants, core, task
from drmdp.workflows import controlexps


class TestLunarLanderControlRun:
    def test_logged_returns_and_evaluations_are_finite(self, tmp_path):
        learning_rate_config, _ = controlexps.derived_learning_rate_config(
            env_name="LunarLander-v3",
            env_args={
                "max_episode_steps": 1000,
                "wrapper": constants.LUNAR_LANDER_REWARD_OBS,
            },
            feats_specs=[
                [
                    {
                        "name": "splice-tile-observation-action-ft",
                        "args": {"tiling_dim": 8, "hash_dim": 2048},
                    }
                ]
            ],
            rule="dim",
        )
        exp_instance = core.ExperimentInstance(
            exp_id="lunar-lander-smoke",
            instance_id=0,
            experiment=core.Experiment(
                env_spec=core.EnvSpec(
                    name="LunarLander-v3",
                    args={
                        "max_episode_steps": 1000,
                        "wrapper": constants.LUNAR_LANDER_REWARD_OBS,
                    },
                    feats_spec=[
                        {
                            "name": "splice-tile-observation-action-ft",
                            "args": {"tiling_dim": 8, "hash_dim": 2048},
                        }
                    ],
                ),
                problem_spec=core.ProblemSpec(
                    policy_type="markovian",
                    reward_mapper={"name": "identity", "args": None},
                    delay_config=None,
                    epsilon=0.1,
                    gamma=0.99,
                    learning_rate_config=learning_rate_config,
                ),
                epochs=1,
            ),
            run_config=core.RunConfig(
                num_runs=1,
                episodes_per_run=20,
                log_episode_frequency=5,
                use_seed=True,
                output_dir=str(tmp_path),
            ),
            context=None,
            export_model=False,
        )

        task.policy_control(exp_instance)

        entries = _read_log(os.path.join(str(tmp_path), "experiment-logs.jsonl"))
        assert entries
        for entry in entries:
            assert np.isfinite(entry["returns"])
            assert entry["steps"] > 0
            assert (
                entry["info"]["policy_eval"]["num_episodes"]
                == task.POLICY_EVAL_EPISODES
            )
            assert np.isfinite(entry["info"]["policy_eval"]["mean"])


def _read_log(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as logged:
        return [json.loads(line) for line in logged if line.strip()]
