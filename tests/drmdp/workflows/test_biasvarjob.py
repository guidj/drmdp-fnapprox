"""Tests for biasvarjob.py — spec builders and job creation."""

from drmdp.workflows import biasvarjob


class TestLeastLfaSpecs:
    def test_cartesian_product_of_episodes_and_feats(self):
        feats = [
            [{"name": "a", "args": None}],
            [{"name": "b", "args": None}],
        ]
        result = biasvarjob.least_lfa_specs(
            attempt_estimation_episodes=(10, 50),
            feat_specs=feats,
        )
        assert len(result) == 4
        episodes_seen = {spec["args"]["attempt_estimation_episode"] for spec in result}
        assert episodes_seen == {10, 50}

    def test_feats_wired_into_spec_args(self):
        feats = [[{"name": "custom-ft", "args": {"dim": 3}}]]
        result = biasvarjob.least_lfa_specs(
            attempt_estimation_episodes=(10,),
            feat_specs=feats,
        )
        assert result[0]["args"]["feats_spec"] == feats[0]


class TestBayesLeastLfaSpecs:
    def test_mode_defaults_to_double(self):
        feats = [[{"name": "scale-observation-ft", "args": None}]]
        result = biasvarjob.bayes_least_lfa_specs(
            init_attempt_estimation_episodes=(10,),
            feat_specs=feats,
        )
        assert result[0]["args"]["mode"] == "double"


class TestCreateAllJobSpecs:
    def test_count_matches_cartesian_product(self):
        feats = [[{"name": "scale-observation-ft", "args": None}]]
        specs = [
            {
                "name": "TestEnv",
                "args": {},
                "feats_specs": feats,
                "rewest": biasvarjob.least_lfa_specs(
                    attempt_estimation_episodes=(10,),
                    feat_specs=feats,
                ),
                "epochs": 1,
            }
        ]
        num_runs = 2
        delays = (3, 5)
        gammas = (0.99,)
        jobs = biasvarjob.create_all_job_specs(
            specs=specs,
            num_runs=num_runs,
            num_episodes=50,
            delays=delays,
            gammas=gammas,
        )
        num_envs = len(specs)
        num_feats = len(feats)
        num_rewest = len(specs[0]["rewest"])
        expected = (
            num_envs * num_feats * num_rewest * len(delays) * len(gammas) * num_runs
        )
        assert len(jobs) == expected

    def test_all_delays_and_gammas_represented(self):
        feats = [[{"name": "a", "args": None}]]
        specs = [
            {
                "name": "Env",
                "args": {},
                "feats_specs": feats,
                "rewest": biasvarjob.least_lfa_specs(
                    attempt_estimation_episodes=(10,),
                    feat_specs=feats,
                ),
                "epochs": 1,
            }
        ]
        jobs = biasvarjob.create_all_job_specs(
            specs=specs,
            num_runs=1,
            num_episodes=50,
            delays=(2, 8),
            gammas=(0.99, 1.0),
        )
        delays_seen = {job.reward_delay for job in jobs}
        gammas_seen = {job.gamma for job in jobs}
        assert delays_seen == {2, 8}
        assert gammas_seen == {0.99, 1.0}


class TestCreateExpInstance:
    def test_episodes_scale_with_epochs(self):
        job_spec = biasvarjob.JobSpec(
            env_name="MountainCar-v0",
            env_args={"max_episode_steps": 200},
            control_feats_spec=[
                {"name": "tile-observation-action-ft", "args": {"tiling_dim": 2}}
            ],
            rewest_method="least-lfa",
            rewest_args={
                "attempt_estimation_episode": 10,
                "feats_spec": [],
                "estimation_buffer_mult": 25,
            },
            reward_delay=3,
            gamma=0.99,
            estimation_episode=10,
            num_episodes=100,
            use_bias=False,
            epochs=3,
            run_id=0,
        )
        instance = biasvarjob.create_exp_instance(job_spec)
        assert instance.run_config.episodes_per_run == 300

    def test_delay_wired_as_fixed(self):
        job_spec = biasvarjob.JobSpec(
            env_name="MountainCar-v0",
            env_args={"max_episode_steps": 200},
            control_feats_spec=[],
            rewest_method="least-lfa",
            rewest_args={},
            reward_delay=7,
            gamma=1.0,
            estimation_episode=10,
            num_episodes=100,
            use_bias=False,
            epochs=1,
            run_id=0,
        )
        instance = biasvarjob.create_exp_instance(job_spec)
        delay_config = instance.experiment.problem_spec.delay_config
        assert delay_config["name"] == "fixed"
        assert delay_config["args"]["delay"] == 7

    def test_run_id_becomes_instance_id(self):
        job_spec = biasvarjob.JobSpec(
            env_name="MountainCar-v0",
            env_args={},
            control_feats_spec=[],
            rewest_method="least-lfa",
            rewest_args={},
            reward_delay=3,
            gamma=1.0,
            estimation_episode=10,
            num_episodes=100,
            use_bias=False,
            epochs=1,
            run_id=42,
        )
        instance = biasvarjob.create_exp_instance(job_spec)
        assert instance.instance_id == 42

    def test_gamma_wired_to_problem_spec(self):
        job_spec = biasvarjob.JobSpec(
            env_name="MountainCar-v0",
            env_args={},
            control_feats_spec=[],
            rewest_method="least-lfa",
            rewest_args={},
            reward_delay=3,
            gamma=0.95,
            estimation_episode=10,
            num_episodes=100,
            use_bias=False,
            epochs=1,
            run_id=0,
        )
        instance = biasvarjob.create_exp_instance(job_spec)
        assert instance.experiment.problem_spec.gamma == 0.95
