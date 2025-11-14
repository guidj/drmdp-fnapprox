import logging
import os
import os.path
import uuid
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from drmdp import algorithms, core, envs, feats, logger, optsol, rewdelay
from drmdp.envs import wrappers

DELAYS: Sequence[type[rewdelay.RewardDelay]] = (
    rewdelay.FixedDelay,
    rewdelay.UniformDelay,
    rewdelay.ClippedPoissonDelay,
)
DELAY_BUILDERS: Mapping[str, type[rewdelay.RewardDelay]] = {
    clz.id(): clz for clz in DELAYS
}


def policy_control(exp_instance: core.ExperimentInstance):
    """
    Entry point running on-policy evaluation for DAAF.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env_spec = exp_instance.experiment.env_spec
    problem_spec = exp_instance.experiment.problem_spec
    proxied_env = create_env(env_spec.name, args=env_spec.args)
    env, monitor = monitor_wrapper(proxied_env.env)
    rew_delay = reward_delay_distribution(problem_spec.delay_config)
    env = delay_wrapper(env, rew_delay)
    env = reward_mapper(
        env,
        proxy_env=proxied_env.proxy,
        mapping_spec=problem_spec.reward_mapper,
    )
    feats_tfx = feats.create_feat_transformer(env=env, **env_spec.feats_spec)
    lr = learning_rate(**problem_spec.learning_rate_config)
    # Create spec using provided name and args for feature spec
    algorithm = create_algorithm(
        env=env,
        feats_transform=feats_tfx,
        delay_reward=rew_delay,
        lr=lr,
        gamma=problem_spec.gamma,
        epsilon=problem_spec.epsilon,
        policy_type=problem_spec.policy_type,
        base_seed=exp_instance.instance_id,
    )

    logging.debug("Starting DRMDP Control Experiments: %s", exp_instance)

    results = algorithm.train(
        env=env, num_episodes=exp_instance.run_config.episodes_per_run, monitor=monitor
    )

    with logger.ExperimentLogger(
        log_dir=exp_instance.run_config.output_dir, experiment_instance=exp_instance
    ) as exp_logger:
        returns = []
        instance_id = f"{exp_instance.exp_id}_{exp_instance.instance_id}"
        try:
            for episode, snapshot in enumerate(results):
                returns.append(snapshot.returns)
                if episode % exp_instance.run_config.log_episode_frequency == 0:
                    exp_logger.log(
                        episode=episode,
                        steps=snapshot.steps,
                        returns=np.mean(returns).item(),
                        info={},
                    )
                    if exp_instance.model_dir:
                        logger.save_model(
                            snapshot.weights,
                            name=f"weights_{episode}",
                            model_dir=os.path.join(exp_instance.model_dir, instance_id),
                        )

            logging.debug(
                "\nReturns for run %d of %s:\n%s",
                exp_instance.instance_id,
                exp_instance.exp_id,
                np.mean(returns),
            )
        except Exception as err:
            logging.error(
                "Task %s, run %s failed: %s",
                exp_instance.exp_id,
                exp_instance.instance_id,
                err,
            )
            raise RuntimeError(
                f"Task {exp_instance.exp_id}, run {exp_instance.instance_id} failed"
            ) from err
    env.close()


def create_env(name: str, args: Optional[Mapping[str, Any]]) -> core.ProxiedEnv:
    """
    Creates an env and a proxy.
    """
    env = envs.make(
        env_name=name,
        **args if args else {},
    )
    proxy = envs.make(
        env_name=name,
        **args if args else {},
    )
    return core.ProxiedEnv(env=env, proxy=proxy)


def create_task_id(task_prefix: str) -> str:
    """
    Creates a task id using a given prefix
    and a generated partial uuid.
    """
    return f"{task_prefix}-{uid()}"


def uid() -> str:
    """
    Generate a uuid from partial uuid
    """
    return next(iter(str(uuid.uuid4()).split("-")))


def generate_experiments_instances(
    experiments: Sequence[core.Experiment],
    num_runs: int,
    num_episodes_per_epoch: int,
    log_episode_frequency: int,
    use_seed: bool,
    output_dir: str,
    task_prefix: str,
    model_dir: Optional[str],
) -> Iterator[core.ExperimentInstance]:
    """
    Parse experiments and creates experiment
    instances from it.
    """
    for experiment in experiments:
        exp_id = "-".join([create_task_id(task_prefix), experiment.env_spec.name])
        for idx in range(num_runs):
            yield core.ExperimentInstance(
                exp_id=exp_id,
                instance_id=idx,
                experiment=experiment,
                run_config=core.RunConfig(
                    num_runs=num_runs,
                    episodes_per_run=num_episodes_per_epoch * experiment.epochs,
                    log_episode_frequency=log_episode_frequency,
                    use_seed=use_seed,
                    # replace run output with run specific values
                    output_dir=os.path.join(
                        output_dir,
                        exp_id,
                        f"run_{idx}",
                        experiment.problem_spec.reward_mapper["name"],
                        uid(),
                    ),
                ),
                context={"dummy": 0},
                model_dir=model_dir,
            )


def bundle(items: Sequence[Any], bundle_size: int) -> Sequence[Sequence[Any]]:
    """
    Bundles items into groups of size `bundle_size`, if possible.
    The last bundle may have fewer items.
    """
    if bundle_size < 1:
        raise ValueError("`bundle_size` must be positive.")

    bundles: List[List[Any]] = []
    bundle_: List[Any] = []
    for idx, item in enumerate(items):
        if idx > 0 and (idx % bundle_size) == 0:
            if bundle_:
                bundles.append(bundle_)
            bundle_ = []
        bundle_.append(item)
    if bundle_:
        bundles.append(bundle_)
    return bundles


def learning_rate(name: str, args: Mapping[str, Any]) -> optsol.LearningRateSchedule:
    """
    Returns a learning rate scheduler.
    """
    if "initial_lr" not in args:
        raise ValueError(f"Missing `initial_lr` from lr config: {args}")
    if name == "constant":
        return optsol.ConstantLRSchedule(args["initial_lr"])
    else:
        raise ValueError(f"Unknown lr {name}")


def reward_delay_distribution(
    delay_config: Optional[Mapping[str, Any]],
) -> Optional[rewdelay.RewardDelay]:
    """
    Returns an instance of a delayed
    reward distribution that can be used
    to sample delays.
    """
    if delay_config:
        name = delay_config["name"]
        args = delay_config["args"]
        if name not in DELAY_BUILDERS:
            raise ValueError(f"Unknown delay type {name}")
        return DELAY_BUILDERS[name](**args)
    return None


def monitor_wrapper(env: gym.Env) -> Tuple[gym.Env, core.EnvMonitor]:
    """
    Wraps the environment in monitor that tracks returns
    based on the underlying rewards.
    """
    mon_env = core.EnvMonitorWrapper(env)
    return mon_env, mon_env.mon


def delay_wrapper(
    env: gym.Env, reward_delay: Optional[rewdelay.RewardDelay]
) -> gym.Env:
    """
    If a delayed reward config is given, wraps
    `env` with the specified mapper.
    """
    if reward_delay:
        return rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)
    return env


def reward_mapper(env: gym.Env, proxy_env: gym.Env, mapping_spec: Mapping[str, Any]):
    """
    Creates a mapper for handling missing rewards.
    """
    name = mapping_spec["name"]
    m_args = dict(**(mapping_spec["args"] or {}))
    enc_feats_spec = m_args.pop("feats_spec") if "feats_spec" in m_args else None
    observation_enc = observation_encoder(proxy_env, feats_spec=enc_feats_spec)

    if name == "identity":
        return env
    elif name == "impute-missing":
        return rewdelay.ImputeMissingRewardWrapper(env, **m_args)
    elif name == "discrete-least-lfa":
        return rewdelay.DiscretisedLeastLfaGenerativeRewardWrapper(
            env=env,
            obs_encoding_wrapper=observation_enc,
            **m_args,
        )
    elif name == "least-lfa":
        # local copy before pop
        return rewdelay.LeastLfaGenerativeRewardWrapper(
            env=env,
            obs_encoding_wrapper=observation_enc,
            **m_args,
        )
    elif name == "bayes-least-lfa":
        # local copy before pop
        return rewdelay.BayesLeastLfaGenerativeRewardWrapper(
            env=env,
            obs_encoding_wrapper=observation_enc,
            **m_args,
        )
    elif name == "cvlps":
        # local copy before pop
        return rewdelay.ConvexSolverGenerativeRewardWrapper(
            env=env,
            obs_encoding_wrapper=observation_enc,
            **m_args,
        )
    elif name == "recurring-cvlps":
        # local copy before pop
        return rewdelay.RecurringConvexSolverGenerativeRewardWrapper(
            env=env,
            obs_encoding_wrapper=observation_enc,
            **m_args,
        )
    raise ValueError(f"Unknown mapping_method: {mapping_spec}")


def observation_encoder(
    env: gym.Env, feats_spec: Optional[Mapping[str, Any]]
) -> Optional[gym.ObservationWrapper]:
    """
    Creates an observation wrapper given a spec.
    """
    if feats_spec is None:
        return None
    return wrappers.wrap(env, wrapper=feats_spec["name"], **(feats_spec["args"] or {}))


def create_algorithm(
    env: gym.Env,
    feats_transform: feats.FeatTransform,
    policy_type: str,
    delay_reward: Optional[rewdelay.RewardDelay],
    lr: optsol.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    base_seed: Optional[int] = None,
):
    """
    Creates an algorithm instance based on the provided
    arguments.
    """
    if policy_type == "markovian":
        return algorithms.SemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.LinearFnApproxPolicy(
                feat_transform=feats_transform, action_space=env.action_space
            ),
            base_seed=base_seed,
        )
    elif policy_type == "uniform-random":
        return algorithms.SemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.RandomFnApproxPolicy(
                feat_transform=feats_transform, action_space=env.action_space
            ),
            base_seed=base_seed,
        )
    elif policy_type == "drop-missing":
        return algorithms.DropMissingSemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.LinearFnApproxPolicy(
                feat_transform=feats_transform, action_space=env.action_space
            ),
            base_seed=base_seed,
        )
    elif policy_type == "options":
        if delay_reward is None:
            raise ValueError("`delay_reward` must be provided")
        return algorithms.OptionsSemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.OptionsLinearFnApproxPolicy(
                feat_transform=feats_transform,
                action_space=env.action_space,
                options_length_range=delay_reward.range(),
            ),
            base_seed=base_seed,
        )
    elif policy_type == "single-action-options":
        if delay_reward is None:
            raise ValueError("`delay_reward` must be provided")
        return algorithms.OptionsSemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.SingleActionOptionsLinearFnApproxPolicy(
                feat_transform=feats_transform,
                action_space=env.action_space,
                options_length_range=delay_reward.range(),
            ),
            base_seed=base_seed,
        )

    raise ValueError(f"Unknown policy_type: {policy_type}")
