import dataclasses
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


def policy_control_run_fn(exp_instance: core.ExperimentInstance):
    """
    Entry point running on-policy evaluation for DAAF.

    Args:
        args: configuration for execution.
    """
    # init env and agent
    env_spec = exp_instance.experiment.env_spec
    problem_spec = exp_instance.experiment.problem_spec
    env = envs.make(
        env_name=env_spec.name,
        **env_spec.args if env_spec.args else {},
    )
    env, monitor = monitor_wrapper(env)
    rew_delay = reward_delay_distribution(problem_spec.delay_config)
    env = delay_wrapper(env, rew_delay)
    env = reward_mapper(
        env,
        mapping_spec=problem_spec.reward_mapper,
        feats_spec=env_spec.feats_spec,
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
    )

    logging.debug("Starting DRMDP Control Experiments: %s", exp_instance)

    results = policy_control(
        env=env,
        algorithm=algorithm,
        num_episodes=exp_instance.run_config.episodes_per_run,
        monitor=monitor,
    )
    with logger.ExperimentLogger(
        log_dir=exp_instance.run_config.output_dir, experiment_instance=exp_instance
    ) as exp_logger:
        returns = []
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

            logging.debug(
                "\nReturns for run %d of %s:\n%s",
                exp_instance.instance_id,
                exp_instance.exp_id,
                np.mean(returns),
            )
        except Exception as err:
            raise RuntimeError(
                f"Task {exp_instance.exp_id}, run {exp_instance.instance_id} failed"
            ) from err
    env.close()


def policy_control(
    env: gym.Env,
    algorithm: algorithms.FnApproxAlgorithm,
    num_episodes: int,
    monitor: core.EnvMonitor,
) -> Iterator[algorithms.PolicyControlSnapshot]:
    """
    Runs policy control with given algorithm, env, and policy spec.
    """
    return algorithm.train(env=env, num_episodes=num_episodes, monitor=monitor)


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
    run_config: core.RunConfig,
    task_prefix: str,
) -> Iterator[core.ExperimentInstance]:
    for experiment in experiments:
        exp_id = "-".join([create_task_id(task_prefix), experiment.env_spec.name])
        for idx in range(run_config.num_runs):
            yield core.ExperimentInstance(
                exp_id=exp_id,
                instance_id=idx,
                experiment=experiment,
                run_config=dataclasses.replace(
                    run_config,
                    # replace run output with run specific values
                    output_dir=os.path.join(
                        run_config.output_dir,
                        exp_id,
                        f"run_{idx}",
                        experiment.problem_spec.reward_mapper["name"],
                        uid(),
                    ),
                ),
                context={"dummy": 0},
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
    if delay_config:
        name = delay_config["name"]
        args = delay_config["args"]
        if name not in DELAY_BUILDERS:
            raise ValueError(f"Unknown delay type {name}")
        return DELAY_BUILDERS[name](**args)
    return None


def monitor_wrapper(env: gym.Env) -> Tuple[gym.Env, core.EnvMonitor]:
    mon_env = core.EnvMonitorWrapper(env)
    return mon_env, mon_env.mon


def delay_wrapper(
    env: gym.Env, reward_delay: Optional[rewdelay.RewardDelay]
) -> gym.Env:
    if reward_delay:
        return rewdelay.DelayedRewardWrapper(env, reward_delay=reward_delay)
    return env


def reward_mapper(
    env: gym.Env, mapping_spec: Mapping[str, Any], feats_spec: Mapping[str, Any]
):
    name = mapping_spec["name"]
    args = mapping_spec["args"]
    if name == "identity":
        return env
    elif name == "zero-impute":
        return rewdelay.ZeroImputeMissingWrapper(env)
    elif name == "least-lfa":
        # local copy before pop
        return rewdelay.LeastLfaMissingWrapper(
            env=env,
            obs_encoding_wrapper=wrappers.wrap(
                env, wrapper=feats_spec["name"], **(feats_spec["args"] or {})
            ),
            **(args if args else {}),
        )
    raise ValueError(f"Unknown mapping_method: {mapping_spec}")


def create_algorithm(
    env: gym.Env,
    feats_transform: feats.FeatTransform,
    policy_type: str,
    delay_reward: Optional[rewdelay.RewardDelay],
    lr: optsol.LearningRateSchedule,
    gamma: float,
    epsilon: float,
):
    if policy_type == "markovian":
        return algorithms.SemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.LinearFnApproxPolicy(
                feat_transform=feats_transform, action_space=env.action_space
            ),
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
        )

    raise ValueError(f"Unknown policy_type: {policy_type}")
