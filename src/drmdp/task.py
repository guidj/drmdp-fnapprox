import logging
import os
import os.path
import uuid
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from drmdp import algorithms, core, envs, logger, optsol, rewdelay, transform
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
    ft_op = transform.transform_pipeline(env=env, specs=env_spec.feats_spec)
    lr = learning_rate(**problem_spec.learning_rate_config)
    # Create spec using provided name and args for feature spec
    algorithm = create_algorithm(
        env=env,
        ft_op=ft_op,
        delay_reward=rew_delay,
        lr=lr,
        gamma=problem_spec.gamma,
        epsilon=problem_spec.epsilon,
        policy_type=problem_spec.policy_type,
        algo_config=problem_spec.learning_rate_config.get("args"),
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
                    if exp_instance.export_model:
                        export_model_snapshot(
                            snapshot.weights,
                            snapshot=episode,
                            max_snapshot=exp_instance.run_config.episodes_per_run,
                            model_dir=exp_instance.run_config.output_dir,
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
    export_model: bool,
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
                export_model=export_model,
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
    enc_feats_spec = m_args.pop("feats_spec") if "feats_spec" in m_args else []
    ft_op = create_ft_ops(proxy_env, feats_spec=enc_feats_spec)

    if name == "identity":
        return env
    elif name == "impute-missing":
        return rewdelay.ImputeMissingRewardWrapper(env, **m_args)
    elif name == "discrete-least-lfa":
        return rewdelay.DiscretisedLeastLfaGenerativeRewardWrapper(
            env=env,
            ft_op=ft_op,
            **m_args,
        )
    elif name == "least-lfa":
        return rewdelay.LeastLfaGenerativeRewardWrapper(
            env=env,
            ft_op=ft_op,
            **m_args,
        )
    elif name == "bayes-least-lfa":
        return rewdelay.BayesLeastLfaGenerativeRewardWrapper(
            env=env,
            ft_op=ft_op,
            **m_args,
        )
    elif name == "cvlps":
        return rewdelay.ConvexSolverGenerativeRewardWrapper(
            env=env,
            ft_op=ft_op,
            **m_args,
        )
    elif name == "recurring-cvlps":
        return rewdelay.RecurringConvexSolverGenerativeRewardWrapper(
            env=env,
            ft_op=ft_op,
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


def create_ft_ops(
    env: gym.Env, feats_spec: Sequence[Mapping[str, Any]]
) -> transform.FTOp:
    """
    Creates a features processing pipeline for an environment.
    """
    return transform.transform_pipeline(env, specs=feats_spec)


def create_algorithm(
    env: gym.Env,
    ft_op: transform.FTOp,
    policy_type: str,
    delay_reward: Optional[rewdelay.RewardDelay],
    lr: optsol.LearningRateSchedule,
    gamma: float,
    epsilon: float,
    algo_config: Optional[Mapping[str, Any]] = None,
    base_seed: Optional[int] = None,
):
    """
    Creates an algorithm instance based on the provided
    arguments.

    Args:
        algo_config: Optional algorithm-specific configuration parameters.
            E.g. neural network hyper-parameters.
    """
    if policy_type == "markovian":
        return algorithms.SemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.LinearFnApproxPolicy(
                ft_op=ft_op, action_space=env.action_space
            ),
            base_seed=base_seed,
        )
    elif policy_type == "uniform-random":
        return algorithms.SemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.RandomFnApproxPolicy(
                ft_op=ft_op, action_space=env.action_space
            ),
            base_seed=base_seed,
        )
    elif policy_type == "drop-missing":
        return algorithms.DropMissingSemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.LinearFnApproxPolicy(
                ft_op=ft_op, action_space=env.action_space
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
                ft_op=ft_op,
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
                ft_op=ft_op,
                action_space=env.action_space,
                options_length_range=delay_reward.range(),
            ),
            base_seed=base_seed,
        )
    elif policy_type == "rrd":
        # RRD requires delayed rewards to learn reward redistribution
        if delay_reward is None:
            raise ValueError(
                "RRD algorithm requires `delay_reward` to be configured. "
                "Set delay_config in ProblemSpec."
            )

        # Extract RRD hyperparameters from algo_config
        config = algo_config or {}

        # Use custom feats_spec for RRD reward network if provided
        # Policy uses the standard ft_op, reward network uses algorithm-specific ft_op
        reward_ft_op = transform.transform_pipeline(env=env, specs=config["feats_spec"])

        return algorithms.RRDSemigradientSARSAFnApprox(
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.LinearFnApproxPolicy(
                ft_op=ft_op, action_space=env.action_space
            ),
            ft_op=reward_ft_op,
            K=config.get("K", 64),
            M=config.get("M", 1),
            reward_network_lr=config.get("reward_network_lr", 0.001),
            reward_update_freq=config.get("reward_update_freq", 10),
            trajectory_buffer_capacity=config.get("trajectory_buffer_capacity", 10000),
            batch_size=config.get("batch_size", 256),
            base_seed=base_seed,
            verbose=True,
        )
    elif policy_type == "hc-decomposition":
        # HC-Decomposition requires delayed rewards for head-critic split
        if delay_reward is None:
            raise ValueError(
                "HC-Decomposition algorithm requires `delay_reward` to be configured. "
                "Set delay_config in ProblemSpec."
            )

        # Extract HC-Decomposition hyperparameters from algo_config
        config = algo_config or {}

        # Use custom feats_spec for HC decomposition if provided
        hc_ft_op = ft_op
        if "feats_spec" in config:
            hc_ft_op = transform.transform_pipeline(env=env, specs=config["feats_spec"])

        # Get separate learning rates for head and critic components
        # Use the main lr as fallback for both
        lr_head_value = config.get("lr_head", config.get("initial_lr", 0.01))
        lr_critic_value = config.get("lr_critic", config.get("initial_lr", 0.01))

        # Create separate LR schedules (both use constant for now)
        lr_head = learning_rate(name="constant", args={"initial_lr": lr_head_value})
        lr_critic = learning_rate(name="constant", args={"initial_lr": lr_critic_value})

        history_window = config.get("history_window", 10)
        # Alpha parameter for coupling strength (default 1.0 for full coupling)
        alpha = config.get("alpha", 1.0)

        return algorithms.HCDecompositionSemigradientSARSAFnApprox(
            lr_head=lr_head,
            lr_critic=lr_critic,
            gamma=gamma,
            epsilon=epsilon,
            policy=algorithms.HCDecompositionLinearFnApproxPolicy(
                ft_op=hc_ft_op,
                action_space=env.action_space,
                history_window=history_window,
                seed=base_seed,
            ),
            base_seed=base_seed,
            verbose=True,
            alpha=alpha,
        )

    raise ValueError(f"Unknown policy_type: {policy_type}")


def export_model_snapshot(
    weights: np.ndarray, snapshot: int, max_snapshot: int, model_dir: str
):
    ndigits = len(str(max_snapshot)) + 1
    logger.save_model(
        weights,
        name=f"weights_{snapshot:0{ndigits}}",
        model_dir=os.path.join(model_dir, "saved_model"),
    )
