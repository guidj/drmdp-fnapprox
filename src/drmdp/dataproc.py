import copy
import uuid
from typing import Any, Mapping

import gymnasium as gym
import pandas as pd
import ray
import ray.data

MAPPERS_NAMES = {
    "identity": "FR",
    "zero-impute": "IMR",
    "least-lfa": "LEAST-LFA",
    "least-bayes-lfa": "LEAST-BAYES-LFA",
}

POLICY_TYPES = {"drop-missing": "DMR", "markovian": "PP", "options": "OP-A", "single-action-options": "OP-S"}


def collection_traj_data(env: gym.Env, steps: int):
    obs, _ = env.reset()
    step = 0
    buffer = []
    while step < steps:
        action = env.action_space.sample()
        (
            next_obs,
            rew,
            term,
            trunc,
            _,
        ) = env.step(action)
        step += 1
        buffer.append((obs, action, next_obs, rew))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    return buffer


def process_data(df_raw):
    def filter_experiment_configs(meta: Mapping[str, Any]):
        del meta
        return True

    def simplify_meta(meta):
        new_meta = dict(**meta, **meta["experiment"])
        new_meta["reward_mapper"] = MAPPERS_NAMES[
            new_meta["problem_spec"]["reward_mapper"]["name"]
        ]
        new_meta["policy_type"] = POLICY_TYPES[new_meta["problem_spec"]["policy_type"]]
        del new_meta["experiment"]
        return new_meta

    def get_method(meta: Mapping[str, Any]):
        method = "/".join([meta["policy_type"], meta["reward_mapper"]])
        if method == "DMR/FR":
            return "PP/DMR"
        return method

    df_proc = copy.deepcopy(df_raw)
    df_proc = df_proc[df_proc["meta"].apply(filter_experiment_configs)]
    df_proc["meta"] = df_proc["meta"].apply(simplify_meta)
    df_proc["method"] = df_proc["meta"].apply(get_method)
    return df_proc


def read_data(files, reader: str = "ray"):
    if reader == "ray":
        with ray.init():
            ds_metrics = ray.data.read_parquet(files)
            df_metrics = ds_metrics.to_pandas()
    elif reader == "pd":
        dfs = []
        for file in files:
            dfs.append(pd.read_parquet(file))
        df_metrics = pd.concat(dfs)
    else:
        raise ValueError(reader)

    return process_data(df_metrics)


def wide_metrics(df_metrics):
    df_raw = df_metrics.drop(["metrics"], axis=1, inplace=False)
    return df_raw.explode("returns")


def get_distinct_envs(df_data: pd.DataFrame):
    envs = {}
    for row in df_data.to_dict("records"):
        env_spec = row["meta"]["env_spec"]
        env_name = env_spec["name"]
        env_args = env_spec["args"]
        envs[env_name] = env_args
    return envs


def drop_duplicate_sets(df_data: pd.DataFrame, keys):
    col = str(uuid.uuid4())
    rows = []
    for row in df_data.to_dict("records"):
        col_set = sorted([row[key] for key in keys])
        new_row = copy.deepcopy(row)
        new_row[col] = col_set
        rows.append(new_row)
    df_raw = pd.DataFrame(rows)
    return df_raw.drop_duplicates(col).drop([col], axis=1)
