import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from rlplg.core import InitState
from rlplg.environments import gridworld, iceworld, redgreen

from drmdp.envs import gridutils, wrappers

DEFAULT_GW_GRID = ["oooooooooooo", "oooooooooooo", "oooooooooooo", "sxxxxxxxxxxg"]
DEFAULT_RG_CURE = ["red", "green", "red", "green", "wait", "green"]
DEFAULT_ICE_MAP = "4x4"
DEFAULT_MC_MAX_EPISODE_STEPS = 10_000
DEFAULT_ACROBOT_MAX_EPISODE_STEPS = 500


class RandomStartGridWorld(gridworld.GridWorld):
    """GridWorld whose initial state is sampled per episode.

    Candidates are non-cliff, non-exit cells that can reach an exit,
    with BFS distance to the nearest exit at or above the median over
    those cells. `reset` draws one uniformly and the drawn cell is the
    episode's initial agent position. Cliff falls send the agent to
    the start given at construction, so the transition table — the
    teleport target included — is the same for every episode. Grids
    where no cell reaches an exit fall back to the textual start.
    """

    def __init__(
        self,
        size: tuple[int, int],
        cliffs: Sequence[tuple[int, int]],
        exits: Sequence[tuple[int, int]],
        start: tuple[int, int],
    ):
        super().__init__(size, cliffs, exits, start)
        # `GridWorld` name-mangles its state-id function; rebuild one
        # for observations created outside the parent class.
        self._state_id_fn = gridworld.create_obs_state_id_fn(
            states=gridworld.states_mapping(size=size, cliffs=tuple(self._cliffs))
        )
        self._start_candidates = _random_start_candidates(
            size=size, cliffs=tuple(self._cliffs), exits=tuple(self._exits)
        ) or (self._start,)

    @property
    def start_candidates(self) -> tuple[tuple[int, int], ...]:
        """Cells `reset` samples from, sorted by position."""
        return self._start_candidates

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> InitState:
        """Starts a new sequence from a uniformly sampled start state."""
        del options
        self.seed(seed)
        episode_start = self._sample_start()
        observation = dict(
            gridworld.create_observation(
                size=self._size,
                start=self._start,
                agent=episode_start,
                cliffs=tuple(self._cliffs),
                exits=tuple(self._exits),
                get_state_id=self._state_id_fn,
            )
        )
        # `create_observation` derives the id from the start cell;
        # the agent begins at the sampled position
        observation[gridworld.OBS_KEY_ID] = self._state_id_fn(episode_start)
        self._observation = observation
        return copy.copy(self._observation), {}

    def _sample_start(self) -> tuple[int, int]:
        idx = int(self._rng.integers(low=0, high=len(self._start_candidates)))
        return self._start_candidates[idx]


class GridWorldObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gridworld.GridWorld):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            high=np.array(
                [
                    env.observation_space["agent"][0].n,
                    env.observation_space["agent"][1].n,
                ]
            ),
            low=np.zeros(shape=2),
            dtype=np.int64,
        )

        self._grid_env = env
        self.states_mapping = gridworld.states_mapping(
            size=env._size, cliffs=tuple(env._cliffs)
        )
        self._get_state_id: Callable[[tuple[int, int]], int] = (
            gridworld.create_obs_state_id_fn(states=self.states_mapping)
        )
        self.num_states = len(env.transition)

    @property
    def transition(self) -> gridworld.MutableEnvTransition:
        """Transition table of the wrapped env.

        Cliff-teleport targets are the start given at construction;
        the table is the same for every episode.
        """
        return self._grid_env.transition

    def observation(self, observation: ObsType):
        return np.array(observation["agent"], dtype=np.int64)

    def get_state_id(self, pos: np.ndarray) -> int:
        """
        Map grid pos to an int.
        """
        return self._get_state_id((pos[0], pos[1]))


class IceworldObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            high=np.array(
                [
                    env.observation_space["agent"][0].n,
                    env.observation_space["agent"][1].n,
                ]
            ),
            low=np.zeros(shape=2),
            dtype=np.int64,
        )

    def observation(self, observation: ObsType):
        return np.array(observation["agent"], dtype=np.int64)


class RedgreenObsAsVectorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            high=np.array([env.observation_space["pos"].n]),
            low=np.zeros(shape=1),
            dtype=np.int64,
        )

    def observation(self, observation: ObsType):
        return np.array([observation["pos"]], dtype=np.int64)


def make(env_name: str, wrapper: str | None = None, **kwargs) -> gym.Env:
    if env_name.startswith("GridWorld-"):
        grid = kwargs.get("grid", DEFAULT_GW_GRID)
        size, cliffs, exits, start = gridworld.parse_grid_from_text(grid)
        env = GridWorldObsAsVectorWrapper(
            RandomStartGridWorld(size, cliffs, exits, start)
        )
        env = episode_steps_limit(env, kwargs.get("max_episode_steps", None))
    elif env_name == "RedGreen-v0":
        cure = kwargs.get("cure", DEFAULT_RG_CURE)
        env = RedgreenObsAsVectorWrapper(redgreen.RedGreenSeq(cure))
        env = episode_steps_limit(env, kwargs.get("max_episode_steps", None))
    elif env_name == "IceWorld-v0":
        map_name = kwargs.get("map_name", DEFAULT_ICE_MAP)
        map_ = iceworld.MAPS[map_name]
        size, lakes, goals, start = iceworld.parse_map_from_text(map_)
        env = IceworldObsAsVectorWrapper(
            iceworld.IceWorld(size, lakes=lakes, goals=goals, start=start)
        )
        env = episode_steps_limit(env, kwargs.get("max_episode_steps", None))
    elif env_name == "MountainCar-v0":
        max_episode_steps = kwargs.get(
            "max_episode_steps", DEFAULT_MC_MAX_EPISODE_STEPS
        )
        env = gym.make("MountainCar-v0", max_episode_steps=max_episode_steps)
    elif env_name == "Acrobot-v1":
        max_episode_steps = kwargs.get(
            "max_episode_steps", DEFAULT_ACROBOT_MAX_EPISODE_STEPS
        )
        env = gym.make("Acrobot-v1", max_episode_steps=max_episode_steps)
    else:
        raise ValueError(f"Environment `{env_name}` unknown")
    return wrappers.wrap(env, wrapper=wrapper, **kwargs)


def episode_steps_limit(env: gym.Env, max_episode_steps: int | None = None):
    """
    Applies a `TimeLimit` wrapper, if `max_episode_steps` is defined.
    """
    if max_episode_steps:
        return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def _random_start_candidates(
    size: tuple[int, int],
    cliffs: Sequence[tuple[int, int]],
    exits: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Cells eligible as random starts.

    A cell is eligible when it is not a cliff, not an exit, and can
    reach an exit: cliff cells are not states, an exit start
    terminates at the first step, and a start from which no exit is
    reachable cannot terminate the episode. Among eligible cells,
    keeps those at or above the median BFS distance to the nearest
    exit, ties at the median included.
    """
    nrows, ncols = size
    grid = np.zeros(shape=size, dtype=np.int8)
    for pos in cliffs:
        grid[pos] = gridutils.CELL_CLIFF
    for pos in exits:
        grid[pos] = gridutils.CELL_GOAL
    exit_cells = [row * ncols + col for row, col in exits]
    distances = gridutils.grid_bfs_distances(grid=grid, sources=exit_cells)
    eligible = [
        ((row, col), int(distances[row * ncols + col]))
        for row in range(nrows)
        for col in range(ncols)
        if grid[row, col] == gridutils.CELL_OPEN and distances[row * ncols + col] >= 0
    ]
    if not eligible:
        return ()
    median_distance = float(np.median([dist for _, dist in eligible]))
    return tuple(sorted(pos for pos, dist in eligible if dist >= median_distance))
