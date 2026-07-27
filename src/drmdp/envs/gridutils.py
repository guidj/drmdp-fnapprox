import collections
import hashlib
from typing import List, Sequence, Tuple

import numpy as np

CELL_OPEN = 0
CELL_START = 1
CELL_GOAL = 2
CELL_CLIFF = 3

GRID_CHAR = {CELL_OPEN: "o", CELL_START: "s", CELL_GOAL: "g", CELL_CLIFF: "x"}


def create_grid(
    size: Tuple[int, int], num_cliffs: int, seed: int = 0
) -> Tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    nrows, ncols = size
    npos = nrows * ncols
    if num_cliffs > npos - 2:
        raise ValueError(f"Too many cliffs. Max: {npos - 2}")

    positions = rng.choice(npos, size=2 + num_cliffs, replace=False)
    start, end = int(positions[0]), int(positions[1])
    cliffs = positions[2:]

    flat_grid = np.zeros(npos, dtype=np.int8)
    flat_grid[start] = CELL_START
    flat_grid[end] = CELL_GOAL
    flat_grid[cliffs] = CELL_CLIFF
    return flat_grid.reshape(nrows, ncols), start, end


def grid_bfs(grid: np.ndarray, source: int, target: int) -> int:
    nrows, ncols = grid.shape
    source_row, source_col = divmod(source, ncols)
    target_row, target_col = divmod(target, ncols)

    if (
        grid[source_row, source_col] == CELL_CLIFF
        or grid[target_row, target_col] == CELL_CLIFF
    ):
        return -1

    visited = np.zeros((nrows, ncols), dtype=bool)
    visited[source_row, source_col] = True
    queue = collections.deque([(source_row, source_col, 0)])

    while queue:
        row, col, dist = queue.popleft()
        if row == target_row and col == target_col:
            return dist
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = row + d_row, col + d_col
            if (
                0 <= next_row < nrows
                and 0 <= next_col < ncols
                and not visited[next_row, next_col]
                and grid[next_row, next_col] != CELL_CLIFF
            ):
                visited[next_row, next_col] = True
                queue.append((next_row, next_col, dist + 1))
    return -1


def grid_reachable_cells(grid: np.ndarray, start: int) -> frozenset[Tuple[int, int]]:
    """All (row, col) positions reachable from *start* via 4-directional moves.

    Cliffs are impassable; every other cell type is traversable.
    """
    nrows, ncols = grid.shape
    start_row, start_col = divmod(start, ncols)
    if grid[start_row, start_col] == CELL_CLIFF:
        return frozenset()

    visited: set[Tuple[int, int]] = {(start_row, start_col)}
    queue = collections.deque([(start_row, start_col)])

    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = row + d_row, col + d_col
            if (
                0 <= next_row < nrows
                and 0 <= next_col < ncols
                and (next_row, next_col) not in visited
                and grid[next_row, next_col] != CELL_CLIFF
            ):
                visited.add((next_row, next_col))
                queue.append((next_row, next_col))
    return frozenset(visited)


def grid_to_strings(grid: np.ndarray) -> List[str]:
    return ["".join(GRID_CHAR[int(cell)] for cell in row) for row in grid]


def grid_max_episode_steps(size: Tuple[int, int]) -> int:
    nrows, ncols = size
    return nrows * ncols


def grid_dead_ohe_indices(
    grid: np.ndarray,
    exits: Sequence[Tuple[int, int]],
    nactions: int,
) -> List[int]:
    """Indices of OHE columns for unobservable states.

    A state is unobservable (permanently zero in the estimation matrix)
    if the agent can never occupy it as a pre-action observation:
    cliff cells, terminal/exit cells, and cells unreachable from the
    start position (e.g. open cells walled off by cliffs).
    """
    nrows, ncols = grid.shape
    nstates = nrows * ncols

    start_positions = list(zip(*np.where(grid == CELL_START)))
    if start_positions:
        start_flat = start_positions[0][0] * ncols + start_positions[0][1]
        reachable = grid_reachable_cells(grid, start_flat)
    else:
        reachable = None

    exit_set = set(exits)
    dead_states: List[int] = []
    for row in range(nrows):
        for col in range(ncols):
            if grid[row, col] == CELL_CLIFF:
                dead_states.append(row * ncols + col)
            elif (row, col) in exit_set:
                dead_states.append(row * ncols + col)
            elif reachable is not None and (row, col) not in reachable:
                dead_states.append(row * ncols + col)
    indices: List[int] = []
    for state_idx in dead_states:
        for action in range(nactions):
            indices.append(action * nstates + state_idx)
    return sorted(indices)


def grid_id(size: Tuple[int, int], seed: int) -> str:
    key = f"{size[0]},{size[1]},{seed}"
    return hashlib.md5(key.encode()).hexdigest()[:6]
