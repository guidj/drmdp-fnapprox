import collections
import hashlib
from typing import List, Tuple

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


def grid_to_strings(grid: np.ndarray) -> List[str]:
    return ["".join(GRID_CHAR[int(cell)] for cell in row) for row in grid]


def grid_max_episode_steps(size: Tuple[int, int]) -> int:
    nrows, ncols = size
    return nrows * ncols


def grid_id(size: Tuple[int, int], seed: int) -> str:
    key = f"{size[0]},{size[1]},{seed}"
    return hashlib.md5(key.encode()).hexdigest()[:6]
