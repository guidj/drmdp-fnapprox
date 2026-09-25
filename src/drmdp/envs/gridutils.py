import collections
import hashlib
import math
from typing import List, Sequence, Tuple

import numpy as np

CELL_OPEN = 0
CELL_START = 1
CELL_GOAL = 2
CELL_CLIFF = 3

GRID_CHAR = {CELL_OPEN: "o", CELL_START: "s", CELL_GOAL: "g", CELL_CLIFF: "x"}


def max_achievable_distance(size: Tuple[int, int], num_cliffs: int) -> int:
    """Upper bound on BFS distance achievable with barrier-based construction."""
    nrows, ncols = size
    manhattan = (nrows - 1) + (ncols - 1)
    cliffs_per_barrier = nrows - 1
    if cliffs_per_barrier == 0:
        return manhattan
    num_pairs = num_cliffs // (2 * cliffs_per_barrier)
    return manhattan + num_pairs * 2 * (nrows - 1)


def create_grid(
    size: Tuple[int, int],
    num_cliffs: int,
    seed: int = 0,
    *,
    strategy: str = "random",
    min_distance: int = 0,
    max_distance: int = 0,
) -> Tuple[np.ndarray, int, int]:
    if strategy == "random":
        return _create_grid_random(size, num_cliffs, seed)
    elif strategy == "path-blocking":
        return _create_grid_by_blocking(
            size, num_cliffs, min_distance, max_distance, seed
        )
    elif strategy == "barriers":
        return _create_grid_with_barriers(
            size, num_cliffs, min_distance, max_distance, seed
        )
    raise ValueError(f"Unknown strategy: {strategy!r}")


def _create_grid_random(
    size: Tuple[int, int], num_cliffs: int, seed: int
) -> Tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    nrows, ncols = size
    npos = nrows * ncols
    if num_cliffs > npos - 2:
        raise ValueError(f"Too many cliffs. Max: {npos - 2}")

    start = 0
    end = npos - 1
    end_rc = divmod(end, ncols)

    flat_grid = np.zeros(npos, dtype=np.int8)
    flat_grid[start] = CELL_START
    flat_grid[end] = CELL_GOAL
    grid = flat_grid.reshape(nrows, ncols)

    candidates = np.arange(1, npos - 1)
    rng.shuffle(candidates)
    for flat_idx in candidates:
        if num_cliffs <= 0:
            break
        row, col = divmod(int(flat_idx), ncols)
        grid[row, col] = CELL_CLIFF
        if end_rc not in grid_reachable_cells(grid, start):
            grid[row, col] = CELL_OPEN
            continue
        num_cliffs -= 1

    return grid, start, end


def _create_grid_by_blocking(
    size: Tuple[int, int],
    num_cliffs: int,
    min_distance: int,
    max_distance: int,
    seed: int,
) -> Tuple[np.ndarray, int, int]:
    """Bridge-and-scatter: short cliff segments plus random individual cliffs."""
    rng = np.random.default_rng(seed)
    nrows, ncols = size
    npos = nrows * ncols
    start = 0
    end = npos - 1
    end_rc = divmod(end, ncols)

    if num_cliffs > npos - 2:
        raise ValueError(f"Too many cliffs. Max: {npos - 2}")

    grid = np.zeros((nrows, ncols), dtype=np.int8)
    grid.flat[start] = CELL_START
    grid.flat[end] = CELL_GOAL

    cliffs_placed = 0
    bridge_budget = int(num_cliffs * 0.6)
    max_attempts = num_cliffs * 4

    for _ in range(max_attempts):
        if cliffs_placed >= bridge_budget:
            break
        length = int(rng.integers(2, 6))
        horizontal = bool(rng.integers(0, 2))
        if horizontal:
            row = int(rng.integers(0, nrows))
            col = int(rng.integers(0, max(1, ncols - length + 1)))
            cells = [(row, col + offset) for offset in range(length)]
        else:
            row = int(rng.integers(0, max(1, nrows - length + 1)))
            col = int(rng.integers(0, ncols))
            cells = [(row + offset, col) for offset in range(length)]

        placeable = [
            (r_cell, c_cell)
            for r_cell, c_cell in cells
            if grid[r_cell, c_cell] == CELL_OPEN
        ]
        if not placeable:
            continue

        for r_cell, c_cell in placeable:
            grid[r_cell, c_cell] = CELL_CLIFF
        if end_rc not in grid_reachable_cells(grid, start):
            for r_cell, c_cell in placeable:
                grid[r_cell, c_cell] = CELL_OPEN
            continue
        cliffs_placed += len(placeable)

    scatter_budget = num_cliffs - cliffs_placed
    if scatter_budget > 0:
        open_cells = [idx for idx in range(npos) if grid.flat[idx] == CELL_OPEN]
        rng.shuffle(open_cells)
        for flat_idx in open_cells:
            if scatter_budget <= 0:
                break
            row, col = divmod(flat_idx, ncols)
            grid[row, col] = CELL_CLIFF
            if end_rc not in grid_reachable_cells(grid, start):
                grid[row, col] = CELL_OPEN
                continue
            scatter_budget -= 1

    return grid, start, end


def _create_grid_with_barriers(
    size: Tuple[int, int],
    num_cliffs: int,
    min_distance: int,
    max_distance: int,
    seed: int,
) -> Tuple[np.ndarray, int, int]:
    """Evenly-spaced vertical barriers with alternating gaps."""
    rng = np.random.default_rng(seed)
    nrows, ncols = size
    npos = nrows * ncols
    start = 0
    end = npos - 1

    cliffs_per_barrier = nrows - 1
    num_barriers = _barriers_needed(nrows, ncols, min_distance)
    actual_cliffs = max(num_cliffs, num_barriers * cliffs_per_barrier)
    if actual_cliffs > npos - 2:
        raise ValueError(
            f"min_distance={min_distance} requires {actual_cliffs} cliffs "
            f"but a {nrows}x{ncols} grid has only {npos - 2} available cells"
        )

    grid = np.zeros((nrows, ncols), dtype=np.int8)
    grid.flat[start] = CELL_START
    grid.flat[end] = CELL_GOAL

    usable_cols = ncols - 2
    num_barriers = min(num_barriers, usable_cols)

    cliffs_placed = 0
    if num_barriers > 0:
        barrier_step = (usable_cols + 1) / (num_barriers + 1)
        for bdx in range(num_barriers):
            col = int(barrier_step * (bdx + 1)) + 1
            col = max(1, min(col, ncols - 2))
            gap_row = (nrows - 1) if bdx % 2 == 0 else 0
            for row in range(nrows):
                if row == gap_row:
                    continue
                if grid[row, col] in (CELL_START, CELL_GOAL, CELL_CLIFF):
                    continue
                grid[row, col] = CELL_CLIFF
                cliffs_placed += 1

    _fill_remaining_cliffs(grid, start, end, actual_cliffs, cliffs_placed, rng)
    return grid, start, end


def _barriers_needed(nrows: int, ncols: int, min_distance: int) -> int:
    """Minimum number of full-height vertical barriers to reach *min_distance*.

    Only barrier *pairs* (with alternating top/bottom gaps) add distance.
    Each pair adds ``2 * (nrows - 1)`` BFS steps, so the result is always
    even.
    """
    manhattan = (nrows - 1) + (ncols - 1)
    extra = max(0, min_distance - manhattan)
    per_pair = 2 * (nrows - 1) if nrows > 1 else 1
    num_pairs = math.ceil(extra / per_pair)
    return 2 * num_pairs


def _fill_remaining_cliffs(
    grid: np.ndarray,
    start: int,
    end: int,
    num_cliffs: int,
    cliffs_placed: int,
    rng: np.random.Generator,
) -> None:
    """Place leftover cliff budget on cells not on the current shortest path."""
    if cliffs_placed >= num_cliffs:
        return
    npos = grid.shape[0] * grid.shape[1]
    _, path = grid_bfs_path(grid, source=start, target=end)
    path_set = set(path)
    open_non_path = [
        idx
        for idx in range(npos)
        if grid.flat[idx] == CELL_OPEN and idx not in path_set
    ]
    rng.shuffle(open_non_path)
    remaining = num_cliffs - cliffs_placed
    for idx in open_non_path[:remaining]:
        row, col = divmod(idx, grid.shape[1])
        grid[row, col] = CELL_CLIFF


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


def grid_bfs_path(
    grid: np.ndarray, source: int, target: int
) -> Tuple[int, Tuple[int, ...]]:
    """BFS shortest path returning both distance and the cell sequence.

    Returns ``(distance, path)`` where *path* is a tuple of flat indices
    from *source* to *target* inclusive.  When *target* is unreachable the
    return value is ``(-1, ())``.
    """
    nrows, ncols = grid.shape
    source_row, source_col = divmod(source, ncols)
    target_row, target_col = divmod(target, ncols)

    if (
        grid[source_row, source_col] == CELL_CLIFF
        or grid[target_row, target_col] == CELL_CLIFF
    ):
        return -1, ()

    if source == target:
        return 0, (source,)

    visited = np.zeros((nrows, ncols), dtype=bool)
    visited[source_row, source_col] = True
    parent = np.full(nrows * ncols, -1, dtype=np.intp)
    queue = collections.deque([(source_row, source_col)])

    found = False
    while queue:
        row, col = queue.popleft()
        if row == target_row and col == target_col:
            found = True
            break
        flat = row * ncols + col
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = row + d_row, col + d_col
            if (
                0 <= next_row < nrows
                and 0 <= next_col < ncols
                and not visited[next_row, next_col]
                and grid[next_row, next_col] != CELL_CLIFF
            ):
                visited[next_row, next_col] = True
                parent[next_row * ncols + next_col] = flat
                queue.append((next_row, next_col))

    if not found:
        return -1, ()

    path: List[int] = []
    current = target
    while current != source:
        path.append(current)
        current = int(parent[current])
    path.append(source)
    path.reverse()
    return len(path) - 1, tuple(path)


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
