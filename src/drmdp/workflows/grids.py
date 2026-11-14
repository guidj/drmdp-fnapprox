"""
Grid assets shared by the control and reward-estimation campaigns.

The MINES layout, the dead-column derivation of the flat-grid OHE
layout, and the loader for the solvable-grid catalogue.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from drmdp.envs import gridutils

# cells: "o" open, "x" cliff, "s" start, "g" goal; the last row is a
# cliff row between the start and goal exits
MINES_GW_GRID = [
    "ooooxooooooo",
    "oooooooooxoo",
    "oxoooooxoooo",
    "oooooxoooooo",
    "ooxooooooxoo",
    "sxxxxxxxxxxg",
]


def grid_dead_ohe_indices(grid: Sequence[str], nactions: int = 4) -> list[int]:
    """
    Returns OHE indices of dead state-action columns.

    Dead states are cliffs, terminal/goal cells, and cells
    unreachable from the start.  Each dead state produces
    ``nactions`` dead columns in the flat-grid-observation-action-ft
    layout (``action * nstates + state_idx``).
    """
    char_map = {
        "o": gridutils.CELL_OPEN,
        "s": gridutils.CELL_START,
        "g": gridutils.CELL_GOAL,
        "x": gridutils.CELL_CLIFF,
    }
    grid_arr = np.array([[char_map[ch] for ch in row] for row in grid], dtype=np.int8)
    exits = list(zip(*np.where(grid_arr == gridutils.CELL_GOAL)))
    return gridutils.grid_dead_ohe_indices(grid_arr, exits=exits, nactions=nactions)


def load_solvable_grids(
    path: str,
    min_passes: int = 4,
) -> Sequence[Mapping[str, Any]]:
    """
    Loads grid specs from a solvable grids JSON file, filtering by
    minimum number of passed solvability trials.
    """
    with open(path, "r") as readable:
        entries: list[Mapping[str, Any]] = json.load(readable)
    passing_grids = [entry for entry in entries if entry["passes"] >= min_passes]
    return passing_grids
