"""
Utils for combinatorial problems.
"""

import math
from typing import Sequence

import numpy as np


def sequence_to_integer(space_size: int, sequence: Sequence[int]) -> int:
    """
    Uses the positional system of integers to generate a unique
    sequence of numbers given represetation integer - `index`.

    Based on https://2ality.com/2013/03/permutations.html.

    Args:
        space_size: the number of possible digits
        sequence_size: the length of the sequence of digits.
        index: the index of the unique sequence.
    """
    id_ = 0
    for idx, value_index in enumerate(reversed(sequence)):
        id_ = id_ + value_index * int(pow(space_size, idx))
    return id_


def interger_to_sequence(
    space_size: int, sequence_length: int, index: int
) -> Sequence[int]:
    """
    Uses the positional system of integers to generate a unique
    sequence of numbers given represetation integer - `index`.

    Based on https://2ality.com/2013/03/permutations.html.

    Args:
        space_size: the number of possible digits
        sequence_length: the length of the sequence of digits.
        index: the index of the unique sequence.
    """
    xs = []
    for pw in reversed(range(sequence_length)):
        if pw == 0:
            xs.append(index)
        else:
            mult = space_size**pw
            digit = math.floor(index / mult)
            xs.append(digit)
            index = index % mult
    return tuple(xs)


def hashtrick(xs, dim: int):
    if dim <= 0:
        raise ValueError("`dim` must be positive")
    # Get indices of non-zero elements directly
    idx = np.nonzero(xs)[0]

    # Use modulo operation on all indices at once
    hashed_idx = idx % dim

    # Use bincount to count occurrences of each index
    # Specify minlength to ensure output has correct size
    return np.bincount(hashed_idx, minlength=dim)
