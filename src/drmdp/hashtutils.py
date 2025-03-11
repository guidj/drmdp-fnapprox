import numpy as np


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
