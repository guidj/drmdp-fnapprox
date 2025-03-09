import numpy as np


def hashtrick(xs, dim: int):
    ys = np.zeros(dim, dtype=np.int32)
    (idx,) = np.where(xs == 1)
    for i in idx:
        ys[i % dim] += 1
    return ys
