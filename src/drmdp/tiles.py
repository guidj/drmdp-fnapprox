import itertools
import math
from typing import List, Optional, Sequence, Union

import numpy as np

basehash = hash


class Tiles:
    def __init__(
        self,
        dims_min: np.ndarray,
        dims_max: np.ndarray,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
    ):
        assert isinstance(dims_min, np.ndarray)
        assert isinstance(dims_max, np.ndarray)
        self.dims_max = dims_max
        self.dims_min = dims_min
        self.tiling_dim = tiling_dim
        self.wrapwidths = [tiling_dim] * np.size(dims_min)

        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions
        self.num_tilings = num_tilings or pow2geq(np.size(dims_min) * 4)
        self.max_size = (tiling_dim ** np.size(dims_min)) * self.num_tilings
        print("Num tilings", self.num_tilings, "\n", "Flat dim:", self.max_size)
        self.iht = IHT(self.max_size)

    def __call__(self, xs):
        xs_scaled_01 = (xs - self.dims_min) / (self.dims_max - self.dims_min)
        repr_ = np.zeros(shape=self.max_size)
        idx = tileswrap(
            self.iht, self.num_tilings, xs_scaled_01 * self.tiling_dim, self.wrapwidths
        )
        repr_[idx] = 1
        return repr_


class IHT:
    """
    Structure to handle collisions
    Source: http://incompleteideas.net/tiles/tiles3.html

    """

    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return (
            "Collision table:"
            + " size:"
            + str(self.size)
            + " overfullCount:"
            + str(self.overfullCount)
            + " dictionary:"
            + str(len(self.dictionary))
            + " items"
        )

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            # TODO: Fail
            if self.overfullCount == 0:
                print("IHT full, starting to allow collisions")
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count


def hashcoords(coordinates, m, readonly=False):
    if isinstance(m, IHT):
        return m.getindex(tuple(coordinates), readonly)
    if isinstance(m, int):
        return basehash(tuple(coordinates)) % m
    if m is None:
        return coordinates


def tiles(
    ihtORsize: Union[IHT, int, None],
    numtilings: int,
    floats: List[float],
    ints: List[int] = [],
    readonly: bool = False,
) -> List[int]:
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [math.floor(f * numtilings) for f in floats]
    tiles_ = []
    for tiling in range(numtilings):
        tiling_x2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tiling_x2
        coords.extend(ints)
        tiles_.append(hashcoords(coords, ihtORsize, readonly))
    return tiles_


def tileswrap(
    ihtORsize: Union[IHT, int, None],
    numtilings: int,
    floats: Sequence[float],
    wrapwidths: Sequence[int],
    ints: Sequence[int] = [],
    readonly: bool = False,
) -> Sequence[int]:
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [math.floor(f * numtilings) for f in floats]
    tiles_ = []
    for tiling in range(numtilings):
        tiling_x2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in itertools.zip_longest(qfloats, wrapwidths):
            c = (q + b % numtilings) // numtilings
            coords.append(c % width if width else c)
            b += tiling_x2
        coords.extend(ints)
        tiles_.append(hashcoords(coords, ihtORsize, readonly))
    return tiles_


def pow2geq(lb: int) -> int:
    exp = 1
    rs = 1
    while True:
        rs = 2**exp
        if rs >= lb:
            break
        exp += 1
    return rs
