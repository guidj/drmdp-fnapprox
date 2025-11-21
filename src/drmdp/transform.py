import abc
import dataclasses
import functools
from typing import (
    Callable,
    Mapping,
    Optional,
    Sequence,
)

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from drmdp import mathutils, tiles


@dataclasses.dataclass(frozen=True)
class Example:
    """
    An observation-action pair.
    """

    observation: ObsType
    action: ActType


@dataclasses.dataclass(frozen=True)
class ExampleSpace:
    """
    Observation and action spaces.
    """

    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space


class FTOp(abc.ABC):
    """
    Abstract Example transformer class.
    """

    def __init__(self, input_space: ExampleSpace):
        self.input_space = input_space
        self.vapply = np.vectorize(self.apply)

    @abc.abstractmethod
    def apply(self, example: Example) -> Example:
        """
        Transforms an example.
        """
        raise NotImplementedError

    def __call__(self, example: Example) -> Example:
        return self.apply(example)

    def batch(self, examples: Sequence[Example]) -> Sequence[Example]:
        """
        Calls `apply` for each example in the batch
        """
        return self.vapply(examples)

    @property
    @abc.abstractmethod
    def output_space(self) -> ExampleSpace:
        """
        Returns:
            Output observation and action spaces.
        """
        raise NotImplementedError

    @classmethod
    def builder(cls, **kwargs) -> Callable[[ExampleSpace], "FTOp"]:
        """
        Returns:
            A constructor for an FTOp.
        """
        if "input_space" in kwargs:
            raise ValueError("Cannot pass `input_space` as an argument")
        return functools.partial(cls, **kwargs)


class FuncFT(FTOp):
    """
    Takes an input function with
    transformation.
    """

    def __init__(
        self,
        input_space: ExampleSpace,
        transform_fn: Callable[[Example], Example],
        output_space: ExampleSpace,
    ):
        super().__init__(input_space=input_space)
        self.transform_fn = transform_fn
        self._output_space = output_space

    def apply(self, example: Example) -> Example:
        """
        Applies the constructor provided transformation
        function.
        """
        return self.transform_fn(example)

    @property
    def output_space(self):
        return self._output_space


class ScaleObservationFT(FTOp):
    """
    Scales observation dimensions [0, 1].
    """

    def __init__(self, input_space: ExampleSpace):
        super().__init__(input_space=input_space)
        if not isinstance(input_space.observation_space, gym.spaces.Box):
            raise ValueError(f"Expected Box observation_space. Got: {input_space}")

        self.input_space = input_space
        self._output_space = dataclasses.replace(
            input_space,
            observation_space=gym.spaces.Box(
                low=np.zeros_like(input_space.observation_space.high),
                high=np.ones_like(input_space.observation_space.high),
            ),
        )
        self._observation_ranges = (
            input_space.observation_space.high - input_space.observation_space.low
        )

    def apply(self, example: Example) -> Example:
        observation_scaled_01 = (
            example.observation - self.input_space.observation_space.low
        ) / (self._observation_ranges)
        return dataclasses.replace(example, observation=observation_scaled_01)

    @property
    def output_space(self):
        return self._output_space


class TileObservationActionFT(FTOp):
    """
    Tiles observation-actions.
    """

    def __init__(
        self,
        input_space: ExampleSpace,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
        hash_dim: Optional[int] = None,
    ):
        super().__init__(input_space=input_space)
        if not isinstance(input_space.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Expected Box observation_space. Got: {input_space.observation_space}"
            )
        if not isinstance(input_space.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"Expected Discrete action_space. Got: {input_space.action_space}"
            )
        self.input_space = input_space
        self.tiling_dim = tiling_dim
        self.wrapwidths = [tiling_dim] * np.size(input_space.observation_space.low)
        self.hash_dim = hash_dim
        self.tiles = tiles.Tiles(
            dims_min=input_space.observation_space.low,
            dims_max=input_space.observation_space.high,
            tiling_dim=tiling_dim,
            num_tilings=num_tilings,
        )
        # For best results,
        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions.
        self.num_tilings = num_tilings or tiles.pow2geq(
            np.size(input_space.observation_space.low) * 4
        )
        self.max_size = (
            (tiling_dim ** np.size(input_space.observation_space.low))
            * self.num_tilings
            * input_space.action_space.n
        )
        self.iht = tiles.IHT(self.max_size)
        self.hash_dim = hash_dim if hash_dim and self.max_size > hash_dim else None

        self._output_space = dataclasses.replace(
            input_space,
            observation_space=gym.spaces.Box(
                low=np.zeros(shape=self.hash_dim or self.max_size),
                high=np.ones(shape=self.hash_dim or self.max_size),
                dtype=np.int64,
            ),
        )

    def apply(self, example: Example) -> Example:
        output = np.zeros(
            shape=self.max_size, dtype=self.output_space.observation_space.dtype
        )
        indices = self._tiles(example.observation, example.action)
        output[indices] = 1
        if self.hash_dim:
            return mathutils.hashtrick(output, self.hash_dim)
        return dataclasses.replace(example, observation=output)

    @property
    def output_space(self):
        """
        Example output shape.
        """
        return self._output_space

    def _tiles(self, scaled_observation: np.ndarray, action: ActType):
        return tiles.tileswrap(
            self.iht,
            numtilings=self.num_tilings,
            floats=scaled_observation * self.tiling_dim,  # type: ignore
            wrapwidths=self.wrapwidths,
            ints=[action] if action is not None else (),
        )


class ActionSliceTileObservationActionFT(FTOp):
    """
    Tiles observation-actions.
    Uses separate tile mappings per action.
    """

    def __init__(
        self,
        input_space: ExampleSpace,
        tiling_dim: int,
        num_tilings: Optional[int] = None,
        hash_dim: Optional[int] = None,
    ):
        super().__init__(input_space=input_space)
        if not isinstance(input_space.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Expected Box observation_space. Got: {input_space.observation_space}"
            )
        if not isinstance(input_space.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"Expected Discrete action_space. Got: {input_space.action_space}"
            )
        self.input_space = input_space
        self.tiling_dim = tiling_dim
        self.wrapwidths = [tiling_dim] * np.size(input_space.observation_space.low)
        self.hash_dim = hash_dim
        self.tiles = tiles.Tiles(
            dims_min=input_space.observation_space.low,
            dims_max=input_space.observation_space.high,
            tiling_dim=tiling_dim,
            num_tilings=num_tilings,
        )
        # For best results,
        # num tilings should a power of 2
        # and at least 4 times greater than
        # the number of dimensions.
        self.num_tilings = num_tilings or tiles.pow2geq(
            np.size(input_space.observation_space.low) * 4
        )
        self.max_size = (
            tiling_dim ** np.size(input_space.observation_space.low)
        ) * self.num_tilings
        self.ihts = {
            action: tiles.IHT(self.max_size)
            for action in range(input_space.action_space.n)
        }
        self.hash_dim = hash_dim if hash_dim and self.max_size > hash_dim else None

        self._output_space = dataclasses.replace(
            input_space,
            observation_space=gym.spaces.Box(
                low=np.zeros(shape=self.hash_dim or self.max_size),
                high=np.ones(shape=self.hash_dim or self.max_size),
                dtype=np.int64,
            ),
        )

    def apply(self, example: Example) -> Example:
        output = np.zeros(shape=self.max_size)
        indices = self._tiles(example.observation, example.action)
        output[indices] = 1
        if self.hash_dim:
            return mathutils.hashtrick(output, self.hash_dim)
        return dataclasses.replace(example, observation=output)

    @property
    def output_space(self):
        """
        Example output shape.
        """
        return self._output_space

    def _tiles(self, scaled_observation: np.ndarray, action: ActType):
        return tiles.tileswrap(
            self.ihts[action],
            numtilings=self.num_tilings,
            floats=scaled_observation * self.tiling_dim,  # type: ignore
            wrapwidths=self.wrapwidths,
        )


class ActionSegmentedObservationFT(FTOp):
    """
    Creates a vector output where the observation
    is positioned according to the action taken.
    """

    def __init__(self, input_space: ExampleSpace, flat: bool = True):
        super().__init__(input_space=input_space)
        if not isinstance(input_space.observation_space, gym.spaces.Box):
            raise ValueError(f"Expected Box observation_space. Got: {input_space}")
        if not isinstance(input_space.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"Expected Discrete action_space. Got: {input_space.action_space}"
            )

        self.input_space = input_space
        self.flat = flat
        obs_low = np.stack(
            [input_space.observation_space.low] * input_space.action_space.n
        )
        obs_high = np.stack(
            [input_space.observation_space.high] * input_space.action_space.n
        )
        self._output_space = dataclasses.replace(
            input_space,
            observation_space=gym.spaces.Box(
                low=obs_low.flatten() if flat else obs_low,
                high=obs_high.flatten() if flat else obs_high,
                dtype=input_space.observation_space.dtype,
            ),
        )

    def apply(self, example: Example) -> Example:
        output = np.zeros(
            shape=(self.input_space.action_space.n,)
            + self.input_space.observation_space.shape
        )
        output[example.action, :] = example.observation
        return dataclasses.replace(
            example, observation=output.flatten() if self.flat else output
        )

    @property
    def output_space(self):
        return self._output_space


class DropObservationDimsFT(FTOp):
    """
    Drops certain dimensions in a specified axis
    in the observation space.
    """

    def __init__(
        self, input_space: ExampleSpace, axis_dims: Mapping[int, Sequence[int]]
    ):
        super().__init__(input_space=input_space)
        if not isinstance(input_space.observation_space, gym.spaces.Box):
            raise ValueError(f"Expected Box observation_space. Got: {input_space}")

        self.input_space = input_space
        self.axis_dims = axis_dims

        obs_low = input_space.observation_space.low
        obs_high = input_space.observation_space.high
        for axis, drop_dims in axis_dims.items():
            obs_low = np.delete(obs_low, drop_dims, axis=axis)
            obs_high = np.delete(obs_high, drop_dims, axis=axis)
        self._output_space = dataclasses.replace(
            input_space,
            observation_space=gym.spaces.Box(
                low=obs_low,
                high=obs_high,
                dtype=input_space.observation_space.dtype,
            ),
        )

    def apply(self, example: Example) -> Example:
        output = example.observation
        for axis, drop_dims in self.axis_dims.items():
            output = np.delete(output, drop_dims, axis=axis)
        return dataclasses.replace(example, observation=output)

    @property
    def output_space(self):
        return self._output_space


class OHEActionFT(FTOp):
    """
    One-hot encodes the action.
    """

    def __init__(self, input_space: ExampleSpace):
        super().__init__(input_space=input_space)
        if not isinstance(input_space.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"Expected Discrete action_space. Got: {input_space.action_space}"
            )

        self.input_space = input_space
        self._output_space = dataclasses.replace(
            input_space,
            action_space=gym.spaces.Box(
                low=np.zeros(input_space.action_space.n),
                high=np.ones(input_space.action_space.n),
                dtype=np.int64,
            ),
        )

    def apply(self, example: Example) -> Example:
        ohe_action = np.zeros(self.input_space.action_space.n)
        ohe_action[example.action] = 1
        return dataclasses.replace(example, action=ohe_action)

    @property
    def output_space(self):
        return self._output_space


class ConcatObservationActionFT(FTOp):
    """
    Drops certain dimensions in a specified axis
    in the observation space.
    """

    def __init__(self, input_space: ExampleSpace):
        super().__init__(input_space=input_space)
        if not isinstance(input_space.observation_space, gym.spaces.Box):
            raise ValueError(f"Expected Box observation_space. Got: {input_space}")
        if not isinstance(input_space.action_space, gym.spaces.Box):
            raise ValueError(
                f"Expected Box action_space. Got: {input_space.action_space}"
            )

        self.input_space = input_space
        self._output_space = dataclasses.replace(
            input_space,
            observation_space=gym.spaces.Box(
                low=np.concatenate(
                    [input_space.observation_space.low, input_space.action_space.low]
                ),
                high=np.concatenate(
                    [input_space.observation_space.high, input_space.action_space.high]
                ),
                dtype=np.concatenate(
                    [
                        input_space.observation_space.sample(),
                        input_space.action_space.sample(),
                    ]
                ).dtype,
            ),
        )

    def apply(self, example: Example) -> Example:
        return dataclasses.replace(
            example, observation=np.concatenate([example.observation, example.action])
        )

    @property
    def output_space(self):
        return self._output_space


class Pipeline(FTOp):
    """
    Creates a pipeline of chained transformers.
    """

    def __init__(self, input_space: ExampleSpace):
        super().__init__(input_space=input_space)
        self.input_space = input_space
        self.ft_op = FuncFT(
            input_space=input_space,
            transform_fn=lambda ex: ex,
            output_space=input_space,
        )

    def map(self, ftop_cls: Callable[[ExampleSpace], FTOp]) -> "Pipeline":
        """
        Chain ops.
        """
        next_ft_op = ftop_cls(self.output_space)
        chained_ftops = FuncFT(
            input_space=self.ft_op.input_space,
            transform_fn=compose2(next_ft_op.apply, self.ft_op.apply),
            output_space=next_ft_op.output_space,
        )
        pipeline = Pipeline(input_space=self.input_space)
        pipeline.ft_op = chained_ftops
        pipeline.vapply = np.vectorize(chained_ftops.apply)
        return pipeline

    def apply(self, example: Example) -> Example:
        """
        Apply transformation pipeline.
        """
        return self.ft_op.apply(example)

    @property
    def output_space(self):
        """
        Returns:
            Example output shape.
        """
        return self.ft_op.output_space


def compose2(f, g):
    """
    Compose two functions, (f o g).
    Returns a function: f(g(x))
    """
    return lambda *a, **kw: f(g(*a, **kw))
