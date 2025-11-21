from typing import Any, Optional, Sequence

import numpy as np
from gymnasium import spaces

from drmdp import tiles, transform


def test_funcft():
    input_space = space(obs_space=spaces.Discrete(2), act_space=spaces.Discrete(2))
    ftop = transform.FuncFT(
        input_space=input_space,
        transform_fn=lambda ex: example(ex.observation * 2, ex.action),
        output_space=input_space,
    )
    assert ftop.output_space == input_space

    inputs = example(obs=1, act=1)
    expected = example(obs=2, act=1)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=1, act=1),
        example(obs=0, act=2),
    ]
    expected = [
        example(obs=2, act=1),
        example(obs=0, act=2),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_scaleobservationft():
    input_space = space(
        obs_space=spaces.Box(arr([0, -10]), arr([10, 0])), act_space=spaces.Discrete(2)
    )
    ftop = transform.ScaleObservationFT(
        input_space=input_space,
    )
    assert ftop.input_space == input_space
    assert ftop.output_space == space(
        obs_space=spaces.Box(arr([0.0, 0.0]), arr([1.0, 1.0])),
        act_space=spaces.Discrete(2),
    )

    inputs = example(obs=arr([4, -8]), act=1)
    expected = example(obs=arr([0.4, 0.2]), act=1)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=arr([4, -8]), act=1),
        example(obs=arr([4, -2]), act=0),
    ]
    expected = [
        example(obs=arr([0.4, 0.2]), act=1),
        example(obs=arr([0.4, 0.8]), act=0),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_tileobservationactionft():
    input_space = space(
        obs_space=spaces.Box(arr([0, 0]), arr([1, 1])), act_space=spaces.Discrete(2)
    )
    iht = tiles.IHT(64)

    def tiles_fn(ex: transform.Example) -> transform.Example:
        output = np.zeros(64, dtype=np.int64)
        indices = tiles.tileswrap(
            iht,
            numtilings=8,
            floats=ex.observation * 2,  # type: ignore
            wrapwidths=[2, 2],
            ints=[ex.action],
        )
        output[indices] = 1
        return example(output, ex.action)

    ftop = transform.TileObservationActionFT(input_space=input_space, tiling_dim=2)
    assert ftop.input_space == input_space
    assert ftop.tiling_dim == 2
    assert ftop.num_tilings == 8
    assert ftop.wrapwidths == [2, 2]
    assert ftop.max_size == 64
    assert ftop.hash_dim is None
    assert ftop.output_space == space(
        obs_space=spaces.Box(arr([0] * 64), arr([1] * 64), dtype=np.int64),
        act_space=spaces.Discrete(2),
    )

    inputs = example(obs=np.array([0.4, 0.2]), act=1)
    expected = tiles_fn(inputs)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=arr([0.4, 0.2]), act=1),
        example(obs=arr([0.5, 0.5]), act=0),
    ]
    expected = [
        tiles_fn(inputs[0]),
        tiles_fn(inputs[1]),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_actionslicetileobservationactionft():
    input_space = space(
        obs_space=spaces.Box(arr([0, 0]), arr([1, 1])), act_space=spaces.Discrete(2)
    )
    ihts = {action: tiles.IHT(32) for action in range(2)}

    def tiles_fn(ex: transform.Example) -> transform.Example:
        output = np.zeros(32, dtype=np.int64)
        indices = tiles.tileswrap(
            ihts[ex.action],
            numtilings=8,
            floats=ex.observation * 2,  # type: ignore
            wrapwidths=[2, 2],
        )
        output[indices] = 1
        return example(output, ex.action)

    ftop = transform.ActionSliceTileObservationActionFT(
        input_space=input_space, tiling_dim=2
    )
    assert ftop.input_space == input_space
    assert ftop.tiling_dim == 2
    assert ftop.num_tilings == 8
    assert ftop.wrapwidths == [2, 2]
    assert ftop.max_size == 32
    assert ftop.hash_dim is None
    assert ftop.output_space == space(
        obs_space=spaces.Box(arr([0] * 32), arr([1] * 32), dtype=np.int64),
        act_space=spaces.Discrete(2),
    )

    inputs = example(obs=np.array([0.4, 0.2]), act=1)
    expected = tiles_fn(inputs)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=np.array([0.4, 0.2]), act=1),
        example(obs=np.array([0.5, 0.5]), act=0),
    ]
    expected = [
        tiles_fn(inputs[0]),
        tiles_fn(inputs[1]),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_actionsegmentedobservationft():
    input_space = space(
        obs_space=spaces.Box(arr([0, -10]), arr([10, 0])), act_space=spaces.Discrete(2)
    )
    ftop = transform.ActionSegmentedObservationFT(input_space=input_space, flat=False)
    assert ftop.output_space == space(
        obs_space=spaces.Box(
            np.stack([getattr(input_space.observation_space, "low")] * 2),
            np.stack([getattr(input_space.observation_space, "high")] * 2),
        ),
        act_space=spaces.Discrete(2),
    )

    inputs = example(obs=arr([0.4, -0.2]), act=1)
    expected = example(obs=arr([[0.0, 0.0], [0.4, -0.2]]), act=1)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=arr([0.4, -0.2]), act=1),
        example(obs=arr([0.5, -0.5]), act=0),
    ]
    expected = [
        example(obs=arr([[0.0, 0.0], [0.4, -0.2]]), act=1),
        example(obs=arr([[0.5, -0.5], [0.0, 0.0]]), act=0),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_actionsegmentedobservationft_with_flat():
    input_space = space(
        obs_space=spaces.Box(arr([0, -10]), arr([10, 0])), act_space=spaces.Discrete(2)
    )
    ftop = transform.ActionSegmentedObservationFT(input_space=input_space)
    assert ftop.output_space == space(
        obs_space=spaces.Box(
            np.stack([getattr(input_space.observation_space, "low")] * 2).flatten(),
            np.stack([getattr(input_space.observation_space, "high")] * 2).flatten(),
        ),
        act_space=spaces.Discrete(2),
    )

    inputs = example(obs=arr([0.4, -0.2]), act=1)
    expected = example(obs=arr([0.0, 0.0, 0.4, -0.2]), act=1)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=arr([0.4, -0.2]), act=1),
        example(obs=arr([0.5, -0.5]), act=0),
    ]
    expected = [
        example(obs=arr([0.0, 0.0, 0.4, -0.2]), act=1),
        example(obs=arr([0.5, -0.5, 0.0, 0.0]), act=0),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_dropobservationdimsft():
    input_space = space(
        obs_space=spaces.Box(arr([0, -10]), arr([10, 0])), act_space=spaces.Discrete(2)
    )
    ftop = transform.DropObservationDimsFT(input_space=input_space, axis_dims={0: [1]})
    assert ftop.output_space == space(
        obs_space=spaces.Box(
            arr([0]), arr([10]), dtype=input_space.observation_space.dtype
        ),
        act_space=spaces.Discrete(2),
    )

    inputs = example(obs=arr([0.4, -0.2]), act=1)
    expected = example(obs=arr([0.4]), act=1)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [example(obs=arr([0.4, -0.2]), act=1), example(obs=arr([0.5, 0.5]), act=0)]
    expected = [
        example(obs=arr([0.4]), act=1),
        example(obs=arr([0.5]), act=0),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_dropobservationdimsft_with_2dspace():
    input_space = space(
        obs_space=spaces.Box(arr([[0, -10], [0, -10]]), arr([[10, 0], [10, 0]])),
        act_space=spaces.Discrete(2),
    )
    ftop = transform.DropObservationDimsFT(input_space=input_space, axis_dims={1: [1]})
    assert ftop.output_space == space(
        obs_space=spaces.Box(
            arr([[0], [0]]),
            arr([[10], [10]]),
            dtype=input_space.observation_space.dtype,
        ),
        act_space=spaces.Discrete(2),
    )

    inputs = example(obs=arr([[0.4, -0.2], [0.3, -0.1]]), act=1)
    expected = example(obs=arr([[0.4], [0.3]]), act=1)
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=arr([[0.4, -0.2], [0.3, -0.1]]), act=1),
        example(obs=arr([[0.5, -0.5], [0.7, -0.2]]), act=0),
    ]
    expected = [
        example(obs=arr([[0.4], [0.3]]), act=1),
        example(obs=arr([[0.5], [0.7]]), act=0),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_oheactionft():
    input_space = space(
        obs_space=spaces.Box(arr([0, -10]), arr([10, 0])), act_space=spaces.Discrete(2)
    )
    ftop = transform.OHEActionFT(input_space=input_space)
    assert ftop.output_space == space(
        obs_space=spaces.Box(arr([0, -10]), arr([10, 0])),
        act_space=spaces.Box(arr([0, 0]), arr([1, 1]), dtype=np.int64),
    )

    inputs = example(obs=arr([0.4, 0.2]), act=1)
    expected = example(obs=arr([0.4, 0.2]), act=arr([0, 1], dtype=np.int64))
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [example(obs=arr([0.4, 0.2]), act=1), example(obs=arr([0.5, 0.5]), act=0)]
    expected = [
        example(obs=arr([0.4, 0.2]), act=arr([0, 1], dtype=np.int64)),
        example(obs=arr([0.5, 0.5]), act=arr([1, 0], dtype=np.int64)),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_concatobservationactionft():
    input_space = space(
        obs_space=spaces.Box(arr([0, -10]), arr([10, 0])),
        act_space=spaces.Box(arr([4, 0]), arr([8, 100])),
    )
    ftop = transform.ConcatObservationActionFT(input_space=input_space)
    assert ftop.output_space == space(
        obs_space=spaces.Box(arr([0, -10, 4, 0]), arr([10, 0, 8, 100])),
        act_space=spaces.Box(arr([4, 0]), arr([8, 100])),
    )

    inputs = example(obs=arr([0.4, 0.2]), act=arr([5, 70]))
    expected = example(obs=arr([0.4, 0.2, 5, 70]), act=arr([5, 70]))
    output = ftop.apply(inputs)
    assert_equal(output, expected)

    inputs = [
        example(obs=arr([0.4, 0.2]), act=arr([5, 70])),
        example(obs=arr([0.5, 0.5]), act=arr([6, 99])),
    ]
    expected = [
        example(obs=arr([0.4, 0.2, 5, 70]), act=arr([5, 70])),
        example(obs=arr([0.5, 0.5, 6, 99]), act=arr([6, 99])),
    ]
    output = ftop.batch(inputs)
    assert_batch(output, expected)


def test_pipeline():
    input_space = space(
        obs_space=spaces.Box(arr([0, 0]), arr([10, 100])), act_space=spaces.Discrete(2)
    )
    output_space = space(
        obs_space=spaces.Box(arr([0, 0, 0, 0]), arr([2, 2, 2, 2])),
        act_space=spaces.Discrete(2),
    )
    # Pass through pipeline
    pipe = transform.Pipeline(input_space)
    assert pipe.input_space == input_space
    assert isinstance(pipe.ft_op, transform.FuncFT)
    assert pipe.output_space == input_space
    inputs = example(obs=arr([4, 30]), act=1)
    expected = example(obs=arr([4, 30]), act=1)
    output = pipe.apply(inputs)
    assert_equal(output, expected)
    inputs = [example(obs=arr([4, 30]), act=1), example(obs=arr([5, 25]), act=0)]
    expected = [example(obs=arr([4, 30]), act=1), example(obs=arr([5, 25]), act=0)]
    output = pipe.batch(inputs)
    assert_batch(output, expected)

    # Pass 'scale -> seg -> double'    #  pipeline
    pipe = (
        pipe.map(transform.ScaleObservationFT.builder())
        .map(transform.ActionSegmentedObservationFT.builder())
        .map(
            transform.FuncFT.builder(
                transform_fn=lambda ex: example(ex.observation * 2, ex.action),
                output_space=output_space,
            )
        )
    )
    assert pipe.input_space == input_space
    assert isinstance(pipe.ft_op, transform.FuncFT)
    assert pipe.output_space == output_space
    inputs = example(obs=arr([4, 30]), act=1)
    expected = example(obs=arr([0, 0, 0.8, 0.6]), act=1)
    output = pipe.apply(inputs)
    assert_equal(output, expected)
    inputs = [example(obs=arr([4, 30]), act=1), example(obs=arr([5, 25]), act=0)]
    expected = [
        example(obs=arr([0, 0, 0.8, 0.6]), act=1),
        example(obs=arr([1.0, 0.5, 0.0, 0.0]), act=0),
    ]
    output = pipe.batch(inputs)
    assert_batch(output, expected)


def example(obs, act) -> transform.Example:
    """
    Creates an example.
    """
    return transform.Example(observation=obs, action=act)


def space(obs_space, act_space) -> transform.ExampleSpace:
    """
    Creates an ExampleSpace.
    """
    return transform.ExampleSpace(observation_space=obs_space, action_space=act_space)


def arr(xs: Sequence[Any], dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Wraps sequence into an array.
    """
    return np.array(xs, dtype=dtype)


def assert_equal(actual: transform.Example, expected: transform.Example) -> None:
    """
    Assert obs and action.
    If either is an array, calling `np.testing.assert_equal` directly
    fails.
    """
    np.testing.assert_equal(actual.observation, expected.observation)
    np.testing.assert_equal(actual.action, expected.action)


def assert_batch(
    actuals: Sequence[transform.Example], outputs: Sequence[transform.Example]
) -> None:
    """
    Calls `assert_equal` on a sequence.
    """
    assert len(actuals) == len(outputs)
    for actual, output in zip(actuals, outputs):
        assert_equal(actual=actual, expected=output)
