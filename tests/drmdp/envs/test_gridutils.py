import numpy as np
import pytest

from drmdp.envs import gridutils


class TestCreateGrid:
    def test_shape(self):
        grid, _, _ = gridutils.create_grid(size=(5, 8), num_cliffs=10, seed=0)
        assert grid.shape == (5, 8)
        assert grid.dtype == np.int8

    def test_unique_positions(self):
        grid, start, end = gridutils.create_grid(size=(10, 10), num_cliffs=30, seed=42)
        assert start != end
        assert grid.flat[start] == gridutils.CELL_START
        assert grid.flat[end] == gridutils.CELL_GOAL
        assert (grid == gridutils.CELL_CLIFF).sum() == 30

    def test_deterministic(self):
        g1, s1, e1 = gridutils.create_grid(size=(5, 5), num_cliffs=5, seed=7)
        g2, s2, e2 = gridutils.create_grid(size=(5, 5), num_cliffs=5, seed=7)
        np.testing.assert_array_equal(g1, g2)
        assert s1 == s2
        assert e1 == e2


class TestGridBfs:
    def test_connected(self):
        grid = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8)
        dist = gridutils.grid_bfs(grid, source=0, target=8)
        assert dist == 4

    def test_unreachable(self):
        grid = np.array([[1, 3, 0], [3, 3, 0], [0, 0, 2]], dtype=np.int8)
        dist = gridutils.grid_bfs(grid, source=0, target=8)
        assert dist == -1

    def test_adjacent(self):
        grid = np.array([[1, 2]], dtype=np.int8)
        dist = gridutils.grid_bfs(grid, source=0, target=1)
        assert dist == 1


class TestGridReachableCells:
    def test_fully_connected(self):
        grid = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8)
        reachable = gridutils.grid_reachable_cells(grid, start=0)
        assert len(reachable) == 9

    def test_isolated_cell(self):
        # Cell (0,2) is open but walled off by cliffs at (0,1) and (1,2)
        grid = np.array([[1, 3, 0], [0, 3, 3], [0, 0, 2]], dtype=np.int8)
        reachable = gridutils.grid_reachable_cells(grid, start=0)
        assert (0, 2) not in reachable
        assert (0, 0) in reachable
        assert (2, 2) in reachable

    def test_start_walled_off(self):
        # Start at (0,0) surrounded by cliffs — only start itself reachable
        grid = np.array([[1, 3, 0], [3, 3, 0], [0, 0, 2]], dtype=np.int8)
        reachable = gridutils.grid_reachable_cells(grid, start=0)
        assert reachable == frozenset({(0, 0)})


class TestGridToStrings:
    def test_roundtrip(self):
        grid = np.array([[0, 1, 3], [2, 0, 0]], dtype=np.int8)
        strings = gridutils.grid_to_strings(grid)
        assert strings == ["osx", "goo"]


class TestGridMaxEpisodeSteps:
    def test_scales_with_grid_size(self):
        assert gridutils.grid_max_episode_steps(size=(5, 5)) == 25
        assert gridutils.grid_max_episode_steps(size=(25, 25)) == 625
        assert gridutils.grid_max_episode_steps(size=(10, 20)) == 200


class TestGridDeadOheIndices:
    def test_no_cliffs_one_exit(self):
        grid = np.array([[0, 0, 0], [1, 0, 2]], dtype=np.int8)
        # exit at (1, 2), state_idx = 1*3 + 2 = 5, nstates = 6
        indices = gridutils.grid_dead_ohe_indices(grid, exits=[(1, 2)], nactions=4)
        assert indices == sorted([0 * 6 + 5, 1 * 6 + 5, 2 * 6 + 5, 3 * 6 + 5])
        assert len(indices) == 4

    def test_cliffs_and_exit(self):
        grid = np.array([[0, 3, 0], [1, 0, 2]], dtype=np.int8)
        # cliff at (0,1) state_idx=1, exit at (1,2) state_idx=5, nstates=6
        indices = gridutils.grid_dead_ohe_indices(grid, exits=[(1, 2)], nactions=4)
        expected_cliff = [action * 6 + 1 for action in range(4)]
        expected_exit = [action * 6 + 5 for action in range(4)]
        assert indices == sorted(expected_cliff + expected_exit)
        assert len(indices) == 8

    def test_unreachable_open_cell(self):
        # Cell (0,2) is open but unreachable (walled by cliffs)
        grid = np.array([[1, 3, 0], [0, 3, 3], [0, 0, 2]], dtype=np.int8)
        # nstates=9, exit at (2,2)=state 8, cliff at (0,1)=1,(1,1)=4,(1,2)=5
        # unreachable open cell at (0,2)=state 2
        indices = gridutils.grid_dead_ohe_indices(grid, exits=[(2, 2)], nactions=4)
        dead_flat = set()
        for idx in indices:
            dead_flat.add(idx % 9)
        assert 2 in dead_flat  # unreachable open cell
        assert 1 in dead_flat  # cliff
        assert 4 in dead_flat  # cliff
        assert 5 in dead_flat  # cliff
        assert 8 in dead_flat  # exit
        assert 0 not in dead_flat  # start — reachable

    def test_indices_in_range(self):
        grid, _, end = gridutils.create_grid(size=(4, 12), num_cliffs=12, seed=0)
        exit_pos = divmod(end, 12)
        indices = gridutils.grid_dead_ohe_indices(grid, exits=[exit_pos], nactions=4)
        nstates = 4 * 12
        assert all(0 <= idx < nstates * 4 for idx in indices)
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)


class TestGridId:
    def test_deterministic(self):
        id1 = gridutils.grid_id(size=(25, 25), seed=0)
        id2 = gridutils.grid_id(size=(25, 25), seed=0)
        assert id1 == id2
        assert len(id1) == 6

    def test_varies(self):
        id1 = gridutils.grid_id(size=(25, 25), seed=0)
        id2 = gridutils.grid_id(size=(25, 25), seed=1)
        assert id1 != id2


class TestMaxAchievableDistance:
    def test_no_cliffs(self):
        assert gridutils.max_achievable_distance((10, 10), 0) == 18

    def test_one_barrier_pair(self):
        dist = gridutils.max_achievable_distance((10, 10), 20)
        assert dist == 18 + 1 * 2 * 9

    def test_25x25(self):
        dist = gridutils.max_achievable_distance((25, 25), 125)
        assert dist < 187


class TestGridBfsPath:
    def test_connected(self):
        grid = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8)
        dist, path = gridutils.grid_bfs_path(grid, source=0, target=8)
        assert dist == 4
        assert path[0] == 0
        assert path[-1] == 8
        assert len(path) == 5

    def test_unreachable(self):
        grid = np.array([[1, 3, 0], [3, 3, 0], [0, 0, 2]], dtype=np.int8)
        dist, path = gridutils.grid_bfs_path(grid, source=0, target=8)
        assert dist == -1
        assert path == ()

    def test_same_cell(self):
        grid = np.array([[1, 0], [0, 2]], dtype=np.int8)
        dist, path = gridutils.grid_bfs_path(grid, source=0, target=0)
        assert dist == 0
        assert path == (0,)

    def test_path_cells_are_neighbors(self):
        grid = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]], dtype=np.int8)
        ncols = 4
        dist, path = gridutils.grid_bfs_path(grid, source=0, target=11)
        assert dist == len(path) - 1
        for idx in range(len(path) - 1):
            r1, c1 = divmod(path[idx], ncols)
            r2, c2 = divmod(path[idx + 1], ncols)
            assert abs(r1 - r2) + abs(c1 - c2) == 1


class TestCreateGridRandom:
    def test_shape_and_dtype(self):
        grid, _, _ = gridutils.create_grid(
            size=(5, 8), num_cliffs=10, strategy="random"
        )
        assert grid.shape == (5, 8)
        assert grid.dtype == np.int8

    def test_cliff_count(self):
        grid, _, _ = gridutils.create_grid(
            size=(10, 10), num_cliffs=30, seed=42, strategy="random"
        )
        assert (grid == gridutils.CELL_CLIFF).sum() == 30

    def test_deterministic(self):
        g1, s1, e1 = gridutils.create_grid(size=(5, 5), num_cliffs=5, seed=7)
        g2, s2, e2 = gridutils.create_grid(size=(5, 5), num_cliffs=5, seed=7)
        np.testing.assert_array_equal(g1, g2)
        assert s1 == s2
        assert e1 == e2

    def test_start_and_end_marked(self):
        grid, start, end = gridutils.create_grid(
            size=(8, 8), num_cliffs=10, seed=0, strategy="random"
        )
        assert grid.flat[start] == gridutils.CELL_START
        assert grid.flat[end] == gridutils.CELL_GOAL

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_goal_reachable(self, seed):
        grid, start, end = gridutils.create_grid(
            size=(10, 10), num_cliffs=20, seed=seed, strategy="random"
        )
        assert gridutils.grid_bfs(grid, source=start, target=end) > 0

    def test_high_cliff_ratio_still_reachable(self):
        grid, start, end = gridutils.create_grid(
            size=(10, 10), num_cliffs=60, seed=42, strategy="random"
        )
        assert gridutils.grid_bfs(grid, source=start, target=end) > 0
        assert (grid == gridutils.CELL_CLIFF).sum() <= 60


class TestCreateGridByBlocking:
    def test_shape_and_dtype(self):
        grid, _, _ = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="path-blocking",
            min_distance=15,
            max_distance=50,
        )
        assert grid.shape == (10, 10)
        assert grid.dtype == np.int8

    def test_cliff_count(self):
        grid, _, _ = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="path-blocking",
            min_distance=0,
            max_distance=200,
        )
        assert (grid == gridutils.CELL_CLIFF).sum() == 20

    def test_reachable(self):
        grid, start, end = gridutils.create_grid(
            size=(15, 15),
            num_cliffs=45,
            seed=42,
            strategy="path-blocking",
            min_distance=0,
            max_distance=200,
        )
        dist = gridutils.grid_bfs(grid, source=start, target=end)
        assert dist > 0

    def test_start_and_end_marked(self):
        grid, start, end = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="path-blocking",
            min_distance=0,
            max_distance=200,
        )
        assert grid.flat[start] == gridutils.CELL_START
        assert grid.flat[end] == gridutils.CELL_GOAL
        assert start == 0
        assert end == 99

    def test_deterministic(self):
        kwargs = dict(
            size=(10, 10),
            num_cliffs=20,
            seed=42,
            strategy="path-blocking",
            min_distance=0,
            max_distance=200,
        )
        g1, s1, e1 = gridutils.create_grid(**kwargs)
        g2, s2, e2 = gridutils.create_grid(**kwargs)
        np.testing.assert_array_equal(g1, g2)

    def test_different_seeds(self):
        common = dict(
            size=(10, 10),
            num_cliffs=20,
            strategy="path-blocking",
            min_distance=0,
            max_distance=200,
        )
        g1, _, _ = gridutils.create_grid(seed=0, **common)
        g2, _, _ = gridutils.create_grid(seed=99, **common)
        assert not np.array_equal(g1, g2)

    def test_distance_increases_with_cliffs(self):
        few, start, end = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=5,
            seed=42,
            strategy="path-blocking",
            min_distance=0,
            max_distance=200,
        )
        many, _, _ = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=30,
            seed=42,
            strategy="path-blocking",
            min_distance=0,
            max_distance=200,
        )
        dist_few = gridutils.grid_bfs(few, source=start, target=end)
        dist_many = gridutils.grid_bfs(many, source=start, target=end)
        assert dist_many >= dist_few


class TestCreateGridWithBarriers:
    def test_shape_and_dtype(self):
        grid, _, _ = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="barriers",
            min_distance=15,
            max_distance=50,
        )
        assert grid.shape == (10, 10)
        assert grid.dtype == np.int8

    def test_cliff_count_at_least_requested(self):
        grid, _, _ = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="barriers",
            min_distance=15,
            max_distance=50,
        )
        assert (grid == gridutils.CELL_CLIFF).sum() >= 20

    def test_distance_in_range(self):
        grid, start, end = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="barriers",
            min_distance=15,
            max_distance=50,
        )
        dist = gridutils.grid_bfs(grid, source=start, target=end)
        assert 15 <= dist < 50

    def test_start_and_end_at_corners(self):
        grid, start, end = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="barriers",
            min_distance=15,
            max_distance=50,
        )
        assert start == 0
        assert end == 99

    def test_deterministic(self):
        kwargs = dict(
            size=(10, 10),
            num_cliffs=20,
            seed=42,
            strategy="barriers",
            min_distance=15,
            max_distance=50,
        )
        g1, _, _ = gridutils.create_grid(**kwargs)
        g2, _, _ = gridutils.create_grid(**kwargs)
        np.testing.assert_array_equal(g1, g2)

    def test_barriers_visible(self):
        grid, _, _ = gridutils.create_grid(
            size=(10, 10),
            num_cliffs=20,
            seed=0,
            strategy="barriers",
            min_distance=15,
            max_distance=50,
        )
        cliff_cols = set()
        for row in range(10):
            for col in range(10):
                if grid[row, col] == gridutils.CELL_CLIFF:
                    cliff_cols.add(col)
        assert len(cliff_cols) >= 1

    def test_exceeds_cliff_floor(self):
        grid, start, end = gridutils.create_grid(
            size=(25, 25),
            num_cliffs=125,
            seed=127,
            strategy="barriers",
            min_distance=187,
            max_distance=250,
        )
        dist = gridutils.grid_bfs(grid, source=start, target=end)
        actual_cliffs = int((grid == gridutils.CELL_CLIFF).sum())
        assert actual_cliffs >= 125
        assert dist >= 187
