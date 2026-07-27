import numpy as np

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
