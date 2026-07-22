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
