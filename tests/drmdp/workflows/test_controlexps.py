from drmdp.workflows import controlexps


class TestGridDeadOheIndices:
    def test_mines_gw_grid(self):
        indices = controlexps._grid_dead_ohe_indices(controlexps.MINES_GW_GRID)
        nrows = len(controlexps.MINES_GW_GRID)
        ncols = len(controlexps.MINES_GW_GRID[0])
        nstates = nrows * ncols
        assert ncols == 12
        cliff_cells = [
            0 * 12 + 4,
            1 * 12 + 9,
            2 * 12 + 1,
            2 * 12 + 7,
            3 * 12 + 5,
            4 * 12 + 2,
            4 * 12 + 9,
        ] + [5 * 12 + col for col in range(1, 11)]
        goal_cells = [5 * 12 + 11]
        dead_cells = cliff_cells + goal_cells
        for state_idx in dead_cells:
            for action in range(4):
                assert action * nstates + state_idx in indices
        assert len(indices) == len(dead_cells) * 4

    def test_simple_grid(self):
        grid = ["sox", "xog"]
        indices = controlexps._grid_dead_ohe_indices(grid)
        nstates = 2 * 3
        cliff_states = [2, 3]
        goal_states = [5]
        dead_states = cliff_states + goal_states
        expected = sorted(
            action * nstates + state for state in dead_states for action in range(4)
        )
        assert sorted(indices) == expected

    def test_no_cliffs(self):
        grid = ["sog"]
        indices = controlexps._grid_dead_ohe_indices(grid)
        nstates = 1 * 3
        goal_states = [2]
        expected = sorted(
            action * nstates + state for state in goal_states for action in range(4)
        )
        assert sorted(indices) == expected
