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


class TestDefaultDelayConfig:
    def test_uniform_ranges(self):
        configs = controlexps.default_delay_config()
        assert [
            (config["args"]["min_delay"], config["args"]["max_delay"])
            for config in configs
        ] == [(3, 5), (9, 11), (15, 17)]

    def test_configs_are_uniform(self):
        for config in controlexps.default_delay_config():
            assert config["name"] == "uniform"

    def test_bounds_are_integers(self):
        # the delay wrapper samples with random.randint, which requires
        # integer bounds
        for config in controlexps.default_delay_config():
            assert isinstance(config["args"]["min_delay"], int)
            assert isinstance(config["args"]["max_delay"], int)

    def test_only_first_range_within_option_budget(self):
        # option specs are derived for delay configs whose minimum delay
        # is within MAX_OPTIONS_DELAY; only [3, 5] qualifies
        within_budget = [
            controlexps._delay_min(config) <= controlexps.MAX_OPTIONS_DELAY
            for config in controlexps.default_delay_config()
        ]
        assert within_budget == [True, False, False]
