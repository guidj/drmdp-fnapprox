import numpy as np
import pytest

from drmdp.identifiability import mrps


def _assert_valid_stochastic_matrix(transition):
    assert transition.ndim == 2
    assert transition.shape[0] == transition.shape[1]
    np.testing.assert_array_less(-1e-12, transition)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)


def _hamming_distance(a, b):
    return bin(a ^ b).count("1")


class TestSolveMrp:
    def test_absorbing_state(self):
        transition = np.array([[1.0, 0.0], [0.0, 1.0]])
        rewards = np.array([5.0, 3.0])
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        np.testing.assert_allclose(values, [50.0, 30.0])

    def test_two_state_chain(self):
        transition = np.array([[0.0, 1.0], [1.0, 0.0]])
        rewards = np.array([1.0, 0.0])
        values = mrps.solve_mrp(transition, rewards, gamma=0.5)
        np.testing.assert_allclose(values, [4 / 3, 2 / 3])

    def test_gamma_zero_returns_immediate_rewards(self):
        transition = np.array([[0.5, 0.5], [0.5, 0.5]])
        rewards = np.array([3.0, 7.0])
        values = mrps.solve_mrp(transition, rewards, gamma=0.0)
        np.testing.assert_allclose(values, rewards)

    def test_single_state(self):
        transition = np.array([[1.0]])
        rewards = np.array([4.0])
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        np.testing.assert_allclose(values, [40.0])

    def test_negative_rewards(self):
        transition = np.array([[0.0, 1.0], [1.0, 0.0]])
        rewards = np.array([-2.0, 3.0])
        values = mrps.solve_mrp(transition, rewards, gamma=0.5)
        # V_0 = -2 + 0.5*V_1, V_1 = 3 + 0.5*V_0
        # V_0 = -2 + 0.5*(3 + 0.5*V_0) = -0.5 + 0.25*V_0 → V_0 = -2/3
        # V_1 = 3 + 0.5*(-2/3) = 8/3
        np.testing.assert_allclose(values, [-2 / 3, 8 / 3])

    @pytest.mark.parametrize(
        "generator,kwargs",
        [
            (mrps.get_dumbbell_mrp, {"clique_size": 3}),
            (mrps.get_cycle_mrp, {"n": 8}),
            (mrps.get_path_mrp, {"n": 8}),
            (mrps.get_hypercube_mrp, {"d": 3}),
            (mrps.get_complete_mrp, {"n": 8}),
            (mrps.get_expander_mrp, {"n": 10, "d": 3, "seed": 0}),
        ],
    )
    def test_bellman_equation_identity(self, generator, kwargs):
        """V must satisfy V = R + γPV for any valid MRP."""
        transition, rewards = generator(**kwargs)
        gamma = 0.9
        values = mrps.solve_mrp(transition, rewards, gamma=gamma)
        expected = rewards + gamma * transition @ values
        np.testing.assert_allclose(values, expected)


class TestMrpGenerators:
    @pytest.mark.parametrize(
        "generator,kwargs,expected_n",
        [
            (mrps.get_dumbbell_mrp, {"clique_size": 3}, 7),
            (mrps.get_dumbbell_mrp, {"clique_size": 4}, 9),
            (mrps.get_cycle_mrp, {"n": 5}, 5),
            (mrps.get_path_mrp, {"n": 6}, 6),
            (mrps.get_hypercube_mrp, {"d": 2}, 4),
            (mrps.get_hypercube_mrp, {"d": 3}, 8),
            (mrps.get_complete_mrp, {"n": 4}, 4),
            (mrps.get_expander_mrp, {"n": 8, "d": 3, "seed": 0}, 8),
        ],
    )
    def test_shape_and_stochasticity(self, generator, kwargs, expected_n):
        transition, rewards = generator(**kwargs)
        assert transition.shape == (expected_n, expected_n)
        assert rewards.shape == (expected_n,)
        _assert_valid_stochastic_matrix(transition)

    def test_dumbbell_rewards_in_clique_a(self):
        transition, rewards = mrps.get_dumbbell_mrp(clique_size=3)
        np.testing.assert_array_equal(rewards[:3], 10.0)
        np.testing.assert_array_equal(rewards[3:], 0.0)

    def test_dumbbell_bridge_transitions(self):
        transition, _ = mrps.get_dumbbell_mrp(clique_size=3)
        bridge = 3
        assert transition[bridge, 0] == pytest.approx(0.5)
        assert transition[bridge, 4] == pytest.approx(0.5)
        assert np.count_nonzero(transition[bridge]) == 2

    def test_dumbbell_within_clique_probs(self):
        clique_size = 4
        bridge_prob = 0.2
        transition, _ = mrps.get_dumbbell_mrp(
            clique_size=clique_size, bridge_prob=bridge_prob
        )
        within_prob = (1 - bridge_prob) / (clique_size - 1)
        for i in range(clique_size):
            assert transition[i, clique_size] == pytest.approx(bridge_prob)
            for j in range(clique_size):
                if j != i:
                    assert transition[i, j] == pytest.approx(within_prob)

    def test_dumbbell_bridge_prob_parameter(self):
        for bridge_prob in [0.1, 0.3, 0.5]:
            transition, _ = mrps.get_dumbbell_mrp(
                clique_size=3, bridge_prob=bridge_prob
            )
            assert transition[0, 3] == pytest.approx(bridge_prob)

    def test_cycle_symmetry(self):
        transition, _ = mrps.get_cycle_mrp(n=6)
        for i in range(6):
            left = (i - 1) % 6
            right = (i + 1) % 6
            assert transition[i, left] == pytest.approx(0.5)
            assert transition[i, right] == pytest.approx(0.5)

    def test_cycle_reward_placement(self):
        _, rewards = mrps.get_cycle_mrp(n=6)
        assert rewards[0] == 10.0
        np.testing.assert_array_equal(rewards[1:], 0.0)

    def test_path_boundary_conditions(self):
        transition, _ = mrps.get_path_mrp(n=5)
        assert transition[0, 1] == pytest.approx(1.0)
        assert transition[4, 3] == pytest.approx(1.0)
        assert transition[0, 0] == 0.0
        assert transition[4, 4] == 0.0

    def test_path_interior_transitions(self):
        transition, _ = mrps.get_path_mrp(n=6)
        for i in range(1, 5):
            assert transition[i, i - 1] == pytest.approx(0.5)
            assert transition[i, i + 1] == pytest.approx(0.5)
            assert np.count_nonzero(transition[i]) == 2

    def test_path_reward_at_far_end(self):
        _, rewards = mrps.get_path_mrp(n=6)
        assert rewards[-1] == 10.0
        np.testing.assert_array_equal(rewards[:-1], 0.0)

    def test_hypercube_degree(self):
        d = 3
        transition, _ = mrps.get_hypercube_mrp(d=d)
        for i in range(2**d):
            nonzero = np.count_nonzero(transition[i])
            assert nonzero == d

    def test_hypercube_neighbors_are_bitflips(self):
        d = 4
        transition, _ = mrps.get_hypercube_mrp(d=d)
        for i in range(2**d):
            neighbors = np.nonzero(transition[i])[0]
            for j in neighbors:
                assert _hamming_distance(i, j) == 1

    def test_hypercube_transition_probabilities(self):
        d = 3
        transition, _ = mrps.get_hypercube_mrp(d=d)
        for i in range(2**d):
            nonzero_values = transition[i, transition[i] > 0]
            np.testing.assert_allclose(nonzero_values, 1.0 / d)

    def test_complete_no_self_loops(self):
        transition, _ = mrps.get_complete_mrp(n=5)
        np.testing.assert_array_equal(np.diag(transition), 0.0)

    def test_complete_transition_probabilities(self):
        n = 6
        transition, _ = mrps.get_complete_mrp(n=n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert transition[i, j] == 0.0
                else:
                    assert transition[i, j] == pytest.approx(1.0 / (n - 1))

    def test_expander_reproducibility(self):
        t1, r1 = mrps.get_expander_mrp(n=8, d=3, seed=42)
        t2, r2 = mrps.get_expander_mrp(n=8, d=3, seed=42)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(r1, r2)

    def test_expander_degree(self):
        transition, _ = mrps.get_expander_mrp(n=10, d=4, seed=0)
        for i in range(10):
            assert np.count_nonzero(transition[i]) == 4

    def test_expander_no_self_loops(self):
        transition, _ = mrps.get_expander_mrp(n=10, d=3, seed=0)
        np.testing.assert_array_equal(np.diag(transition), 0.0)

    def test_expander_different_seeds_differ(self):
        t1, r1 = mrps.get_expander_mrp(n=10, d=3, seed=0)
        t2, r2 = mrps.get_expander_mrp(n=10, d=3, seed=99)
        assert not np.array_equal(t1, t2) or not np.array_equal(r1, r2)


class TestValueFunctionProperties:
    def test_path_monotonic_values(self):
        transition, rewards = mrps.get_path_mrp(n=10)
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

    def test_cycle_symmetric_values(self):
        n = 8
        transition, rewards = mrps.get_cycle_mrp(n=n)
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        for i in range(1, n):
            np.testing.assert_allclose(values[i], values[n - i])

    def test_complete_near_flat_values(self):
        n = 8
        transition, rewards = mrps.get_complete_mrp(n=n)
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        assert values[0] > values[1]
        non_reward_values = values[1:]
        np.testing.assert_allclose(non_reward_values, non_reward_values[0])

    def test_hypercube_hamming_distance_ordering(self):
        d = 3
        transition, rewards = mrps.get_hypercube_mrp(d=d)
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        by_distance = {}
        for i in range(2**d):
            dist = _hamming_distance(i, 0)
            by_distance.setdefault(dist, []).append(values[i])
        for dist, vals in by_distance.items():
            np.testing.assert_allclose(vals, vals[0])
        distances = sorted(by_distance.keys())
        for k in range(len(distances) - 1):
            assert by_distance[distances[k]][0] > by_distance[distances[k + 1]][0]

    def test_dumbbell_clique_a_higher_values(self):
        clique_size = 4
        transition, rewards = mrps.get_dumbbell_mrp(clique_size=clique_size)
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        clique_a_mean = np.mean(values[:clique_size])
        clique_b_mean = np.mean(values[clique_size + 1 :])
        assert clique_a_mean > clique_b_mean

    @pytest.mark.parametrize(
        "generator,kwargs",
        [
            (mrps.get_dumbbell_mrp, {"clique_size": 3}),
            (mrps.get_cycle_mrp, {"n": 8}),
            (mrps.get_path_mrp, {"n": 8}),
            (mrps.get_hypercube_mrp, {"d": 3}),
            (mrps.get_complete_mrp, {"n": 8}),
            (mrps.get_expander_mrp, {"n": 10, "d": 3, "seed": 0}),
        ],
    )
    def test_all_generators_positive_values(self, generator, kwargs):
        transition, rewards = generator(**kwargs)
        values = mrps.solve_mrp(transition, rewards, gamma=0.9)
        np.testing.assert_array_less(0, values)


class TestIsIrreducible:
    def test_complete_graph(self):
        transition, _ = mrps.get_complete_mrp(5)
        assert mrps.is_irreducible(transition)

    def test_cycle(self):
        transition, _ = mrps.get_cycle_mrp(6)
        assert mrps.is_irreducible(transition)

    def test_disconnected_components(self):
        transition = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )
        assert not mrps.is_irreducible(transition)

    def test_absorbing_state(self):
        transition = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.0, 1.0],
            ]
        )
        assert not mrps.is_irreducible(transition)

    def test_single_state(self):
        transition = np.array([[1.0]])
        assert mrps.is_irreducible(transition)

    def test_all_generators_are_irreducible(self):
        generators = [
            mrps.get_dumbbell_mrp(),
            mrps.get_cycle_mrp(),
            mrps.get_path_mrp(),
            mrps.get_hypercube_mrp(),
            mrps.get_complete_mrp(),
            mrps.get_expander_mrp(seed=0),
        ]
        for transition, _ in generators:
            assert mrps.is_irreducible(transition)


class TestIsAperiodic:
    def test_complete_graph_aperiodic(self):
        transition, _ = mrps.get_complete_mrp(5)
        assert mrps.is_aperiodic(transition)

    def test_even_cycle_periodic(self):
        transition, _ = mrps.get_cycle_mrp(4)
        assert not mrps.is_aperiodic(transition)

    def test_odd_cycle_aperiodic(self):
        transition, _ = mrps.get_cycle_mrp(3)
        assert mrps.is_aperiodic(transition)

    def test_self_loop_makes_aperiodic(self):
        transition = np.array(
            [
                [0.1, 0.9],
                [0.9, 0.1],
            ]
        )
        assert mrps.is_aperiodic(transition)

    def test_single_state(self):
        transition = np.array([[1.0]])
        assert mrps.is_aperiodic(transition)

    def test_path_graph_periodic(self):
        transition, _ = mrps.get_path_mrp(4)
        assert not mrps.is_aperiodic(transition)

    def test_dumbbell_aperiodic(self):
        transition, _ = mrps.get_dumbbell_mrp(clique_size=3)
        assert mrps.is_aperiodic(transition)

    def test_hypercube_periodic(self):
        transition, _ = mrps.get_hypercube_mrp(d=3)
        assert not mrps.is_aperiodic(transition)

    def test_expander_aperiodic(self):
        transition, _ = mrps.get_expander_mrp(n=10, d=3, seed=0)
        assert mrps.is_aperiodic(transition)
