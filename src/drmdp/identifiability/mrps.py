from collections import deque
from math import gcd
from typing import Optional

import numpy as np


def solve_mrp(transition, rewards, gamma=0.9):
    """
    Computes the Value Function V = (I - γP)^-1 * R.
    V represents the expected discounted cumulative reward starting from each state.
    """
    n = transition.shape[0]
    identity = np.eye(n)
    # Solving the Bellman Equation: V = R + γPV -> (I - γP)V = R
    values = np.linalg.solve(identity - gamma * transition, rewards)
    return values


# 1. THE DUMBBELL CHAIN (The Bottleneck Model)
# Ref: Diaconis, P., & Saloff-Coste, L. (1996). "Nash inequalities for finite Markov chains."
# DOI: 10.1007/BF02214660
def get_dumbbell_mrp(clique_size: int = 3, bridge_prob: float = 0.2):
    """Two cliques connected by a single bridge node, forming a bottleneck.

    Args:
        clique_size: Number of nodes in each clique. Must be >= 2.
        bridge_prob: Probability of transitioning from a clique node to the bridge.
            The remaining probability is split uniformly among other clique members.

    Total states = 2 * clique_size + 1 (two cliques + one bridge).
    The bridge transitions equally to one node in each clique. Reward is
    concentrated in clique A (R=10), making the value function sensitive to
    the bottleneck's effect on state reachability.

    Source: Diaconis, P., & Saloff-Coste, L. (1996). "Nash inequalities for finite
    Markov chains." Journal of Theoretical Probability, 9, 459-510.
    DOI: https://doi.org/10.1007/BF02214660
    """
    n = 2 * clique_size + 1
    transition = np.zeros((n, n))
    clique_a = list(range(clique_size))
    bridge = clique_size
    clique_b = list(range(clique_size + 1, n))
    within_prob = (1 - bridge_prob) / (clique_size - 1)
    for i in clique_a:
        transition[i, [x for x in clique_a if x != i]] = within_prob
        transition[i, bridge] = bridge_prob
    for i in clique_b:
        transition[i, [x for x in clique_b if x != i]] = within_prob
        transition[i, bridge] = bridge_prob
    transition[bridge, clique_a[0]], transition[bridge, clique_b[0]] = 0.5, 0.5

    rewards = np.zeros(n)
    rewards[clique_a] = 10
    return transition, rewards


# 2. THE N-CYCLE (The Closed Loop Model)
# Ref: Levin, D. A., & Peres, Y. (2017). "Markov Chains and Mixing Times."
# ISBN: 978-1-4704-2962-1
def get_cycle_mrp(num_states=8):
    """Symmetric random walk on a cycle graph of num_states states.

    Each state transitions to its two neighbors with equal probability (0.5).
    A single rewarding state (R=10 at state 0) creates a value gradient that
    decays symmetrically around the cycle, with the antipodal state having the
    lowest value.

    Source: Levin, D. A., & Peres, Y. (2017). "Markov Chains and Mixing Times"
    (2nd ed.). American Mathematical Society.
    ISBN: 978-1-4704-2962-1
    """
    transition = np.zeros((num_states, num_states))
    for i in range(num_states):
        transition[i, (i - 1) % num_states] = 0.5
        transition[i, (i + 1) % num_states] = 0.5
    rewards = np.zeros(num_states)
    rewards[0] = 10
    return transition, rewards


# 3. THE PATH GRAPH (The Line Model)
# Ref: Diaconis, P., & Saloff-Coste, L. (1993). "Comparison theorems for reversible Markov chains."
# DOI: 10.1214/aoap/1177005359
def get_path_mrp(num_states=8):
    """Random walk on a path (line) graph of num_states states with reflecting boundaries.

    Interior states transition to each neighbor with probability 0.5. Boundary
    states deterministically transition to their sole neighbor. Reward is placed
    at the far end (R=10 at state num_states-1), producing a monotonically increasing
    value function along the path.

    Source: Diaconis, P., & Saloff-Coste, L. (1993). "Comparison theorems for
    reversible Markov chains." The Annals of Applied Probability, 3(3), 696-730.
    DOI: https://doi.org/10.1214/aoap/1177005359
    """
    transition = np.zeros((num_states, num_states))
    for i in range(num_states):
        if i == 0:
            transition[i, 1] = 0.5
            transition[i, i] = 0.5
        elif i == num_states - 1:
            transition[i, num_states - 2] = 0.5
            transition[i, i] = 0.5
        else:
            transition[i, i - 1], transition[i, i + 1] = 0.5, 0.5
    rewards = np.zeros(num_states)
    rewards[-1] = 10
    return transition, rewards


# 4. THE HYPERCUBE (The High-Dimensional Model)
# Ref: Diaconis, P., Graham, R. L., & Morrison, J. A. (1990). "Asymptotic analysis of a random walk on a hypercube."
# DOI: 10.1214/aop/1176990628
def get_hypercube_mrp(dimensions=3):
    """Random walk on a dimensions-dimensional binary hypercube {0,1}^dimensions.

    States are the 2^dimensions binary strings of length dimensions. Each step
    flips one bit uniformly at random, so each state has exactly dimensions
    neighbors. Reward is placed at state 0 (the all-zeros vertex). The Hamming
    distance from state 0 determines the value gradient.

    Source: Diaconis, P., Graham, R. L., & Morrison, J. A. (1990). "Asymptotic
    analysis of a random walk on a hypercube with many dimensions." Annals of
    Probability, 18(3), 1296-1314.
    DOI: https://doi.org/10.1214/aop/1176990628
    """
    num_states = 2**dimensions
    transition = np.zeros((num_states, num_states))
    for i in range(num_states):
        for bit in range(dimensions):
            neighbor = i ^ (1 << bit)
            transition[i, neighbor] = 1.0 / dimensions
    rewards = np.zeros(num_states)
    rewards[0] = 10
    return transition, rewards


# 5. THE COMPLETE GRAPH (The Fully Connected Model)
# Ref: Aldous, D., & Fill, J. (2002). "Reversible Markov Chains and Random Walks on Graphs."
# URL: https://www.stat.berkeley.edu/~aldous/RWG/book.html
def get_complete_mrp(num_states=8):
    """Random walk on a complete graph of num_states states (fully connected, no self-loops).

    Each state transitions uniformly to any of the other num_states-1 states. Reward is
    placed at state 0. Because every state is one step away from every other,
    the value function is nearly flat: the rewarding state has a slightly higher
    value, while all others share the same value.

    Source: Aldous, D., & Fill, J. (2002). "Reversible Markov Chains and Random
    Walks on Graphs." Unfinished monograph, UC Berkeley.
    URL: https://www.stat.berkeley.edu/~aldous/RWG/book.html
    """
    transition = np.full((num_states, num_states), 1.0 / (num_states - 1))
    np.fill_diagonal(transition, 0)
    rewards = np.zeros(num_states)
    rewards[0] = 10
    return transition, rewards


# 6. RANDOM EXPANDER GRAPH (The Spectral Gap Model)
# Ref: Hoory, S., Linial, N., & Wigderson, A. (2006). "Expander graphs and their applications."
# DOI: 10.1090/S0273-0979-06-01126-8
def get_expander_mrp(num_states=10, degree=3, seed: Optional[int] = None):
    """Random walk on a random degree-regular-like graph of num_states states.

    Each state is connected to degree randomly chosen neighbors (not necessarily
    symmetric), approximating an expander graph. Rewards are drawn uniformly
    at random in [0, 10]. The large spectral gap typical of expanders leads
    to rapid mixing and a relatively uniform value function.

    Source: Hoory, S., Linial, N., & Wigderson, A. (2006). "Expander graphs and
    their applications." Bulletin of the AMS, 43(4), 439-561.
    DOI: https://doi.org/10.1090/S0273-0979-06-01126-8
    """
    rng = np.random.default_rng(seed)
    transition = np.zeros((num_states, num_states))
    for i in range(num_states):
        targets = rng.choice(
            [j for j in range(num_states) if i != j], degree, replace=False
        )
        transition[i, targets] = 1.0 / degree
    rewards = rng.uniform(0, 10, num_states)
    return transition, rewards


def is_irreducible(transition: np.ndarray) -> bool:
    """Check if the Markov chain with the given transition matrix is irreducible."""
    n = transition.shape[0]
    adjacency = (transition > 0).astype(float)
    reachability = np.linalg.matrix_power(np.eye(n) + adjacency, n - 1)
    return bool(np.all(reachability > 0))


def is_aperiodic(transition: np.ndarray) -> bool:
    """Check if the Markov chain with the given transition matrix is aperiodic."""
    n = transition.shape[0]
    dist = np.full(n, -1)
    dist[0] = 0
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if transition[u, v] > 0 and dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    period = 0
    for u in range(n):
        if dist[u] == -1:
            continue
        for v in range(n):
            if transition[u, v] > 0 and dist[v] != -1:
                cycle_len = dist[u] + 1 - dist[v]
                if cycle_len > 0:
                    period = gcd(period, cycle_len)
    return period == 1


def is_ergodic(transition: np.ndarray):
    return is_irreducible(transition) and is_aperiodic(transition)
