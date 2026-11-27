# GridWorld-MINES: Gaussian illustration

The agent navigates a grid to the exit. Entering a cliff costs 100 and returns the agent to a fixed cell without ending the episode. This is the repository's [random-start GridWorld](../../src/drmdp/envs/gympg.py), not Gymnasium's CliffWalking environment.

## State and observation space

The raw observation is the agent's integer position $(i,j)$ on a six-row, twelve-column grid, indexed from zero at the upper left. The [MINES layout](../../src/drmdp/workflows/grids.py) has 54 cells from which actions can be taken, 17 cliffs, and one exit. Here `s` marks the cliff return cell, `g` the exit, `x` a cliff, and `o` an open cell.

```text
ooooxooooooo
oooooooooxoo
oxoooooxoooo
oooooxoooooo
ooxooooooxoo
sxxxxxxxxxxg
```

The wrapper emits $(i,j,\Phi(s))$. Let $d(s)$ be the length of a shortest cliff-free path to the exit and $D=17$ the largest such distance. The appended potential is $\Phi(s)=-d(s)^2/D$.

## Action space

`Discrete(4)`: 0 moves left, 1 right, 2 up, and 3 down. A move past the boundary leaves the agent in place.

## Starting state and episode end

Reset samples uniformly from reachable non-cliff, non-exit cells whose distance to the exit is at least the median over those cells, including ties. A cliff returns the agent to $(5,0)$, **not** to the sampled episode start. Entry into the exit at $(5,11)$ terminates. The time limit is 200 steps.

## Reward

An ordinary move, boundary collision, or exit entry pays $-1$ before shaping; cliff entry pays $-100$ instead. Let $C$ indicate cliff entry and $g$ be zero on termination and one otherwise. For current cell $s$, successor cell $s'$, and discount $\gamma\in\{0.99,1\}$, the reward before [Gaussian noise](README.md) is

$$r_*=-1-99C-\Phi(s)+\gamma g\Phi(s').$$

On a cliff transition, the successor potential is that of $(5,0)$, not the cliff. At the exit it is zero. The [reward census](../../agents/analyses/2026-11-08-gw-shaped-reward.md) tabulates all 216 live state-action pairs at $\gamma=1$.

**Per-step range before noise.** Enumerating the four actions from each of the 54 live cells gives the following extrema. Cliff entry accounts for the minimum: the teleport changes the potential as well as incurring the base penalty. Exit entry from distance one always pays $-1+1/17=-16/17$.

| Discount | Minimum | Maximum |
|---:|---:|---:|
| $1$ | $-1985/17$ (about $-116.765$) | $16/17$ (about $0.941$) |
| $0.99$ | $-198211/1700$ (about $-116.595$) | $464/425$ (about $1.092$) |

## Reward learning representation

LEAST and Bayes use [`gw_potential_curr_succ`](../../src/drmdp/workflows/encodings.py). Its four entries are $x=(\Phi(s),g\Phi(s'),1,C)$, with weights $w=(-1,\gamma,-1,-99)$. The encoder obtains $C$ by comparing the actual successor position with the proposed unit move clamped to the grid. A cliff teleport differs from that move; a boundary collision does not. The column therefore identifies the penalty without reading the reward.

| Coordinate | Meaning | Per-step values |
| --- | --- | --- |
| $x_1=\Phi(s)$ | Potential of the cell from which the move starts | $-k^2/17$ for integer distances $k=1,\ldots,17$ |
| $x_2=g\Phi(s')$ | Potential of the cell reached, zero at the exit | $-k^2/17$ for $k=0,\ldots,17$ |
| $x_3=1$ | Move count | $\{1\}$ |
| $x_4=C$ | Cliff entry indicator | $\{0,1\}$ |

## Control representation

Control drops the potential and tile-codes the current row and column with the candidate action in the tile key. Five tiles per coordinate and eight offset tilings give $5^2\times8\times4=800$ entries. This is not a one-hot table of the 216 live state-action pairs.
