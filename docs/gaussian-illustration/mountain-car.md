# MountainCar-v0: Gaussian illustration

The car must build momentum to reach the flag on the right hill. This experiment changes [Gymnasium Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) by adding height-based reward shaping, exposing the height potential, and extending the time limit.

## State and observation space

The raw state is position and velocity, $s=(p,v)$, with $p\in[-1.2,0.6]$ and $v\in[-0.07,0.07]$. The experiment emits a three-coordinate Box observation $(p,v,\Phi(s))$, where $\Phi(s)=\sin(3p)-\sin(1.5)$ measures height relative to the flag.

## Action space

`Discrete(3)`: 0 accelerates left, 1 applies no engine force, and 2 accelerates right.

## Starting state and episode end

Reset samples $p$ uniformly from $[-0.6,-0.4]$ and sets $v=0$. The goal is $p\geq0.5$ with $v\geq0$. The experiment truncates at 2500 steps, rather than Gymnasium's usual 200.

## Reward

The base reward is $-1$ on every step, including entry into the goal. Let $g=0$ if the step terminates and $g=1$ otherwise. For current state $s$, successor $s'$, and discount $\gamma\in\{0.99,1\}$, the shaped reward before [Gaussian noise](README.md) is

$$r_*=-1-\Phi(s)+\gamma g\Phi(s').$$

The goal can be overshot, so the successor's observed potential need not be zero. The shaping wrapper omits it on termination, but retains it at a time limit.

**Per-step range before noise.** At $\gamma=1$, the shaped reward lies in $[-1.21,-0.79]$; at $\gamma=0.99$, it lies in $[-1.21,-0.77]$. These are bounds: position changes by at most $0.07$ per step and the potential slope has magnitude at most $3$, so the potential changes by at most $0.21$. Before goal entry the potential lies in $[-2,0]$, so discounting the successor can add at most $0.02$. On goal entry the same displacement bound applies with the potential evaluated at the flag, where it is zero.

## Reward learning representation

LEAST and Bayes use [`mc_potential_curr_succ`](../../src/drmdp/workflows/encodings.py). Its three entries are $x=(\Phi(s),1,g\Phi(s'))$. Weights $w=(-1,-1,\gamma)$ give $x^\top w=r_*$. The constant counts every step in an aggregate window, including the last one.

| Coordinate | Meaning | Per-step range |
| --- | --- | --- |
| $x_1=\Phi(s)$ | Current height relative to the flag | $[-1-\sin(1.5),0)$ |
| $x_2=1$ | Step count | $\{1\}$ |
| $x_3=g\Phi(s')$ | Successor height relative to the flag, zero on termination | $[-1-\sin(1.5),0]$ |

The lowest hill height is $-1$. From the reset region, the first crossing of the flag terminates the episode, so every state from which an action is taken has $p<0.5$.

## Control representation

Control drops the appended potential and tile-codes $(p,v)$ with the candidate action as part of the tile key. With four tiles per coordinate and eight offset tilings, the vector has $4^2\times8\times3=384$ entries. Velocity distinguishes states that have the same height but require different actions.
