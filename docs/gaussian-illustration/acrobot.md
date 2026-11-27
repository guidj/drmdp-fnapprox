# Acrobot-v1: Gaussian illustration

The agent applies torque at the joint between two links to swing the free end above the target height. The experiment uses [Gymnasium Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) with tip-height reward shaping.

## State and observation space

The physical state consists of two joint angles and their angular velocities. The six raw Box coordinates are $(\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\dot\theta_1,\dot\theta_2)$; $\theta_1$ is measured from the downward vertical and $\theta_2$ relative to the first link. The trigonometric coordinates are in $[-1,1]$ and the velocity bounds are $\pm4\pi$ and $\pm9\pi$, respectively. The seventh observation coordinate is $\Phi(s)=h(s)-1$, where $h(s)=-\cos\theta_1-\cos(\theta_1+\theta_2)$ is the tip height.

## Action space

`Discrete(3)`: actions 0, 1, and 2 apply torques $-1$, $0$, and $+1$, respectively, at the actuated joint.

## Starting state and episode end

Reset independently samples each angle and each angular velocity uniformly from $[-0.1,0.1]$, near the hanging configuration. The episode terminates when $h(s)>1$ and truncates at 500 steps if the goal has not been reached.

## Reward

The base reward is $-1$ per nonterminal step and $0$ on reaching the goal. Let $g=0$ on termination and $g=1$ otherwise. For current state $s$, successor $s'$, and discount $\gamma\in\{0.99,1\}$, the shaped reward before [Gaussian noise](README.md) is

$$r_*=-g-\Phi(s)+\gamma g\Phi(s').$$

Thus the terminal **base** reward is zero, but the terminal **shaped** reward is $-\Phi(s)$. Truncation alone does not suppress either the step cost or the successor term.

**Per-step range before noise.** On a nonterminal step, both tip heights lie in $[-2,1]$, so both potentials lie in $[-3,0]$. Substitution gives $-1-3\gamma\le r_*\le 2$: a bound of $[-4,2]$ at $\gamma=1$ or $[-3.97,2]$ at $\gamma=0.99$. On goal entry, $r_*=-\Phi(s)\in[0,3]$ at either discount. These are analytical bounds; the dynamics need not attain the endpoints in one step.

## Reward learning representation

LEAST and Bayes use [`acrobot_potential_curr_succ`](../../src/drmdp/workflows/encodings.py). Its three entries are $x=(\Phi(s),g\Phi(s'),g)$ with weights $w=(-1,\gamma,-1)$. The last column counts only nonterminal steps in an aggregate window, unlike MountainCar's constant column.

| Coordinate | Meaning | Per-step range |
| --- | --- | --- |
| $x_1=\Phi(s)$ | Current tip height relative to the target | $[-3,0]$ |
| $x_2=g\Phi(s')$ | Successor tip height relative to the target, zero on termination | $[-3,0]$ |
| $x_3=g$ | Nonterminal step indicator | $\{0,1\}$ |

The two links give $h(s)\in[-2,2]$, so the observation can contain a potential as high as $1$. A positive potential occurs only above the target, where the episode terminates. It therefore never contributes to the reward feature row.

## Control representation

Control drops the seventh coordinate and tile-codes the six raw coordinates with the candidate action in the tile key. The default uses three tiles per coordinate and 32 offset tilings. The operator hashes the $3^6\times32\times3=69{,}984$ nominal indices into 8192 entries; when indices coincide, the entry contains their count.
