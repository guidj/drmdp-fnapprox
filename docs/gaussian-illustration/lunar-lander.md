# LunarLander-v3: Gaussian illustration

The agent fires thrusters to land a spacecraft on terrain generated at reset. This experiment uses discrete [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) with wind disabled. It exposes terms from the native reward rather than adding a shaping reward.

## State and observation space

The eight raw Box coordinates are $s=(p_x,p_y,v_x,v_y,\theta,\omega,b_L,b_R)$: position relative to the pad, linear velocity, hull angle, angular velocity, and left and right leg ground-contact indicators. The declared position bounds are $\pm2.5$, velocity bounds $\pm10$, angle bounds $\pm2\pi$, and angular velocity bounds $\pm10$. Each contact is zero or one.

The [observation wrapper](../../src/drmdp/envs/lunar_lander.py) **prepends** five terms to the raw state. The resulting 13-coordinate observation is $(\psi(s),s)$, where

$$\psi(s)=(\lVert p\rVert_2,\lVert v\rVert_2,|\theta|,b_L,b_R).$$

## Action space

`Discrete(4)`: 0 does nothing, 1 fires the left orientation engine, 2 fires the main engine, and 3 fires the right orientation engine.

## Starting state and episode end

The lander starts at the top center of the viewport with a random force applied to its center of mass. A hull crash, leaving the horizontal flight area ($|p_x|\geq1$), or the lander coming to rest (Box2D puts the body to sleep) terminates the episode. The time limit is 1000 steps.

## Reward

Let $u=(-100,-100,-100,10,10)$ and $f=(0,-0.03,-0.30,-0.03)$ in action order. The scalar potential is $\Phi(s)=u^\top\psi(s)$, the weighted sum of the five observation terms. For a nonterminal step from $s$ to $s'$ under action $a$, the native reward before [Gaussian noise](README.md) is

$$r_*=u^\top[\psi(s')-\psi(s)]+f_a.$$

Moving closer to the pad, slowing down, or becoming more upright increases the corresponding term. On termination, a sleeping lander receives $+100$; any other termination receives $-100$. This payment **replaces** the potential difference and fuel cost. Time-limit truncation alone retains the nonterminal formula. The native difference does not use the control discount.

**Per-step range before noise.** The terminal reward set is exactly $\{-100,100\}$. The nonterminal formula has no explicit reward clip, so the terminal payments do not bound every step. In the [100-episode holdout per policy](../../agents/analyses/2026-11-08-lunar-lander-step-reward-distributions.md), the observed range including terminal steps was $[-100,100]$ for greedy linear SARSA and $[-100,86.322]$ for uniform random actions. These are sample extrema; neither establishes a global nonterminal range.

## Reward learning representation

LEAST and Bayes use [`LunarLanderRewardFeatures`](../../src/drmdp/rewardfeat.py). Let $g$ indicate a nonterminal step, $e_a$ be the four-entry one-hot action vector, and $L$ and $C$ indicate a sleeping and an awake termination, respectively. The 16 entries and their noise-free weights are

$$x=(g\psi(s'),-g\psi(s),g e_a,L,C),\qquad w=(u,u,f,100,-100).$$

The table gives per-step ranges for observations inside the declared Box. The encoder does not clip the position, speed, or angle magnitudes to these bounds.

| Coordinates | Meaning, in order | Range of each coordinate, in order |
| --- | --- | --- |
| $x_1,x_2,x_3$ | Successor distance to pad, speed, angle magnitude | $[0,2.5\sqrt{2}]$, $[0,10\sqrt{2}]$, $[0,2\pi]$ |
| $x_4,x_5$ | Successor left and right leg contacts | $\{0,1\}$ each |
| $x_6,x_7,x_8$ | Negated current distance to pad, speed, angle magnitude | $[-2.5\sqrt{2},0]$, $[-10\sqrt{2},0]$, $[-2\pi,0]$ |
| $x_9,x_{10}$ | Negated current left and right leg contacts | $\{-1,0\}$ each |
| $x_{11},\ldots,x_{14}$ | Indicators for actions 0 through 3, respectively | $\{0,1\}$ each |
| $x_{15},x_{16}$ | Landing and crash indicators, respectively | $\{0,1\}$ each |

On termination, the first fourteen entries are zero. Only one terminal column survives on termination. The [terminal-branch analysis](../../agents/analyses/2026-11-14-lunar-lander-terminal-branch-identifiability.md) explains why leg contacts cannot identify this branch: the wrapper supplies the Box2D sleep status through `info["awake"]`, not through the observation.

## Control representation

Unlike the other three experiments, control keeps the **entire** wrapped observation. `splice-tile-observation-action-ft` tiles all 13 coordinates with eight tiles per coordinate and 32 offset tilings. The candidate action selects its own tile-index table. Indices are hashed into one 2048-entry output vector, not four concatenated vectors; collisions contribute counts. Control does not read the successor, terminal columns, or `info["awake"]`.
