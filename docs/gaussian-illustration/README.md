# Gaussian illustration experiments

Each page describes one environment as configured by [`gaussian_illustration_experiment_specs`](../../src/drmdp/workflows/control/exps.py). The five noise settings change the reward, not the observation or either feature map. Dimensions below use the default tiling parameters.

| Experiment | Wrapped observation width | LEAST and Bayes feature width | Control feature width |
| --- | ---: | ---: | ---: |
| [MountainCar-v0](mountain-car.md) | 3 | 3 | 384 |
| [Acrobot-v1](acrobot.md) | 7 | 3 | 8192 |
| [GridWorld-MINES](gridworld-mines.md) | 3 | 4 | 800 |
| [LunarLander-v3](lunar-lander.md) | 13 | 16 | 2048 |

**Observed reward.** Each page gives the per-step reward before noise. On steps that do not terminate, the [noise wrapper](../../src/drmdp/envs/wrappers.py) adds a draw from $\mathcal N(0,\sigma^2)$ clipped to $[-1.96\sigma,1.96\sigma]$, with $\sigma\in\{0,0.25,0.5,1,2\}$. Termination disables noise; time-limit truncation alone does not. For the three environments with potential-based reward shaping, the shaping discount is the control discount, $\gamma\in\{1,0.99\}$.

**Reading the per-step ranges.** The [MountainCar](mountain-car.md#reward) and [Acrobot](acrobot.md#reward) pages give analytical bounds; [GridWorld](gridworld-mines.md#reward) gives exact extrema from its transition table; [LunarLander](lunar-lander.md#reward) distinguishes terminal payments from rollout extrema. All are before noise and window aggregation. A clean nonterminal interval $[L,U]$ gives a training bound of $[L-1.96\sigma,U+1.96\sigma]$ at the selected noise level. At the largest setting this widens each end by $3.92$; terminal rewards remain unchanged.

**Reward learning.** LEAST (`least-lfa`) and Bayes (by default `blade-td-nig`, using the posterior mean) receive the same [transition features](../../src/drmdp/workflows/encodings.py). For a delayed window $W$, the regression input is the sum of its feature rows, $X_W=\sum_{t\in W}x_t$, and the target is the observed reward sum, $R_W=\sum_{t\in W}r_t$. The fitted weights predict a step reward as $x_t^\top\hat w$. The weights given on the pages reproduce the reward **before noise**; they are not measured fits. A constant in a step row counts steps when summed, unlike a regression intercept. See the [window-sum analysis](../../agents/analyses/2026-11-08-window-sum-fit-window-length.md).

**Control.** Linear SARSA estimates action values from a different vector, $z(o,a)$, evaluated for each candidate action using only the current observation $o$. Neither the next observation nor the terminal flag is available at action selection. The [tile operators](../../src/drmdp/transform.py) scale coordinates using observation-space bounds and select one index per offset tiling. The feature widths above count all output coordinates, not the number selected on one evaluation.
