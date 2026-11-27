# Farama Foundation Environments — Reward Structure Survey

## Complete Environment Table

| Project | Category / Environment | Agent Type | Reward per Step | Dense r(s,a)? |
|---|---|---|---|---|
| **Gymnasium** | Classic Control — CartPole | Single | +1 constant while upright | No (constant) |
| **Gymnasium** | Classic Control — Pendulum | Single | -(θ² + 0.1·θ̇² + 0.001·τ²) | **Yes** |
| **Gymnasium** | Classic Control — Acrobot | Single | -1 constant per step | No (constant+sparse) |
| **Gymnasium** | Classic Control — MountainCar | Single | -1 constant per step | No (constant) |
| **Gymnasium** | Classic Control — MountainCarContinuous | Single | -1 constant per step | No (constant) |
| **Gymnasium** | Box2D — LunarLander | Single | f(position, velocity, angle, leg contact, fuel) | **Yes** |
| **Gymnasium** | Box2D — BipedalWalker | Single | f(forward_progress) - f(torque) | **Yes** |
| **Gymnasium** | Box2D — BipedalWalkerHardcore | Single | Same as BipedalWalker | **Yes** |
| **Gymnasium** | Box2D — CarRacing | Single | -0.1/frame + f(tiles visited) | **Yes** |
| **Gymnasium** | Toy Text — FrozenLake | Single | +1 at goal, 0 elsewhere | No (sparse) |
| **Gymnasium** | Toy Text — Taxi | Single | -1/step, +20 delivery, -10 illegal | No (sparse) |
| **Gymnasium** | MuJoCo — HalfCheetah | Single | f(velocity) - f(ctrl) | **Yes** |
| **Gymnasium** | MuJoCo — Hopper | Single | 1 + f(velocity) - f(ctrl) | **Yes** |
| **Gymnasium** | MuJoCo — Walker2d | Single | 1 + f(velocity) - f(ctrl) | **Yes** |
| **Gymnasium** | MuJoCo — Ant | Single | 1 + f(velocity) - f(ctrl) - f(contact) | **Yes** |
| **Gymnasium** | MuJoCo — Swimmer | Single | f(velocity) - f(ctrl) | **Yes** |
| **Gymnasium** | MuJoCo — Humanoid | Single | 5 + f(velocity) - f(ctrl) - f(contact) | **Yes** |
| **Gymnasium** | MuJoCo — HumanoidStandup | Single | f(height, velocity, ctrl, contact) | **Yes** |
| **Gymnasium** | MuJoCo — Reacher | Single | -‖fingertip − target‖₂ - f(ctrl) | **Yes** |
| **Gymnasium** | MuJoCo — Pusher | Single | -f(obj_dist) - f(ctrl) - f(tip_dist) | **Yes** |
| **Gymnasium** | MuJoCo — InvertedPendulum | Single | +1 constant while upright | No (constant) |
| **Gymnasium** | MuJoCo — InvertedDoublePendulum | Single | 10 - f(dist) - f(velocity) | **Yes** |
| **Gymnasium-Robotics** | Fetch tasks (default) | Single | -1 or 0 (binary) | No (sparse) |
| **Gymnasium-Robotics** | Fetch tasks (Dense variant) | Single | -‖obj − goal‖₂ | **Yes** |
| **Gymnasium-Robotics** | Hand tasks (default) | Single | -1 or 0 (binary) | No (sparse) |
| **Gymnasium-Robotics** | Hand tasks (Dense variant) | Single | -‖state − goal‖₂ | **Yes** |
| **Gymnasium-Robotics** | Maze envs | Single | MuJoCo locomotion r(s,a) + f(dist_to_goal) | **Yes** |
| **MO-Gymnasium** | MO-* variants | Single | Vectorized; same density as base env | Inherits |
| **HighwayEnv** | Highway, Merge, Roundabout, etc. | Single | f(speed, lane_pos, collision) | **Yes** |
| **Metaworld** | 50 manipulation tasks | Single | Binary success/failure | No (sparse) |
| **MiniGrid** | Grid navigation variants | Single | Reward only at goal | No (sparse) |
| **MiniWoB++** | Web interaction tasks | Single | 0 during episode; terminal only | No (sparse) |
| **ALE** | Atari 2600 (60+ games) | Single | Game-specific; mostly score events | No (mostly sparse) |
| **Stable-Retro** | Retro games (1000+) | Single | Score deltas; sparse by design | No (sparse) |
| **Procgen2** | Procedural side-scrollers | Single | Game scoring (docs unavailable) | Unknown |
| **PettingZoo** | Atari (23 games) | Multi | Game-specific | No (mostly sparse) |
| **PettingZoo** | Classic (Chess, Go, etc.) | Multi | +1/-1 at game end | No (sparse) |
| **PettingZoo** | Butterfly — Coop Pong | Multi | ~0.11/step - 10·miss | Partially |
| **PettingZoo** | Butterfly — Pistonball | Multi | f(ball_position_change) - 0.1 | **Yes** |
| **PettingZoo** | Butterfly — KAZ | Multi | +1 per kill | No (sparse) |
| **PettingZoo** | SISL — Multiwalker | Multi | f(package_pos_change) + penalties | **Yes** |
| **PettingZoo** | SISL — Pursuit | Multi | 5·surround + 0.01·touch - 0.1 | Partially |
| **MAgent2** | Battle, Pursuit, etc. | Multi | -0.005/step + f(attack, kill events) | Partially |
| **MOMAland** | Multi-obj multi-agent | Multi | Vectorized; varies by env | Configurable |
| **MPE2** | Communication envs | Multi | Varies; some have f(distance) | Configurable |
| **ViZDoom** | Doom scenarios | Single | User-defined per config | Configurable |
| **Shimmy** | Wrappers | Varies | Inherits from wrapped env | Varies |
| **Minari** | Offline datasets | N/A | Not an environment | N/A |

---

## Filtered: Environments with Dense r(s,a) Rewards

Reward is a non-trivial function of state and/or action at every step (not constant, not zero, not only at termination).

### Single-Agent

| Project | Environment | Obs Dim | Act Dim | Reward Formula |
|---|---|---|---|---|
| **Gymnasium** | Pendulum | 3 (cont) | 1 (cont) | -(θ² + 0.1·θ̇² + 0.001·τ²) |
| **Gymnasium** | LunarLander | 8 (cont) | 4 (disc) or 2 (cont) | f(pos, vel, angle, legs, fuel) |
| **Gymnasium** | BipedalWalker | 24 (cont) | 4 (cont) | f(dx/dt) - f(torque) |
| **Gymnasium** | BipedalWalkerHardcore | 24 (cont) | 4 (cont) | f(dx/dt) - f(torque) |
| **Gymnasium** | CarRacing | 96×96×3 (img) | 3 (cont) | -0.1 + f(tiles) |
| **Gymnasium** | HalfCheetah | 17 (cont) | 6 (cont) | f(velocity) - f(ctrl) |
| **Gymnasium** | Hopper | 11 (cont) | 3 (cont) | 1 + f(velocity) - f(ctrl) |
| **Gymnasium** | Walker2d | 17 (cont) | 6 (cont) | 1 + f(velocity) - f(ctrl) |
| **Gymnasium** | Ant | 27 (cont) | 8 (cont) | 1 + f(velocity) - f(ctrl) - f(contact) |
| **Gymnasium** | Swimmer | 8 (cont) | 2 (cont) | f(velocity) - f(ctrl) |
| **Gymnasium** | Humanoid | 376 (cont) | 17 (cont) | 5 + f(velocity) - f(ctrl) - f(contact) |
| **Gymnasium** | HumanoidStandup | 376 (cont) | 17 (cont) | f(height, vel, ctrl, contact) |
| **Gymnasium** | Reacher | 11 (cont) | 2 (cont) | -‖tip − target‖ - f(ctrl) |
| **Gymnasium** | Pusher | 23 (cont) | 7 (cont) | -f(obj_dist) - f(ctrl) - f(tip_dist) |
| **Gymnasium** | InvertedDoublePendulum | 11 (cont) | 1 (cont) | 10 - f(site_dist) - f(velocity) |
| **Gym-Robotics** | FetchReachDense | dict (cont) | 4 (cont) | -‖gripper − target‖₂ |
| **Gym-Robotics** | FetchPushDense | dict (cont) | 4 (cont) | -‖obj − target‖₂ |
| **Gym-Robotics** | FetchSlideDense | dict (cont) | 4 (cont) | -‖obj − target‖₂ |
| **Gym-Robotics** | FetchPickAndPlaceDense | dict (cont) | 4 (cont) | -‖obj − target‖₂ |
| **Gym-Robotics** | HandReachDense | dict (cont) | 20 (cont) | -‖fingertips − target‖₂ |
| **Gym-Robotics** | HandManipulate*Dense | dict (cont) | 20 (cont) | -‖state − goal‖₂ |
| **Gym-Robotics** | Maze (Ant/Humanoid/Point) | cont | cont | locomotion r(s,a) + f(dist_to_goal) |
| **HighwayEnv** | Highway | ~25 (cont) | 5 (disc) | f(speed, lane, collision) |
| **HighwayEnv** | Merge | ~25 (cont) | 5 (disc) | f(speed, lane, collision) |
| **HighwayEnv** | Roundabout | ~25 (cont) | 5 (disc) | f(speed, lane, collision) |
| **HighwayEnv** | Intersection | ~25 (cont) | 5 (disc) | f(speed, collision) |
| **HighwayEnv** | Parking | ~12 (cont) | 2 (cont) | f(position, heading) to target |
| **HighwayEnv** | Racetrack | ~25 (cont) | 5 (disc) | f(speed, lane) |

### Multi-Agent (with dense r(s,a))

| Project | Environment | Reward Formula |
|---|---|---|
| **PettingZoo** | Pistonball | f(ball_position_change) per step |
| **PettingZoo** | Multiwalker | f(package_position_change) per step |
