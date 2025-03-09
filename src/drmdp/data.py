import gymnasium as gym


def collection_traj_data(env: gym.Env, steps: int):
    obs, _ = env.reset()
    step = 0
    buffer = []
    while step < steps:
        action = env.action_space.sample()
        (
            next_obs,
            rew,
            term,
            trunc,
            _,
        ) = env.step(action)
        step += 1
        buffer.append((obs, action, next_obs, rew))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    return buffer
