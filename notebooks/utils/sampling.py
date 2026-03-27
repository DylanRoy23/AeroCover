from pettingzoo.mpe import simple_spread_v3


def collect_observations(config, n_seeds=20, rollout_steps=10):
    env = simple_spread_v3.parallel_env(
        N=config["n_agents"],
        local_ratio=0.0,
        max_cycles=config["max_steps"],
        continuous_actions=False,
    )

    samples = []

    for seed in range(5000, 5000 + n_seeds):
        obs, _ = env.reset(seed=seed)

        # Initial observations
        for agent in sorted(obs):
            samples.append(obs[agent].copy())

        # Rollout
        for _ in range(rollout_steps):
            actions = {
                a: env.action_space(a).sample()
                for a in env.agents
            }
            obs, *_ = env.step(actions)

            for agent in sorted(obs):
                samples.append(obs[agent].copy())

    env.close()
    return samples