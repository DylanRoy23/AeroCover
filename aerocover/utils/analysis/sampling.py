from __future__ import annotations

from typing import Dict

import numpy as np
from pettingzoo.mpe import simple_spread_v3

from aerocover.utils.environments import apply_landmark_drift, reset_landmark_drift


def collect_observations(config: Dict, n_seeds: int = 20, rollout_steps: int = 10):
    env = simple_spread_v3.parallel_env(
        N=config["n_agents"],
        local_ratio=0.0,
        max_cycles=config["max_steps"],
        continuous_actions=False,
    )

    samples = []
    moving = config.get("moving_landmarks", False)
    drift_speed = config.get("drift_speed", 0.02)

    for seed in range(5000, 5000 + n_seeds):
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        if moving:
            reset_landmark_drift(env, rng, drift_speed)

        for agent in sorted(obs):
            samples.append(obs[agent].copy())

        for _ in range(rollout_steps):
            if moving:
                apply_landmark_drift(env, rng, drift_speed)

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, *_ = env.step(actions)

            for agent in sorted(obs):
                samples.append(obs[agent].copy())

    env.close()
    return samples

def sample_team_states(
    config: Dict,
    n_samples: int = 200,
    seed: int = 99,
    drift_speed: float | None = None,
) -> np.ndarray:
    env = simple_spread_v3.parallel_env(
        N=config["n_agents"],
        local_ratio=0.0,
        max_cycles=config["max_steps"],
        continuous_actions=False,
    )
    rng = np.random.default_rng(seed)
    states = []
    drift = config.get("drift_speed", 0.02) if drift_speed is None else drift_speed

    while len(states) < n_samples:
        obs, _ = env.reset(seed=int(rng.integers(0, 10000)))
        if config.get("moving_landmarks", True):
            reset_landmark_drift(env, rng, drift)
        agents = env.agents[:]

        for _ in range(config["max_steps"]):
            if config.get("moving_landmarks", True):
                apply_landmark_drift(env, rng, drift)

            stacked = np.stack(
                [np.asarray(obs[agent], dtype=np.float32) for agent in sorted(obs)]
            )
            states.append(stacked.reshape(-1).astype(np.float32))

            if len(states) >= n_samples:
                break

            actions = {agent: env.action_space(agent).sample() for agent in agents}
            obs, _, terminations, truncations, _ = env.step(actions)

            if all(terminations.values()) or all(truncations.values()):
                break

    env.close()
    return np.stack(states[:n_samples])