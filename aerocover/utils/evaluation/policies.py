from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from pettingzoo.mpe import simple_spread_v3

from aerocover.env_adapters.mpe_state import compute_covered_mask, reconstruct_positions
from aerocover.utils.environments import (
    apply_landmark_drift,
    policy_actions,
    reset_landmark_drift,
)


def evaluate_deep_policy(
    policy_fn: Callable,
    n_episodes: int = 10,
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_steps: int = 50,
    cover_dist: float = 0.30,
    seed_start: int = 5000,
    continuous_actions: bool = False,
    moving_landmarks: bool = False,
    drift_speed: float = 0.02,
) -> Dict:
    coverages = []
    efficiencies = []

    for episode_idx in range(n_episodes):
        env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=0.0,
            max_cycles=max_steps,
            continuous_actions=continuous_actions,
        )

        seed = seed_start + episode_idx
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        if moving_landmarks:
            reset_landmark_drift(env, rng, drift_speed)
        agents = env.agents[:]
        episode_coverage = []

        for _ in range(max_steps):
            if moving_landmarks:
                apply_landmark_drift(env, rng, drift_speed)

            actions = policy_actions(policy_fn, obs, env, agents)
            next_obs, _, terminations, truncations, _ = env.step(
                {agent: actions[idx] for idx, agent in enumerate(agents)}
            )

            agent_positions, landmark_positions = reconstruct_positions(
                next_obs,
                n_landmarks,
                n_agents,
            )
            coverage_mask = compute_covered_mask(
                agent_positions,
                landmark_positions,
                cover_dist,
            )
            episode_coverage.append(bin(coverage_mask).count("1"))
            obs = next_obs

            if all(terminations.values()) or all(truncations.values()):
                break

        env.close()
        coverages.append(float(np.mean(episode_coverage)))
        full_coverage_steps = sum(c == n_landmarks for c in episode_coverage)
        efficiencies.append(full_coverage_steps / max(len(episode_coverage), 1))

    return {
        "avg_coverage_mean": float(np.mean(coverages)),
        "avg_coverage_std": float(np.std(coverages)),
        "efficiency_mean": float(np.mean(efficiencies)),
        "efficiency_std": float(np.std(efficiencies)),
    }

def evaluate_deep_policy_moving(
    policy_fn: Callable,
    n_episodes: int = 10,
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_steps: int = 50,
    cover_dist: float = 0.15,
    seed_start: int = 5000,
    continuous_actions: bool = False,
    drift_speed: float = 0.02,
    moving: bool = True,
) -> Dict:
    return evaluate_deep_policy(
        policy_fn,
        n_episodes=n_episodes,
        n_agents=n_agents,
        n_landmarks=n_landmarks,
        max_steps=max_steps,
        cover_dist=cover_dist,
        seed_start=seed_start,
        continuous_actions=continuous_actions,
        moving_landmarks=moving,
        drift_speed=drift_speed,
    )