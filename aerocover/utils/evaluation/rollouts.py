from __future__ import annotations

from typing import Callable, List

import numpy as np
from pettingzoo.mpe import simple_spread_v3

from aerocover.env_adapters.mpe_state import compute_covered_mask, reconstruct_positions
from aerocover.utils.environments import (
    apply_landmark_drift,
    policy_actions,
    reset_landmark_drift,
)


def collect_coverage_trajectory(
    policy_fn: Callable,
    *,
    seed: int = 2026,
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_steps: int = 50,
    cover_dist: float = 0.30,
    continuous_actions: bool = False,
    moving_landmarks: bool = False,
    drift_speed: float = 0.02,
) -> List[int]:
    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.0,
        max_cycles=max_steps,
        continuous_actions=continuous_actions,
    )

    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    if moving_landmarks:
        reset_landmark_drift(env, rng, drift_speed)
    agents = env.agents[:]
    coverage_over_time = []

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
        coverage_over_time.append(bin(coverage_mask).count("1"))
        obs = next_obs

        if all(terminations.values()) or all(truncations.values()):
            break

    env.close()
    return coverage_over_time

def collect_coverage_trajectory_moving(
    policy_fn: Callable,
    seed: int = 2026,
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_steps: int = 50,
    cover_dist: float = 0.15,
    continuous_actions: bool = False,
    drift_speed: float = 0.02,
    moving: bool = True,
) -> List[int]:
    return collect_coverage_trajectory(
        policy_fn,
        seed=seed,
        n_agents=n_agents,
        n_landmarks=n_landmarks,
        max_steps=max_steps,
        cover_dist=cover_dist,
        continuous_actions=continuous_actions,
        moving_landmarks=moving,
        drift_speed=drift_speed,
    )

def collect_rollout_auto(
    policy_fn,
    is_continuous,
    seed=2026,
    n_agents=2,
    n_landmarks=2,
    max_steps=50,
    moving_landmarks=False,
    drift_speed=0.02,
):
    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.0,
        max_cycles=max_steps,
        continuous_actions=is_continuous,
    )
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    if moving_landmarks:
        reset_landmark_drift(env, rng, drift_speed)
    agents = env.agents[:]

    traj = []
    for _ in range(max_steps):
        agent_pos, landmarks = reconstruct_positions(obs, n_landmarks, n_agents)
        traj.append((agent_pos, landmarks))

        if moving_landmarks:
            apply_landmark_drift(env, rng, drift_speed)

        actions = policy_actions(policy_fn, obs, env, agents)
        action_dict = {agent: actions[idx] for idx, agent in enumerate(agents)}
        obs, _, terms, truncs, _ = env.step(action_dict)

        if all(terms.values()) or all(truncs.values()):
            break

    env.close()
    return traj