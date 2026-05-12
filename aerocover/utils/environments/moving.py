from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from pettingzoo.mpe import simple_spread_v3

from aerocover.env_adapters.landmark_motion import (
    move_landmarks,
    reset_landmark_motion,
)


def apply_landmark_drift(env, rng: np.random.Generator, drift_speed: float) -> None:
    move_landmarks(env, rng, drift_speed)

def reset_landmark_drift(env, rng: np.random.Generator, drift_speed: float) -> None:
    reset_landmark_motion(env, rng, drift_speed)

def policy_actions(policy_fn: Callable, obs: Dict, env, agents: List[str]) -> Tuple:
    actions = policy_fn(obs) if policy_fn else None
    if actions is not None:
        return actions
    return tuple(env.action_space(agent).sample() for agent in agents)

def build_moving_sb3_env(
    n_agents: int,
    n_landmarks: int,
    max_steps: int,
    cover_dist: float,
    seed: int,
    n_envs: int = 1,
    continuous: bool = False,
    *,
    moving_landmarks: bool = True,
    drift_speed: float = 0.02,
):
    from aerocover.deep.sb3.env import make_sb3_env

    return make_sb3_env(
        n_agents=n_agents,
        n_landmarks=n_landmarks,
        max_steps=max_steps,
        cover_dist=cover_dist,
        seed=seed,
        n_envs=n_envs,
        continuous_actions=continuous,
        moving_landmarks=moving_landmarks,
        drift_speed=drift_speed,
    )

def make_mappo_env(
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_steps: int = 50,
    cover_dist: float = 0.30,
    seed: int = 42,
    shaped_reward: bool = True,
    moving_landmarks: bool = False,
    drift_speed: float = 0.02,
):
    from aerocover.deep.sb3.env import CoverageRewardWrapper, MovingLandmarksWrapper

    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.0,
        max_cycles=max_steps,
        continuous_actions=False,
        render_mode="rgb_array",
    )
    if moving_landmarks:
        env = MovingLandmarksWrapper(env, drift_speed=drift_speed, seed=seed)
    if shaped_reward:
        env = CoverageRewardWrapper(env, n_landmarks, n_agents, cover_dist)
    return env

def patch_rl_utils_build_env(rl_utils_module, config: Dict) -> Callable:
    original_build_env = rl_utils_module.build_env

    def build_env_with_moving(
        n_agents,
        n_landmarks,
        max_steps,
        cover_dist,
        seed,
        n_envs=1,
        continuous=False,
    ):
        return build_moving_sb3_env(
            n_agents=n_agents,
            n_landmarks=n_landmarks,
            max_steps=max_steps,
            cover_dist=cover_dist,
            seed=seed,
            n_envs=n_envs,
            continuous=continuous,
            moving_landmarks=config["moving_landmarks"],
            drift_speed=config["drift_speed"],
        )

    rl_utils_module.build_env = build_env_with_moving
    return original_build_env
