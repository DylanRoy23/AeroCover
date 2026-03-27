from __future__ import annotations
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from stable_baselines3 import DQN

from aerocover.deep.sb3_utils import (
    make_eval_policy_fn,
    CoverageLoggerCallback,
)

from .rl_utils import (
    build_env,
    train_model,
    build_return,
)



@dataclass
class DQNConfig:
    total_timesteps: int = 50_000
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    target_update_interval: int = 500
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05
    net_arch: List[int] = field(default_factory=lambda: [128, 128])
    seed: int = 42


def train_dqn(
    cfg: DQNConfig,
    n_agents: int = 2,
    n_landmarks: int = 2,
    cover_dist: float = 0.30,
    max_steps: int = 50,
    n_envs: int = 1,
) -> Tuple[DQN, List[float], Dict]:
    env = build_env(
        n_agents, 
        n_landmarks, 
        max_steps, 
        cover_dist, 
        cfg.seed, 
        n_envs
    )
    cb = CoverageLoggerCallback()

    model = DQN(
        "MlpPolicy", env,
        learning_rate=cfg.lr,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        policy_kwargs=dict(net_arch=cfg.net_arch),
        seed=cfg.seed,
        verbose=0,
    )
    train_model(model, cfg, cb)
    env.close()

    return model, cb.episode_rewards, build_return(model, env, continuous=False)


def dqn_policy_fn(model: DQN):
    return make_eval_policy_fn(model)