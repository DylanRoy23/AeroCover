from __future__ import annotations
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from stable_baselines3 import PPO

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
class PPOConfig:
    total_timesteps: int = 80_000
    gamma: float = 0.95
    lr: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch_pi: List[int] = field(default_factory=lambda: [128, 128])
    net_arch_vf: List[int] = field(default_factory=lambda: [128, 128])
    seed: int = 42


def train_ppo(
    cfg: PPOConfig,
    n_agents: int = 2,
    n_landmarks: int = 2,
    cover_dist: float = 0.30,
    max_steps: int = 50,
    n_envs: int = 1,
) -> Tuple[PPO, List[float], Dict]:
    env = build_env(
        n_agents, 
        n_landmarks, 
        max_steps, 
        cover_dist, 
        cfg.seed, 
        n_envs
    )
    cb = CoverageLoggerCallback()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=cfg.lr,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=dict(net_arch=dict(pi=cfg.net_arch_pi, vf=cfg.net_arch_vf)),
        seed=cfg.seed,
        verbose=0,
    )
    
    train_model(model, cfg, cb)
    env.close()

    return model, cb.episode_rewards, build_return(model, env, continuous=False)


def ppo_policy_fn(model: PPO):
    return make_eval_policy_fn(model)