from __future__ import annotations

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam

from aerocover.deep.mappo.core import (
    ObsDict,
    SharedActorCentralCritic,
    collect_rollout,
    ordered_agents,
    select_device,
    stack_obs,
    update_model,
)
from aerocover.utils.environments import make_mappo_env


@dataclass
class MAPPOConfig:
    total_timesteps: int = 200_000
    rollout_steps: int = 512
    gamma: float = 0.95
    gae_lambda: float = 0.95
    lr: float = 3e-4
    n_epochs: int = 4
    minibatch_size: int = 128
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    seed: int = 42
    device: str = "auto"
    log_interval: int = 5

def train_mappo(
    cfg: MAPPOConfig,
    n_agents: int = 2,
    n_landmarks: int = 2,
    cover_dist: float = 0.30,
    max_steps: int = 50,
    shaped_reward: bool = True,
    moving_landmarks: bool = False,
    drift_speed: float = 0.02,
) -> Tuple[SharedActorCentralCritic, List[float], Dict]:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = select_device(cfg.device)
    env = make_mappo_env(
        n_agents=n_agents,
        n_landmarks=n_landmarks,
        max_steps=max_steps,
        cover_dist=cover_dist,
        seed=cfg.seed,
        shaped_reward=shaped_reward,
        moving_landmarks=moving_landmarks,
        drift_speed=drift_speed,
    )

    obs, _ = env.reset(seed=cfg.seed)
    agents = ordered_agents(obs)
    obs_dim = int(np.asarray(obs[agents[0]]).shape[0])
    state_dim = obs_dim * len(agents)
    n_actions = int(env.action_space(agents[0]).n)

    model = SharedActorCentralCritic(
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_sizes=cfg.hidden_sizes,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    episode_rewards: List[float] = []
    current_episode_reward = 0.0
    timesteps = 0
    rollout_idx = 0

    while timesteps < cfg.total_timesteps:
        rollout = collect_rollout(
            env=env,
            model=model,
            obs=obs,
            agents=agents,
            cfg=cfg,
            device=device,
            initial_episode_reward=current_episode_reward,
        )
        obs = rollout["last_obs"]
        timesteps += rollout["steps"] * len(agents)
        episode_rewards.extend(rollout["episode_rewards"])
        current_episode_reward = rollout["current_episode_reward"]
        update_model(model, optimizer, rollout, cfg, device)
        rollout_idx += 1

        if cfg.log_interval > 0 and rollout_idx % cfg.log_interval == 0:
            recent = episode_rewards[-20:] if episode_rewards else [0.0]
            print(
                f"  Rollout {rollout_idx:4d} | "
                f"Frames {timesteps:7d}/{cfg.total_timesteps} | "
                f"Episodes {len(episode_rewards):4d} | "
                f"Mean reward (last 20): {np.mean(recent):.2f}"
            )

    env.close()

    info = {
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "n_agents": len(agents),
        "config": asdict(cfg),
        "unfinished_episode_reward": current_episode_reward,
        "n_rollouts": rollout_idx,
        "n_episodes_completed": len(episode_rewards),
    }
    return model, episode_rewards, info

def mappo_policy_fn(
    model: SharedActorCentralCritic,
    deterministic: bool = True,
    device: str = "auto",
) -> Callable[[ObsDict], Tuple[int, ...]]:
    eval_device = select_device(device)
    model.to(eval_device)
    model.eval()

    def policy_fn(obs_dict: ObsDict) -> Tuple[int, ...]:
        agents = ordered_agents(obs_dict)
        obs = torch.as_tensor(
            stack_obs(obs_dict, agents),
            dtype=torch.float32,
            device=eval_device,
        )
        with torch.no_grad():
            logits = model.actor(obs)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)
                actions = dist.sample()
        return tuple(int(action) for action in actions.cpu().numpy())

    return policy_fn

def save_mappo(
    model: SharedActorCentralCritic,
    path: str | Path,
    info: Dict | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": model.obs_dim,
            "state_dim": model.state_dim,
            "n_actions": model.n_actions,
            "hidden_sizes": model.hidden_sizes,
            "info": info or {},
        },
        path,
    )

def load_mappo(
    path: str | Path,
    device: str = "auto",
) -> Tuple[SharedActorCentralCritic, Dict]:
    eval_device = select_device(device)
    checkpoint = torch.load(Path(path), map_location=eval_device, weights_only=False)
    model = SharedActorCentralCritic(
        obs_dim=int(checkpoint["obs_dim"]),
        state_dim=int(checkpoint["state_dim"]),
        n_actions=int(checkpoint["n_actions"]),
        hidden_sizes=list(checkpoint["hidden_sizes"]),
    ).to(eval_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("info", {})