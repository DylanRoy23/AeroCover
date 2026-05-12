from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

ObsDict = Dict[str, np.ndarray]


class SharedActorCentralCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        hidden_sizes: List[int],
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_sizes = list(hidden_sizes)
        self.actor = mlp(obs_dim, hidden_sizes, n_actions)
        self.critic = mlp(state_dim, hidden_sizes, 1)

    def action_distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

def mlp(input_dim: int, hidden_sizes: List[int], output_dim: int) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_size))
        layers.append(nn.Tanh())
        prev_dim = hidden_size
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def ordered_agents(obs: ObsDict) -> List[str]:
    return sorted(obs.keys())

def stack_obs(obs: ObsDict, agents: List[str]) -> np.ndarray:
    return np.stack([np.asarray(obs[agent], dtype=np.float32) for agent in agents])

def global_state(agent_obs: np.ndarray) -> np.ndarray:
    return agent_obs.reshape(-1).astype(np.float32)

def collect_rollout(
    env,
    model: SharedActorCentralCritic,
    obs: ObsDict,
    agents: List[str],
    cfg: Any,
    device: torch.device,
    initial_episode_reward: float = 0.0,
) -> Dict:
    obs_buf: List[np.ndarray] = []
    state_buf: List[np.ndarray] = []
    action_buf: List[np.ndarray] = []
    logprob_buf: List[np.ndarray] = []
    reward_buf: List[float] = []
    done_buf: List[float] = []
    value_buf: List[float] = []
    episode_rewards: List[float] = []
    current_episode_reward = initial_episode_reward

    for _ in range(cfg.rollout_steps):
        agent_obs = stack_obs(obs, agents)
        state = global_state(agent_obs)

        obs_t = torch.as_tensor(agent_obs, dtype=torch.float32, device=device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            dist = model.action_distribution(obs_t)
            actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)
            value_t = model.value(state_t).squeeze(0)

        action_dict = {
            agent: int(actions_t[idx].cpu().item())
            for idx, agent in enumerate(agents)
        }
        next_obs, rewards, terminations, truncations, _ = env.step(action_dict)

        done = all(terminations.values()) or all(truncations.values())
        team_reward = float(np.mean(list(rewards.values())))
        current_episode_reward += team_reward

        obs_buf.append(agent_obs)
        state_buf.append(state)
        action_buf.append(actions_t.cpu().numpy().astype(np.int64))
        logprob_buf.append(log_probs_t.cpu().numpy())
        reward_buf.append(team_reward)
        done_buf.append(float(done))
        value_buf.append(float(value_t.cpu().item()))

        obs = next_obs
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs, _ = env.reset()

    last_agent_obs = stack_obs(obs, agents)
    last_state = global_state(last_agent_obs)
    with torch.no_grad():
        last_state_t = torch.as_tensor(
            last_state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        last_value = float(model.value(last_state_t).cpu().item())

    advantages, returns = compute_gae(
        rewards=np.asarray(reward_buf, dtype=np.float32),
        dones=np.asarray(done_buf, dtype=np.float32),
        values=np.asarray(value_buf, dtype=np.float32),
        last_value=last_value,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "states": np.asarray(state_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.int64),
        "log_probs": np.asarray(logprob_buf, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "last_obs": obs,
        "steps": len(reward_buf),
        "episode_rewards": episode_rewards,
        "current_episode_reward": current_episode_reward,
    }

def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = last_value

    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
        next_value = values[step]

    returns = advantages + values
    return advantages, returns.astype(np.float32)

def update_model(
    model: SharedActorCentralCritic,
    optimizer: Adam,
    rollout: Dict,
    cfg: Any,
    device: torch.device,
):
    obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=device)
    states = torch.as_tensor(rollout["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(
        rollout["log_probs"], dtype=torch.float32, device=device
    )
    advantages = torch.as_tensor(
        rollout["advantages"], dtype=torch.float32, device=device
    )
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n_steps, n_agents, obs_dim = obs.shape
    indices = np.arange(n_steps)

    for _ in range(cfg.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n_steps, cfg.minibatch_size):
            batch_idx = indices[start:start + cfg.minibatch_size]

            batch_obs = obs[batch_idx].reshape(-1, obs_dim)
            batch_actions = actions[batch_idx].reshape(-1)
            batch_old_log_probs = old_log_probs[batch_idx].reshape(-1)
            batch_advantages = (
                advantages[batch_idx]
                .unsqueeze(-1)
                .expand(-1, n_agents)
                .reshape(-1)
            )

            dist = model.action_distribution(batch_obs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - cfg.clip_range,
                1.0 + cfg.clip_range,
            ) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            values = model.value(states[batch_idx])
            value_loss = 0.5 * (returns[batch_idx] - values).pow(2).mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()