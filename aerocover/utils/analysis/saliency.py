from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import torch


def _grad_dqn(model, obs):
    device = next(model.q_net.parameters()).device
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad_(True)
    q_vals = model.q_net(x)
    q_vals.max().backward()
    return x.grad

def _grad_sb3_ac(model, obs):
    policy = model.policy
    obs_tensor = policy.obs_to_tensor(obs)[0]
    obs_tensor.requires_grad_(True)
    dist = policy.get_distribution(obs_tensor)
    dist.distribution.logits.max().backward()
    return obs_tensor.grad

def _grad_sb3_continuous(model, obs):
    policy = model.policy
    obs_tensor = policy.obs_to_tensor(obs)[0]
    obs_tensor.requires_grad_(True)
    action = policy.actor(obs_tensor)
    action.abs().sum().backward()
    return obs_tensor.grad

def compute_saliency(model, obs_samples, method):
    grads = []

    for obs in obs_samples:
        try:
            if method == "dqn":
                grad = _grad_dqn(model, obs)
            elif method == "sb3_ac":
                grad = _grad_sb3_ac(model, obs)
            elif method == "sb3_continuous":
                grad = _grad_sb3_continuous(model, obs)
            else:
                continue

            grad_np = grad.detach().abs().squeeze(0).cpu().numpy()
            grads.append(grad_np)
        except Exception:
            continue

    if not grads:
        raise ValueError("No gradients computed")

    return np.stack(grads).mean(axis=0)

def compute_mappo_critic_saliency(model, team_states: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    states = torch.as_tensor(team_states, dtype=torch.float32, device=device)
    states.requires_grad_(True)
    values = model.value(states)
    grads = torch.autograd.grad(values.sum(), states)[0]
    return grads.detach().abs().mean(dim=0).cpu().numpy()

def mappo_team_obs_labels(
    obs_dim: int,
    n_agents: int,
    n_landmarks: int = 2,
) -> List[str]:
    labels_per_agent = ["vel_x", "vel_y", "pos_x", "pos_y"]
    for landmark_idx in range(n_landmarks):
        labels_per_agent.extend([f"lm{landmark_idx}_rel_x", f"lm{landmark_idx}_rel_y"])
    labels_per_agent.extend(["other_rel_x", "other_rel_y", "comm_0", "comm_1"])
    labels_per_agent = labels_per_agent[:obs_dim]
    return [
        f"A{agent_idx}_{label}"
        for agent_idx in range(n_agents)
        for label in labels_per_agent
    ]

def saliency_group_means(labels: Iterable[str], saliency: Iterable[float]) -> Dict[str, float]:
    groups = defaultdict(list)
    for label, value in zip(labels, saliency):
        if "vel" in label:
            group = "velocity"
        elif "pos" in label and "rel" not in label:
            group = "self_pos"
        elif "lm" in label:
            group = "landmark"
        elif "other" in label:
            group = "other_agent"
        elif "comm" in label:
            group = "communication"
        else:
            group = "?"
        groups[group].append(float(value))

    return {group: float(np.mean(values)) for group, values in groups.items()}
