from __future__ import annotations
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from typing import Dict, List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from aerocover.env_adapters.mpe_state import (
    reconstruct_positions,
    compute_covered_mask
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
        obs, _ = env.reset(seed=seed)

        agents = env.agents[:]

        episode_coverage = []


        for step_idx in range(max_steps):

            actions = policy_fn(obs)

            if actions is None:
                sampled_actions = []
                for agent in agents:
                    act = env.action_space(agent).sample()
                    sampled_actions.append(act)
                actions = tuple(sampled_actions)

            action_dict = {}
            for idx, agent in enumerate(agents):
                action_dict[agent] = actions[idx]

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            agent_positions, landmark_positions = reconstruct_positions(
                next_obs,
                n_landmarks,
                n_agents
            )

            coverage_mask = compute_covered_mask(
                agent_positions,
                landmark_positions,
                cover_dist
            )

            covered_count = bin(coverage_mask).count("1")
            episode_coverage.append(covered_count)

            obs = next_obs

            done = all(terminations.values()) or all(truncations.values())
            if done:
                break

        env.close()

        mean_coverage = np.mean(episode_coverage)

        full_coverage_steps = 0
        for c in episode_coverage:
            if c == n_landmarks:
                full_coverage_steps += 1

        efficiency = full_coverage_steps / len(episode_coverage)

        coverages.append(mean_coverage)
        efficiencies.append(efficiency)

    return {
        "avg_coverage_mean": np.mean(coverages),
        "avg_coverage_std": np.std(coverages),
        "efficiency_mean": np.mean(efficiencies),
        "efficiency_std": np.std(efficiencies),
    }

def collect_coverage_trajectory(
    policy_fn: Callable,
    *,
    seed: int = 2026,
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_steps: int = 50,
    cover_dist: float = 0.30,
    continuous_actions: bool = False,
) -> List[int]:

    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.0,
        max_cycles=max_steps,
        continuous_actions=continuous_actions,
    )

    obs, _ = env.reset(seed=seed)
    agents = env.agents[:]

    coverage_over_time = []


    for step_idx in range(max_steps):

        actions = policy_fn(obs)

        if actions is None:
            sampled_actions = []
            for agent in agents:
                act = env.action_space(agent).sample()
                sampled_actions.append(act)
            actions = tuple(sampled_actions)

        action_dict = {}
        for idx, agent in enumerate(agents):
            action_dict[agent] = actions[idx]

        next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

        agent_positions, landmark_positions = reconstruct_positions(
            next_obs,
            n_landmarks,
            n_agents
        )

        coverage_mask = compute_covered_mask(
            agent_positions,
            landmark_positions,
            cover_dist
        )

        covered_count = bin(coverage_mask).count("1")
        coverage_over_time.append(covered_count)

        obs = next_obs

        done = all(terminations.values()) or all(truncations.values())
        if done:
            break

    env.close()

    return coverage_over_time


def plot_learning_curves(
    results: Dict[str, List[float]],
    window: int = 50,
    figsize: Tuple[int, int] = (16, 5),
):
    n_methods = len(results)

    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=figsize,
        sharey=True
    )

    if n_methods == 1:
        axes = [axes]

    for ax, (name, returns) in zip(axes, results.items()):

        ax.plot(
            returns,
            alpha=0.15,
            color="steelblue"
        )

        if len(returns) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(returns, kernel, mode="valid")

            x_vals = range(window - 1, len(returns))

            ax.plot(
                x_vals,
                smoothed,
                color="darkblue",
                linewidth=2
            )

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Reward")

    plt.suptitle(
        "Training Reward Curves",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    return fig


def plot_coverage_comparison_v2(
    results: Dict[str, List[int]],
    n_landmarks: int = 2,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    for name, coverage in results.items():
        ax.plot(
            coverage,
            label=name,
            linewidth=2
        )

    ax.axhline(
        n_landmarks,
        linestyle="--",
        color="green",
        alpha=0.5,
        label="Full"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Covered")
    ax.set_title("Coverage Over Time")

    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

def collect_rollout_auto(
    policy_fn,
    is_continuous,
    seed=2026,
    n_agents=2,
    n_landmarks=2,
    max_steps=50,
):
    """Collect a rollout that works for both discrete and continuous policies."""
    env = simple_spread_v3.parallel_env(
        N=n_agents, local_ratio=0.0,
        max_cycles=max_steps, continuous_actions=is_continuous,
    )
    obs, _ = env.reset(seed=seed)
    agents = env.agents[:]

    traj = []
    for _ in range(max_steps):
        agent_pos, landmarks = reconstruct_positions(obs, n_landmarks, n_agents)
        traj.append((agent_pos, landmarks))

        actions = policy_fn(obs) if policy_fn else None

        if actions is None:
            action_dict = {a: env.action_space(a).sample() for a in agents}
        else:
            action_dict = {a: actions[i] for i, a in enumerate(agents)}

        obs, _, terms, truncs, _ = env.step(action_dict)
        if all(terms.values()) or all(truncs.values()):
            break

    env.close()
    return traj