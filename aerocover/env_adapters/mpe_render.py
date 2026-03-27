from __future__ import annotations
import random
from itertools import product
from typing import Callable, Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pettingzoo.mpe import simple_spread_v3
from aerocover.env_adapters.mpe_state import reconstruct_positions, discretize_state, MDPState

def all_joint_actions(n_agents: int, per_agent_actions: int = 5):
    return list(product(range(per_agent_actions), repeat=n_agents))

def action_tuple_to_dict(agents: List[str], a: Tuple) -> Dict:
    return {agent: a[i] for i, agent in enumerate(agents)}

def policy_from_state_action_map(
    policy: Dict[MDPState, Tuple[int, ...]],
    n_landmarks: int,
    n_agents: int,
    grid: int,
    cover_dist: float,
):
    def policy_fn(obs_dict):
        s = discretize_state(obs_dict, n_landmarks, n_agents, grid, cover_dist)
        return policy.get(s, None)
    return policy_fn

def collect_rollout(
    policy_fn: Optional[Callable[[Dict[str, np.ndarray]], Optional[Tuple[int, ...]]]] = None,
    seed: int = 0,
    max_steps: int = 50,
    n_agents: int = 2,
    n_landmarks: int = 2,
):
    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.0,
        max_cycles=max_steps,
        continuous_actions=False
    )

    obs, infos = env.reset(seed=seed)
    agents = env.agents[:]
    actions_all = all_joint_actions(n_agents, 5)

    traj = []
    for _ in range(max_steps):
        agent_pos, landmarks = reconstruct_positions(obs, n_landmarks, n_agents)
        traj.append((agent_pos, landmarks))

        if policy_fn is None:
            a = random.choice(actions_all)
        else:
            a = policy_fn(obs)
            if a is None:
                a = random.choice(actions_all)

        obs, rewards, terminations, truncations, infos = env.step(action_tuple_to_dict(agents, a))
        done = all(terminations.values()) or all(truncations.values())
        if done:
            break

    env.close()
    return traj

def _clamp_points(P, lo=-1.0, hi=1.0):
    P = np.asarray(P, dtype=float)
    P[:, 0] = np.clip(P[:, 0], lo, hi)
    P[:, 1] = np.clip(P[:, 1], lo, hi)
    return P

def _coverage_mask(agent_xy: np.ndarray, lm_xy: np.ndarray, cover_dist: float) -> int:
    mask = 0
    for i in range(lm_xy.shape[0]):
        dists = np.linalg.norm(agent_xy - lm_xy[i], axis=1)
        if np.any(dists <= cover_dist):
            mask |= (1 << i)
    return mask

def animate_traj(traj, cover_dist: float = 0.15, interval_ms: int = 120):
    all_positions = []
    for agent_pos, landmarks in traj:
        all_positions.extend(agent_pos.values())
        all_positions.extend(landmarks)
    
    positions_array = np.array(all_positions)
    margin = 0.2
    x_min = positions_array[:, 0].min() - margin
    x_max = positions_array[:, 0].max() + margin
    y_min = positions_array[:, 1].min() - margin
    y_max = positions_array[:, 1].max() + margin
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("AeroCover — MPE simple_spread rollout", fontsize=14, fontweight='bold')

    # Arena boundary for reference (true MPE arena is [-1, 1])
    ax.add_patch(plt.Rectangle(
        (-1, -1), 2, 2, 
        fill=False, 
        linewidth=1.5, 
        color='gray', 
        linestyle='--',
        alpha=0.5
    ))

    # Scatter plots for agents and landmarks
    agent_scat = ax.scatter([], [], s=150, c='blue', label='Agents', edgecolors='black', linewidths=1.5)
    lm_scat = ax.scatter([], [], marker="X", s=200, c='red', label='Landmarks', edgecolors='black', linewidths=1.5)
    
    # Info text box
    info = ax.text(
        0.02, 0.98, "", 
        transform=ax.transAxes, 
        va="top", 
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace'
    )

    # Coverage circles (created for each landmark)
    n_landmarks = len(traj[0][1]) if traj else 0
    circles = []
    for _ in range(n_landmarks):
        c = plt.Circle(
            (0, 0), 
            cover_dist, 
            fill=False, 
            alpha=0.4, 
            color='green', 
            linestyle=':',
            linewidth=2
        )
        ax.add_patch(c)
        circles.append(c)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    def init():
        agent_scat.set_offsets(np.zeros((0, 2)))
        lm_scat.set_offsets(np.zeros((0, 2)))
        info.set_text("")
        for c in circles:
            c.center = (0, 0)
        return [agent_scat, lm_scat, info, *circles]

    def update(frame):
        agent_pos, landmarks = traj[frame]
        agents_sorted = sorted(agent_pos.keys())

        # Get positions as arrays
        A = np.array([agent_pos[a] for a in agents_sorted], dtype=float)
        L = np.array(landmarks, dtype=float)

        # Clamp for plotting to prevent visual drift beyond arena
        A = _clamp_points(A, -1.0, 1.0)
        L = _clamp_points(L, -1.0, 1.0)

        # Update scatter plots
        agent_scat.set_offsets(A)
        lm_scat.set_offsets(L)

        # Update coverage circles to center on landmarks
        for i, c in enumerate(circles):
            c.center = (float(L[i, 0]), float(L[i, 1]))

        # Compute coverage
        mask = _coverage_mask(A, L, cover_dist)
        covered = bin(mask).count("1")
        
        # Format coverage mask as binary string
        mask_binary = f"{mask:0{n_landmarks}b}"
        
        # Update info text
        info.set_text(
            f"t = {frame:3d}\n"
            f"covered = {covered}/{n_landmarks}\n"
            f"mask = {mask_binary}"
        )

        return [agent_scat, lm_scat, info, *circles]

    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(traj), 
        init_func=init, 
        interval=interval_ms, 
        blit=False  # Changed to False for compatibility
    )
    
    plt.close(fig)
    return ani