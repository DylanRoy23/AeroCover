from __future__ import annotations
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aerocover.env_adapters.mpe_state import MDPState

def plot_value_function_heatmap(
    mdp,
    grid_size: int = 7,
    covered_mask: int = 0,
    agent_idx: int = 0,
    fixed_other_cell: int = 0,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    values = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell = i * grid_size + j
            
            if agent_idx == 0:
                state = MDPState(a1_cell=cell, a2_cell=fixed_other_cell, covered_mask=covered_mask)
            else:
                state = MDPState(a1_cell=fixed_other_cell, a2_cell=cell, covered_mask=covered_mask)
            
            values[i, j] = mdp.V.get(state, 0)
    
    im = ax.imshow(values, origin='lower', cmap='viridis', aspect='auto')
    ax.set_xlabel(f'Agent {agent_idx+1} x-position (cell)', fontsize=11)
    ax.set_ylabel(f'Agent {agent_idx+1} y-position (cell)', fontsize=11)
    
    # Format covered_mask as binary string
    mask_str = f"{covered_mask:02b}" if covered_mask < 4 else f"{covered_mask:b}"
    ax.set_title(f'Value Function (coverage={mask_str}, other_agent@cell_{fixed_other_cell})', 
                 fontsize=12, pad=10)
    
    plt.colorbar(im, ax=ax, label='Value')
    return ax

def plot_policy_directions(
    mdp,
    grid_size: int = 7,
    covered_mask: int = 0,
    agent_idx: int = 0,
    fixed_other_cell: int = 0,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Action to direction mapping
    action_to_dir = {
        0: (0, 0),      # no-op
        1: (-1, 0),     # left
        2: (1, 0),      # right
        3: (0, -1),     # down
        4: (0, 1)       # up
    }
    
    X, Y, U, V = [], [], [], []
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell = i * grid_size + j
            
            if agent_idx == 0:
                state = MDPState(a1_cell=cell, a2_cell=fixed_other_cell, covered_mask=covered_mask)
            else:
                state = MDPState(a1_cell=fixed_other_cell, a2_cell=cell, covered_mask=covered_mask)
            
            joint_action = mdp.policy.get(state, None)
            if joint_action is None:
                continue
            
            # Extract this agent's action from joint action
            agent_action = joint_action[agent_idx]
            dx, dy = action_to_dir.get(agent_action, (0, 0))
            
            X.append(j)
            Y.append(i)
            U.append(dx * 0.4)  # Scale for visibility
            V.append(dy * 0.4)
    
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xlabel(f'Agent {agent_idx+1} x-position (cell)', fontsize=11)
    ax.set_ylabel(f'Agent {agent_idx+1} y-position (cell)', fontsize=11)
    ax.set_title(f'Policy Directions (coverage={covered_mask:02b})', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax

def plot_coverage_comparison(
    learned_coverage: List[int],
    random_coverage: List[int],
    n_landmarks: int,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    timesteps = range(len(learned_coverage))
    
    ax.plot(timesteps, learned_coverage, label='Learned Policy', 
            linewidth=2, marker='o', markersize=4, markevery=5)
    ax.plot(timesteps, random_coverage, label='Random Policy', 
            linewidth=2, marker='s', markersize=4, markevery=5, alpha=0.7)
    
    ax.axhline(y=n_landmarks, color='green', linestyle='--', 
               label='Full Coverage', alpha=0.5)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Landmarks Covered', fontsize=12)
    ax.set_title('Coverage Performance: Learned vs Random Policy', fontsize=14, pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, n_landmarks + 0.2)
    
    return ax

def plot_value_iteration_convergence(
    mdp,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    values = list(mdp.V.values())
    
    ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('State Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of State Values', fontsize=14, pad=15)
    ax.axvline(x=np.mean(values), color='red', linestyle='--', 
               label=f'Mean: {np.mean(values):.2f}')
    ax.axvline(x=np.median(values), color='green', linestyle='--', 
               label=f'Median: {np.median(values):.2f}')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax

def plot_state_space_analysis(
    state_stats: Dict,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Visited', 'Unvisited']
    values = [
        state_stats['observed_states'],
        state_stats['total_possible_states'] - state_stats['observed_states']
    ]
    colors = ['#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=categories,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )
    
    ax.set_title(
        f"State Space Coverage\n"
        f"{state_stats['observed_states']:,} / {state_stats['total_possible_states']:,} states visited",
        fontsize=13,
        pad=20
    )
    
    return ax

def create_analysis_figure(
    mdp,
    learned_coverage: List[int],
    random_coverage: List[int],
    state_stats: Dict,
    n_landmarks: int,
    grid_size: int = 7
):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Coverage comparison
    ax1 = fig.add_subplot(gs[0, :2])
    plot_coverage_comparison(learned_coverage, random_coverage, n_landmarks, ax=ax1)
    
    # State space coverage
    ax2 = fig.add_subplot(gs[0, 2])
    plot_state_space_analysis(state_stats, ax=ax2)
    
    # Value function - no coverage
    ax3 = fig.add_subplot(gs[1, 0])
    plot_value_function_heatmap(mdp, grid_size=grid_size, covered_mask=0, 
                                agent_idx=0, ax=ax3)
    
    # Value function - full coverage
    ax4 = fig.add_subplot(gs[1, 1])
    full_mask = (2 ** n_landmarks) - 1  # All landmarks covered
    plot_value_function_heatmap(mdp, grid_size=grid_size, covered_mask=full_mask,
                                agent_idx=0, ax=ax4)
    
    # Value distribution
    ax5 = fig.add_subplot(gs[1, 2])
    plot_value_iteration_convergence(mdp, ax=ax5)
    
    fig.suptitle('AeroCover V1: Comprehensive Policy Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig