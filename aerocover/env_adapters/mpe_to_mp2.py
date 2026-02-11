from __future__ import annotations
import random
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from .mpe_state import MDPState, discretize_state, reconstruct_positions

def all_joint_actions(n_agents: int, per_agent_actions: int = 5) -> List[Tuple[int, ...]]:
    return list(product(range(per_agent_actions), repeat=n_agents))

def action_tuple_to_dict(agents: List[str], a: Tuple[int, ...]) -> Dict[str, int]:
    return {agent: int(a[i]) for i, agent in enumerate(agents)}

def biased_action_selection(
    obs_dict,
    n_landmarks: int,
    n_agents: int,
    joint_actions: List[Tuple[int, ...]],
    bias_prob: float = 0.3
):
    if random.random() > bias_prob:
        # Random action
        return random.choice(joint_actions)
    
    # Greedy: move agents toward nearest uncovered landmark
    agent_pos, landmarks = reconstruct_positions(obs_dict, n_landmarks, n_agents)
    agents_sorted = sorted(agent_pos.keys())
    
    # Action mapping: 0=noop, 1=left, 2=right, 3=down, 4=up
    action_deltas = {
        0: (0, 0),
        1: (-0.1, 0),   # left
        2: (0.1, 0),    # right
        3: (0, -0.1),   # down
        4: (0, 0.1)     # up
    }
    
    best_action = []
    for i, agent_name in enumerate(agents_sorted):
        agent_p = agent_pos[agent_name]
        
        # Find nearest landmark
        min_dist = float('inf')
        for lm in landmarks:
            dist = np.linalg.norm(agent_p - lm)
            min_dist = min(min_dist, dist)
        
        # Choose action that reduces distance most
        best_a = 0
        best_improvement = -float('inf')
        
        for a in range(5):
            dx, dy = action_deltas[a]
            new_pos = agent_p + np.array([dx, dy])
            
            # Check distance to nearest landmark from new position
            new_min_dist = float('inf')
            for lm in landmarks:
                dist = np.linalg.norm(new_pos - lm)
                new_min_dist = min(new_min_dist, dist)
            
            improvement = min_dist - new_min_dist
            if improvement > best_improvement:
                best_improvement = improvement
                best_a = a
        
        best_action.append(best_a)
    
    return tuple(best_action)

def build_mp2_tables_from_mpe(
    episodes: int = 400,
    max_steps: int = 50,
    grid: int = 7,
    cover_dist: float = 0.15,
    n_agents: int = 2,
    n_landmarks: int = 2,
    seed: int = 0,
    step_penalty: float = 0.01,
    coverage_bonus: float = 1.0,
):
    random.seed(seed)
    np.random.seed(seed)

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    reward_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.0,
        max_cycles=max_steps,
        continuous_actions=False
    )

    joint_actions = all_joint_actions(n_agents, 5)

    for ep in range(episodes):
        obs, infos = env.reset(seed=seed + ep)
        agents = env.agents[:]
        
        # Curriculum: Gradually increase bias toward coverage
        # Early episodes: mostly random (bias=0.1)
        # Late episodes: mostly greedy (bias=0.5)
        exploration_progress = ep / episodes
        bias_prob = 0.1 + 0.4 * exploration_progress  # 0.1 → 0.5

        for step_num in range(max_steps):
            s = discretize_state(obs, n_landmarks, n_agents, grid, cover_dist)

            # Use biased action selection
            a = biased_action_selection(
                obs, n_landmarks, n_agents, joint_actions, bias_prob=bias_prob
            )
            
            next_obs, rewards_dict, terminations, truncations, infos = env.step(
                action_tuple_to_dict(agents, a)
            )

            sp = discretize_state(next_obs, n_landmarks, n_agents, grid, cover_dist)

            # balcane reward structure
            newly_covered_bits = sp.covered_mask & (~s.covered_mask)
            newly_covered = bin(newly_covered_bits).count("1")
            total_covered = bin(sp.covered_mask).count("1")
            prev_covered = bin(s.covered_mask).count("1")
            
            # Base coverage with multiplier
            base_coverage_reward = 2.0 * coverage_bonus * total_covered
            if total_covered == n_landmarks:
                coverage_reward = base_coverage_reward * 1.5
            else:
                coverage_reward = base_coverage_reward
            
            # New coverage bonus
            new_coverage_bonus = 3.0 * coverage_bonus * newly_covered
            
            # Full coverage achievement bonus
            full_coverage_bonus = 0.0
            if total_covered == n_landmarks and prev_covered < n_landmarks:
                full_coverage_bonus = 5.0 * coverage_bonus
            
            # Loss penalty
            lost_coverage_bits = s.covered_mask & (~sp.covered_mask)
            lost_coverage = bin(lost_coverage_bits).count("1")
            loss_penalty = -2.0 * coverage_bonus * lost_coverage
            
            # Time penalty
            time_penalty = -step_penalty
            
            # Distance shaping
            team_r = float(sum(rewards_dict.values()))
            scaled_distance_penalty = 0.2 * team_r
            
            # Total reward
            r = (coverage_reward + 
                 new_coverage_bonus + 
                 full_coverage_bonus + 
                 loss_penalty + 
                 time_penalty + 
                 scaled_distance_penalty)

            counts[s][a][sp] += 1
            reward_sums[s][a][sp] += r

            obs = next_obs

            done = all(terminations.values()) or all(truncations.values())
            if done:
                break

    env.close()

    # Build complete observed state set
    observed_states = set()
    for s in counts.keys():
        observed_states.add(s)
        for a in counts[s].keys():
            for sp in counts[s][a].keys():
                observed_states.add(sp)

    states = list(observed_states)
    actions = joint_actions

    # Print exploration statistics
    states_by_mask = defaultdict(int)
    for s in states:
        states_by_mask[s.covered_mask] += 1
    
    print("\nExploration Statistics:")
    for mask in sorted(states_by_mask.keys()):
        count = states_by_mask[mask]
        mask_bin = f"{mask:0{n_landmarks}b}"
        n_covered = bin(mask).count("1")
        pct = (count / len(states)) * 100
        print(f"  Coverage {mask_bin} ({n_covered}/{n_landmarks}): {count} states ({pct:.1f}%)")

    # MP2 dict format
    transitions: Dict[MDPState, Dict[Tuple[int, ...], List[Tuple[float, MDPState]]]] = {}
    rewards: Dict[MDPState, Dict[Tuple[int, ...], float]] = {}

    for s in states:
        if s not in counts:
            continue

        transitions[s] = {}
        rewards[s] = {}

        for a, next_counts in counts[s].items():
            total = sum(next_counts.values())
            if total == 0:
                continue

            trans_list: List[Tuple[float, MDPState]] = []
            exp_r = 0.0

            for sp, c in next_counts.items():
                p = c / total
                trans_list.append((p, sp))

                mean_r = reward_sums[s][a][sp] / c
                exp_r += p * mean_r

            transitions[s][a] = trans_list
            rewards[s][a] = exp_r

    return states, actions, transitions, rewards