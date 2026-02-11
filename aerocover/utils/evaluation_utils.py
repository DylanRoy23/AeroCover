from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from aerocover.env_adapters.mpe_state import compute_covered_mask


def evaluate_coverage_over_time(
    traj: List[Tuple[Dict[str, np.ndarray], List[np.ndarray]]],
    n_landmarks: int,
    cover_dist: float
) -> List[int]:
    coverage_over_time = []
    for agent_pos, landmarks in traj:
        mask = compute_covered_mask(agent_pos, landmarks, cover_dist)
        coverage_over_time.append(bin(mask).count('1'))
    return coverage_over_time


def compute_trajectory_metrics(
    traj: List[Tuple[Dict[str, np.ndarray], List[np.ndarray]]],
    n_landmarks: int,
    cover_dist: float
) -> Dict[str, float]:
    coverage = evaluate_coverage_over_time(traj, n_landmarks, cover_dist)
    
    # Time to full coverage
    time_to_full = -1
    for t, c in enumerate(coverage):
        if c == n_landmarks:
            time_to_full = t
            break
    
    # Other metrics
    avg_coverage = np.mean(coverage)
    final_coverage = coverage[-1] if coverage else 0
    
    # Efficiency: what proportion of time was fully covered
    full_coverage_steps = sum(1 for c in coverage if c == n_landmarks)
    efficiency = full_coverage_steps / len(coverage) if coverage else 0
    
    return {
        'time_to_full_coverage': time_to_full,
        'avg_coverage': avg_coverage,
        'final_coverage': final_coverage,
        'coverage_efficiency': efficiency,
        'trajectory_length': len(coverage)
    }


def compare_policies(
    learned_traj: List,
    random_traj: List,
    n_landmarks: int,
    cover_dist: float
) -> Dict[str, Dict[str, float]]:
    learned_metrics = compute_trajectory_metrics(learned_traj, n_landmarks, cover_dist)
    random_metrics = compute_trajectory_metrics(random_traj, n_landmarks, cover_dist)
    
    return {
        'learned': learned_metrics,
        'random': random_metrics,
        'improvement': {
            'time_to_full': (
                (random_metrics['time_to_full_coverage'] - learned_metrics['time_to_full_coverage']) 
                / max(random_metrics['time_to_full_coverage'], 1) * 100
                if random_metrics['time_to_full_coverage'] > 0 and learned_metrics['time_to_full_coverage'] > 0
                else 0
            ),
            'avg_coverage': (
                (learned_metrics['avg_coverage'] - random_metrics['avg_coverage'])
                / random_metrics['avg_coverage'] * 100
                if random_metrics['avg_coverage'] > 0
                else 0
            ),
            'efficiency': (
                (learned_metrics['coverage_efficiency'] - random_metrics['coverage_efficiency'])
                / max(random_metrics['coverage_efficiency'], 0.01) * 100
            )
        }
    }


def evaluate_state_space_coverage(
    states: List,
    grid_size: int,
    n_agents: int,
    n_landmarks: int
) -> Dict[str, float]:
    # Total possible states: grid^(2*n_agents) * 2^n_landmarks
    # For 2 agents on 7x7 grid with 2 landmarks: 7^4 * 4 = 9604
    cells_per_agent = grid_size * grid_size
    total_cells = cells_per_agent ** n_agents
    coverage_states = 2 ** n_landmarks
    total_possible = total_cells * coverage_states
    
    observed = len(states)
    
    # Unique coverage masks seen
    unique_masks = len(set(s.covered_mask for s in states))
    
    return {
        'total_possible_states': total_possible,
        'observed_states': observed,
        'coverage_percentage': (observed / total_possible) * 100,
        'unique_coverage_masks': unique_masks,
        'possible_coverage_masks': coverage_states
    }