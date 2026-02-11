from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class MDPState:
    a1_cell: int
    a2_cell: int
    covered_mask: int  # bit i = landmark i covered

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def clamp_vec(v, lo=-1.0, hi=1.0):
    v = np.asarray(v, dtype=float).copy()
    v[0] = clamp(float(v[0]), lo, hi)
    v[1] = clamp(float(v[1]), lo, hi)
    return v

def pos_to_cell(x: float, y: float, grid: int, lo: float = -1.0, hi: float = 1.0) -> int:
    x = clamp(x, lo, hi)
    y = clamp(y, lo, hi)
    ix = int((x - lo) / (hi - lo) * grid)
    iy = int((y - lo) / (hi - lo) * grid)
    ix = min(grid - 1, max(0, ix))
    iy = min(grid - 1, max(0, iy))
    return iy * grid + ix

def unpack_obs(obs: np.ndarray, n_landmarks: int, n_agents: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs).reshape(-1)
    self_pos = obs[2:4]
    start = 4
    lm_rel = obs[start : start + 2 * n_landmarks]
    start += 2 * n_landmarks
    other_rel = obs[start : start + 2 * (n_agents - 1)] if n_agents > 1 else np.array([])
    return self_pos, lm_rel, other_rel

def reconstruct_positions(obs_dict: Dict[str, np.ndarray], n_landmarks: int, n_agents: int):
    agents = sorted(obs_dict.keys())
    a0 = agents[0]
    a1 = agents[1] if len(agents) > 1 else None

    a0_pos, a0_lm_rel, a0_other_rel = unpack_obs(obs_dict[a0], n_landmarks, n_agents)

    landmarks: List[np.ndarray] = []
    for i in range(n_landmarks):
        rel = a0_lm_rel[2 * i : 2 * i + 2]
        landmarks.append(a0_pos + rel)

    agent_pos: Dict[str, np.ndarray] = {a0: a0_pos.copy()}

    if n_agents == 2 and a1 is not None and a0_other_rel.size >= 2:
        agent_pos[a1] = a0_pos + a0_other_rel[:2]
    elif a1 is not None:
        a1_pos, _, _ = unpack_obs(obs_dict[a1], n_landmarks, n_agents)
        agent_pos[a1] = a1_pos.copy()

    for k in list(agent_pos.keys()):
        agent_pos[k] = clamp_vec(agent_pos[k])

    landmarks = [clamp_vec(lm) for lm in landmarks]

    return agent_pos, landmarks

def compute_covered_mask(agent_pos: Dict[str, np.ndarray], landmarks: List[np.ndarray], cover_dist: float) -> int:
    mask = 0
    for i, lm in enumerate(landmarks):
        for p in agent_pos.values():
            if np.linalg.norm(p - lm) <= cover_dist:
                mask |= (1 << i)
                break
    return mask

def discretize_state(
    obs_dict: Dict[str, np.ndarray],
    n_landmarks: int,
    n_agents: int,
    grid: int,
    cover_dist: float
) -> MDPState:
    agent_pos, landmarks = reconstruct_positions(obs_dict, n_landmarks, n_agents)
    agents_sorted = sorted(agent_pos.keys())
    a0 = agents_sorted[0]
    a1 = agents_sorted[1] if len(agents_sorted) > 1 else agents_sorted[0]

    c0 = pos_to_cell(float(agent_pos[a0][0]), float(agent_pos[a0][1]), grid=grid)
    c1 = pos_to_cell(float(agent_pos[a1][0]), float(agent_pos[a1][1]), grid=grid)
    m = compute_covered_mask(agent_pos, landmarks, cover_dist=cover_dist)

    return MDPState(a1_cell=c0, a2_cell=c1, covered_mask=m)