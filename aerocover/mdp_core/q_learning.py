from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Tuple
import numpy as np

State = Hashable
Action = Hashable

@dataclass
class QLearningConfig:
    gamma: float = 0.90
    alpha: float = 0.20
    epsilon: float = 0.15
    episodes: int = 2000
    max_steps: int = 50
    seed: int = 0

def _epsilon_greedy_action(rng: np.random.Generator,
                          Q_s: Dict[Action, float],
                          actions: List[Action],
                          epsilon: float) -> Action:
    if rng.random() < epsilon:
        return actions[rng.integers(0, len(actions))]
    # Greedy with random tie-break
    best_val = None
    best_actions: List[Action] = []
    for a in actions:
        v = Q_s.get(a, 0.0)
        if best_val is None or v > best_val:
            best_val = v
            best_actions = [a]
        elif v == best_val:
            best_actions.append(a)
    return best_actions[rng.integers(0, len(best_actions))]

def _sample_next_state(rng: np.random.Generator,
                       trans_list: List[Tuple[float, State]]) -> State:
    # trans_list: [(prob, next_state), ...]
    probs = np.array([p for p, _ in trans_list], dtype=float)
    probs = probs / probs.sum()  # safety normalize
    idx = rng.choice(len(trans_list), p=probs)
    return trans_list[idx][1]

def train_q_learning(
    states: Iterable[State],
    actions: Iterable[Action],
    transitions: Dict[State, Dict[Action, List[Tuple[float, State]]]],
    rewards: Dict[State, Dict[Action, float]],
    cfg: QLearningConfig,
) -> Tuple[Dict[State, Dict[Action, float]], Dict[State, Action], List[float]]:
    rng = np.random.default_rng(cfg.seed)
    states_list = list(states)
    actions_list = list(actions)

    Q: Dict[State, Dict[Action, float]] = {}
    episode_returns: List[float] = []

    for _ep in range(cfg.episodes):
        s = states_list[rng.integers(0, len(states_list))]
        G = 0.0

        for _t in range(cfg.max_steps):
            # If we have no transitions from this state, end episode
            if s not in transitions:
                break

            Q_s = Q.setdefault(s, {})
            a = _epsilon_greedy_action(rng, Q_s, actions_list, cfg.epsilon)

            # If action never observed from this state, end (or you can skip/penalize)
            trans_sa = transitions[s].get(a)
            if not trans_sa:
                break

            r = rewards.get(s, {}).get(a, 0.0)
            s_next = _sample_next_state(rng, trans_sa)

            # Q-learning update
            Q_next = Q.get(s_next, {})
            max_next = 0.0
            if Q_next:
                max_next = max(Q_next.get(a2, 0.0) for a2 in actions_list)

            old = Q_s.get(a, 0.0)
            Q_s[a] = old + cfg.alpha * (r + cfg.gamma * max_next - old)

            G += r
            s = s_next

        episode_returns.append(G)

    # Greedy policy extraction
    policy_map: Dict[State, Action] = {}
    for s, Q_s in Q.items():
        # choose argmax_a Q(s,a) with tie-break
        best_val = None
        best_actions: List[Action] = []
        for a in actions_list:
            v = Q_s.get(a, 0.0)
            if best_val is None or v > best_val:
                best_val = v
                best_actions = [a]
            elif v == best_val:
                best_actions.append(a)
        if best_actions:
            policy_map[s] = best_actions[rng.integers(0, len(best_actions))]

    return Q, policy_map, episode_returns