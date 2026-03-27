
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple
import math
import numpy as np

State  = Hashable
Action = Hashable



@dataclass
class TDConfig:
    gamma: float = 0.90
    alpha: float = 0.10
    epsilon: float = 0.15
    lam: float = 0.80          # lambda for eligibility traces
    n_step: int = 4            # n for forward-view n-step methods
    episodes: int = 2000
    max_steps: int = 50
    seed: int = 0
    exploration: str = "epsilon_greedy"  # "epsilon_greedy" | "boltzmann" | "ucb"
    temperature: float = 1.0   # for Boltzmann
    ucb_c: float = 2.0         # for UCB



def _epsilon_greedy(rng, Q_s, actions, epsilon):
    if rng.random() < epsilon:
        return actions[rng.integers(len(actions))]
    best_v = -float("inf")
    best = []
    for a in actions:
        v = Q_s.get(a, 0.0)
        if v > best_v:
            best_v = v
            best = [a]
        elif v == best_v:
            best.append(a)
    return best[rng.integers(len(best))]


def _boltzmann(rng, Q_s, actions, temperature):
    qs = np.array([Q_s.get(a, 0.0) for a in actions])
    qs = qs - qs.max()                         # numerical stability
    probs = np.exp(qs / max(temperature, 1e-8))
    probs /= probs.sum()
    return actions[rng.choice(len(actions), p=probs)]


def _ucb(rng, Q_s, actions, visit_counts, total_steps, c):
    best_v = -float("inf")
    best = []
    for a in actions:
        n_sa = visit_counts.get(a, 0)
        q = Q_s.get(a, 0.0)
        if n_sa == 0:
            return a                            # explore unvisited
        bonus = c * math.sqrt(math.log(max(total_steps, 1)) / n_sa)
        v = q + bonus
        if v > best_v:
            best_v = v
            best = [a]
        elif v == best_v:
            best.append(a)
    return best[rng.integers(len(best))]


def select_action(rng, Q_s, actions, cfg: TDConfig,
                  visit_counts=None, total_steps=0):
    if cfg.exploration == "boltzmann":
        return _boltzmann(rng, Q_s, actions, cfg.temperature)
    if cfg.exploration == "ucb":
        return _ucb(rng, Q_s, actions, visit_counts or {}, total_steps, cfg.ucb_c)
    return _epsilon_greedy(rng, Q_s, actions, cfg.epsilon)


def _sample_next(rng, trans_list):
    probs = np.array([p for p, _ in trans_list], dtype=float)
    probs /= probs.sum()
    idx = rng.choice(len(trans_list), p=probs)
    return trans_list[idx][1]


def _greedy_policy(Q, actions):
    policy = {}
    for s, q_s in Q.items():
        best_v = -float("inf")
        best_a = actions[0]
        for a in actions:
            v = q_s.get(a, 0.0)
            if v > best_v:
                best_v = v
                best_a = a
        policy[s] = best_a
    return policy


def train_monte_carlo(
    states, actions, transitions, rewards, cfg: TDConfig,
    first_visit: bool = True
):

    rng = np.random.default_rng(cfg.seed)
    states_list = list(states)
    actions_list = list(actions)

    Q: Dict[State, Dict[Action, float]] = {}
    N: Dict[State, Dict[Action, int]]   = {}    # visit counts
    episode_returns: List[float] = []

    for ep in range(cfg.episodes):
        s = states_list[rng.integers(len(states_list))]
        trajectory: List[Tuple[State, Action, float]] = []

        for _ in range(cfg.max_steps):
            if s not in transitions:
                break
            Q_s = Q.setdefault(s, {})
            a = select_action(rng, Q_s, actions_list, cfg)
            trans = transitions[s].get(a)
            if not trans:
                break
            r = rewards.get(s, {}).get(a, 0.0)
            trajectory.append((s, a, r))
            s = _sample_next(rng, trans)

        G = 0.0
        visited = set()
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = cfg.gamma * G + r_t

            if first_visit and (s_t, a_t) in visited:
                continue
            visited.add((s_t, a_t))

            N.setdefault(s_t, {})
            N[s_t][a_t] = N[s_t].get(a_t, 0) + 1
            Q.setdefault(s_t, {})
            old = Q[s_t].get(a_t, 0.0)
            Q[s_t][a_t] = old + (G - old) / N[s_t][a_t]

        episode_returns.append(G)

    policy = _greedy_policy(Q, actions_list)
    return Q, policy, episode_returns

def train_td_n(
    states, actions, transitions, rewards, cfg: TDConfig
):
    """Forward-view n-step TD prediction with n-step bootstrapping."""
    rng = np.random.default_rng(cfg.seed)
    states_list = list(states)
    actions_list = list(actions)
    n = cfg.n_step

    V: Dict[State, float] = {s: 0.0 for s in states}
    # Need a Q for action selection; bootstrap from V
    Q: Dict[State, Dict[Action, float]] = {}
    episode_returns: List[float] = []

    _INF = cfg.max_steps + n + 10 

    for ep in range(cfg.episodes):
        s0 = states_list[rng.integers(len(states_list))]
        buf_s = [s0]
        buf_r: List[float] = []
        buf_a: List[Action] = []

        s = s0
        T_end = _INF
        t = 0

        while True:
            if t < T_end:
                if s not in transitions:
                    T_end = t
                else:
                    Q_s = Q.setdefault(s, {})
                    a = select_action(rng, Q_s, actions_list, cfg)
                    trans = transitions[s].get(a)
                    if not trans:
                        T_end = t
                    else:
                        r = rewards.get(s, {}).get(a, 0.0)
                        s_next = _sample_next(rng, trans)
                        buf_r.append(r)
                        buf_a.append(a)
                        buf_s.append(s_next)
                        Q.setdefault(s, {})[a] = r + cfg.gamma * V.get(s_next, 0.0)
                        s = s_next
                        if t + 1 >= cfg.max_steps:
                            T_end = t + 1

            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                upper = min(tau + n, T_end)
                for i in range(tau, min(upper, len(buf_r))):
                    G += (cfg.gamma ** (i - tau)) * buf_r[i]
                if tau + n < T_end and tau + n < len(buf_s):
                    G += (cfg.gamma ** n) * V.get(buf_s[tau + n], 0.0)

                s_tau = buf_s[tau]
                V[s_tau] = V.get(s_tau, 0.0) + cfg.alpha * (G - V.get(s_tau, 0.0))

            t += 1
            if tau >= T_end - 1:
                break

        episode_returns.append(sum(buf_r))

    policy = {}
    for s in states:
        if s not in transitions:
            continue
        best_a, best_v = None, -float("inf")
        for a in transitions[s]:
            q = rewards.get(s, {}).get(a, 0.0)
            for p, sp in transitions[s][a]:
                q += cfg.gamma * p * V.get(sp, 0.0)
            if q > best_v:
                best_v = q
                best_a = a
        if best_a is not None:
            policy[s] = best_a

    return V, policy, episode_returns

def train_td_lambda(
    states, actions, transitions, rewards, cfg: TDConfig
):
    """Backward-view TD(lambda) with accumulating eligibility traces."""    
    rng = np.random.default_rng(cfg.seed)
    states_list = list(states)
    actions_list = list(actions)

    V: Dict[State, float] = {s: 0.0 for s in states}
    Q: Dict[State, Dict[Action, float]] = {}   # for action selection
    episode_returns: List[float] = []

    for ep in range(cfg.episodes):
        s = states_list[rng.integers(len(states_list))]
        E: Dict[State, float] = {}              # eligibility traces
        G = 0.0

        for t in range(cfg.max_steps):
            if s not in transitions:
                break
            Q_s = Q.setdefault(s, {})
            a = select_action(rng, Q_s, actions_list, cfg)
            trans = transitions[s].get(a)
            if not trans:
                break

            r = rewards.get(s, {}).get(a, 0.0)
            s_next = _sample_next(rng, trans)

            delta = r + cfg.gamma * V.get(s_next, 0.0) - V.get(s, 0.0)

            # accumulating trace
            E[s] = E.get(s, 0.0) + 1.0

            # update all visited states
            for st in list(E.keys()):
                V[st] += cfg.alpha * delta * E[st]
                E[st] *= cfg.gamma * cfg.lam
                if E[st] < 1e-10:
                    del E[st]

            # refresh Q for action selection
            Q.setdefault(s, {})[a] = r + cfg.gamma * V.get(s_next, 0.0)

            G += r
            s = s_next

        episode_returns.append(G)

    # greedy policy from V
    policy = {}
    for s in states:
        if s not in transitions:
            continue
        best_a, best_v = None, -float("inf")
        for a in transitions[s]:
            q = rewards.get(s, {}).get(a, 0.0)
            for p, sp in transitions[s][a]:
                q += cfg.gamma * p * V.get(sp, 0.0)
            if q > best_v:
                best_v = q
                best_a = a
        if best_a is not None:
            policy[s] = best_a

    return V, policy, episode_returns

def train_sarsa_n(
    states, actions, transitions, rewards, cfg: TDConfig
):
    """Forward-view n-step SARSA (on-policy)."""
    rng = np.random.default_rng(cfg.seed)
    states_list = list(states)
    actions_list = list(actions)
    n = cfg.n_step

    Q: Dict[State, Dict[Action, float]] = {}
    episode_returns: List[float] = []

    _INF = cfg.max_steps + n + 10

    for ep in range(cfg.episodes):
        s0 = states_list[rng.integers(len(states_list))]
        Q_s0 = Q.setdefault(s0, {})
        a0 = select_action(rng, Q_s0, actions_list, cfg)

        buf_s = [s0]
        buf_a = [a0]
        buf_r: List[float] = []

        s, a = s0, a0
        T_end = _INF
        t = 0

        while True:
            if t < T_end:
                trans = transitions.get(s, {}).get(a)
                if trans is None:
                    T_end = t
                    buf_r.append(0.0)
                else:
                    r = rewards.get(s, {}).get(a, 0.0)
                    s_next = _sample_next(rng, trans)
                    buf_r.append(r)
                    buf_s.append(s_next)

                    if s_next not in transitions or t + 1 >= cfg.max_steps:
                        T_end = t + 1
                        buf_a.append(None)
                    else:
                        Q_sn = Q.setdefault(s_next, {})
                        a_next = select_action(rng, Q_sn, actions_list, cfg)
                        buf_a.append(a_next)
                        s, a = s_next, a_next

            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                upper = min(tau + n, T_end)
                for i in range(tau, min(upper, len(buf_r))):
                    G += (cfg.gamma ** (i - tau)) * buf_r[i]
                if tau + n < T_end and tau + n < len(buf_a) and buf_a[tau + n] is not None:
                    G += (cfg.gamma ** n) * Q.get(buf_s[tau + n], {}).get(buf_a[tau + n], 0.0)

                s_tau = buf_s[tau]
                a_tau = buf_a[tau]
                Q.setdefault(s_tau, {})
                old = Q[s_tau].get(a_tau, 0.0)
                Q[s_tau][a_tau] = old + cfg.alpha * (G - old)

            t += 1
            if tau >= T_end - 1:
                break

        episode_returns.append(sum(buf_r))

    policy = _greedy_policy(Q, actions_list)
    return Q, policy, episode_returns

def train_sarsa_lambda(
    states, actions, transitions, rewards, cfg: TDConfig
):
    """Backward-view SARSA(lambda) with accumulating eligibility traces (on-policy)."""
    rng = np.random.default_rng(cfg.seed)
    states_list = list(states)
    actions_list = list(actions)

    Q: Dict[State, Dict[Action, float]] = {}
    episode_returns: List[float] = []

    for ep in range(cfg.episodes):
        E: Dict[Tuple[State, Action], float] = {}
        s = states_list[rng.integers(len(states_list))]
        if s not in transitions:
            episode_returns.append(0.0)
            continue

        Q_s = Q.setdefault(s, {})
        a = select_action(rng, Q_s, actions_list, cfg)
        G = 0.0

        for t in range(cfg.max_steps):
            trans = transitions.get(s, {}).get(a)
            if trans is None:
                break

            r = rewards.get(s, {}).get(a, 0.0)
            s_next = _sample_next(rng, trans)

            Q_sn = Q.setdefault(s_next, {})
            if s_next in transitions:
                a_next = select_action(rng, Q_sn, actions_list, cfg)
            else:
                a_next = actions_list[rng.integers(len(actions_list))]

            q_sa = Q.get(s, {}).get(a, 0.0)
            q_next = Q.get(s_next, {}).get(a_next, 0.0)
            delta = r + cfg.gamma * q_next - q_sa

            # accumulating trace
            E[(s, a)] = E.get((s, a), 0.0) + 1.0

            for (st, at) in list(E.keys()):
                Q.setdefault(st, {})
                old = Q[st].get(at, 0.0)
                Q[st][at] = old + cfg.alpha * delta * E[(st, at)]
                E[(st, at)] *= cfg.gamma * cfg.lam
                if E[(st, at)] < 1e-10:
                    del E[(st, at)]

            G += r
            s, a = s_next, a_next

        episode_returns.append(G)

    policy = _greedy_policy(Q, actions_list)
    return Q, policy, episode_returns