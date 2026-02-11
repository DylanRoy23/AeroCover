# AeroCover: Multi-Agent Coordination via Reinforcement Learning

**Author:** Dylan Roy  
**Course:** Reinforcement Learning — Spring 2026  
**Professor:** Dr. Alexander Lowenstein  
**Environment:** PettingZoo MPE `simple_spread_v3`

---

## Project Overview

AeroCover addresses the multi-agent coverage problem: autonomous agents must coordinate to efficiently cover multiple landmarks. The project progresses from classical tabular RL through deep RL to scalable graph-based architectures across four versions.

| Version | Focus | Status |
|---------|-------|--------|
| V1 | Tabular RL on empirical MDP | Complete |
| V2 | Deep Q-Networks (DQN) with continuous state | Planned |
| V3 | Multi-agent methods (MADDPG / QMIX) | Planned |
| V4 | Graph Neural Networks for scalable coordination | Planned |

## Repository Structure

```
aerocover/
├── env_adapters/            # Environment interface & state discretization
│   ├── mpe_state.py         # MDPState dataclass, discretization, coverage masks
│   ├── mpe_to_mp2.py        # Empirical MDP construction from MPE rollouts
│   └── mpe_render.py        # Rollout collection & trajectory animation
├── mdp_core/                # MDP solvers & RL algorithms
│   ├── mp2_mdp.py           # MarkovDecisionProcess: VI, PI, Q-table, save/load
│   ├── q_learning.py        # Tabular Q-learning with epsilon-greedy
│   └── td_methods.py        # MC, TD(n), TD(lambda), Sarsa(n), Sarsa(lambda), exploration strategies
├── utils/
│   ├── evaluation_utils.py  # Coverage metrics, trajectory analysis, policy comparison
│   └── notebook_helpers.py  # Printing utilities
└── viz/
    └── visualization_utils.py  # Heatmaps, policy arrows, convergence plots
notebooks/
└── v1_notebook.ipynb        # V1 main notebook (full pipeline & analysis)
requirements.txt
README.md
```

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/aerocover.git
cd aerocover

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install aerocover package in editable mode
pip install -e .

# Run the V1 notebook
jupyter notebook notebooks/v1_notebook.ipynb
```

**Python:** 3.10+  
**Key dependencies:** PettingZoo (MPE), NumPy, Matplotlib, Seaborn

## V1: Tabular Reinforcement Learning

### Problem Formulation

- **State:** `(agent1_cell, agent2_cell, coverage_mask)` — 2 agent grid positions + binary landmark coverage
- **Actions:** 25 joint actions (5 per agent × 2 agents: no-op, left, right, down, up)
- **Transitions:** Learned empirically from 600 episodes with curriculum exploration
- **Rewards:** Shaped coverage bonus + new discovery bonus + full coverage achievement − step penalty − coverage loss penalty

### Algorithms Implemented

**Dynamic Programming:**
- Value Iteration (converges in ~162 iterations)
- Policy Iteration (converges in ~8 iterations, 100% policy agreement with VI)

**Model-Free Methods:**
- First-Visit Monte Carlo
- Q-Learning (off-policy, tabular)
- TD(n) — forward-view n-step bootstrapping
- TD(lambda) — backward-view with eligibility traces
- Sarsa(n) — on-policy n-step
- Sarsa(lambda) — on-policy with eligibility traces

**Exploration Strategies:**
- Epsilon-greedy
- Boltzmann (softmax)
- Upper Confidence Bound (UCB)

### Training & Evaluation

```python
# Build empirical MDP
from aerocover.env_adapters.mpe_to_mp2 import build_mp2_tables_from_mpe
states, actions, transitions, rewards = build_mp2_tables_from_mpe(episodes=600, grid=5)

# Solve with value iteration
from aerocover.mdp_core.mp2_mdp import MarkovDecisionProcess
mdp = MarkovDecisionProcess(states, actions, transitions, rewards, gamma=0.9)
mdp.value_iteration()
mdp.save_best("best_policy_vi.pkl")

# Evaluate
from aerocover.env_adapters.mpe_render import collect_rollout, policy_from_state_action_map
policy_fn = policy_from_state_action_map(mdp.policy, n_landmarks=2, n_agents=2, grid=5, cover_dist=0.30)
traj = collect_rollout(policy_fn=policy_fn, seed=42)
```

### V1 Results Summary

Best method: Sarsa(4) with 0.306 avg coverage (~2.5× random baseline of 0.124). All tabular methods struggle due to discretization information loss, sparse joint-action coverage, and flat value landscapes. See the notebook for full analysis.

### Saved Artifacts

- `best_policy_vi.pkl` Value Iteration policy kernel + value function
- `best_policy_pi.pkl` Policy Iteration policy kernel + value function

## Roadmap

### V2: Deep Q-Networks
- Continuous state representation (no grid discretization)
- Experience replay buffer for sample efficiency
- Target network for stable training
- Per-agent DQN with shared reward signal

### V3: Multi-Agent Deep RL
- MADDPG or QMIX for centralized training / decentralized execution
- Proper credit assignment via agent-specific critics
- Continuous action spaces

### V4: Graph Neural Networks
- Agent-landmark interaction graphs
- Message-passing for scalable coordination
- Variable team sizes and landmark counts

## References & Citations

**Algorithms:**
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

**Environment:**
- Terry, J. K., et al. (2021). PettingZoo: Gym for Multi-Agent Reinforcement Learning. *NeurIPS 2021*.
- Mordatch, I., & Abbeel, P. (2018). Emergence of Grounded Compositional Language in Multi-Agent Populations. *AAAI 2018*.

**Tools:** Python 3.x, NumPy, Matplotlib, PettingZoo, Seaborn

**Collaborators:** Discussion with course peers on reward shaping approaches.

**AI Assistance:** Claude (Anthropic) used for code review and documentation assistance.
