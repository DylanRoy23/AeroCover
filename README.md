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
| V2 | Deep RL (DQN, PPO, SAC, etc.) with continuous state | Complete |
| V3 | Multi-agent methods (MADDPG / QMIX) | Planned |
| V4 | Graph Neural Networks for scalable coordination | Planned |

## Repository Structure
```
aerocover/
├── env_adapters/ # Environment interface & state discretization
│   ├── mpe_state.py # MDPState dataclass, discretization, coverage masks
│   ├── mpe_to_mp2.py # Empirical MDP construction from MPE rollouts
│   └── mpe_render.py # Rollout collection & trajectory animation
├── tabular/  # V1: Classical RL algorithms
│   ├── mp2_mdp.py # MarkovDecisionProcess: VI, PI, Q-table, save/load
│   ├── q_learning.py # Tabular Q-learning with epsilon-greedy
│   └── td_methods.py # MC, TD(n), TD(λ), Sarsa(n), Sarsa(λ), exploration
├── deep/ # V2: Deep RL via Stable Baselines3
│   ├── sb3_utils.py # PettingZoo→SB3 bridge, reward shaping, callbacks
│   ├── rl_utils.py # Shared builders (env, noise, train, checkpoint)
│   ├── dqn.py # DQN
│   ├── ppo.py # PPO
│   ├── td3.py # TD3
│   └── sac.py # SAC
├── utils/
│   ├── evaluation_utils.py # Coverage metrics, trajectory analysis
│   ├── deep_eval_utils.py # V2 evaluation, coverage trajectories, rollout helpers
│   ├── notebook_helpers.py # Printing utilities
│   ├── saliency.py # Gradient-based saliency analysis
│   ├── sampling.py # Observation collection for saliency
│   └── plotting.py # Saliency heatmaps and grouped bar charts
└── viz/
    └── visualization_utils.py # V1 heatmaps, policy arrows, convergence plots
scripts/
├── replay_manager.py # Replay buffer storage, cleanup, size enforcement
└── checkpoint_manager.py # Model checkpointing with hyperparameter logging
docs/
└── technical-challenges.md # Bugs, surprises, stuck points
notebooks/
├── v1_notebook.ipynb # V1 full pipeline & analysis
└── v2_notebook.ipynb # V2 deep RL training, evaluation, saliency
checkpoints/ # Organized by algo/task/tag with config.json
replay/ # Organized by algo/task with metadata.json
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
**Key dependencies:** PettingZoo (MPE), Stable-Baselines3, sb3-contrib, SuperSuit, PyTorch, NumPy, Matplotlib, Seaborn

## V1: Tabular Reinforcement Learning

### Problem Formulation

- **State:** `(agent1_cell, agent2_cell, coverage_mask)` — 2 agent grid positions + binary landmark coverage
- **Actions:** 25 joint actions (5 per agent × 2 agents: no-op, left, right, down, up)
- **Transitions:** Learned empirically from 600 episodes with curriculum exploration
- **Rewards:** Shaped coverage bonus + new discovery bonus + full coverage achievement − step penalty − coverage loss penalty

## V2: Deep Reinforcement Learning

### Motivation

V1's best tabular method (Sarsa(4), avg coverage 0.31) hit a ceiling due to discretization information loss and sparse joint-action coverage. V2 eliminates
discretization entirely, operating on raw 10-dimensional continuous observations via neural network function approximation.

### Algorithms Implemented

**Value-Based (Discrete Actions):**
- DQN with experience replay and target network

**Actor-Critic (Discrete Actions):**
- PPO with clipped surrogate objective

**Actor-Critic (Continuous Actions):**
- TD3 (twin critics, delayed policy updates)
- SAC (maximum entropy framework)

### Key Design Choices

- **SuperSuit** bridges PettingZoo parallel envs to SB3 VecEnv, giving parameter sharing across agents for free
- **Shaped reward** (coverage bonus + discovery bonus + loss penalty) identical to V1 for fair cross-version comparison
- **cover_dist=0.15** (tightened from V1's 0.30) makes the task harder and better differentiates methods

## Algorithm Justification

### Why four methods?

The AeroCover environment has properties that stress-test different algorithm families differently, making a comprehensive comparison scientifically valuable:

**Off-policy value-based (DQN):** The coverage task has sparse rewards so agents must reach precise positions within cover_dist=0.15. Replay
buffers let these methods learn from rare successes repeatedly, which proved decisive. DQN achieved the highest coverage overall.

**On-policy actor-critic (PPO):** This os a standard baseline for any RL task. PPO's stability makes it a natural choice. Their underperformance here (vs DQN) is informative: on-policy methods discard data after each update, which is wasteful when coverage events are rare.

**Continuous-action (TD3, SAC):** The MPE environment supports continuous actions, allowing smoother agent movement. SAC and TD3 both performed strongly.


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
from aerocover.tabular.mp2_mdp import MarkovDecisionProcess
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

### V3: Multi-Agent Deep RL (Planned)
- MADDPG: Centralized critic, decentralized actors
- QMIX: Factored Q with monotonicity constraints
- Communication protocols: learned message-passing
- Moving landmarks: dynamic coverage zones

### V4: Graph Neural Networks (Planned)
- Agent-landmark interaction graphs
- Message-passing for scalable coordination
- Variable team sizes and landmark counts

## References & Citations

**Algorithms:**
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature* 518.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *arXiv:1802.09477*. (TD3)
- Haarnoja et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with Stochastic Actor. *arXiv:1801.01290*.
- Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *arXiv:1506.02438*.

**Libraries:**
- Raffin et al. (2021). Stable-Baselines3. *JMLR* 22(268).
- Terry et al. (2021). PettingZoo. *NeurIPS Datasets & Benchmarks*.
- SuperSuit (PettingZoo wrapper library).

**Environment:**
- Terry, J. K., et al. (2021). PettingZoo: Gym for Multi-Agent Reinforcement Learning. *NeurIPS 2021*.
- Mordatch, I., & Abbeel, P. (2018). Emergence of Grounded Compositional Language in Multi-Agent Populations. *AAAI 2018*.

**Tools:** Python 3.12, NumPy, Matplotlib, PettingZoo, Seaborn

**Collaborators:** Discussion with course peers on reward shaping approaches.

**AI Assistance:** Claude (Anthropic) used for code review, visualizations and documentation assistance.
