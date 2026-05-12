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
| V1 | Tabular RL on empirical MDP, static landmarks | Complete |
| V2 | Deep RL (DQN, PPO, TD3, SAC) as independent learners, static landmarks | Complete |
| Final | MAPPO with centralized critic, moving landmarks, N-invariant reward shaping | Complete |

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
├── deep/
│   ├── sb3/ # DQN/PPO/SAC/TD3 plus SB3 env bridge/common helpers
│   └── mappo/ # MAPPO public API plus network/rollout/update internals
├── utils/
│   ├── analysis/ # Sampling and gradient-based saliency
│   ├── environments/ # Moving-landmark env helpers
│   ├── evaluation/ # Coverage metrics, policy eval, rollout collection
│   ├── storage/ # Checkpoints and replay buffers
│   └── notebooks.py # Printing helpers
└── viz/
    └── visualization_utils.py # Plots and visualization helpers
docs/
└── technical-challenges.md # Bugs, surprises, stuck points
notebooks/
├── v1_notebook.ipynb # V1 full pipeline & analysis
├── v2_notebook.ipynb # V2 deep RL training, evaluation, saliency
└── final_notebook.ipynb # Final: MAPPO + moving landmarks
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

### V1 Results Summary

Best method: Sarsa(4) with 0.306 avg coverage (~2.5× random baseline of 0.124). All tabular methods struggle due to discretization information loss, sparse joint-action coverage, and flat value landscapes. See the notebook for full analysis.

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

## Final: MAPPO + Moving Landmarks

### Motivation

V2 solved the static-landmark task at N=2 with independent learners. The final asks whether centralized training with decentralized execution (CTDE) helps under a non-stationary version of the task with drifting landmarks.

### What Changed

- **Hand-rolled MAPPO** (`aerocover/deep/mappo/`) same actor and same PPO clipped surrogate as V2's PPO, but the critic takes the concatenated team observation as input.
- **Moving landmarks** (`env_adapters/landmark_motion.py`) persistent unit-direction velocity with small angular noise and reflective walls. Drift speed `0.01` keeps tracking feasible relative to the agents' max speed (~0.04).
- **N-invariant reward shaping** coefficients in `CoverageRewardWrapper` rebalanced so the per-step shaped reward at full coverage is constant in `n_landmarks`, making the same hyperparameters transfer to N=3+ without re-tuning.
- **Per-agent contribution bonus** each agent receives `+1.0` per step for each landmark where it is the closest covering agent. Addresses the lazy-agent failure mode where one drone parks on a landmark and the other gives up.
- **Generalized `reconstruct_positions`** was hardcoded for N=2; now handles arbitrary N.

### Result

DQN dominates the coverage metric on this task because the replay buffers plus the dense shaped reward are well-matched to the moving-landmark setup. MAPPO produces partial two-drone coordination but the lazy-agent pattern is reduced rather than eliminated. PPO's behavior under the layered non-stationarity (peer policies plus landmark drift) does not converge to a coherent strategy. See `notebooks/final_notebook.ipynb` for the full comparison and `docs/technical-challenges.md` for the bug/surprise/stuck-point notes.

## Algorithm Justification

### Why four methods?

The AeroCover environment has properties that stress-test different algorithm families differently, making a comprehensive comparison scientifically valuable:

**Off-policy value-based (DQN):** The coverage task has sparse rewards so agents must reach precise positions within cover_dist=0.15. Replay
buffers let these methods learn from rare successes repeatedly, which proved decisive. DQN achieved the highest coverage overall.

**On-policy actor-critic (PPO):** This os a standard baseline for any RL task. PPO's stability makes it a natural choice. Their underperformance here (vs DQN) is informative: on-policy methods discard data after each update, which is wasteful when coverage events are rare.

**Continuous-action (TD3, SAC):** The MPE environment supports continuous actions, allowing smoother agent movement. SAC and TD3 both performed strongly.


## Future Work

The project is complete as of now. The two natural next steps if continued:

**MADDPG.** With hindsight, MADDPG would have been the better algorithmic choice over MAPPO. Per-agent critics solve the credit-assignment problem (the lazy-agent pathology we addressed via reward shaping) without requiring custom shaping logic. Off-policy with replay also handles non-stationarity better than on-policy methods, which DQN's strong performance on this task suggests matters.

**Graph neural networks for variable team sizes.** A permutation-equivariant critic via message passing over agent-landmark graphs would let one trained model handle arbitrary `n_agents` and `n_landmarks` at inference time. The MAPPO formulation here is the foundation that change would build on because it would be the same algorithm, same loss, only the critic's network is the change.

## References & Citations

**Algorithms:**
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature* 518.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *arXiv:1802.09477*. (TD3)
- Haarnoja et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with Stochastic Actor. *arXiv:1801.01290*.
- Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *arXiv:1506.02438*.
- Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. *NeurIPS 2017*. (MADDPG, foundational CTDE work)
- Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. *NeurIPS Datasets and Benchmarks*. (MAPPO)

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
