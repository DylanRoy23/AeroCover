# Neural Network Architectures

## Discrete-Action Methods (DQN, PPO, A2C)
- **Input:** 12-dim continuous observation (vel, pos, landmark delta, agent delta, comm)
- **Architecture:** MLP [128, 128] with ReLU activations
- **Output:** 5 discrete actions (no-op, left, right, down, up)
- Configured via `net_arch` in SB3 policy_kwargs

## Continuous-Action Methods (TD3, SAC)
- **Input:** 12-dim continuous observation
- **Architecture:** MLP [128, 128] with ReLU activations
- **Output:** 5-dim continuous action vector
- TD3/SAC use twin critics

## REINFORCE (Custom)
- **PolicyNet:** MLP [128, 128] to Categorical (discrete) or Normal (continuous)
- **Baseline:** MLP [128, 128] to scalar state-value estimate
- Defined in `aerocover/deep/reinforce.py`

## SB3 Architecture Notes
- All SB3 models use `MlpPolicy` with shared config
- Actor-critic methods (PPO, A2C) have separate pi/vf heads
- DQN uses a single Q-network + target network (EMA copy)