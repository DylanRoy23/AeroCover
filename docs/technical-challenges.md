# Technical Challenges & Surprises

## Bugs
- SuperSuit + SB3 `VecEnvWrapper.seed` incompatibility so it required monkey-patch
- `torch._dynamo` crash on Adam optimizer init so got it fixed with `TORCHDYNAMO_DISABLE=1`
- PettingZoo `mpe` to `mpe2` migration deprecation warnings
- `action_tuple_to_dict` crashed on continuous actions (numpy arrays) so I removed `int()` cast
- Multi-env (`n_envs=16`) silently degraded performance due to effective batch size scaling

## Surprises
- Off-policy methods (DQN, SAC) outperformed on-policy (PPO) at tight cover_dist=0.15 because replay buffer reuses rare success events
- V1 tabular methods collapsed from ~0.31 to ~0.04 coverage when cover_dist halved from 0.30 to 0.15 proving discretization can't represent precision
- PPO is learning (reward curve trends up) but converges 3-4x slower than DQN on this sparse-reward task
- Saliency maps show PPO specializes (attends to one landmark), while DQN spreads attention across both landmarks + other agent position

## Stuck Points
- Reward shaping took significant iteration because initial coverage-only reward gave no learning signal for any method
- Getting SB3 to work with PettingZoo required SuperSuit wrappers + CoverageRewardWrapper that properly implements BaseParallelWrapper