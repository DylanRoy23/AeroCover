# Technical Challenges & Surprises V2

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


# Technical Challenges & Surprises

## Bugs
- `MovingLandmarksWrapper` originally wasn’t re-seeding its RNG during `reset(seed=...)`, so every episode during training ended up using the exact same landmark drift pattern. Training looked stable, but only because the agents were seeing the same sequence every time. Fixed by rebuilding `self._rng` inside `reset`.

- `reconstruct_positions` was accidentally hardcoded for `N=2`. When switching to `N=3`, the third drone was silently ignored. This caused evaluation coverage to be undercounted, and the animations only showed two drones. Itwas eventually noticed because one drone had “vanished” from the visualization.

- `evaluate_deep_policy` was rebuilding the environment without the moving-landmark wrapper, meaning evaluation was happening on static landmarks even though training used moving ones.

- Early on, we monkey-patched `rl_utils.build_env` to pass through the `moving_landmarks` argument. Later this became unnecessary once `train_ppo` and `train_dqn` supported the kwarg directly.

## Surprises
- The original `drift_speed=0.08` made the task basically impossible. The landmarks were moving around twice as fast as the drones could realistically chase them. At first it looked like the algorithms were failing badly, but the real issue was just that the environment itself was unwinnable. Dropping drift speed to `0.01` fixed this.

- Reward shaping from the V2 setup didn’t scale cleanly to `N=3`. The per-landmark coverage rewards naturally increased with more agents/landmarks, while the “full coverage” bonus stayed fixed. What ended up happening was a redesigning pf the reward terms to be more invariant to team size, which should also make future `N=4+` experiments easier.

- MAPPO developed a funny “lazy agent” behavior. Once one drone found a landmark, the other often stopped trying because the shared team reward was already good enough.

- Adding a per-agent contribution bonus (`+1.0` for being the closest covering agent) reduced the worst lazy-agent cases, but didn’t completely solve coordination. A common pattern became one drone parking on a target while the other only partially participated.

- DQN ended up performing the best on coverage despite being the least “fancy” algorithm in the project. The combination of replay buffers, off-policy learning, and dense reward shaping just happened to fit this problem really well.

- MAPPO’s training reward looked much stronger than PPO’s (roughly ~38 vs ~15–20), but actual evaluation coverage barely improved. It seems MAPPO learned how to optimize reward locally without actually improving the team’s real objective.

- PPO struggled the most with the layered non-stationarity of the environment. The agents had to adapt both to each other’s changing policies and to drifting landmarks, and the learned behavior never really stabilized into a coherent strategy.

## Stuck Points
- Picking a reasonable drift speed took several iterations. It wasn’t immediately obvious that the drones’ max movement speed (~0.04) meant landmarks had to drift *well below* that value to remain catchable over time.

- The first `N=3` experiments produced completely flat learning curves. This turned out to be caused by *two separate issues at once*: the `reconstruct_positions` bug and the poorly-scaled reward shaping.

- It was originally planned to compare against BenchMARL as a second MAPPO implementation, but ended up cutting it. TorchRL’s `PettingZooEnv` wrapper didn’t properly inherit custom `BaseParallelWrapper`s like `MovingLandmarksWrapper`, so getting a fair apples-to-apples comparison would have required writing a custom task wrapper from scratch.

- It was briefly considered to try the Muon optimizer, but decided against it. Muon is mainly designed around attention-heavy architectures, while our models were just simple 2-layer MLPs. Adding it also would have complicated the experiment too much this late in the project.

- Last-minute hyperparameter tuning mostly made results worse instead of better. In the end, the safest choice was keeping PPO and DQN close to their stable V2 settings and focusing tuning effort only on MAPPO.