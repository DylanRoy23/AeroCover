from __future__ import annotations
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from typing import Dict, List, Callable
import numpy as np
import supersuit as ss
from stable_baselines3.common.callbacks import BaseCallback
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.wrappers import BaseParallelWrapper
from aerocover.env_adapters.landmark_motion import (
    move_landmarks,
    reset_landmark_motion,
)
from aerocover.env_adapters.mpe_state import reconstruct_positions, compute_covered_mask

def _ensure_vec_env_seed(env, seed: int):

    stack = []
    current = env
    while current is not None and id(current) not in stack:
        stack.append(id(current))
        if not hasattr(current, "seed"):
            def seed_fn(s=None, _env=current):
                if hasattr(_env, "reset"):
                    _env.reset(seed=s)
                return [s]

            current.seed = seed_fn
        current = getattr(current, "venv", None)

    env.seed(seed)
    return env

class MovingLandmarksWrapper(BaseParallelWrapper):
    def __init__(self, env, drift_speed=0.02, seed=42):
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self.drift_speed = drift_speed

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        reset_landmark_motion(self.env, self._rng, self.drift_speed)
        return obs, info

    def step(self, actions):
        move_landmarks(self.env, self._rng, self.drift_speed)
        return self.env.step(actions)
    
class CoverageRewardWrapper(BaseParallelWrapper):
    def __init__(
        self,
        env,
        n_landmarks: int,
        n_agents: int,
        cover_dist: float,
        n_landmarks_norm: bool = True,
    ):
        super().__init__(env)
        self.n_landmarks = n_landmarks
        self.n_agents = n_agents
        self.cover_dist = cover_dist
        self.n_landmarks_norm = n_landmarks_norm
        self._prev_mask = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_mask = 0
        return obs, info

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        try:
            agent_pos, landmarks = reconstruct_positions(
                obs, self.n_landmarks, self.n_agents,
            )
            mask = compute_covered_mask(agent_pos, landmarks, self.cover_dist)
        except Exception:
            return obs, rewards, terminations, truncations, infos

        covered = bin(mask).count("1")
        newly = bin(mask & ~self._prev_mask).count("1")
        lost = bin(self._prev_mask & ~mask).count("1")

        if self.n_landmarks_norm:
            coverage_frac = covered / self.n_landmarks
            mpe_avg = float(np.mean(list(rewards.values())))
            shaped = (
                2.0 * coverage_frac
                + 3.0 * newly
                + (5.0 if covered == self.n_landmarks else 0.0)
                - 2.0 * lost
                - 0.01
                + 0.2 * mpe_avg
            )
        else:
            # ---------- V2 (legacy, N=2-tuned) ----------
            shaped = (
                2.0 * covered
                + 3.0 * newly
                + (5.0 if covered == self.n_landmarks else 0.0)
                - 2.0 * lost
                - 0.01
                + 0.2 * sum(rewards.values())
            )

        team_per_agent = shaped / self.n_agents

        per_agent_rewards = {a: team_per_agent for a in rewards}
        for lm in landmarks:
            dists = {a: float(np.linalg.norm(agent_pos[a] - lm)) for a in agent_pos}
            closest = min(dists, key=dists.get)
            if dists[closest] <= self.cover_dist:
                per_agent_rewards[closest] += 1.0

        self._prev_mask = mask
        return obs, per_agent_rewards, terminations, truncations, infos

def make_sb3_env(
    n_agents: int = 2,
    n_landmarks: int = 2,
    max_steps: int = 50,
    cover_dist: float = 0.30,
    seed: int = 42,
    shaped_reward: bool = True,
    continuous_actions: bool = False,
    n_envs: int = 1,
    moving_landmarks: bool = False,
    drift_speed: float = 0.02,
):
    env = simple_spread_v3.parallel_env(
        N=n_agents, 
        local_ratio=0.0,
        max_cycles=max_steps, 
        continuous_actions=continuous_actions,
        render_mode="rgb_array"
    )
    if moving_landmarks:
        env = MovingLandmarksWrapper(env, drift_speed=drift_speed, seed=seed)
    if shaped_reward:
        env = CoverageRewardWrapper(env, n_landmarks, n_agents, cover_dist)
        
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, n_envs, base_class="stable_baselines3")

    return _ensure_vec_env_seed(env, seed)
        
class CoverageLoggerCallback(BaseCallback):
    """Tracks per-episode reward and prints periodic summaries."""
    def __init__(self, log_interval: int = 100, verbose: bool = True):
        super().__init__(verbose=0)
        self.log_interval = log_interval
        self._verbose = verbose
        self.episode_rewards: List[float] = []
        self._current_rewards: List[float] = []
        self._ep_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for r, d in zip(rewards, dones):
            self._current_rewards.append(float(r))
            if d:
                self.episode_rewards.append(sum(self._current_rewards))
                self._current_rewards = []
                self._ep_count += 1
                if self._verbose and self._ep_count % self.log_interval == 0:
                    recent = self.episode_rewards[-self.log_interval:]
                    print(f"  Ep {self._ep_count:5d} | "
                          f"Mean reward (last {self.log_interval}): "
                          f"{np.mean(recent):.2f}")
        return True


def make_eval_policy_fn(model, is_continuous: bool = False) -> Callable:
    """
    Wrap any SB3 model as a policy compatible with the V1 evaluation utils.
    """
    def policy_fn(obs_dict: Dict[str, np.ndarray]):
        actions = []
        for agent in sorted(obs_dict):
            act, _ = model.predict(obs_dict[agent], deterministic=True)
            if is_continuous:
                actions.append(act)
            else:
                actions.append(int(act))
        return tuple(actions)
    return policy_fn
