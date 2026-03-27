import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from aerocover.deep.sb3_utils import make_sb3_env


def build_env(
    n_agents: int,
    n_landmarks: int,
    max_steps: int,
    cover_dist: float,
    seed: int,
    n_envs: int = 1,
    continuous: bool = False
):
    env = make_sb3_env(
        n_agents,
        n_landmarks,
        max_steps,
        cover_dist,
        seed,
        n_envs=n_envs,
        continuous_actions=continuous
    )
    return env

def build_action_noise(env):
    n_actions = env.action_space.shape[-1]

    noise_mean = np.zeros(n_actions)
    noise_sigma = 0.1 * np.ones(n_actions)

    action_noise = NormalActionNoise(
        mean=noise_mean,
        sigma=noise_sigma
    )

    return action_noise, n_actions

def train_model(model, cfg, callback):
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback
    )

def build_return(model, env, continuous: bool):
    obs_dim = model.observation_space.shape[0]

    if continuous:
        n_actions = env.action_space.shape[-1]
    else:
        n_actions = model.action_space.n

    return {
        "obs_dim": obs_dim,
        "n_actions": n_actions,
    }