from __future__ import annotations

import numpy as np


_VELOCITY_ATTR = "_aerocover_landmark_velocities"

def _unit_vectors(rng: np.random.Generator, n_vectors: int) -> np.ndarray:
    vectors = rng.normal(size=(n_vectors, 2)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return vectors / norms

def reset_landmark_motion(env, rng: np.random.Generator, speed: float) -> None:
    world = env.unwrapped.world
    velocities = _unit_vectors(rng, len(world.landmarks)) * float(speed)
    setattr(world, _VELOCITY_ATTR, velocities.astype(np.float32))

def move_landmarks(env, rng: np.random.Generator, speed: float) -> None:
    if speed <= 0.0:
        return

    world = env.unwrapped.world
    n_landmarks = len(world.landmarks)
    velocities = getattr(world, _VELOCITY_ATTR, None)

    if velocities is None or np.asarray(velocities).shape != (n_landmarks, 2):
        reset_landmark_motion(env, rng, speed)
        velocities = getattr(world, _VELOCITY_ATTR)

    # Small direction noise keeps the motion from becoming perfectly periodic.
    velocities = np.asarray(velocities, dtype=np.float32)
    velocities += rng.normal(scale=float(speed) * 0.10, size=velocities.shape)
    norms = np.maximum(np.linalg.norm(velocities, axis=1, keepdims=True), 1e-6)
    velocities = velocities / norms * float(speed)

    for idx, landmark in enumerate(world.landmarks):
        pos = np.asarray(landmark.state.p_pos, dtype=np.float32) + velocities[idx]

        for dim in range(2):
            if pos[dim] < -1.0:
                pos[dim] = -1.0 + (-1.0 - pos[dim])
                velocities[idx, dim] = abs(velocities[idx, dim])
            elif pos[dim] > 1.0:
                pos[dim] = 1.0 - (pos[dim] - 1.0)
                velocities[idx, dim] = -abs(velocities[idx, dim])

        landmark.state.p_pos = np.clip(pos, -1.0, 1.0)

    setattr(world, _VELOCITY_ATTR, velocities.astype(np.float32))