import numpy as np
import torch



def _grad_dqn(model, obs):
    device = next(model.q_net.parameters()).device
    x = torch.FloatTensor(obs).unsqueeze(0).to(device).requires_grad_(True)

    q_vals = model.q_net(x)
    target = q_vals.max()

    target.backward()
    return x.grad


def _grad_sb3_ac(model, obs):
    policy = model.policy
    obs_tensor = policy.obs_to_tensor(obs)[0].requires_grad_(True)

    dist = policy.get_distribution(obs_tensor)
    target = dist.distribution.logits.max()

    target.backward()
    return obs_tensor.grad


def _grad_sb3_continuous(model, obs):
    policy = model.policy
    obs_tensor = policy.obs_to_tensor(obs)[0].requires_grad_(True)

    action = policy.actor(obs_tensor)
    target = action.abs().sum()

    target.backward()
    return obs_tensor.grad


# ==============================
# Main API
# ==============================
def compute_saliency(model, obs_samples, method):
    grads = []

    for obs in obs_samples:
        try:
            if method == "dqn":
                grad = _grad_dqn(model, obs)

            elif method == "sb3_ac":
                grad = _grad_sb3_ac(model, obs)

            elif method == "sb3_continuous":
                grad = _grad_sb3_continuous(model, obs)

            else:
                continue

            grad_np = grad.detach().abs().squeeze(0).cpu().numpy()
            grads.append(grad_np)

        except Exception:
            continue

    if not grads:
        raise ValueError("No gradients computed")

    return np.stack(grads).mean(axis=0)