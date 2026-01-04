from __future__ import annotations
import torch
from torch.distributions import Categorical

def discount_cumsum(rewards, gamma: float):
    out = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

def rollout_episode(env, policy, device, gamma: float, max_steps: int, entropy_bonus: float = 0.0):
    obs, _ = env.reset()
    rewards = []
    log_probs = []
    entropies = []
    steps = 0

    done = False
    while not done and steps < max_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        logits = policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        rewards.append(float(reward))
        done = terminated or truncated
        steps += 1

    returns = discount_cumsum(rewards, gamma).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs_t = torch.stack(log_probs)
    ent_t = torch.stack(entropies)

    # REINFORCE loss (maximize returns)
    loss = -(log_probs_t * returns).sum() - entropy_bonus * ent_t.sum()
    total_reward = float(sum(rewards))
    return loss, total_reward, steps
