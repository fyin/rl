import os

import gymnasium as gym

from collections import deque

import numpy as np
import torch
from torch import optim

from policy_gradient.config import cartpole_hyperparameters
from policy_gradient.evaluate import evaluate_policy, record_video
from policy_gradient.model import Policy


def train_policy(policy, optimizer, n_training_episodes, max_steps, gamma, env, device):
    scores = []
    for i_episode in range(1, n_training_episodes + 1):
        log_probs = []
        rewards = []
        state = env.reset()
        for step in range(max_steps):
            action, log_prob = policy.act(state, device, True)
            log_probs.append(log_prob)
            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated:
                break
        scores.append(sum(rewards))

        returns = deque(maxlen=max_steps)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...
        for step in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[step])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        # add eps (small value) to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    return scores

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    cartpole_policy = Policy(
        s_size,
        a_size,
        cartpole_hyperparameters["h_size"],
    ).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

    scores = train_policy(
        cartpole_policy,
        cartpole_optimizer,
        cartpole_hyperparameters["n_training_episodes"],
        cartpole_hyperparameters["max_steps_per_episode"],
        cartpole_hyperparameters["gamma"],
        env,
        device
    )

    mean_reward, std_reward = evaluate_policy(
        env, cartpole_hyperparameters["max_steps_per_episode"],
        cartpole_hyperparameters["n_evaluation_episodes"],
        cartpole_policy,
        device
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    out_directory = cartpole_hyperparameters["output_dir"]
    os.makedirs(out_directory, exist_ok=True)
    record_video(env, cartpole_policy, out_directory + "/cartpole.gif", device, fps=30)