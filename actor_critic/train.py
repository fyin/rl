import numpy as np
import torch
from torch import optim, nn

from actor_critic.config import get_config
from actor_critic.model import Agent
import gymnasium as gym

from actor_critic.visualizer import plot_score

def train_ppo_agent(config:dict, device):
    update_nums = config["update_nums"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    epsilon = config["epsilon"]
    max_terminal_steps = config["max_terminal_steps"]
    gamma = config["gamma"]
    lambda_val = config["lambda"]
    lr = config["learning_rate"]

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=lr, amsgrad=True)
    scores = []

    for update in range(update_nums):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        rewards = []
        transition_buffer = []  # Store transitions

        # Collect experience data
        for step in range(0, max_terminal_steps):
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(state)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                transition_buffer.append((state, action, log_prob, reward, value, terminated))
                state = next_state
                rewards.append(reward)

                if terminated or truncated:
                    state, _ = env.reset()
                    state = torch.tensor(state, dtype=torch.float32, device=device)
                    scores.append(sum(rewards))

        # Process collected data
        states, actions, log_probs, rewards, values, dones = zip(*transition_buffer)
        states = torch.stack(states)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()

        # Compute Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(state).squeeze()
            values = torch.cat([values, next_value.unsqueeze(0)])  # Append final value for GAE computation
            advantages, returns = compute_gae(rewards, values, dones, gamma, lambda_val)
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Use PPO-CLIP algorithm to update the agent parameters
        # epoch: number of gradient updates to perform on the same batch of collected experiences before collecting new data
        for _ in range(epochs):
            indices = np.random.permutation(len(transition_buffer))
            for i in range(0, len(transition_buffer), batch_size):
                batch_idx = indices[i:i + batch_size]

                batch_states = states[batch_idx]
                batch_log_probs = log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_states)
                ratio = (new_log_probs - batch_log_probs).exp()

                # Compute clipped surrogate loss
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Value function loss (MSE)
                value_loss = nn.functional.mse_loss(new_values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Loss: {loss.item()}")

    env.close()
    return scores

def compute_gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" \
        if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    config = get_config()
    scores = train_ppo_agent(config, device)
    plot_score(scores, title="PPO training result")