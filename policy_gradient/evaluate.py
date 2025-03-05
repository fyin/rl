import numpy as np
import gymnasium as gym
import torch

from policy_gradient.config import cartpole_hyperparameters
from policy_gradient.model import Policy


def evaluate_policy(env, max_steps, n_eval_episodes, policy, device):
    """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    """
    with torch.no_grad():
        episode_rewards = []
        for episode in range(n_eval_episodes):
            state = env.reset()
            total_rewards_ep = 0

            for step in range(max_steps):
                action, _ = policy.act(state, device, False)
                new_state, reward, terminated, truncated, _ = env.step(action)
                total_rewards_ep += reward
                done = terminated or truncated

                if done:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    cartpole_policy = Policy(
        s_size,
        a_size,
        cartpole_hyperparameters["h_size"],
    ).to(device)
    cartpole_policy.load_state_dict(
        torch.load(cartpole_hyperparameters["checkpoint_path"]+"/checkpoint_last.pth", map_location=device, weights_only=True)
    )
    mean_reward, std_reward = evaluate_policy(
        env,
        cartpole_hyperparameters["max_steps_per_episode"],
        cartpole_hyperparameters["n_evaluation_episodes"],
        cartpole_policy,
        device)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")