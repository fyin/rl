import imageio
import numpy as np
import gymnasium as gym


def evaluate_policy(env, max_steps, n_eval_episodes, policy, device):
    """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state, device, False)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward

            if terminated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def record_video(env, policy, out_path, device, fps=30):
    """
    Generate a replay video of the agent
    """
    images = []
    env = gym.make("CartPole-v1", render_mode="rgb_array")  # Set render mode here
    state = env.reset()
    img = env.render()
    images.append(img)
    terminated = False
    while not terminated:
        action, _ = policy.act(state, device, False)
        new_state, reward, terminated, truncated, _  = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_path, [np.array(img) for i, img in enumerate(images)], fps=fps)