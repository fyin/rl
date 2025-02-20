import torch
import gymnasium as gym
from dqn.config import get_config
from dqn.model import DQN


def inference(config:dict, device):
    # Load trained DQN model
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    state, info = env.reset()
    print(f"state shape: {state.shape}, state type: {type(state)}, state: {state}")
    n_observations = len(state)
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load(config["model_path"]+"/best_model.pth",  map_location=device, weights_only=True))
    policy_net.eval()  # Set model to evaluation mode

    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # Get Q-values from policy network
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
        observation, reward, terminated, truncated, _ = env.step(action)

        # Accumulate reward
        total_reward += reward
        print(f"Total Accumulated Reward: {total_reward}")

        if terminated or truncated:
            state, info = env.reset()
            done = True
        else:
            state = observation

    env.close()
    return total_reward

if __name__ == "__main__":
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    total_reward = inference(config, device)
    print(f"Total Accumulated Reward: {total_reward}")
