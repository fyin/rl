import math
import os
from itertools import count

import gymnasium as gym

import random
from collections import deque, namedtuple

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from dqn.config import get_config
from model import DQN
from visualizer import plot_durations

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_dqn(config: dict, device, relay_memory: ReplayMemory, policy_net: DQN, target_net: DQN, optimizer):
    batch_size = int(config["batch_size"])

    if len(relay_memory) < batch_size:
        return
    transitions = relay_memory.sample(batch_size)
    # zip(*transitions) unzips the list of Transition tuples into four separate lists (states, actions, next_states, rewards).
    # Transition(*zip(*transitions)) reconstructs them back into a Transition object, each element containing a list of separate elements.
    # Ex: Transition(s1, a1, s1', r1), Transition(s2, a2, s2', r2), Transition(s3, a3, s3', r3)
    # Transition(*zip(*transitions)): Transition(states = [s1, s2, s3], actions = [a1, a2, a3], next_states = [s1', s2', s3'], rewards = [r1, r2, r3])
    batch = Transition(*zip(*transitions))

    # final state is the one where the agent terminates the episode due to reaching a terminal state,
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Select the Q-values only for the chosen actions in action_batch.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    # Use DNQ model to get the Q-values starting from each non-final next state.
    # No need to compute gradient as we're not training the target network.
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * float(config["gamma"])) + reward_batch

    # Use Huber loss to compute the loss to stabilize learning when minimizing TD error
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Use clipping technique to cap the gradients to avoid gradient explosion.
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss

"""
Select an action according to epsilon-greedy policy
"""
def select_action(config:dict, state, policy_net: DQN, steps_done, device, env:gym.Env):
    sample = random.random()
    eps_threshold = float(config["epsilon_end"]) + (float(config["epsilon_start"]) - float(config["epsilon_end"])) * \
        math.exp(-1. * steps_done / float(config["epsilon_decay"]))
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def train_dqn(config:dict, device):
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    # Policy network: Main network in DQN, update every training step using gradient descent, calculate Q(s,a) and used for action selection.
    # Target network: Copy of the policy network, used to stabilize Q-value targets in the loss function.
    # Its weights are updated at a slower rate or via soft updates.
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=float(config["learning_rate"]), amsgrad=True)
    relay_memory = ReplayMemory(int(config["replay_buffer_size"]))
    writer = SummaryWriter(config['tensorboard_path'])
    os.makedirs(config["model_path"], exist_ok=True)
    best_avg_reward = -float("inf")

    episode_durations = []
    for i_episode in range(int(config["num_episodes"])):
        steps_done = 0
        total_reward = 0
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for step in count():
            action = select_action(config, state, policy_net, steps_done, device, env)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            total_reward += reward
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            relay_memory.push(state, action, next_state, reward)
            state = next_state
            # Perform optimization on the policy network
            optimize_dqn(config, device, relay_memory, policy_net, target_net, optimizer)

            # Soft update the target network's weights using parameters from the policy network
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            tau = float(config["tau"])
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(step + 1)
                plot_durations(True, episode_durations)
                break

        writer.add_scalar('Reward', total_reward, i_episode)
        writer.flush()
        if i_episode % 100 == 0:
            checkpoint_path = f"{config['model_path']}/checkpoint_{i_episode}.pth"
            torch.save(policy_net.state_dict(), checkpoint_path)

            # Save the best-performing model
        avg_reward = total_reward  # (Use a moving average for better stability)
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_path = f"{config['model_path']}/best_model.pth"
            torch.save(policy_net.state_dict(), best_model_path)

    # Save final model
    final_model_path = f"{config['model_path']}/final_model.pth"
    torch.save(policy_net.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    env.close()

if __name__ == "__main__":
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    train_dqn(config, device)
    plt.show()
