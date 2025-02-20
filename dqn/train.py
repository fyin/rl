import math
from itertools import count

import gymnasium as gym

import random
from collections import deque, namedtuple

import torch
from torch import nn, optim

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

def optimize_model(config: dict, device, relay_memory: ReplayMemory, policy_net: DQN, target_net: DQN, optimizer):
    batch_size = config.get("batch_size")

    if len(relay_memory) < batch_size:
        return
    transitions = relay_memory.sample(batch_size)
    # zip(*transitions) unzips the list of tuples into four separate lists (states, actions, next_states, rewards).
    # Transition(*zip(*transitions)) reconstructs them back into a Transition object batch containing lists of each component.
    batch = Transition(*zip(*transitions))

    # final state is the one where the agent terminates the episode due to reaching a terminal state,
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * config.get("gamma")) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Use clipping technique to cap the gradients to avoid gradient explosion.
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

"""
Select an action according to epsilon-greedy policy
"""
def select_action(config:dict, state, policy_net: DQN, steps_done, device, env:gym.Env):
    sample = random.random()
    eps_threshold = float(config.get("epsilon_end")) + (float(config.get("epsilon_start")) - float(config.get("epsilon_end"))) * \
        math.exp(-1. * steps_done / float(config.get("epsilon_decay")))
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def train(config:dict, device):
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    # Policy network: Main network in DQN, update every training step using gradient descent, calculate Q(s,a) and used for action selection.
    # Target network: Copy of the policy network, used to stabilize Q-value targets in the loss function. Its weights are updated at a slower rate or via soft updates.
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=config.get("learning_rate"), amsgrad=True)
    relay_memory = ReplayMemory(config.get("replay_buffer_size"))

    episode_durations = []
    for i_episode in range(config.get("num_episodes")):
        steps_done = 0
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for step in count():
            action = select_action(config, state, policy_net,steps_done, device, env)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            relay_memory.push(state, action, next_state, reward)
            state = next_state
            # Perform optimization on the policy network
            optimize_model(config, device, relay_memory, policy_net, target_net, optimizer)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            tau = config.get("tau")
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(step + 1)
                plot_durations(True, episode_durations)
                break

if __name__ == "__main__":
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    train(config, device)
