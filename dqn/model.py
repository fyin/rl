import torch.nn as nn
import torch.nn.functional as F

'''
DQN model takes non_final_next_states (next states that are not terminal) and returns Q-values for all possible actions.
So that it could be used to get the max value for epsilon-greedy action selection.
'''
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)