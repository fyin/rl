import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(s_size, h_size)
        self.layer2 = nn.Linear(h_size, h_size)
        self.layer3 = nn.Linear(h_size, a_size)

    def forward(self, x):
       x = F.relu(self.layer1(x))
       x = F.relu(self.layer2(x))
       return F.softmax(self.layer3(x), dim=1)

    def act(self, state, device, training=True):
        if isinstance(state, tuple):
            state = state[0] # get only state part
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        if (training):
            # Use stochastic action selection for exploration during training.
            # If always selecting the action with the highest probability (argmax), the policy becomes deterministic too early.
            # This could lead to premature convergence and suboptimal policies because the agent might not explore better actions.
            action = m.sample()
        else:
            # Use deterministic action selection for evaluation.
            action = torch.argmax(probs)
        return action.item(), m.log_prob(action)