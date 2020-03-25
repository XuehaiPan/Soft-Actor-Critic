import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from common.network_base import VanillaNeuralNetwork


DEVICE_CPU = torch.device('cpu')


class ValueNetwork(VanillaNeuralNetwork):
    def __init__(self, state_dim, hidden_dims, activation=F.relu, device=DEVICE_CPU):
        super().__init__(n_dims=[state_dim, *hidden_dims, 1],
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim

    def forward(self, state):
        return super().forward(state)


class SoftQNetwork(VanillaNeuralNetwork):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=F.relu, device=DEVICE_CPU):
        super().__init__(n_dims=[state_dim + action_dim, *hidden_dims, 1],
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state, action):
        return super().forward(torch.cat([state, action], dim=-1))


class PolicyNetwork(VanillaNeuralNetwork):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=F.relu, device=DEVICE_CPU,
                 log_std_min=-20, log_std_max=2):
        super().__init__(n_dims=[state_dim, *hidden_dims, 2 * action_dim],
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        mean, log_std = super().forward(state).chunk(chunks=2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def evaluate(self, state, epsilon=1E-6):
        mean, std = self(state)

        z = Normal(0, 1).sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1.0 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(dim=0).to(self.device)
            mean, std = self(state)

            if deterministic:
                action = torch.tanh(mean)
            else:
                z = Normal(0, 1).sample()
                action = torch.tanh(mean + std * z)
        action = action.cpu().numpy()[0]
        return action

    def random_action(self):
        return np.random.uniform(low=-1, high=1, size=self.action_dim)
