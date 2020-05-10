import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from common.network import NetworkBase, MultilayerPerceptron


__all__ = [
    'StateEncoderWrapper',
    'DimensionScaler', 'ValueNetwork', 'SoftQNetwork',
    'PolicyNetwork'
]


class StateEncoderWrapper(NetworkBase):
    def __init__(self, encoder, device=None):
        super().__init__()

        self.encoder = encoder

        self.to(device)

    def forward(self, *input, **kwargs):
        return self.encoder.forward(*input, **kwargs)

    def encode(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(self.device)
            encoded = self(observation)
        encoded = encoded.cpu().numpy()[0]
        return encoded


class DimensionScaler(NetworkBase):
    def __init__(self, input_dim, output_dim, device=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.scaler = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        nn.init.zeros_(self.scaler.weight)
        nn.init.zeros_(self.scaler.bias)
        with torch.no_grad():
            for _, i, o in zip(range(max(input_dim, output_dim)),
                               itertools.cycle(range(input_dim)),
                               itertools.cycle(range(output_dim))):
                self.scaler.weight[o, i] = 1.0

        self.to(device)

    def forward(self, action):
        return self.scaler.forward(action)

    def plot(self):
        input_dim = self.input_dim
        output_dim = self.output_dim

        weight = self.scaler.weight.detach().cpu().numpy()
        bias = self.scaler.bias.detach().cpu().numpy()
        bias = np.expand_dims(bias, axis=1)

        weight = np.concatenate([weight, np.zeros_like(bias), bias], axis=1)

        vmax = np.abs(weight).max()

        fig, ax = plt.subplots(figsize=(input_dim + 3.5, output_dim))

        im = ax.imshow(weight, origin='upper', aspect='equal', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        fig.colorbar(im, ax=ax)
        for (i, j), v in np.ndenumerate(weight):
            if j != input_dim:
                ax.text(j, i, s=f'{v:.2f}',
                        horizontalalignment='center',
                        verticalalignment='center')

        for j in range(output_dim + 1):
            alpha = 0.5
            linestyle = ':'
            if j == 0 or j == output_dim:
                alpha = 1.0
                linestyle = '-'
            ax.axhline(j - 0.5, xmin=0.05 / (input_dim + 2.1), xmax=1.0 - 2.05 / (input_dim + 2.1),
                       color='black', alpha=alpha, linestyle=linestyle, linewidth=1.0)
            ax.axhline(j - 0.5, xmin=1.0 - 1.05 / (input_dim + 2.1), xmax=1.0 - 0.05 / (input_dim + 2.1),
                       color='black', alpha=alpha, linestyle=linestyle, linewidth=1.0)
        for i in range(input_dim + 1):
            alpha = 0.5
            linestyle = ':'
            if i == 0 or i == input_dim:
                alpha = 1.0
                linestyle = '-'
            ax.axvline(i - 0.5, ymin=0.05 / (output_dim + 0.1), ymax=1.0 - 0.05 / (output_dim + 0.1),
                       color='black', alpha=alpha, linestyle=linestyle, linewidth=1.0)
        ax.add_artist(plt.Rectangle(xy=(input_dim - 0.5, -0.5),
                                    width=1, height=output_dim,
                                    color=fig.get_facecolor()))
        ax.axvline(input_dim + 0.5, ymin=0.05 / (output_dim + 0.1), ymax=1.0 - 0.05 / (output_dim + 0.1),
                   color='black', linestyle='-', linewidth=1.0)
        ax.axvline(input_dim + 1.5, ymin=0.05 / (output_dim + 0.1), ymax=1.0 - 0.05 / (output_dim + 0.1),
                   color='black', linestyle='-', linewidth=1.0)

        ax.tick_params(top=False, bottom=False, left=False, right=False)

        ax.set_xlim(left=-0.55, right=input_dim + 1.55)
        ax.set_ylim(top=-0.55, bottom=output_dim - 0.45)

        ax.set_xticks(ticks=[(input_dim - 1) / 2, input_dim + 1])
        ax.set_xticklabels(labels=['$w$', '$b$'])
        ax.set_yticks(ticks=[])

        for spline in ax.spines.values():
            spline.set_visible(False)

        fig.tight_layout()
        return fig


class ValueNetwork(MultilayerPerceptron):
    def __init__(self, state_dim, hidden_dims, activation=F.relu, device=None):
        super().__init__(n_dims=[state_dim, *hidden_dims, 1],
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim

    def forward(self, state):
        return super().forward(state)


class SoftQNetwork(MultilayerPerceptron):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=F.relu, device=None):
        scaled_action_dim = max(state_dim, action_dim)

        super().__init__(n_dims=[state_dim + scaled_action_dim, *hidden_dims, 1],
                         activation=activation,
                         output_activation=None,
                         device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scaled_action_dim = scaled_action_dim

        self.action_scaler = DimensionScaler(input_dim=action_dim,
                                             output_dim=scaled_action_dim,
                                             device=device)

    def forward(self, state, action):
        return super().forward(torch.cat([state, self.action_scaler(action)], dim=-1))


class PolicyNetwork(MultilayerPerceptron):
    def __init__(self, state_dim, action_dim, hidden_dims, activation=F.relu, device=None,
                 log_std_min=np.log(1E-8), log_std_max=np.log(20.0)):
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

        distribution = Normal(mean, std)
        u = distribution.rsample()
        action = torch.tanh(u)
        log_prob = distribution.log_prob(u) - torch.log(1.0 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, distribution

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
