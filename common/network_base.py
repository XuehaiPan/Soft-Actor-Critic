import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE_CPU = torch.device('cpu')


class NetworkBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.device = None

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()


class VanillaNeuralNetwork(NetworkBase):
    def __init__(self, n_dims, activation=F.relu, output_activation=None, device=DEVICE_CPU):
        super().__init__()
        self.device = device

        self.activation = activation
        self.output_activation = output_activation

        self.linear_layers = nn.ModuleList()
        for i in range(len(n_dims) - 1):
            self.linear_layers.append(module=nn.Linear(in_features=n_dims[i],
                                                       out_features=n_dims[i + 1],
                                                       bias=True))

        self.to(device)

    def forward(self, x):
        n_layers = len(self.linear_layers)
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < n_layers - 1:
                x = self.activation(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class VanillaLSTMNetwork(NetworkBase):
    def __init__(self, n_dims_before_lstm, n_dims_lstm_hidden, n_dims_after_lstm, skip_connection,
                 activation=F.relu, output_activation=None, device=DEVICE_CPU):
        assert len(n_dims_lstm_hidden) > 0

        super().__init__()
        self.device = device

        n_dims_lstm_hidden = [n_dims_before_lstm[-1], *n_dims_lstm_hidden]
        n_dims_after_lstm = [n_dims_lstm_hidden[-1], *n_dims_after_lstm]

        self.skip_connection = skip_connection
        if skip_connection:
            n_dims_after_lstm[0] += n_dims_before_lstm[-1]

        self.activation = activation
        self.output_activation = output_activation

        self.linear_layers_before_lstm = VanillaNeuralNetwork(n_dims=n_dims_before_lstm,
                                                              activation=activation,
                                                              output_activation=activation)
        self.lstm_layers = nn.ModuleList()
        self.init_hiddens = nn.ParameterList()
        self.init_cells = nn.ParameterList()
        for i in range(len(n_dims_lstm_hidden) - 1):
            self.lstm_layers.append(module=nn.LSTM(input_size=n_dims_lstm_hidden[i],
                                                   hidden_size=n_dims_lstm_hidden[i + 1],
                                                   num_layers=1, bias=True,
                                                   batch_first=False, bidirectional=False))
            bound = 1 / np.sqrt(n_dims_lstm_hidden[i])
            hidden = nn.Parameter(torch.Tensor(1, 1, n_dims_lstm_hidden[i + 1]))
            cell = nn.Parameter(torch.Tensor(1, 1, n_dims_lstm_hidden[i + 1]))
            nn.init.uniform_(hidden, -bound, bound)
            nn.init.uniform_(cell, -bound, bound)
            self.init_hiddens.append(hidden)
            self.init_cells.append(cell)

        self.linear_layers_after_lstm = VanillaNeuralNetwork(n_dims=n_dims_after_lstm,
                                                             activation=activation,
                                                             output_activation=output_activation)

        self.to(device)

    def forward(self, x, hx=None):
        if hx is None:
            batch_size = x.shape[1]
            hx = list(zip(map(lambda tensor: tensor.repeat(1, batch_size, 1), self.init_hiddens),
                          map(lambda tensor: tensor.repeat(1, batch_size, 1), self.init_cells)))
        identity = x = self.linear_layers_before_lstm(x)
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, hx[i] = lstm_layer(x, hx[i])
        if self.skip_connection:
            x = torch.cat([x, identity], dim=-1)
        x = self.linear_layers_after_lstm(x)
        return x, hx
