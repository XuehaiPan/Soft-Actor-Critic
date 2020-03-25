import numpy as np
import torch
import torch.functional
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


class LSTMHidden(object):
    def __init__(self, hidden):
        self.hidden = hidden

    def __str__(self):
        return str(self.hidden)

    def __repr__(self):
        return repr(self.hidden)

    def __getitem__(self, item):
        new_hidden = []
        for h, c in self.hidden:
            new_hidden.append((h[item], c[item]))
        return LSTMHidden(hidden=new_hidden)

    def __getattr__(self, item):
        attr = getattr(torch.Tensor, item)

        if callable(attr):
            self_hidden = self.hidden

            def func(*args, **kwargs):
                new_hidden = []
                for h, c in self_hidden:
                    new_hidden.append((getattr(h, item)(*args, **kwargs), getattr(c, item)(*args, **kwargs)))
                return LSTMHidden(hidden=new_hidden)

            return func
        else:
            new_hidden = []
            for h, c in self.hidden:
                new_hidden.append((getattr(h, item), getattr(c, item)))
            return LSTMHidden(hidden=new_hidden)

    @staticmethod
    def cat(hiddens, dim=0):
        hiddens = [hidden.hidden for hidden in hiddens]
        new_hidden = []
        for ith_layer_hiddens in zip(*hiddens):
            hidden = torch.cat(list(map(lambda hc: hc[0], ith_layer_hiddens)), dim=dim)
            cell = torch.cat(list(map(lambda hc: hc[1], ith_layer_hiddens)), dim=dim)
            new_hidden.append((hidden, cell))
        return LSTMHidden(hidden=new_hidden)


class VanillaLSTMNetwork(NetworkBase):
    def __init__(self, n_dims_before_lstm, n_dims_lstm_hidden, n_dims_after_lstm,
                 skip_connection, trainable_initial_hidden=True, activation=F.relu,
                 output_activation=None, device=DEVICE_CPU):
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
        for i in range(len(n_dims_lstm_hidden) - 1):
            self.lstm_layers.append(module=nn.LSTM(input_size=n_dims_lstm_hidden[i],
                                                   hidden_size=n_dims_lstm_hidden[i + 1],
                                                   num_layers=1, bias=True,
                                                   batch_first=False, bidirectional=False))
        if trainable_initial_hidden:
            self.init_hiddens = nn.ParameterList()
            self.init_cells = nn.ParameterList()
            for i in range(len(n_dims_lstm_hidden) - 1):
                bound = 1 / np.sqrt(n_dims_lstm_hidden[i])
                hidden = nn.Parameter(torch.Tensor(1, 1, n_dims_lstm_hidden[i + 1]))
                cell = nn.Parameter(torch.Tensor(1, 1, n_dims_lstm_hidden[i + 1]))
                nn.init.uniform_(hidden, -bound, bound)
                nn.init.uniform_(cell, -bound, bound)
                self.init_hiddens.append(hidden)
                self.init_cells.append(cell)
        else:
            self.init_hiddens = []
            self.init_cells = []
            for i in range(len(n_dims_lstm_hidden) - 1):
                self.init_hiddens.append(torch.zeros(1, 1, n_dims_lstm_hidden[i + 1],
                                                     device=self.device, requires_grad=False))
                self.init_cells.append(torch.zeros(1, 1, n_dims_lstm_hidden[i + 1],
                                                   device=self.device, requires_grad=False))

        self.linear_layers_after_lstm = VanillaNeuralNetwork(n_dims=n_dims_after_lstm,
                                                             activation=activation,
                                                             output_activation=output_activation)

        self.to(device)

    def forward(self, x, hx=None):
        if hx is None:
            hx = self.initial_hiddens(batch_size=x.size(1))
        assert isinstance(hx, LSTMHidden)

        identity = x = self.linear_layers_before_lstm(x)

        new_hx = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            new_hx.append(None)
            x, new_hx[-1] = lstm_layer(x, hx.hidden[i])

        if self.skip_connection:
            x = torch.cat([x, identity], dim=-1)
        x = self.linear_layers_after_lstm(x)
        return x, LSTMHidden(hidden=new_hx)

    def initial_hiddens(self, batch_size=1):
        init_hidden = LSTMHidden(hidden=list(zip(self.init_hiddens, self.init_cells)))
        return init_hidden.repeat(1, batch_size, 1)
