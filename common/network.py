import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'build_encoder',
    'NetworkBase',
    'VanillaNeuralNetwork', 'VanillaNN',
    'MultilayerPerceptron', 'MLP',
    'LSTMHidden', 'cat_hidden',
    'RecurrentNeuralNetwork', 'RNN',
    'ConvolutionalNeuralNetwork', 'CNN'
]

DEVICE_CPU = torch.device('cpu')


def build_encoder(config):
    state_dim = (config.state_dim or config.observation_dim)
    state_encoder = nn.Identity()
    if config.FC_encoder:
        if config.state_dim is not None or len(config.encoder_hidden_dims) > 0:
            state_encoder = VanillaNeuralNetwork(n_dims=[config.observation_dim,
                                                         *config.encoder_hidden_dims,
                                                         config.state_dim],
                                                 activation=config.activation,
                                                 output_activation=None)
    elif config.RNN_encoder:
        state_encoder = RecurrentNeuralNetwork(n_dims_before_lstm=[config.observation_dim,
                                                                   *config.encoder_hidden_dims_before_lstm],
                                               n_dims_lstm_hidden=config.encoder_hidden_dims_lstm,
                                               n_dims_after_lstm=[*config.encoder_hidden_dims_after_lstm,
                                                                  config.state_dim],
                                               skip_connection=config.skip_connection,
                                               activation=config.activation,
                                               output_activation=None)
    elif config.CNN_encoder:
        state_encoder = ConvolutionalNeuralNetwork(input_channels=config.observation_dim,
                                                   output_dim=config.state_dim,
                                                   n_hidden_channels=config.encoder_hidden_channels,
                                                   batch_normalization=False,
                                                   output_activation=None,
                                                   **config.build_from_keys(['kernel_sizes',
                                                                             'strides',
                                                                             'paddings',
                                                                             'activation']))

    config.state_encoder = state_encoder
    config.state_dim = state_dim

    return state_encoder


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


cat_hidden = LSTMHidden.cat


class RecurrentNeuralNetwork(NetworkBase):
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


class ConvolutionalNeuralNetwork(NetworkBase):
    def __init__(self, input_channels, output_dim,
                 n_hidden_channels, kernel_sizes, strides, paddings,
                 batch_normalization, activation=F.relu,
                 output_activation=None, device=DEVICE_CPU):
        assert len(n_hidden_channels) == len(kernel_sizes)
        assert len(n_hidden_channels) == len(strides)
        assert len(n_hidden_channels) == len(paddings)

        super().__init__()
        self.device = device

        n_hidden_channels = [input_channels, *n_hidden_channels, output_dim]
        kernel_sizes = [*kernel_sizes, 1]
        strides = [*strides, 1]
        paddings = [*paddings, 0]

        self.activation = activation
        self.output_activation = output_activation

        self.conv_layers = nn.ModuleList()
        for i in range(len(n_hidden_channels) - 1):
            self.conv_layers.append(module=nn.Conv2d(n_hidden_channels[i],
                                                     n_hidden_channels[i + 1],
                                                     kernel_size=kernel_sizes[i],
                                                     stride=strides[i],
                                                     padding=paddings[i],
                                                     bias=True))

        self.batch_normalization = batch_normalization
        if batch_normalization:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(1, len(n_hidden_channels)):
                self.batch_norm_layers.append(module=nn.BatchNorm2d(n_hidden_channels[i],
                                                                    affine=True))

        self.global_average_pooling_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.to(device)

    def forward(self, x):
        input_size = x.size()
        x = x.view(np.prod(input_size[:-3]), *input_size[-3:])

        n_layers = len(self.conv_layers)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if self.batch_normalization:
                x = self.batch_norm_layers[i](x)
            if i < n_layers - 1:
                x = self.activation(x)

        x = self.global_average_pooling_layer(x)
        x = x.view(x.size()[:-2])
        if self.output_activation is not None:
            x = self.output_activation(x)

        x = x.view(*input_size[:-3], -1)
        return x


MLP = MultilayerPerceptron = VanillaNN = VanillaNeuralNetwork
RNN = RecurrentNeuralNetwork
CNN = ConvolutionalNeuralNetwork
