#!/usr/bin/env python3

import argparse
import os
import random
import sys

import matplotlib as mpl
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from common.config import Config
from common.environment import initialize_environment
from common.network import build_encoder
from common.utils import check_devices, check_logging
from sac import build_model, train, test


mpl.use('Agg')

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_config():
    def gpu_type(device):
        try:
            return int(device)
        except ValueError:
            return device

    parser = argparse.ArgumentParser(description='Train or test Soft Actor-Critic controller.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='mode (default: train)')
    parser.add_argument('--gpu', type=gpu_type, default=None, nargs='+', metavar='CUDA_DEVICE',
                        help="GPU device indexes "
                             "(int for CUDA device or 'c'/'cpu' for CPU) "
                             "(use 'cuda:0' if no following arguments; use CPU if not present)")
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='environment to train on (default: Pendulum-v0)')
    parser.add_argument('--n-frames', type=int, default=1,
                        help='concatenate original N consecutive observations as a new observation (default: 1)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--vision-observation', action='store_true',
                        help='use rendered images as observation')
    parser.add_argument('--image-size', type=int, default=96, metavar='SIZE',
                        help='image size of vision observation (default: 96)')
    parser.add_argument('--hidden-dims', type=int, default=[], nargs='+', metavar='DIM',
                        help='hidden dimensions of FC controller')
    parser.add_argument('--activation', type=str, choices=['ReLU', 'LeakyReLU'], default='ReLU',
                        help='activation function in controller networks (default: ReLU)')
    encoder_group = parser.add_argument_group('state encoder')
    encoder_group.add_argument('--encoder-arch', type=str, choices=['FC', 'RNN', 'CNN'], default='FC',
                               help='architecture of state encoder network (default: FC)')
    encoder_group.add_argument('--state-dim', type=int, default=None, metavar='DIM',
                               help='target state dimension of encoded state '
                                    '(use env.observation_space.shape if not present)')
    encoder_group.add_argument('--encoder-activation', type=str, choices=['ReLU', 'LeakyReLU'], metavar='ACTIVATION',
                               help='activation function in state encoder networks '
                                    '(use activation function in controller if not present)')
    fc_encoder_group = parser.add_argument_group('FC state encoder')
    fc_encoder_group.add_argument('--encoder-hidden-dims', type=int, default=[], nargs='+', metavar='DIM',
                                  help='hidden dimensions of FC state encoder')
    rnn_encoder_group = parser.add_argument_group('RNN state encoder')
    rnn_encoder_group.add_argument('--encoder-hidden-dims-before-rnn', type=int, default=[], nargs='+', metavar='DIM',
                                   help='hidden FC dimensions before GRU layers in RNN state encoder')
    rnn_encoder_group.add_argument('--encoder-hidden-dims-rnn', type=int, default=[], nargs='+', metavar='DIM',
                                   help='GRU hidden dimensions of RNN state encoder')
    rnn_encoder_group.add_argument('--encoder-hidden-dims-after-rnn', type=int, default=[], nargs='+', metavar='DIM',
                                   help='hidden FC dimensions after GRU layers in RNN state encoder')
    rnn_encoder_group.add_argument('--skip-connection', action='store_true', default=False,
                                   help='add skip connection beside GRU layers in RNN state encoder')
    rnn_encoder_group.add_argument('--trainable-hidden', action='store_true', default=False,
                                   help='set initial hidden of GRU layers trainable '
                                        '(use zeros as initial hidden if not present)')
    rnn_encoder_group.add_argument('--step-size', type=int, default=16,
                                   help='number of continuous steps for update (default: 16)')
    cnn_encoder_group = parser.add_argument_group('CNN state encoder')
    cnn_encoder_group.add_argument('--encoder-hidden-channels', type=int, default=[], nargs='+', metavar='CHN',
                                   help='channels of hidden conv layers in CNN state encoder')
    cnn_encoder_group.add_argument('--kernel-sizes', type=int, default=[], nargs='+', metavar='K',
                                   help='kernel sizes of conv layers in CNN state encoder (defaults: 3)')
    cnn_encoder_group.add_argument('--strides', type=int, default=[], nargs='+', metavar='S',
                                   help='strides of conv layers in CNN state encoder (defaults: 1)')
    cnn_encoder_group.add_argument('--paddings', type=int, default=[], nargs='+', metavar='P',
                                   help='paddings of conv layers in CNN state encoder (defaults: K // 2)')
    cnn_encoder_group.add_argument('--poolings', type=int, default=[], nargs='+', metavar='K',
                                   help='max pooling kernel size after activation function in CNN state encoder (defaults: 1)')
    cnn_encoder_group.add_argument('--batch-normalization', action='store_true', default=False,
                                   help='use batch normalization in CNN state encoder')
    parser.add_argument('--max-episode-steps', type=int, default=10000,
                        help='max steps per episode (default: 10000)')
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='number of training epochs (default: 1000)')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='number of test episodes (default: 100)')
    parser.add_argument('--n-updates', type=int, default=256,
                        help='number of learning updates per epoch (default: 256)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--n-samplers', type=int, default=4,
                        help='number of parallel samplers (default: 4)')
    parser.add_argument('--buffer-capacity', type=int, default=1000000, metavar='CAPACITY',
                        help='capacity of replay buffer (default: 1000000)')
    parser.add_argument('--update-sample-ratio', type=float, default=2.0, metavar='RATIO',
                        help='speed ratio of training and sampling '
                             '(sample speed <= training speed / ratio (ratio should be larger than 1.0)) '
                             '(default: 2.0)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--soft-tau', type=float, default=0.01, metavar='TAU',
                        help='soft update factor for target networks (default: 0.01)')
    parser.add_argument('--normalize-rewards', action='store_true',
                        help='normalize rewards for training')
    parser.add_argument('--reward-scale', type=float, default=1.0, metavar='SCALE',
                        help='reward scale factor for normalized rewards (default: 1.0)')
    parser.add_argument('--deterministic', action='store_true', help='deterministic in evaluation')
    lr_group = parser.add_argument_group('learning rate')
    lr_group.add_argument('--lr', type=float, default=1E-4,
                          help='learning rate (can be override by the following specific learning rate) (default: 0.0001)')
    lr_group.add_argument('--critic-lr', type=float, default=None,
                          help='learning rate for critic networks (use LR above if not present)')
    lr_group.add_argument('--actor-lr', type=float, default=None,
                          help='learning rate for actor networks (use LR above if not present)')
    alpha_group = parser.add_argument_group('temperature parameter')
    alpha_group.add_argument('--alpha-lr', type=float, default=None,
                             help='learning rate for temperature parameter (use ACTOR_LR above if not present)')
    alpha_group.add_argument('--initial-alpha', type=float, default=1.0, metavar='ALPHA',
                             help='initial value of temperature parameter (default: 1.0)')
    alpha_group.add_argument('--adaptive-entropy', action='store_true',
                             help='auto update temperature parameter while training')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--clip-gradient', action='store_true',
                        help='clip gradient on optimizer step')
    parser.add_argument('--random-seed', type=int, default=0, metavar='SEED',
                        help='random seed (default: 0)')
    parser.add_argument('--log-episode-video', action='store_true',
                        help='save rendered episode videos to TensorBoard logs')
    parser.add_argument('--log-dir', type=str, default=os.path.join(ROOT_DIR, 'logs'),
                        help='folder to save TensorBoard logs')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join(ROOT_DIR, 'checkpoints'),
                        help='folder to save checkpoint')
    parser.add_argument('--load-checkpoint', action='store_true',
                        help='load latest checkpoint in checkpoint dir')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    config = Config(vars(args))

    return config


def initialize(config):
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    initialize_hyperparameters(config)
    initialize_environment(config)
    build_encoder(config)
    check_devices(config)
    check_logging(config)


def initialize_hyperparameters(config):
    config.activation = {
        'ReLU': nn.ReLU(inplace=True),
        'LeakyReLU': nn.LeakyReLU(negative_slope=0.01, inplace=True)
    }.get(config.activation)
    config.encoder_activation = {
        'ReLU': nn.ReLU(inplace=True),
        'LeakyReLU': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        None: config.activation
    }.get(config.encoder_activation)

    config.FC_encoder = (config.encoder_arch == 'FC')
    config.RNN_encoder = (config.encoder_arch == 'RNN')
    config.CNN_encoder = (config.encoder_arch == 'CNN')

    if config.CNN_encoder:
        kernel_sizes = config.kernel_sizes
        strides = config.strides
        paddings = config.paddings
        poolings = config.poolings
        while len(kernel_sizes) < len(config.encoder_hidden_channels):
            kernel_sizes.append(3)
        while len(strides) < len(kernel_sizes):
            strides.append(1)
        while len(paddings) < len(kernel_sizes):
            paddings.append(kernel_sizes[len(paddings)] // 2)
        while len(poolings) < len(kernel_sizes):
            poolings.append(1)

    config.n_samples_per_update = config.batch_size
    if config.RNN_encoder:
        config.n_samples_per_update *= config.step_size

    config.critic_lr = (config.critic_lr or config.lr)
    config.actor_lr = (config.actor_lr or config.lr)
    config.alpha_lr = (config.alpha_lr or config.actor_lr)


def main():
    config = get_config()
    initialize(config)

    model = build_model(config)
    if config.mode == 'train' and config.initial_epoch < config.n_epochs:
        train(model, config)
    elif config.mode == 'test':
        test(model, config)


if __name__ == '__main__':
    main()
