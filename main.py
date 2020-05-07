#!/usr/bin/env python3

import argparse
import glob
import os
import random
import re
import sys
import time
from collections import OrderedDict

import gym
import matplotlib as mpl
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from setproctitle import setproctitle
from torch.utils.tensorboard import SummaryWriter

from common.config import Config
from common.environment import FlattenedAction, NormalizedAction, \
    FlattenedObservation, VisionObservation, ConcatenatedObservation
from common.network_base import VanillaNeuralNetwork, VanillaRecurrentNeuralNetwork, \
    VanillaConvolutionalNetwork


mpl.use('Agg')

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_config():
    parser = argparse.ArgumentParser(description='Train or test Soft Actor-Critic controller.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='mode (default: train)')
    parser.add_argument('--gpu', type=int, default=None, nargs='+', metavar='CUDA_DEVICE',
                        help='GPU devices (use CPU if not present)')
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='environment to train on (default: Pendulum-v0)')
    parser.add_argument('--n-frames', type=int, default=1,
                        help='concatenate original N consecutive observations as a new observation (default: 1)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--vision-observation', action='store_true',
                        help='use rendered images as observation')
    parser.add_argument('--image-size', type=int, default=128, metavar='SIZE',
                        help='image size of vision observation (default: 128)')
    parser.add_argument('--hidden-dims', type=int, default=[], nargs='+', metavar='DIM',
                        help='hidden dimensions of FC controller')
    parser.add_argument('--activation', type=str, choices=['ReLU', 'LeakyReLU'], default='ReLU',
                        help='activation function in networks (default: ReLU)')
    encoder_group = parser.add_argument_group('state encoder')
    encoder_group.add_argument('--encoder-arch', type=str, choices=['FC', 'RNN', 'CNN'], default='FC',
                               help='architecture of state encoder network (default: FC)')
    encoder_group.add_argument('--state-dim', type=int, default=None, metavar='DIM',
                               help='target state dimension of encoded state (use env.observation_space.shape if not present)')
    fc_encoder_group = parser.add_argument_group('FC state encoder')
    fc_encoder_group.add_argument('--encoder-hidden-dims', type=int, default=[], nargs='+', metavar='DIM',
                                  help='hidden dimensions of FC state encoder')
    rnn_encoder_group = parser.add_argument_group('RNN state encoder')
    rnn_encoder_group.add_argument('--encoder-hidden-dims-before-lstm', type=int, default=[], nargs='+', metavar='DIM',
                                   help='hidden FC dimensions before LSTM layers in RNN state encoder')
    rnn_encoder_group.add_argument('--encoder-hidden-dims-lstm', type=int, default=[], nargs='+', metavar='DIM',
                                   help='LSTM hidden dimensions of RNN controller')
    rnn_encoder_group.add_argument('--encoder-hidden-dims-after-lstm', type=int, default=[], nargs='+', metavar='DIM',
                                   help='hidden FC dimensions after LSTM layers in RNN state encoder')
    rnn_encoder_group.add_argument('--skip-connection', action='store_true', default=False,
                                   help='add skip connection beside LSTM layers in RNN state encoder')
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
    cnn_encoder_group.add_argument('--batch-normalization', action='store_true', default=False,
                                   help='use batch normalization in CNN state encoder')
    parser.add_argument('--max-episode-steps', type=int, default=10000,
                        help='max steps per episode (default: 10000)')
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='number of training epochs (default: 1000)')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='number of test episodes (default: 100)')
    parser.add_argument('--n-updates', type=int, default=32,
                        help='number of learning updates per epoch (default: 32)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--n-samplers', type=int, default=4,
                        help='number of parallel samplers (default: 4)')
    parser.add_argument('--buffer-capacity', type=int, default=1000000, metavar='CAPACITY',
                        help='capacity of replay buffer (default: 1000000)')
    parser.add_argument('--update-sample-ratio', type=float, default=2.0, metavar='RATIO',
                        help='speed ratio of training and sampling (default: 2.0)')
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
    lr_group.add_argument('--soft-q-lr', type=float, default=None,
                          help='learning rate for Soft Q Networks (use LR above if not present)')
    lr_group.add_argument('--policy-lr', type=float, default=None,
                          help='learning rate for Policy Networks (use LR above if not present)')
    alpha_group = parser.add_argument_group('temperature parameter')
    alpha_group.add_argument('--alpha-lr', type=float, default=None,
                             help='learning rate for temperature parameter (use POLICY_LR above if not present)')
    alpha_group.add_argument('--initial-alpha', type=float, default=1.0, metavar='ALPHA',
                             help='initial value of temperature parameter (default: 1.0)')
    alpha_group.add_argument('--adaptive-entropy', action='store_true',
                             help='auto update temperature parameter while training')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
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

    initialize_variables(config)
    build_env(config)
    build_encoder(config)


def initialize_variables(config):
    config.activation = {'ReLU': F.relu, 'LeakyReLU': F.leaky_relu}.get(config.activation)

    config.FC_encoder = (config.encoder_arch == 'FC')
    config.RNN_encoder = (config.encoder_arch == 'RNN')
    config.CNN_encoder = (config.encoder_arch == 'CNN')

    config.n_samples_per_update = config.batch_size
    if config.RNN_encoder:
        config.n_samples_per_update *= config.step_size

    config.soft_q_lr = (config.soft_q_lr or config.lr)
    config.policy_lr = (config.policy_lr or config.lr)
    config.alpha_lr = (config.alpha_lr or config.policy_lr)

    check_devices(config)
    check_logging(config)


def check_devices(config):
    if config.gpu is not None and torch.cuda.is_available():
        if len(config.gpu) == 0:
            config.gpu = [0]
        devices = [torch.device(f'cuda:{cuda_device}') for cuda_device in config.gpu]
    else:
        devices = [torch.device('cpu')]

    config.devices = devices

    return devices


def check_logging(config):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint_regex = re.compile(r'^(.*/)?[\w-]*-(?P<epoch>\d+)\.pkl$')
    if config.mode == 'test' or config.load_checkpoint:
        initial_checkpoint = max(glob.iglob(os.path.join(config.checkpoint_dir, '*.pkl')),
                                 key=lambda path: int(checkpoint_regex.search(path).group('epoch')),
                                 default=None)
    else:
        initial_checkpoint = None
    if initial_checkpoint is not None:
        initial_epoch = int(checkpoint_regex.search(initial_checkpoint).group('epoch'))
    else:
        initial_epoch = 0

    config.initial_checkpoint = initial_checkpoint
    config.initial_epoch = initial_epoch


def build_env(config):
    if not isinstance(config.env, gym.Env):
        env = gym.make(config.env)
        env.seed(config.random_seed)

        env = NormalizedAction(FlattenedAction(env))
        if config.vision_observation:
            env = VisionObservation(env, image_size=(config.image_size, config.image_size))
        else:
            env = FlattenedObservation(env)
        if config.n_frames > 1:
            env = ConcatenatedObservation(env, n_frames=config.n_frames, dim=0)

        config.env = env
        config.observation_dim = env.observation_space.shape[0]
        config.action_dim = env.action_space.shape[0]
        try:
            config.max_episode_steps = min(config.max_episode_steps, env.spec.max_episode_steps)
        except AttributeError:
            pass
        except TypeError:
            pass

    return config.env


def build_encoder(config):
    build_env(config)

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
        state_encoder = VanillaRecurrentNeuralNetwork(n_dims_before_lstm=[config.observation_dim,
                                                                          *config.encoder_hidden_dims_before_lstm],
                                                      n_dims_lstm_hidden=config.encoder_hidden_dims_lstm,
                                                      n_dims_after_lstm=[*config.encoder_hidden_dims_after_lstm,
                                                                         config.state_dim],
                                                      skip_connection=config.skip_connection,
                                                      activation=config.activation,
                                                      output_activation=None)
    elif config.CNN_encoder:
        n_hidden_channels = config.encoder_hidden_channels
        kernel_sizes = config.kernel_sizes
        strides = config.strides
        paddings = config.paddings
        while len(kernel_sizes) < len(n_hidden_channels):
            kernel_sizes.append(3)
        while len(strides) < len(kernel_sizes):
            strides.append(1)
        while len(paddings) < len(kernel_sizes):
            paddings.append(kernel_sizes[len(paddings)] // 2)
        state_encoder = VanillaConvolutionalNetwork(input_channels=config.observation_dim,
                                                    output_dim=config.state_dim,
                                                    n_hidden_channels=n_hidden_channels,
                                                    batch_normalization=False,
                                                    output_activation=None,
                                                    **config.build_dict_from_keys(['kernel_sizes',
                                                                                   'strides',
                                                                                   'paddings',
                                                                                   'activation']))

    config.state_encoder = state_encoder
    config.state_dim = state_dim

    return state_encoder


def build_model(config):
    model_kwargs = config.build_dict_from_keys(['env',
                                                'state_encoder',
                                                'state_dim',
                                                'action_dim',
                                                'hidden_dims',
                                                'activation',
                                                'initial_alpha',
                                                'n_samplers',
                                                'buffer_capacity',
                                                'devices',
                                                'random_seed'])
    if config.mode == 'train':
        model_kwargs.update(config.build_dict_from_keys(['soft_q_lr',
                                                         'policy_lr',
                                                         'alpha_lr',
                                                         'weight_decay']))

        if not config.RNN_encoder:
            from sac.model import Trainer as Model
        else:
            from sac.rnn.model import Trainer as Model
    else:
        if not config.RNN_encoder:
            from sac.model import Tester as Model
        else:
            from sac.rnn.model import Tester as Model

    model = Model(**model_kwargs)
    model.print_info()

    if config.initial_checkpoint is not None:
        model.load_model(path=config.initial_checkpoint)

    return model


def train_loop(model, config, update_kwargs):
    with SummaryWriter(log_dir=os.path.join(config.log_dir, 'trainer'), comment='trainer') as writer:
        n_initial_samples = model.collector.n_total_steps
        while model.collector.n_total_steps == n_initial_samples:
            time.sleep(0.1)

        setproctitle(title='trainer')
        for epoch in range(config.initial_epoch + 1, config.n_epochs + 1):
            epoch_soft_q_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_alpha = 0.0
            with tqdm.trange(config.n_updates, desc=f'Training {epoch}/{config.n_epochs}') as pbar:
                for i in pbar:
                    soft_q_loss, policy_loss, alpha, info = model.update(**update_kwargs)

                    buffer_size = model.replay_buffer.size
                    try:
                        update_sample_ratio = (config.n_samples_per_update * model.global_step) / \
                                              (model.collector.n_total_steps - n_initial_samples)
                    except ZeroDivisionError:
                        update_sample_ratio = config.update_sample_ratio
                    epoch_soft_q_loss += (soft_q_loss - epoch_soft_q_loss) / (i + 1)
                    epoch_policy_loss += (policy_loss - epoch_policy_loss) / (i + 1)
                    epoch_alpha += (alpha - epoch_alpha) / (i + 1)
                    writer.add_scalar(tag='train/soft_q_loss', scalar_value=soft_q_loss,
                                      global_step=model.global_step)
                    writer.add_scalar(tag='train/policy_loss', scalar_value=policy_loss,
                                      global_step=model.global_step)
                    writer.add_scalar(tag='train/temperature_parameter', scalar_value=alpha,
                                      global_step=model.global_step)
                    writer.add_scalar(tag='train/buffer_size', scalar_value=buffer_size,
                                      global_step=model.global_step)
                    writer.add_scalar(tag='train/update_sample_ratio', scalar_value=update_sample_ratio,
                                      global_step=model.global_step)
                    pbar.set_postfix(OrderedDict([('global_step', model.global_step),
                                                  ('soft_q_loss', epoch_soft_q_loss),
                                                  ('policy_loss', epoch_policy_loss),
                                                  ('n_samples', f'{model.collector.n_total_steps:.2E}'),
                                                  ('update/sample', f'{update_sample_ratio:.1f}')]))
                    if update_sample_ratio < config.update_sample_ratio:
                        model.collector.pause()
                    else:
                        model.collector.resume()

            writer.add_scalar(tag='epoch/soft_q_loss', scalar_value=epoch_soft_q_loss, global_step=epoch)
            writer.add_scalar(tag='epoch/policy_loss', scalar_value=epoch_policy_loss, global_step=epoch)
            writer.add_scalar(tag='epoch/temperature_parameter', scalar_value=epoch_alpha, global_step=epoch)

            writer.add_figure(tag='epoch/action_scaler_1',
                              figure=model.soft_q_net_1.action_scaler.plot(),
                              global_step=epoch)
            writer.add_figure(tag='epoch/action_scaler_2',
                              figure=model.soft_q_net_2.action_scaler.plot(),
                              global_step=epoch)

            writer.flush()
            if epoch % 10 == 0:
                model.save_model(path=os.path.join(config.checkpoint_dir, f'checkpoint-{epoch}.pkl'))


def train(model, config):
    update_kwargs = config.build_dict_from_keys(['batch_size',
                                                 'normalize_rewards',
                                                 'reward_scale',
                                                 'adaptive_entropy',
                                                 'gamma',
                                                 'soft_tau'])
    update_kwargs.update(target_entropy=-1.0 * config.action_dim)

    print(f'Start parallel sampling using {config.n_samplers} samplers '
          f'at {tuple(map(str, model.collector.devices))}.')

    model.collector.eval()
    while model.replay_buffer.size < 10 * config.n_samples_per_update:
        model.sample(n_episodes=10,
                     max_episode_steps=config.max_episode_steps,
                     deterministic=False,
                     random_sample=config.RNN_encoder,
                     render=config.render)

    model.collector.train()
    samplers = model.async_sample(n_episodes=np.inf,
                                  deterministic=False,
                                  random_sample=False,
                                  **config.build_dict_from_keys(['max_episode_steps',
                                                                 'render',
                                                                 'log_episode_video',
                                                                 'log_dir']))

    try:
        train_loop(model, config, update_kwargs)
    except KeyboardInterrupt:
        pass
    except Exception:
        raise
    finally:
        for sampler in samplers:
            if sampler.is_alive():
                sampler.terminate()
            sampler.join()


def test(model, config):
    with SummaryWriter(log_dir=config.log_dir) as writer:
        print(f'Start parallel sampling using {config.n_samplers} samplers '
              f'at {tuple(map(str, model.collector.devices))}.')

        model.sample(random_sample=False,
                     **config.build_dict_from_keys([
                         'n_episodes',
                         'max_episode_steps',
                         'deterministic',
                         'render',
                         'log_episode_video',
                         'log_dir'
                     ]))

        episode_steps = np.asanyarray(model.collector.episode_steps)
        episode_rewards = np.asanyarray(model.collector.episode_rewards)
        average_reward = episode_rewards / episode_steps
        writer.add_histogram(tag='test/cumulative_reward', values=episode_rewards)
        writer.add_histogram(tag='test/average_reward', values=average_reward)
        writer.add_histogram(tag='test/episode_steps', values=episode_steps)

        results = {
            'Metrics': ['Cumulative Reward', 'Average Reward', 'Episode Steps'],
            'Mean': list(map(np.mean, [episode_rewards, average_reward, episode_steps])),
            'Stddev': list(map(np.std, [episode_rewards, average_reward, episode_steps])),
        }
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
        except ImportError:
            for metric, mean, stddev in zip(results['Metrics'], results['Mean'], results['Stddev']):
                print(f'{metric}: {dict(mean=mean, stddev=stddev)}')


config = get_config()


def main():
    global config

    initialize(config)

    model = build_model(config)
    if config.mode == 'train' and config.initial_epoch < config.n_epochs:
        train(model, config)
    elif config.mode == 'test':
        test(model, config)


if __name__ == '__main__':
    main()
