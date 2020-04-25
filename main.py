import argparse
import glob
import os
import random
import re
import time
from collections import OrderedDict

import gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.environment import FlattenedAction, NormalizedAction, \
    FlattenedObservation, ConcatenatedObservation
from common.network_base import VanillaNeuralNetwork


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Train or test Soft Actor-Critic controller.')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                    help='mode (default: train)')
parser.add_argument('--gpu', type=int, default=None, nargs='+', metavar='CUDA_DEVICE',
                    help='GPU devices (use CPU if not present)')
parser.add_argument('--env', type=str, default='BipedalWalker-v3',
                    help='environment to train on (default: BipedalWalker-v3)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--net', type=str, choices=['FC', 'RNN'], default='FC',
                    help='architecture of controller network')
parser.add_argument('--activation', type=str, choices=['ReLU', 'LeakyReLU'], default='ReLU',
                    help='activation function in networks (default: ReLU)')
parser.add_argument('--deterministic', action='store_true', help='deterministic in evaluation')
fc_group = parser.add_argument_group('FC controller')
fc_group.add_argument('--hidden-dims', type=int, default=[], nargs='+',
                      help='hidden dimensions of FC controller')
rnn_group = parser.add_argument_group('RNN controller')
rnn_group.add_argument('--hidden-dims-before-lstm', type=int, default=[], nargs='+',
                       help='hidden FC dimensions before LSTM layers in RNN controller')
rnn_group.add_argument('--hidden-dims-lstm', type=int, default=[], nargs='+',
                       help='LSTM hidden dimensions of RNN controller')
rnn_group.add_argument('--hidden-dims-after-lstm', type=int, default=[], nargs='+',
                       help='hidden FC dimensions after LSTM layers in RNN controller')
rnn_group.add_argument('--skip-connection', action='store_true', default=False,
                       help='add skip connection beside LSTM layers in RNN controller')
rnn_group.add_argument('--step-size', type=int, default=16,
                       help='number of continuous steps for update (default: 16)')
encoder_group = parser.add_argument_group('state encoder')
encoder_group.add_argument('--n-frames', type=int, default=1,
                           help='concatenate original N consecutive observations as a new observation (default: 1)')
encoder_group.add_argument('--state-dim', type=int, default=None,
                           help='target state dimension of encoded state (use env.observation_space.shape if not present)')
encoder_group.add_argument('--encoder-hidden-dims', type=int, default=[], nargs='+',
                           help='hidden dimensions of FC state encoder')
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
parser.add_argument('--buffer-capacity', type=int, default=1000000,
                    help='capacity of replay buffer (default: 1000000)')
parser.add_argument('--update-sample-ratio', type=float, default=2.0,
                    help='speed ratio of training and sampling (default: 2.0)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--soft-tau', type=float, default=0.01,
                    help='soft update factor for target networks (default: 0.01)')
parser.add_argument('--normalize-rewards', action='store_true',
                    help='normalize rewards for training')
parser.add_argument('--reward-scale', type=float, default=1.0,
                    help='reward scale factor for normalized rewards (default: 1.0)')
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
alpha_group.add_argument('--initial-alpha', type=float, default=1.0,
                         help='initial value of temperature parameter (default: 1.0)')
alpha_group.add_argument('--adaptive-entropy', action='store_true',
                         help='auto update temperature parameter while training')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--random-seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--log-dir', type=str, default=os.path.join(ROOT_DIR, 'logs'),
                    help='folder to save tensorboard logs')
parser.add_argument('--checkpoint-dir', type=str, default=os.path.join(ROOT_DIR, 'checkpoints'),
                    help='folder to save checkpoint from')
parser.add_argument('--load-checkpoint', action='store_true',
                    help='load latest checkpoint in checkpoint dir')
args = parser.parse_args()

MODE = args.mode

RANDOM_SEED = args.random_seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

USE_LSTM = (args.net == 'RNN')
if USE_LSTM:
    HIDDEN_DIMS_BEFORE_LSTM = args.hidden_dims_before_lstm
    HIDDEN_DIMS_AFTER_LSTM = args.hidden_dims_after_lstm
    HIDDEN_DIMS_LSTM = args.hidden_dims_lstm
    SKIP_CONNECTION = args.skip_connection
    STEP_SIZE = args.step_size
else:
    HIDDEN_DIMS = args.hidden_dims

if args.activation == 'ReLU':
    ACTIVATION = F.relu
else:
    ACTIVATION = F.leaky_relu

ENV_NAME = args.env
ENV = gym.make(ENV_NAME)
ENV.seed(RANDOM_SEED)
ENV = FlattenedObservation(NormalizedAction(FlattenedAction(ENV)))
ACTION_DIM = ENV.action_space.shape[0]
if args.n_frames > 1:
    ENV = ConcatenatedObservation(ENV, n_frames=args.n_frames, dim=0)
ENV_OBSERVATION_DIM = ENV.observation_space.shape[0]
STATE_DIM = (args.state_dim or ENV_OBSERVATION_DIM)

if args.state_dim is not None or len(args.encoder_hidden_dims) > 0:
    STATE_ENCODER = VanillaNeuralNetwork(n_dims=[ENV_OBSERVATION_DIM, *args.encoder_hidden_dims, STATE_DIM],
                                         activation=ACTIVATION, output_activation=None)
else:
    STATE_ENCODER = nn.Identity()
MAX_EPISODE_STEPS = args.max_episode_steps
try:
    MAX_EPISODE_STEPS = min(MAX_EPISODE_STEPS, ENV.spec.max_episode_steps)
except AttributeError:
    pass
RENDER = args.render

N_EPISODES = args.n_episodes
N_EPOCHS = args.n_epochs
N_SAMPLERS = args.n_samplers
BUFFER_CAPACITY = args.buffer_capacity
N_UPDATES = args.n_updates
BATCH_SIZE = args.batch_size
UPDATE_SAMPLE_RATIO = args.update_sample_ratio
GAMMA = args.gamma
SOFT_TAU = args.soft_tau
NORMALIZE_REWARDS = args.normalize_rewards
REWARD_SCALE = args.reward_scale

N_SAMPLES_PER_UPDATE = BATCH_SIZE
if USE_LSTM:
    N_SAMPLES_PER_UPDATE *= STEP_SIZE

DETERMINISTIC = args.deterministic

LR = args.lr
SOFT_Q_LR = (args.soft_q_lr or LR)
POLICY_LR = (args.policy_lr or LR)
ALPHA_LR = (args.alpha_lr or POLICY_LR)

INITIAL_ALPHA = args.initial_alpha
WEIGHT_DECAY = args.weight_decay
ADAPTIVE_ENTROPY = args.adaptive_entropy

if args.gpu is not None and torch.cuda.is_available():
    if len(args.gpu) == 0:
        args.gpu = [0]
    DEVICES = [torch.device(f'cuda:{cuda_device}') for cuda_device in args.gpu]
else:
    DEVICES = [torch.device('cpu')]

RANDOM_SEED = args.random_seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
ENV.seed(RANDOM_SEED)

LOG_DIR = args.log_dir
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_REGEX = re.compile(r'^(.*/)?[\w-]*-(?P<epoch>\d+)\.pkl$')
if MODE == 'test' or args.load_checkpoint:
    INITIAL_CHECKPOINT = max(glob.iglob(os.path.join(CHECKPOINT_DIR, '*.pkl')),
                             key=lambda path: int(CHECKPOINT_REGEX.search(path).group('epoch')),
                             default=None)
else:
    INITIAL_CHECKPOINT = None
if INITIAL_CHECKPOINT is not None:
    INITIAL_EPOCH = int(CHECKPOINT_REGEX.search(INITIAL_CHECKPOINT).group('epoch'))
else:
    INITIAL_EPOCH = 0


def main():
    model_kwargs = {}
    update_kwargs = {}
    initial_random_sample = True
    if not USE_LSTM:
        model_kwargs.update({'hidden_dims': HIDDEN_DIMS})
    else:
        model_kwargs.update({
            'hidden_dims_before_lstm': HIDDEN_DIMS_BEFORE_LSTM,
            'hidden_dims_lstm': HIDDEN_DIMS_LSTM,
            'hidden_dims_after_lstm': HIDDEN_DIMS_AFTER_LSTM,
            'skip_connection': SKIP_CONNECTION
        })

    if MODE == 'train':
        model_kwargs.update({
            'soft_q_lr': SOFT_Q_LR,
            'policy_lr': POLICY_LR,
            'alpha_lr': ALPHA_LR,
            'weight_decay': WEIGHT_DECAY
        })

        if not USE_LSTM:
            from sac.model import Trainer as Model
        else:
            from sac.rnn.model import Trainer as Model
            initial_random_sample = False
            update_kwargs.update({'step_size': STEP_SIZE})
    else:
        if not USE_LSTM:
            from sac.model import Tester as Model
        else:
            from sac.rnn.model import Tester as Model

    model = Model(env=ENV,
                  state_encoder=STATE_ENCODER,
                  state_dim=STATE_DIM,
                  action_dim=ACTION_DIM,
                  activation=ACTIVATION,
                  initial_alpha=INITIAL_ALPHA,
                  n_samplers=N_SAMPLERS,
                  buffer_capacity=BUFFER_CAPACITY,
                  devices=DEVICES,
                  random_seed=RANDOM_SEED,
                  **model_kwargs)

    model.print_info()

    if INITIAL_CHECKPOINT is not None:
        model.load_model(path=INITIAL_CHECKPOINT)

    print(f'Start parallel sampling using {N_SAMPLERS} samplers at {tuple(map(str, model.collector.devices))}.')
    if MODE == 'train' and INITIAL_EPOCH < N_EPOCHS:
        model.collector.eval()
        while model.replay_buffer.size < 10 * N_SAMPLES_PER_UPDATE:
            model.sample(n_episodes=10,
                         max_episode_steps=MAX_EPISODE_STEPS,
                         deterministic=False,
                         random_sample=initial_random_sample,
                         render=RENDER)

        model.collector.train()
        samplers = model.async_sample(n_episodes=np.inf,
                                      max_episode_steps=MAX_EPISODE_STEPS,
                                      deterministic=False,
                                      random_sample=False,
                                      render=RENDER,
                                      log_dir=LOG_DIR)

        try:
            train_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'trainer'), comment='trainer')
            n_initial_samples = model.collector.n_total_steps
            while model.collector.n_total_steps == n_initial_samples:
                time.sleep(0.1)

            for epoch in range(INITIAL_EPOCH + 1, N_EPOCHS + 1):
                epoch_soft_q_loss = 0.0
                epoch_policy_loss = 0.0
                epoch_alpha = 0.0
                with tqdm.trange(N_UPDATES, desc=f'Training {epoch}/{N_EPOCHS}') as pbar:
                    for i in pbar:
                        soft_q_loss, policy_loss, alpha, info = model.update(batch_size=BATCH_SIZE,
                                                                             normalize_rewards=NORMALIZE_REWARDS,
                                                                             reward_scale=REWARD_SCALE,
                                                                             adaptive_entropy=ADAPTIVE_ENTROPY,
                                                                             target_entropy=-1.0 * model.action_dim,
                                                                             gamma=GAMMA,
                                                                             soft_tau=SOFT_TAU,
                                                                             **update_kwargs)

                        buffer_size = model.replay_buffer.size
                        try:
                            update_sample_ratio = (N_SAMPLES_PER_UPDATE * model.global_step) / \
                                                  (model.collector.n_total_steps - n_initial_samples)
                        except ZeroDivisionError:
                            update_sample_ratio = UPDATE_SAMPLE_RATIO
                        epoch_soft_q_loss += (soft_q_loss - epoch_soft_q_loss) / (i + 1)
                        epoch_policy_loss += (policy_loss - epoch_policy_loss) / (i + 1)
                        epoch_alpha += (alpha - epoch_alpha) / (i + 1)
                        train_writer.add_scalar(tag='train/soft_q_loss', scalar_value=soft_q_loss,
                                                global_step=model.global_step)
                        train_writer.add_scalar(tag='train/policy_loss', scalar_value=policy_loss,
                                                global_step=model.global_step)
                        train_writer.add_scalar(tag='train/temperature_parameter', scalar_value=alpha,
                                                global_step=model.global_step)
                        train_writer.add_scalar(tag='train/buffer_size', scalar_value=buffer_size,
                                                global_step=model.global_step)
                        train_writer.add_scalar(tag='train/update_sample_ratio', scalar_value=update_sample_ratio,
                                                global_step=model.global_step)
                        pbar.set_postfix(OrderedDict([('global_step', model.global_step),
                                                      ('soft_q_loss', epoch_soft_q_loss),
                                                      ('policy_loss', epoch_policy_loss),
                                                      ('n_samples', f'{model.collector.n_total_steps:.2E}'),
                                                      ('update/sample', f'{update_sample_ratio:.1f}')]))
                        if update_sample_ratio < UPDATE_SAMPLE_RATIO:
                            model.collector.pause()
                        else:
                            model.collector.resume()

                train_writer.add_scalar(tag='epoch/soft_q_loss', scalar_value=epoch_soft_q_loss, global_step=epoch)
                train_writer.add_scalar(tag='epoch/policy_loss', scalar_value=epoch_policy_loss, global_step=epoch)
                train_writer.add_scalar(tag='epoch/temperature_parameter', scalar_value=epoch_alpha, global_step=epoch)

                train_writer.flush()
                if epoch % 10 == 0:
                    model.save_model(path=os.path.join(CHECKPOINT_DIR, f'checkpoint-{epoch}.pkl'))

            train_writer.close()
        except KeyboardInterrupt:
            pass
        except Exception:
            raise
        finally:
            for sampler in samplers:
                if sampler.is_alive():
                    sampler.terminate()
                sampler.join()

    elif MODE == 'test':
        test_writer = SummaryWriter(log_dir=LOG_DIR)

        model.sample(n_episodes=N_EPISODES,
                     max_episode_steps=MAX_EPISODE_STEPS,
                     deterministic=DETERMINISTIC,
                     random_sample=False,
                     render=RENDER,
                     log_dir=LOG_DIR)
        episode_steps = np.asanyarray(model.collector.episode_steps)
        episode_rewards = np.asanyarray(model.collector.episode_rewards)
        average_reward = episode_rewards / episode_steps
        test_writer.add_histogram(tag='test/cumulative_reward', values=episode_rewards)
        test_writer.add_histogram(tag='test/average_reward', values=average_reward)
        test_writer.add_histogram(tag='test/episode_steps', values=episode_steps)

        try:
            import pandas as pd
            df = pd.DataFrame({
                'Metrics': ['Cumulative Reward', 'Average Reward', 'Episode Steps'],
                'Mean': list(map(np.mean, [episode_rewards, average_reward, episode_steps])),
                'Std': list(map(np.std, [episode_rewards, average_reward, episode_steps])),
            })
            print(df.to_string(index=False))
        except ImportError:
            pass

        test_writer.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        raise
    finally:
        ENV.close()
