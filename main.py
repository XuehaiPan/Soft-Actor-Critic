import argparse
import glob
import os
import random
import re
from collections import OrderedDict
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.environment import NormalizedActions


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Train or test Soft Actor-Critic controller.')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                    help='mode (default: train)')
parser.add_argument('--gpu', type=int, nargs='?', const=0, default=None, metavar='CUDA_DEVICE',
                    help='center GPU device (use CPU when not present)')
parser.add_argument('--env', type=str, default='BipedalWalker-v3',
                    help='environment to train on (default: BipedalWalker-v3)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--net', type=str, choices=['FC', 'RNN'], default='FC',
                    help='network architecture')
parser.add_argument('--activation', type=str, choices=['ReLU', 'LeakyReLU'], default='ReLU',
                    help='activation function in networks (default: ReLU)')
parser.add_argument('--deterministic', action='store_true', help='deterministic in evaluation')
fc_group = parser.add_argument_group('FC')
fc_group.add_argument('--hidden-dims', type=int, default=[512], nargs='+',
                      help='hidden dimensions')
rnn_group = parser.add_argument_group('RNN')
rnn_group.add_argument('--hidden-dims-before-lstm', type=int, default=[512], nargs='+',
                       help='hidden FC dimensions before LSTM')
rnn_group.add_argument('--hidden-dims-lstm', type=int, default=[512], nargs='+',
                       help='LSTM hidden dimensions')
rnn_group.add_argument('--hidden-dims-after-lstm', type=int, default=[512], nargs='+',
                       help='hidden FC dimensions after LSTM')
rnn_group.add_argument('--skip-connection', action='store_true', default=False,
                       help='add skip connection beside LSTM')
rnn_group.add_argument('--max-step-size', type=int, default=32,
                       help='max continuous steps for update (default: 32)')
parser.add_argument('--max-episodes', type=int, default=1000,
                    help='max learning episodes (default: 1000)')
parser.add_argument('--max-episode-steps', type=int, default=1000,
                    help='max steps per episode (default: 1000)')
parser.add_argument('--n-updates', type=int, default=32,
                    help='number of learning updates after sample a new episode (default: 32)')
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size (default: 256)')
parser.add_argument('--buffer-capacity', type=int, default=1000000,
                    help='capacity of replay buffer (default: 1000000)')
parser.add_argument('--lr', type=float, default=1E-4,
                    help='learning rate (default: 0.0001)')
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

USE_LSTM = (args.net == 'RNN')
if USE_LSTM:
    HIDDEN_DIMS_BEFORE_LSTM = args.hidden_dims_before_lstm
    HIDDEN_DIMS_AFTER_LSTM = args.hidden_dims_after_lstm
    HIDDEN_DIMS_LSTM = args.hidden_dims_lstm
    SKIP_CONNECTION = args.skip_connection
    MAX_STEP_SIZE = args.max_step_size
else:
    HIDDEN_DIMS = args.hidden_dims

if args.activation == 'ReLU':
    ACTIVATION = F.relu
else:
    ACTIVATION = F.leaky_relu

ENV_NAME = args.env
ENV = NormalizedActions(gym.make(ENV_NAME))
MAX_EPISODE_STEPS = args.max_episode_steps
try:
    MAX_EPISODE_STEPS = min(MAX_EPISODE_STEPS, ENV.spec.max_episode_steps)
except AttributeError:
    pass
RENDER = args.render

MAX_EPISODES = args.max_episodes
BUFFER_CAPACITY = args.buffer_capacity
N_UPDATES = args.n_updates
BATCH_SIZE = args.batch_size

DETERMINISTIC = args.deterministic

LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay

if args.gpu is not None:
    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cpu')

RANDOM_SEED = args.random_seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
ENV.seed(RANDOM_SEED)

CURRENT_TIME = datetime.now().strftime('%Y-%m-%d-%T')
LOG_DIR = os.path.join(args.log_dir, CURRENT_TIME)
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_REGEX = re.compile(r'^(.*/)?[\w-]*-(?P<epoch>\d+)\.pkl$')
if args.load_checkpoint:
    INITIAL_CHECKPOINT = max(glob.iglob(os.path.join(CHECKPOINT_DIR, '*.pkl')),
                             key=lambda path: int(CHECKPOINT_REGEX.search(path).group('epoch')),
                             default=None)
else:
    INITIAL_CHECKPOINT = None
if INITIAL_CHECKPOINT is not None:
    INITIAL_EPISODES = int(CHECKPOINT_REGEX.search(INITIAL_CHECKPOINT).group('epoch'))
else:
    INITIAL_EPISODES = 0


def main():
    writer = SummaryWriter(log_dir=LOG_DIR)
    update_kwargs = {}
    if not USE_LSTM:
        from sac.trainer import Trainer
        trainer = Trainer(env=ENV,
                          state_dim=ENV.observation_space.shape[0],
                          action_dim=ENV.action_space.shape[0],
                          hidden_dims=HIDDEN_DIMS,
                          activation=ACTIVATION,
                          soft_q_lr=LEARNING_RATE,
                          policy_lr=LEARNING_RATE,
                          alpha_lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY,
                          buffer_capacity=BUFFER_CAPACITY,
                          device=DEVICE)
    else:
        from sac.rnn.trainer import Trainer
        trainer = Trainer(env=ENV,
                          state_dim=ENV.observation_space.shape[0],
                          action_dim=ENV.action_space.shape[0],
                          hidden_dims_before_lstm=HIDDEN_DIMS_BEFORE_LSTM,
                          hidden_dims_lstm=HIDDEN_DIMS_LSTM,
                          hidden_dims_after_lstm=HIDDEN_DIMS_AFTER_LSTM,
                          activation=ACTIVATION,
                          skip_connection=SKIP_CONNECTION,
                          soft_q_lr=LEARNING_RATE,
                          policy_lr=LEARNING_RATE,
                          alpha_lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY,
                          buffer_capacity=BUFFER_CAPACITY,
                          device=DEVICE)
        update_kwargs.update({'max_step_size': MAX_STEP_SIZE})
    trainer.print_info()

    if INITIAL_CHECKPOINT is not None:
        trainer.load_model(path=INITIAL_CHECKPOINT)
        trainer.train()

    if INITIAL_EPISODES < MAX_EPISODES:
        while trainer.replay_buffer.size < 10 * BATCH_SIZE:
            trainer.env_sample(n_episodes=1,
                               max_episode_steps=MAX_EPISODE_STEPS,
                               random_sample=True,
                               render=RENDER)
        global_step = 0
        for epoch in range(INITIAL_EPISODES + 1, MAX_EPISODES + 1):
            trainer.env_sample(n_episodes=1,
                               max_episode_steps=MAX_EPISODE_STEPS,
                               deterministic=DETERMINISTIC,
                               random_sample=False,
                               render=RENDER,
                               writer=writer)
            q_value_loss_list = []
            policy_loss_list = []
            with tqdm.trange(N_UPDATES, desc=f'Training {epoch}/{MAX_EPISODES}') as pbar:
                for i in pbar:
                    q_value_loss_1, q_value_loss_2, policy_loss = trainer.update(batch_size=BATCH_SIZE,
                                                                                 normalize_rewards=True,
                                                                                 auto_entropy=True,
                                                                                 target_entropy=-1.0 * trainer.action_dim,
                                                                                 soft_tau=0.01,
                                                                                 **update_kwargs)
                    global_step += 1
                    q_value_loss_list.append(q_value_loss_1)
                    q_value_loss_list.append(q_value_loss_2)
                    policy_loss_list.append(policy_loss)
                    writer.add_scalar(tag='train/q_value_loss_1', scalar_value=q_value_loss_1, global_step=global_step)
                    writer.add_scalar(tag='train/q_value_loss_2', scalar_value=q_value_loss_2, global_step=global_step)
                    writer.add_scalar(tag='train/policy_loss', scalar_value=policy_loss, global_step=global_step)
                    pbar.set_postfix(OrderedDict([('global_step', global_step),
                                                  ('q_value_loss', np.mean(q_value_loss_list)),
                                                  ('policy_loss', np.mean(policy_loss_list))]))

            writer.flush()
            if epoch % 50 == 0:
                trainer.save_model(path=os.path.join(CHECKPOINT_DIR, f'checkpoint-{epoch}.pkl'))

    ENV.close()


if __name__ == '__main__':
    main()
