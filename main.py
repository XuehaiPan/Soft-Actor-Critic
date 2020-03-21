import glob
import os
import random
import re
from collections import OrderedDict
from datetime import datetime

import gym
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.environment import NormalizedActions


USE_LSTM = True
if not USE_LSTM:
    ALGORITHM_NAME = 'SAC'
else:
    ALGORITHM_NAME = 'SAC-LSTM'

ENV_NAME = 'Pendulum-v0'
ENV = NormalizedActions(gym.make(ENV_NAME))
try:
    MAX_EPISODE_STEPS = min(500, ENV.spec.max_episode_steps)
except AttributeError:
    MAX_EPISODE_STEPS = 500

N_EPISODES_EACH_SAMPLE = 1
N_UPDATES_EACH_SAMPLE = 256
BATCH_SIZE = 256
if USE_LSTM:
    N_UPDATES_EACH_SAMPLE = 32
    BATCH_SIZE = 16

DETERMINISTIC = False

TOTAL_EPOCHS = 2000
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 1E-4
BUFFER_CAPACITY = 100000

GPU = True
DEVICE_IDX = 0
LOAD_CHECKPOINT = False
RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
ENV.seed(RANDOM_SEED)

if GPU:
    DEVICE = torch.device(f'cuda:{DEVICE_IDX}' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cpu')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR_ROOT = os.path.join(ROOT_DIR, 'logs', ENV_NAME, ALGORITHM_NAME)
CURRENT_TIME = datetime.now().strftime('%Y-%m-%d-%T')
LOG_DIR = os.path.join(LOG_DIR_ROOT, CURRENT_TIME)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints', ENV_NAME, ALGORITHM_NAME)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.system(f'ln -sfn {CURRENT_TIME} {os.path.join(LOG_DIR_ROOT, "latest")}')

CHECKPOINT_REGEX = re.compile(r'^(.*/)?[\w-]*-(?P<epoch>\d+)\.pkl$')
if LOAD_CHECKPOINT:
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
    global MAX_EPISODE_STEPS

    writer = SummaryWriter(log_dir=LOG_DIR)
    if not USE_LSTM:
        from sac.trainer import Trainer
        trainer = Trainer(env=ENV,
                          state_dim=ENV.observation_space.shape[0],
                          action_dim=ENV.action_space.shape[0],
                          hidden_dims=[512, 512, 512],
                          soft_q_lr=LEARNING_RATE,
                          policy_lr=LEARNING_RATE,
                          alpha_lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY,
                          buffer_capacity=BUFFER_CAPACITY,
                          writer=writer,
                          device=DEVICE)
    else:
        from sac.rnn.trainer import Trainer
        trainer = Trainer(env=ENV,
                          state_dim=ENV.observation_space.shape[0],
                          action_dim=ENV.action_space.shape[0],
                          hidden_dims_before_lstm=[512, 512],
                          hidden_dims_lstm=[512],
                          hidden_dims_after_lstm=[512, 512],
                          soft_q_lr=LEARNING_RATE,
                          policy_lr=LEARNING_RATE,
                          alpha_lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY,
                          buffer_capacity=BUFFER_CAPACITY,
                          writer=writer,
                          device=DEVICE)
    trainer.print_info()

    if INITIAL_CHECKPOINT is not None:
        trainer.load_model(path=INITIAL_CHECKPOINT)

    if INITIAL_EPOCH < TOTAL_EPOCHS:
        while trainer.replay_buffer.size < BATCH_SIZE - 1:
            trainer.env_sample(n_episodes=N_EPISODES_EACH_SAMPLE,
                               max_episode_steps=MAX_EPISODE_STEPS,
                               deterministic=DETERMINISTIC,
                               epsilon=1.0)
        global_step = 0
        for epoch in range(INITIAL_EPOCH + 1, TOTAL_EPOCHS + 1):
            if epoch >= 1000:
                try:
                    MAX_EPISODE_STEPS = ENV.spec.max_episode_steps
                except AttributeError:
                    pass
            trainer.env_sample(n_episodes=N_EPISODES_EACH_SAMPLE,
                               max_episode_steps=MAX_EPISODE_STEPS,
                               deterministic=DETERMINISTIC,
                               epsilon=0.0)
            q_value_loss_list = []
            policy_loss_list = []
            with tqdm.trange(N_UPDATES_EACH_SAMPLE, desc=f'Training {epoch}/{TOTAL_EPOCHS}') as pbar:
                for i in pbar:
                    q_value_loss_1, q_value_loss_2, policy_loss = trainer.update(batch_size=BATCH_SIZE,
                                                                                 reward_scale=10,
                                                                                 target_entropy=-1.0 * trainer.action_dim,
                                                                                 soft_tau=0.01)
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
            if epoch % 20 == 0:
                trainer.save_model(path=os.path.join(CHECKPOINT_DIR, f'checkpoint-{epoch}.pkl'))

    ENV.close()


if __name__ == '__main__':
    main()
