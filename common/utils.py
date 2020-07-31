import copy
import glob
import json
import os
import re
from functools import partial

import numpy as np
import torch
import torch.nn as nn


CHECKPOINT_FORMAT = '{prefix}epoch({epoch})-reward({reward:+.2E}){suffix}.pkl'
CHECKPOINT_FORMAT = partial(CHECKPOINT_FORMAT.format, prefix='', suffix='')
CHECKPOINT_PATTERN = re.compile(r'^(.*/)?[\w-]*epoch\((?P<epoch>\d+)\)-reward\((?P<reward>[\-+Ee\d.]+)\)[\w-]*\.pkl$')


def clone_network(src_net, device=None):
    if device is None:
        device = getattr(src_net, 'device', None)

    dst_net = copy.deepcopy(src_net)

    if device is not None:
        dst_net.to(device)

    return dst_net


def sync_params(src_net, dst_net, soft_tau=1.0):
    assert 0.0 <= soft_tau <= 1.0
    assert type(src_net) == type(dst_net)

    if soft_tau == 0.0:
        return
    elif soft_tau == 1.0:
        for src_param, dst_param in zip(src_net.parameters(), dst_net.parameters()):
            dst_param.data.copy_(src_param.data)
    else:  # 0.0 < soft_tau < 1.0
        for src_param, dst_param in zip(src_net.parameters(), dst_net.parameters()):
            dst_param.data.copy_(dst_param.data * (1.0 - soft_tau) + src_param.data * soft_tau)


def init_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        n_params = 0
        for param in param_group['params']:
            n_params += param.size().numel()
        param_group['n_params'] = n_params


def clip_grad_norm(optimizer, max_norm=None, norm_type=2):
    for param_group in optimizer.param_groups:
        max_norm_x = max_norm
        if max_norm_x is None and 'n_params' in param_group:
            max_norm_x = 0.1 * np.sqrt(param_group['n_params'])
        if max_norm_x is not None:
            nn.utils.clip_grad.clip_grad_norm_(param_group['params'],
                                               max_norm=max_norm,
                                               norm_type=norm_type)


def check_devices(config):
    if config.gpu is not None and torch.cuda.is_available():
        if len(config.gpu) == 0:
            config.gpu = [0]
        devices = []
        for device in config.gpu:
            if isinstance(device, int):
                devices.append(torch.device(f'cuda:{device}'))
            elif device in ('c', 'cpu', 'C', 'CPU'):
                devices.append(torch.device('cpu'))
    else:
        devices = [torch.device('cpu')]

    config.devices = devices

    return devices


def get_checkpoint(checkpoint_dir, by='epoch'):
    try:
        checkpoints = glob.iglob(os.path.join(checkpoint_dir, '*.pkl'))
        matches = filter(None, map(CHECKPOINT_PATTERN.match, checkpoints))
        max_match = max(matches, key=lambda match: float(match.group(by)), default=None)
        return max_match.group()
    except AttributeError:
        return None


def check_logging(config):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    try:
        from IPython.core.formatters import PlainTextFormatter
        formatter = PlainTextFormatter()
    except ImportError:
        formatter = str

    for directory in (config.log_dir, config.checkpoint_dir):
        with open(file=os.path.join(directory, 'config.json'), mode='w') as file:
            json.dump(config, file, indent=4, default=formatter)

    if config.mode == 'test' or config.load_checkpoint:
        initial_checkpoint = get_checkpoint(config.checkpoint_dir, by='epoch')
    else:
        initial_checkpoint = None
    if initial_checkpoint is not None:
        initial_epoch = int(CHECKPOINT_PATTERN.search(initial_checkpoint).group('epoch'))
    else:
        initial_epoch = 0

    config.initial_checkpoint = initial_checkpoint
    config.initial_epoch = initial_epoch
