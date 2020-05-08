import copy
import glob
import os
import re

import torch


def clone_network(src_net, device=None):
    if device is None and hasattr(src_net, 'device'):
        device = src_net.device

    dst_net = copy.deepcopy(src_net)

    if device is not None:
        try:
            dst_net.device = device
        except AttributeError:
            pass
        dst_net.to(device)

    return dst_net


def sync_params(src_net, dst_net, soft_tau=1.0):
    if soft_tau == 1.0:
        for src_param, dst_param in zip(src_net.parameters(), dst_net.parameters()):
            dst_param.data.copy_(src_param.data)
    else:
        for src_param, dst_param in zip(src_net.parameters(), dst_net.parameters()):
            dst_param.data.copy_(dst_param.data * (1.0 - soft_tau) + src_param.data * soft_tau)


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
