import copy
import glob
import json
import os
import re

import torch


CHECKPOINT_REGEX = re.compile(r'^(.*/)?[\w-]*-(?P<epoch>\d+)-(?P<reward>[\-+Ee\d.]+)\.pkl$')


def clone_network(src_net, device=None):
    if device is None and hasattr(src_net, 'device'):
        device = src_net.device

    dst_net = copy.deepcopy(src_net)

    if device is not None:
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
        return max(glob.iglob(os.path.join(checkpoint_dir, '*.pkl')),
                   key=lambda path: float(CHECKPOINT_REGEX.search(path).group(by)),
                   default=None)
    except AttributeError:
        return None


def check_logging(config):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for directory in (config.log_dir, config.checkpoint_dir):
        with open(file=os.path.join(directory, 'args.json'), mode='w') as file:
            json.dump(config, file, indent=4, default=str)

    if config.mode == 'test' or config.load_checkpoint:
        initial_checkpoint = get_checkpoint(config.checkpoint_dir, by='epoch')
    else:
        initial_checkpoint = None
    if initial_checkpoint is not None:
        initial_epoch = int(CHECKPOINT_REGEX.search(initial_checkpoint).group('epoch'))
    else:
        initial_epoch = 0

    config.initial_checkpoint = initial_checkpoint
    config.initial_epoch = initial_epoch
