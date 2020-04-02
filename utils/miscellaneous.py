import errno
import os
import yaml
import torch
import shutil
import numpy as np
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import PIL

from torch.utils.collect_env import get_pretty_env_info

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_configs(file_path):
    cfg_file = file_path
    with open(cfg_file, 'r') as config_file:
        cfgs = yaml.safe_load(config_file)

    return cfgs


def read_npy(file_path):
    data = np.load(file_path)
    print(data)
    print(data.shape)

    return data



def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_pil_version()
    return env_str


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')