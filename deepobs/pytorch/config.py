# -*- coding: utf-8 -*-
import torch

DATA_DIR = "data_deepobs/pytorch"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_dir():
    return DATA_DIR

def set_data_dir(data_dir):
    global DATA_DIR
    DATA_DIR = data_dir

def get_default_device():
    return DEFAULT_DEVICE

def set_default_device(device):
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = torch.device(device)