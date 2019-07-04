# -*- coding: utf-8 -*-
import torch

DATA_DIR = "data_deepobs/pytorch"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0

def get_num_workers():
    return NUM_WORKERS

def set_num_workers(num_workers):
    global NUM_WORKERS
    NUM_WORKERS = num_workers

def get_data_dir():
    return DATA_DIR

def set_data_dir(data_dir):
    global DATA_DIR
    DATA_DIR = data_dir

def get_default_device():
    return DEFAULT_DEVICE

def set_default_device(device):
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device