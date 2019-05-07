# -*- coding: utf-8 -*-

DATA_DIR = "data_deepobs/pytorch"
BASELINE_DIR = "baselines_deepobs"


def get_data_dir():
    return DATA_DIR


def set_data_dir(data_dir):
    global DATA_DIR
    DATA_DIR = data_dir


def get_baseline_dir():
    return BASELINE_DIR


def set_baseline_dir(baseline_dir):
    global BASELINE_DIR
    BASELINE_DIR = baseline_dir