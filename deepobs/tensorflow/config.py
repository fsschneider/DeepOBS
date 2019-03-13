# -*- coding: utf-8 -*-

import tensorflow as tf

DATA_DIR = "data_deepobs"
BASELINE_DIR = "baselines_deepobs"
TF_FLOAT_DTYPE = tf.float32


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


def get_float_dtype():
    return TF_FLOAT_DTYPE


def set_float_dtype(dtype):
    global TF_FLOAT_DTYPE
    TF_FLOAT_DTYPE = dtype
