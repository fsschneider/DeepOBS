# -*- coding: utf-8 -*-

import tensorflow as tf

TF_FLOAT_DTYPE = tf.float32


def get_float_dtype():
    return TF_FLOAT_DTYPE


def set_float_dtype(dtype):
    global TF_FLOAT_DTYPE
    TF_FLOAT_DTYPE = dtype
