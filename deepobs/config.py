# -*- coding: utf-8 -*-

BASELINE_DIR = "baselines_deepobs"

def get_baseline_dir():
    return BASELINE_DIR

def set_baseline_dir(baseline_dir):
    global BASELINE_DIR
    BASELINE_DIR = baseline_dir
