# -*- coding: utf-8 -*-
FRAMEWORK = 'pytorch'
BASELINE_DIR = "baselines_deepobs"
SMALL_TEST_SET = ['quadratic_deep', 'mnist_vae', 'fmnist_2c2d', 'cifar10_3c3d']
LARGE_TEST_SET = ['fmnist_vae', 'cifar100_allcnnc', 'svhn_wrn_164', 'tolstoi_char_rnn']

def get_framework():
    return FRAMEWORK
def set_framework(framework):
    global FRAMEWORK
    FRAMEWORK = framework

def get_baseline_dir():
    return BASELINE_DIR
def set_baseline_dir(baseline_dir):
    global BASELINE_DIR
    BASELINE_DIR = baseline_dir

def get_small_test_set():
    return SMALL_TEST_SET
def set_small_test_set(testset):
    global SMALL_TEST_SET
    SMALL_TEST_SET = testset

def get_large_test_set():
    return LARGE_TEST_SET
def set_large_test_set(testset):
    global LARGE_TEST_SET
    LARGE_TEST_SET = testset

# TODO at defaults for all testproblems
DEFAULT_TEST_PROBLEMS_SETTINGS = {
        'quadratic_deep': {
            'batch_size': 128,
            'num_epochs': 100
            },
        'mnist_vae': {
            'batch_size': 64,
            'num_epochs': 50
            },
        'fmnist_2c2d': {
            'batch_size': 128,
            'num_epochs': 100
            },
        'cifar10_3c3d': {
            'batch_size': 128,
            'num_epochs': 100
            },
        'fmnist_vae': {
            'batch_size': 64,
            'num_epochs': 100
            },
        'cifar100_allcnnc': {
            'batch_size': 256,
            'num_epochs': 350
            },
        'svhn_wrn164': {
            'batch_size': 128,
            'num_epochs': 160
            },
        'tolstoi_char_rnn': {
            'batch_size': 50,
            'num_epochs': 200
            },
        'mnist_2c2d': {
                'batch_size': 128,
                'num_epochs': 100
                },
        'mnist_mlp': {
            'batch_size': 128,
            'num_epochs': 100
            },
        'fmnist_mlp': {
            'batch_size': 128,
            'num_epochs': 100
            }
        }

def get_testproblem_default_setting(testproblem):
    return DEFAULT_TEST_PROBLEMS_SETTINGS[testproblem]