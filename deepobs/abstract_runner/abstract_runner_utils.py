# -*- coding: utf-8 -*-
"""Utility functions for running optimizers."""
import argparse


def float2str(x):
    s = "{:.10e}".format(x)
    mantissa, exponent = s.split("e")
    return mantissa.rstrip("0") + "e" + exponent


def _add_hp_to_argparse(parser, optimizer_name, hp_specification, hp_name):
    if hp_specification['type'] == bool:
        if 'default' in hp_specification:
            parser.add_argument(
                "--{0:s}".format(hp_name),
                default=hp_specification['default'],
                help='Hyperparameter {0:s} of {1:s} ({2:s}). Defaults to {3:s}).'.format(hp_name, optimizer_name, str(hp_specification['type']), str(hp_specification['default'])),
                action='store_true')
        else:
            parser.add_argument(
                "--{0:s}".format(hp_name),
                required=True,
                help='Hyperparameter {0:s} of {1:s} ({2:s}).'.format(hp_name, optimizer_name, str(hp_specification['type'])),
                action='store_true')
    else:
        if 'default' in hp_specification:
            parser.add_argument(
                "--{0:s}".format(hp_name),
                default=hp_specification['default'],
                type = hp_specification['type'],
                help='Hyperparameter {0:s} of {1:s} ({2:s}). Defaults to {3:s}).'.format(hp_name, optimizer_name, str(hp_specification['type']), str(hp_specification['default'])))
        else:
            parser.add_argument(
                "--{0:s}".format(hp_name),
                required=True,
                type=hp_specification['type'],
                help='Hyperparameter {0:s} of {1:s} ({2:s}).'.format(hp_name, optimizer_name, str(hp_specification['type'])))
