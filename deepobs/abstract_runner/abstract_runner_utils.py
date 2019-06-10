# -*- coding: utf-8 -*-
"""Utility functions for running optimizers."""
import argparse

def float2str(x):
    s = "{:.10e}".format(x)
    mantissa, exponent = s.split("e")
    return mantissa.rstrip("0") + "e" + exponent

class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         # TODO split at different character
         for kv in values.split(",,"):
             k,v = kv.split("=")
             my_dict[k] = eval(v)
         setattr(namespace, self.dest, my_dict)