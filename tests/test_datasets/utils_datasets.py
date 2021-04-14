"""Utility functions for testing the DeepOBS data sets."""


import numpy as np


def denormalize_image(img):
    """Convert a normalized (float) image back to unsigned 8-bit images."""
    img -= np.min(img)
    img /= np.max(img)
    img *= 255.0
    return np.round(img).astype(np.uint8)
