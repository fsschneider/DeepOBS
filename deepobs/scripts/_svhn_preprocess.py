# -*- coding: utf-8 -*-
"""
Script to convert the SVHN data provided in .mat files (available from
http://ufldl.stanford.edu/housenumbers/) to a CIFAR-style binary format.
"""

import os
import scipy.io


def preprocess(file_path="", save_path=""):
    """Preprocesses the train and test dataset of the street view house numbers
    from mat files to a CIFAR-style binary format.

    Args:
        file_path (str): Path to the .mat files.
        save_path (str): Path where the function should save the .bin files to.
    """
    # Convert train images

    train_file = os.path.join(file_path, "train_32x32.mat")
    read_input = scipy.io.loadmat(train_file)
    j = 0
    output_file = open(save_path + '/data_batch_%d.bin' % j, 'ab')

    for i in range(0, 64000):

        # create new bin file
        if i > 0 and i % 12800 == 0:
            output_file.close()
            j = j + 1
            output_file = open(save_path + '/data_batch_%d.bin' % j, 'ab')

        # Write to bin file
        if read_input['y'][i] == 10:
            read_input['y'][i] = 0
        read_input['y'][i].astype('uint8').tofile(output_file)
        read_input['X'][:, :, :, i].astype('uint8').tofile(output_file)

    output_file.close()

    # Convert test images
    test_file = os.path.join(file_path, "test_32x32.mat")
    read_input = scipy.io.loadmat(test_file)
    output_file = open(save_path + '/test_batch.bin', 'ab')

    for i in range(0, 26032):
        # Write to bin file
        if read_input['y'][i] == 10:
            read_input['y'][i] = 0
        read_input['y'][i].astype('uint8').tofile(output_file)
        read_input['X'][:, :, :, i].astype('uint8').tofile(output_file)

    output_file.close()
