# -*- coding: utf-8 -*-
"""Tests for the ImageNet dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from deepobs.tensorflow import datasets


class ImagenetTest(unittest.TestCase):
    """Tests for the ImageNet dataset."""

    def setUp(self):
        """Sets up ImageNet dataset for the tests."""
        self.batch_size = 128
        self.imagenet = datasets.imagenet(self.batch_size)

        self.image_size = 224
        self.image_channels = 3
        self.classes = 1001

        self.size_train_set = 1281167

        self.print_batch_every = 100

    def test_dataset(self):
        """Tests several aspects of the data set class including:

        - Make sure the ``phase`` variable is not trainable
        - Size of input (images) and outputs (labels)
        - That the one-hot vector of labels sums to one
        - Size (number of data points) for each data set.
        """
        with tf.Session() as sess:
            # Check Phase variable
            phase = self.imagenet.phase
            self.assertFalse(phase.trainable)

            # Loop over init operations (aka data sets)
            init_ops = [
                self.imagenet.train_init_op,
                self.imagenet.train_eval_init_op,
                self.imagenet.valid_init_op,
                self.imagenet.test_init_op,
            ]
            data_set_sizes = [
                self.size_train_set - self.imagenet._train_eval_size,
                self.imagenet._train_eval_size,
                self.imagenet._train_eval_size,
                self.imagenet._train_eval_size,
            ]
            for init_op, data_set_size in zip(init_ops, data_set_sizes):
                sess.run(init_op)
                print("***", init_op.name, "initialized")
                size = 0
                while True:
                    try:
                        if size == 0:
                            x_, y_ = sess.run(self.imagenet.batch)
                            self.assertEqual(
                                x_.shape,
                                (
                                    self.batch_size,
                                    self.image_size,
                                    self.image_size,
                                    self.image_channels,
                                ),
                            )
                            self.assertEqual(
                                y_.shape, (self.batch_size, self.classes)
                            )
                            self.assertTrue(
                                np.allclose(
                                    np.sum(y_, axis=1), np.ones(self.batch_size)
                                )
                            )
                            print("Finished first size checks")
                        else:
                            sess.run(self.imagenet.batch)
                            if size % self.print_batch_every == 0:
                                print("Fetched batch", str(size))
                        size += 1
                    except tf.errors.OutOfRangeError:
                        print(
                            "Data set size for",
                            init_op.name,
                            ":",
                            size * self.batch_size,
                        )
                        self.assertEqual(size, data_set_size // self.batch_size)
                        break


if __name__ == "__main__":
    unittest.main()
