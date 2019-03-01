# -*- coding: utf-8 -*-
"""CIFAR-100 DeepOBS dataset."""

import os
import tensorflow as tf
from . import dataset
from .. import config


class cifar100(dataset.DataSet):
    """DeepOBS data set class for the `CIFAR-100\
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``50 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    data_augmentation (bool): If ``True`` some data augmentation operations
        (random crop window, horizontal flipping, lighting augmentation) are
        applied to the training data (but not the test data).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.

  Attributes:
    batch: A tuple ``(x, y)`` of tensors, yielding batches of CIFAR-100 images
        (``x`` with shape ``(batch_size, 32, 32, 3)``) and corresponding one-hot
        label vectors (``y`` with shape ``(batch_size, 100)``). Executing these
        tensors raises a ``tf.errors.OutOfRangeError`` after one epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``
        or ``test``, depending on the current phase. This can be used by testproblems
        to adapt their behavior to this phase.
  """

    def __init__(self,
                 batch_size,
                 data_augmentation=True,
                 train_eval_size=10000):
        """Creates a new CIFAR-100 instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``50 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      data_augmentation (bool): If ``True`` some data augmentation operations
          (random crop window, horizontal flipping, lighting augmentation) are
          applied to the training data (but not the test data).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "cifar100"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(cifar100, self).__init__(batch_size)

    def _make_dataset(self,
                      binaries_fname_pattern,
                      data_augmentation=False,
                      shuffle=True):
        """Creates a CIFAR-100 data set (helper used by ``.make_*_datset`` below).

    Args:
        binaries_fname_pattern (str): Pattern of the ``.bin`` files from which
            to load images and labels (e.g. ``some/path/data_batch_*.bin``).
        data_augmentation (bool): Whether to apply data augmentation operations.
        shuffle (bool):  Switch to turn on or off shuffling of the data set.
            Defaults to ``True``.

    Returns:
        A tf.data.Dataset yielding batches of CIFAR-100 data.
    """
        # Set number of bytes to read.
        label_bytes = 1
        label_offset = 1
        num_classes = 100
        depth = 3
        image_size = 32
        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes

        def parse_func(raw_record):
            """Function parsing data from raw binary records."""
            # Decode raw_record.
            record = tf.reshape(
                tf.decode_raw(raw_record, tf.uint8), [record_bytes])
            label = tf.cast(
                tf.slice(record, [label_offset], [label_bytes]), tf.int32)
            depth_major = tf.reshape(
                tf.slice(record, [label_bytes], [image_bytes]),
                [depth, image_size, image_size])
            image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

            # Add image pre-processing.
            if data_augmentation:
                image = tf.image.resize_image_with_crop_or_pad(
                    image, image_size + 4, image_size + 4)
                image = tf.random_crop(image, [32, 32, 3])
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=63. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)

            image = tf.image.per_image_standardization(image)
            label = tf.squeeze(tf.one_hot(label, depth=num_classes))
            return image, label

        with tf.name_scope(self._name):
            with tf.device('/cpu:0'):
                filenames = tf.matching_files(binaries_fname_pattern)
                filenames = tf.random_shuffle(filenames)
                data = tf.data.FixedLengthRecordDataset(
                    filenames=filenames, record_bytes=record_bytes)
                data = data.map(
                    parse_func,
                    num_parallel_calls=(8 if data_augmentation else 4))
                if shuffle:
                    data = data.shuffle(
                        buffer_size=20000)
                data = data.batch(self._batch_size, drop_remainder=True)
                data = data.prefetch(
                    buffer_size=4)
                return data

    def _make_train_dataset(self):
        """Creates the CIFAR-100 training dataset.

    Returns:
      A tf.data.Dataset instance with batches of training data.
    """
        pattern = os.path.join(config.get_data_dir(), "cifar-100", "train.bin")
        return self._make_dataset(
            pattern, data_augmentation=self._data_augmentation, shuffle=True)

    def _make_train_eval_dataset(self):
        """Creates the CIFAR-100 train eval dataset.

    Returns:
      A tf.data.Dataset instance with batches of training eval data.
    """
        return self._train_dataset.take(
            self._train_eval_size // self._batch_size)

    def _make_test_dataset(self):
        """Creates the CIFAR-100 test dataset.

    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        pattern = os.path.join(config.get_data_dir(), "cifar-100", "test.bin")
        return self._make_dataset(
            pattern, data_augmentation=False, shuffle=False)
