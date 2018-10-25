# -*- coding: utf-8 -*-
"""
This module contains TensorFlow data loading functionality for SVHN.
"""

import os
import tensorflow as tf
from .. import dataset_utils


class data_loading:
    """Class providing the data loading functionality for the SVHN data set.

    Args:
        batch_size (int): Batch size of the input-output pairs. No default value is given.

    Attributes:
        batch_size (int): Batch size of the input-output pairs.
        train_eval_size (int): Number of data points to evaluate during the `train eval` phase. Currently set to ``26032`` the size of the test set.
        D_train (tf.data.Dataset): The training data set.
        D_train_eval (tf.data.Dataset): The training evaluation data set. It is the same data as `D_train` but we go through it separately.
        D_test (tf.data.Dataset): The test data set.
        phase (tf.Variable): Variable to describe which phase we are currently in. Can be "train", "train_eval" or "test". The phase variable can determine the behaviour of the network, for example deactivate dropout during evaluation.
        iterator (tf.data.Iterator): A single iterator for all three data sets. We us the initialization operators (see below) to switch this iterator to the data sets.
        X (tf.Tensor): Tensor holding the SVHN images. It has dimension `batch_size` x ``32`` (image size) x ``32`` (image size) x ``3`` (rgb).
        y (tf.Tensor): Label of the SVHN images. It has dimension `batch_size` x ``10`` (number of classes).
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch. It sets the `phase` variable to "train" and initializes the iterator to the training data set.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval phase. It sets the `phase` variable to "train_eval" and initializes the iterator to the training eval data set.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase. It sets the `phase` variable to "test" and initializes the iterator to the test data set.

    """
    def __init__(self, batch_size):
        """Initializes the data loading class.

        Args:
            batch_size (int): Batch size of the input-output pairs. No default value is given.

        """
        self.train_eval_size = 26032  # The size of the test set
        self.batch_size = batch_size
        self.D_train = self.train_dataset(batch_size)
        self.D_train_eval = self.train_eval_dataset(batch_size)
        self.D_test = self.test_dataset(batch_size)
        self.phase = tf.Variable("train", name="phase", trainable=False)

        # Reinitializable iterator given types and shapes of the outputs (needs to be the same for train and test of course)
        self.iterator = tf.data.Iterator.from_structure(
            self.D_train.output_types, self.D_train.output_shapes)
        self.X, self.y = self.iterator.get_next()

        # Operations to do when switching the phase (initialize iterator and assign phase to phase variable)
        self.train_init_op = tf.group([self.iterator.make_initializer(
            self.D_train), tf.assign(self.phase, "train")], name="train_init_op")
        self.train_eval_init_op = tf.group([self.iterator.make_initializer(
            self.D_train_eval), tf.assign(self.phase, "train_eval")], name="train_eval_init_op")
        self.test_init_op = tf.group([self.iterator.make_initializer(
            self.D_test), tf.assign(self.phase, "test")], name="test_init_op")

    def load(self):
        """Returns the data (`X` (images) and `y` (labels)) and the phase variable.

        Returns:
            tupel: Tupel consisting of the images (`X`), the label (`y`) and the phase variable (`phase`).

        """
        return self.X, self.y, self.phase

    def train_dataset(self, batch_size, data_augmentation=True):
        """Creates the training data set.

        Args:
            batch_size (int): Batch size of the input-output pairs.
            data_augmentation (bool): Switch to turn basic data augmentation on or off while training. Defaults to ``true``.

        Returns:
            tf.data.Dataset: The training data set.

        """

        pattern = os.path.join(dataset_utils.get_data_dir(),
                               "svhn", "data_batch_*.bin")
        if data_augmentation:
            D = self.make_dataset(
                binaries_fname_pattern=pattern,
                batch_size=batch_size,
                crop_size=32,
                per_image_standardization=True,
                random_crop=True,
                pad_before_random_crop=4,
                random_flip_left_right=True,
                lighting_augmentation=True,
                one_hot=True,
                shuffle=True,
                shuffle_buffer_size=10000,
                num_prefetched_batches=3,
                num_preprocessing_threads=8)
        else:
            D = self.make_dataset(
                binaries_fname_pattern=pattern,
                batch_size=batch_size,
                crop_size=32,
                per_image_standardization=True,
                random_crop=False,
                pad_before_random_crop=0,
                random_flip_left_right=False,
                lighting_augmentation=False,
                one_hot=True,
                shuffle=True,
                shuffle_buffer_size=10000,
                num_prefetched_batches=3,
                num_preprocessing_threads=8)
        return D

    def train_eval_dataset(self, batch_size, data_augmentation=True):
        """Creates the train eval data set.

        Args:
            batch_size (int): Batch size of the input-output pairs.
            data_augmentation (bool): Switch to turn basic data augmentation on or off while evaluating the training data set. Defaults to ``true``.

        Returns:
            tf.data.Dataset: The train eval data set.

        """

        pattern = os.path.join(dataset_utils.get_data_dir(),
                               "svhn", "data_batch_*.bin")
        if data_augmentation:
            D = self.make_dataset(
                binaries_fname_pattern=pattern,
                batch_size=batch_size,
                crop_size=32,
                per_image_standardization=True,
                random_crop=True,
                pad_before_random_crop=4,
                random_flip_left_right=True,
                lighting_augmentation=True,
                one_hot=True,
                shuffle=True,
                shuffle_buffer_size=73257,
                num_prefetched_batches=3,
                num_preprocessing_threads=8, data_set_size=self.train_eval_size)
        else:
            D = self.make_dataset(
                binaries_fname_pattern=pattern,
                batch_size=batch_size,
                crop_size=32,
                per_image_standardization=True,
                random_crop=False,
                pad_before_random_crop=0,
                random_flip_left_right=False,
                lighting_augmentation=False,
                one_hot=True,
                shuffle=True,
                shuffle_buffer_size=73257,
                num_prefetched_batches=3,
                num_preprocessing_threads=8, data_set_size=self.train_eval_size)
        return D

    def test_dataset(self, batch_size):
        """Creates the test data set.

        Args:
            batch_size (int): Batch size of the input-output pairs.

        Returns:
            tf.data.Dataset: The test data set.

        """

        pattern = os.path.join(dataset_utils.get_data_dir(),
                               "svhn", "test_batch.bin")
        return self.make_dataset(
            binaries_fname_pattern=pattern,
            batch_size=batch_size,
            crop_size=32,
            per_image_standardization=True,
            random_crop=False,
            pad_before_random_crop=0,
            random_flip_left_right=False,
            lighting_augmentation=False,
            one_hot=True,
            shuffle=False,
            shuffle_buffer_size=-1,
            num_prefetched_batches=3,
            num_preprocessing_threads=4)

    def make_dataset(self, binaries_fname_pattern, batch_size, crop_size=32, per_image_standardization=True, random_crop=False, pad_before_random_crop=0, random_flip_left_right=False, lighting_augmentation=False, one_hot=True, shuffle=True, shuffle_buffer_size=10000, num_prefetched_batches=3, num_preprocessing_threads=8, data_set_size=-1):
        """Creates a data set from a pattern of the images and label files.

        Args:
            binaries_fname_pattern (str): Pattern of the ``.bin`` files containing the images and labels.
            batch_size (int): Batch size of the input-output pairs.
            crop_size (int): Crop size of each image. Defaults to ``32``.
            per_image_standardization (bool): Switch to standardize each image to have zero mean and unit norm. Defaults to ``True``.
            random_crop (bool): Switch if random crops should be used. Defaults to ``False``.
            pad_before_random_crop (int): Defines the added padding before a random crop is applied. Defaults to ``0``.
            random_flip_left_right (bool): Switch to randomly flip the images horizontally. Defaults to ``False``.
            lighting_augmentation (bool): Switch to use random brightness, saturation and contrast on each image. Defaults to ``False``.
            one_hot (bool): Switch to turn on or off one-hot encoding of the labels. Defaults to ``True``.
            shuffle (bool):  Switch to turn on or off shuffling of the data set. Defaults to ``True``.
            shuffle_buffer_size (int): Size of the shuffle buffer. Defaults to ``10000``.
            num_prefetched_batches (int): Number of prefeteched batches, defaults to ``3``.
            num_preprocessing_threads (int): The number of elements to process in parallel while applying the image transformations. Defaults to ``8``.
            data_set_size (int): Size of the data set to extract from the images and label files. Defaults to ``-1`` meaning that the full data set is used.

        Returns:
            tf.data.Dataset: Data set object created from the images and label files.

        """

        # Set number of bytes to read
        label_bytes = 1
        label_offset = 0
        num_classes = 10
        depth = 3
        image_size = 32
        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes

        def _parse_func(raw_record):
            # Decode raw_record
            record = tf.reshape(tf.decode_raw(
                raw_record, tf.uint8), [record_bytes])

            label = tf.cast(
                tf.slice(record, [label_offset], [label_bytes]), tf.int32)

            # Convert from string to [depth * height * width] to [depth, height, width].
            depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                                     [depth, image_size, image_size])

            # Convert from [depth, height, width] to [height, width, depth].
            image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

            # Add image pre-processing
            if random_crop:
                image = tf.image.resize_image_with_crop_or_pad(
                    image, image_size + pad_before_random_crop, image_size + pad_before_random_crop)
                image = tf.random_crop(image, [crop_size, crop_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(
                    image, crop_size, crop_size)

            # Randomly flip left right if desired
            if random_flip_left_right:
                image = tf.image.random_flip_left_right(image)

            # Add random brightness, saturation, contrast, if desired
            if lighting_augmentation:
                image = tf.image.random_brightness(image, max_delta=63. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

            # Standardize images if desired
            if per_image_standardization:
                image = tf.image.per_image_standardization(image)

            # Label to one-hot vector
            if one_hot:
                label = tf.squeeze(tf.one_hot(label, depth=num_classes))

            return image, label

        with tf.name_scope("svhn"):
            with tf.device('/cpu:0'):
                filenames = tf.matching_files(binaries_fname_pattern)
                filenames = tf.random_shuffle(filenames)
                D = tf.data.FixedLengthRecordDataset(
                    filenames=filenames,
                    record_bytes=record_bytes)
                D = D.map(_parse_func, num_preprocessing_threads)
                if shuffle:
                    D = D.shuffle(buffer_size=shuffle_buffer_size)
                D = D.take(data_set_size)
                D = D.batch(batch_size)
                D = D.prefetch(buffer_size=num_prefetched_batches)
                return D
