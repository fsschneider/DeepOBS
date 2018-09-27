# -*- coding: utf-8 -*-
"""
This module contains TensorFlow data loading functionality for CIFAR-10.
"""

import os
import tensorflow as tf
from .. import dataset_utils


class data_loading:
    def __init__(self, batch_size, data_augmentation=True):
        self.train_eval_size = 10000  # The size of the test set
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.D_train = self.train_dataset(batch_size, data_augmentation)
        self.D_train_eval = self.train_eval_dataset(
            batch_size, data_augmentation)
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
        return self.X, self.y, self.phase

    def train_dataset(self, batch_size, data_augmentation=True):
        """Create a ``tf.data.Dataset`` for the CIFAR-10 training data."""

        pattern = os.path.join(dataset_utils.get_data_dir(),
                               "cifar-10", "data_batch_*.bin")
        if data_augmentation:
            D = self._make_dataset(
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
            D = self._make_dataset(
                binaries_fname_pattern=pattern,
                batch_size=batch_size, crop_size=32,
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
        """Create a ``tf.data.Dataset`` for the CIFAR-10 training evaluation data."""

        pattern = os.path.join(dataset_utils.get_data_dir(),
                               "cifar-10", "data_batch_*.bin")
        if data_augmentation:
            D = self._make_dataset(
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
                shuffle_buffer_size=60000,
                num_prefetched_batches=3,
                num_preprocessing_threads=8,
                data_set_size=self.train_eval_size)
        else:
            D = self._make_dataset(
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
                shuffle_buffer_size=60000,
                num_prefetched_batches=3,
                num_preprocessing_threads=8,
                data_set_size=self.train_eval_size)
        return D

    def test_dataset(self, batch_size):
        """Create a ``tf.data.Dataset`` for the CIFAR-10 test data."""

        pattern = os.path.join(dataset_utils.get_data_dir(),
                               "cifar-10", "test_batch.bin")
        return self._make_dataset(
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

    def _make_dataset(self,
                      binaries_fname_pattern,
                      batch_size,
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
                      num_preprocessing_threads=8,
                      data_set_size=-1):
        """Produce CIFAR dataset."""

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

        with tf.name_scope("cifar10"):
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
