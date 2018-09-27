"""
This module contains TensorFlow data loading functionality for ImageNet.
"""

import os
import tensorflow as tf

from .. import dataset_utils


class data_loading:
    def __init__(self, batch_size):
        self.train_eval_size = 50000  # The size of the test set
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
        return self.X, self.y, self.phase

    def train_dataset(self, batch_size, data_augmentation=True):
        """Create a ``tf.data.Dataset`` for the Imagenet training data."""

        filenames = [os.path.join(dataset_utils.get_data_dir(),
                                  "imagenet", "train-" + str(i).zfill(5) + "-of-01024")
                     for i in range(1024)]
        if data_augmentation:
            D = self._make_dataset(
                filenames,
                batch_size,
                per_image_standardization=True,
                crop_size=224,
                random_crop=True,
                random_flip_left_right=True,
                distort_color=False,
                shuffle=True,
                shuffle_buffer_size=15000,
                num_prefetched_batches=8,
                num_preprocessing_threads=16)
        else:
            D = self._make_dataset(
                filenames,
                batch_size,
                per_image_standardization=True,
                crop_size=224,
                random_crop=False,
                random_flip_left_right=False,
                distort_color=False,
                shuffle=True,
                shuffle_buffer_size=15000,
                num_prefetched_batches=8,
                num_preprocessing_threads=16)
        return D

    def train_eval_dataset(self, batch_size, data_augmentation=True):
        """Create a ``tf.data.Dataset`` for the Imagenet training evaluation data."""

        filenames = [os.path.join(dataset_utils.get_data_dir(),
                                  "imagenet", "train-" + str(i).zfill(5) + "-of-01024")
                     for i in range(1024)]
        if data_augmentation:
            D = self._make_dataset(
                filenames,
                batch_size,
                per_image_standardization=True,
                crop_size=224,
                random_crop=True,
                random_flip_left_right=True,
                distort_color=True,
                shuffle=False,
                shuffle_buffer_size=-1,
                num_prefetched_batches=4,
                num_preprocessing_threads=8, data_set_size=self.train_eval_size)
        else:
            D = self._make_dataset(
                filenames,
                batch_size,
                per_image_standardization=True,
                crop_size=224,
                random_crop=False,
                random_flip_left_right=False,
                distort_color=False,
                shuffle=False,
                shuffle_buffer_size=-1,
                num_prefetched_batches=4,
                num_preprocessing_threads=8, data_set_size=self.train_eval_size)
        return D

    def test_dataset(self, batch_size):
        """Create a ``tf.data.Dataset`` for the Imagenet test data."""

        filenames = [os.path.join(dataset_utils.get_data_dir(),
                                  "imagenet", "validation-" + str(i).zfill(5) + "-of-00128")
                     for i in range(128)]
        return self._make_dataset(
            filenames,
            batch_size,
            per_image_standardization=True,
            crop_size=224,
            random_crop=False,
            random_flip_left_right=False,
            distort_color=False,
            shuffle=False,
            shuffle_buffer_size=-1,
            num_prefetched_batches=4,
            num_preprocessing_threads=8)

    def _make_dataset(self,
                      filenames,
                      batch_size,
                      per_image_standardization=True,
                      crop_size=224,
                      random_crop=False,
                      random_flip_left_right=False,
                      distort_color=False,
                      shuffle=True,
                      shuffle_buffer_size=15000,
                      one_hot=True,
                      num_prefetched_batches=8,
                      num_preprocessing_threads=16,
                      data_set_size=-1):

        num_classes = 1000

        # Define parse function depending on the above arguments and map the dataset
        # through it
        def _parse_func(example_serialized):
            # Parse example proto, decode image and resize while preserving aspect
            image_buffer, label, _ = self._parse_example_proto(
                example_serialized)
            image = self._decode_jpeg(image_buffer)
            image = self._aspect_preserving_resize(
                image, target_smaller_side=256)

            # Crop to 224x224, either randomly or centered according to arguments
            if random_crop:
                image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
                image = tf.random_crop(image, [224, 224, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

            # Optionally perform random flip
            if random_flip_left_right:
                image = tf.image.random_flip_left_right(image)

            # Optionally distort color
            if distort_color:
                image = self._color_distortion(image)

            # Normalize
            if per_image_standardization:
                image = tf.image.per_image_standardization(image)

            # Convert label to shape [] (instead of) [1,] such that the label vector for
            # a mini-batch will later be of shape [batch_size,]
            label = tf.reshape(label, [])
            # Label to one-hot vector
            if one_hot:
                label = tf.squeeze(tf.one_hot(label, depth=num_classes))

            return image, label

        with tf.name_scope("imagenet_input"):
            with tf.device('/cpu:0'):
                # TODO: buffer_size, compression_type?
                filenames = tf.random_shuffle(filenames)
                D = tf.data.TFRecordDataset(filenames)
                D = D.map(_parse_func, num_preprocessing_threads)
                if shuffle:
                    D = D.shuffle(buffer_size=shuffle_buffer_size)
                D = D.take(data_set_size)
                D = D.batch(batch_size)
                D = D.prefetch(buffer_size=num_prefetched_batches)
                return D

    def _parse_example_proto(self, example_serialized):
        """Parses an Example proto containing a training example of an image.
        The output of the build_image_data.py image preprocessing script is a dataset
        containing serialized Example protocol buffers. Each Example proto contains
        the following fields:
          image/height: 462
          image/width: 581
          image/colorspace: 'RGB'
          image/channels: 3
          image/class/label: 615
          image/class/synset: 'n03623198'
          image/class/text: 'knee pad'
          image/format: 'JPEG'
          image/filename: 'ILSVRC2012_val_00041207.JPEG'
          image/encoded: <JPEG encoded string>
        Args:
          example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.
        Returns:
          image_buffer: Tensor tf.string containing the contents of a JPEG file.
          label: Tensor tf.int32 containing the label.
          text: Tensor tf.string containing the human-readable label.
        """
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                    default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                   default_value=''),
        }

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        return features['image/encoded'], label, features['image/class/text']

    def _decode_jpeg(self, image_buffer, scope=None):
        """Decode a JPEG string into one 3-D float image Tensor.
        Args:
          image_buffer: scalar string Tensor.
          scope: Optional scope for name_scope.
        Returns:
          3-D float Tensor with values ranging from [0, 1).
        """
        with tf.name_scope(values=[image_buffer], name=scope,
                           default_name='decode_jpeg'):
            # Decode the string as an RGB JPEG.
            # Note that the resulting image contains an unknown height and width
            # that is set dynamically by decode_jpeg. In other words, the height
            # and width of image is unknown at compile-time.
            image = tf.image.decode_jpeg(image_buffer, channels=3)

            # After this point, all image pixels reside in [0,1)
            # until the very end, when they're rescaled to (-1, 1).  The various
            # adjust_* ops all require this range for dtype float.
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image

    def _aspect_preserving_resize(self, image, target_smaller_side):
        """Resize image such that the smaller size has size ``target_smaller_sider``
        while preserving the aspect ratio."""

        shape = tf.shape(image)
        height = tf.to_float(shape[0])
        width = tf.to_float(shape[1])
        smaller_side = tf.reduce_min(shape[0:2])
        scale = tf.divide(target_smaller_side, tf.to_float(smaller_side))
        new_height = tf.to_int32(tf.round(scale * height))
        new_width = tf.to_int32(tf.round(scale * width))
        # TODO: resize method?
        resized_image = tf.image.resize_images(image, [new_height, new_width])

        return resized_image

    def _color_distortion(self, image, scope=None):
        """Distort the color of the image.

        Args:
          image: Tensor containing single image.
          scope: Optional scope for name_scope.
        Returns:
          color-distorted image
        """
        with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

            # The random_* ops do not necessarily clamp.
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image
