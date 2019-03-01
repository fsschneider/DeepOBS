# -*- coding: utf-8 -*-
"""ImageNet DeepOBS dataset."""

import os
import tensorflow as tf
from . import dataset
from .. import config


class imagenet(dataset.DataSet):
    """DeepOBS data set class for the `ImageNet\
    <http://www.image-net.org/>`_ data set.

  .. NOTE::
    We use ``1001`` classes  which includes an additional `background` class,
    as it is used for example by the inception net.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size the remainder is dropped in each
        epoch (after shuffling).
    data_augmentation (bool): If ``True`` some data augmentation operations
        (random crop window, horizontal flipping, lighting augmentation) are
        applied to the training data (but not the test data).
    train_eval_size (int): Size of the train eval dataset.
        Defaults to ``10 000``.

  Attributes:
    batch: A tuple ``(x, y)`` of tensors, yielding batches of ImageNet images
        (``x`` with shape ``(batch_size, 224, 224, 3)``) and corresponding one-hot
        label vectors (``y`` with shape ``(batch_size, 1001)``).  Executing these
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
                 train_eval_size=50000):
        """Creates a new ImageNet instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size the remainder is dropped in each
          epoch (after shuffling).
      data_augmentation (bool): If ``True`` some data augmentation operations
          (random crop window, horizontal flipping, lighting augmentation) are
          applied to the training data (but not the test data).
      train_eval_size (int): Size of the train eval dataset (default: 10k).
    """
        self._name = "imagenet"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(imagenet, self).__init__(batch_size)

    def _make_dataset(self,
                      pattern,
                      per_image_standardization=True,
                      random_crop=False,
                      random_flip_left_right=False,
                      distort_color=False,
                      shuffle=True):
        """Creates an ImageNet data set (helper used by ``.make_*_datset`` below).

        Args:
            pattern (str): Pattern of the files from which
                to load images and labels (e.g. ``some/path/train-00000-of-01024``).
            per_image_standardization (bool): Switch to standardize each image
                to have zero mean and unit norm. Defaults to ``True``.
            random_crop (bool): Switch if random crops should be used.
                Defaults to ``False``.
            random_flip_left_right (bool): Switch to randomly flip the images
                horizontally. Defaults to ``False``.
            distort_color (bool): Switch to use random brightness, saturation,
                hue and contrast on each image. Defaults to ``False``.
            shuffle (bool):  Switch to turn on or off shuffling of the data set.
                Defaults to ``True``.

        Returns:
            A tf.data.Dataset yielding batches of ImageNet data.
        """
        num_classes = 1001  # Class 0 is for Background. Therefore we have 1001

        def parse_func(example_serialized):
            """Parse function depending on the above arguments and map the
            data set through it
            """
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
            label = tf.squeeze(tf.one_hot(label, depth=num_classes))

            return image, label

        with tf.name_scope(self._name):
            with tf.device('/cpu:0'):
                filenames = tf.matching_files(pattern)
                filenames = tf.random_shuffle(filenames)
                data = tf.data.TFRecordDataset(filenames)
                data = data.map(
                    parse_func,
                    num_parallel_calls=(8 if self._data_augmentation else 4))
                if shuffle:
                    data = data.shuffle(buffer_size=20000)
                data = data.batch(self._batch_size, drop_remainder=True)
                data = data.prefetch(buffer_size=4)
                return data

    def _make_train_dataset(self):
        """Creates the ImageNet training dataset.

    Returns:
      A tf.data.Dataset instance with batches of training data.
    """
        pattern = os.path.join(config.get_data_dir(), "imagenet", "train-*")
        return self._make_dataset(
            pattern,
            per_image_standardization=True,
            random_crop=self._data_augmentation,
            random_flip_left_right=self._data_augmentation,
            distort_color=False,
            shuffle=True)

    def _make_train_eval_dataset(self):
        """Creates the ImageNet train eval dataset.

    Returns:
      A tf.data.Dataset instance with batches of training eval data.
    """
        return self._train_dataset.take(
            self._train_eval_size // self._batch_size)

    def _make_test_dataset(self):
        """Creates the ImageNet test dataset.

    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        pattern = os.path.join(config.get_data_dir(), "imagenet",
                               "validation-*")
        return self._make_dataset(
            pattern,
            per_image_standardization=True,
            random_crop=False,
            random_flip_left_right=False,
            distort_color=False,
            shuffle=False)

    def _parse_example_proto(self, example_serialized):
        """Parses an Example proto containing a training example of an image.
        The output of the build_image_data.py image preprocessing script is a
        dataset containing serialized Example protocol buffers. Each Example
        proto contains the following fields:
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
          example_serialized (tf.string): Scalar Tensor tf.string containing a
          serialized Example protocol buffer.

        Returns:
          tupel: Tupel of image_buffer (tf.string) containing the contents of a
          JPEG file, the label (tf.int32) containing the label and text
          (tf.string) containing the human-readable label.
        """
        # Dense features in Example proto.
        feature_map = {
            'image/encoded':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/class/label':
            tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
            'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        }

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        return features['image/encoded'], label, features['image/class/text']

    def _decode_jpeg(self, image_buffer, scope=None):
        """Decode a JPEG string into one 3-D float image Tensor.

        Args:
          image_buffer (tf.string): scalar string Tensor.
          scope (str): Optional scope for name_scope.
        Returns:
          tf.Tensor: 3-D float Tensor with values ranging from [0, 1).
        """
        with tf.name_scope(
            values=[image_buffer], name=scope, default_name='decode_jpeg'):
            # Decode the string as an RGB JPEG.
            # Note that the resulting image contains an unknown height and width
            # that is set dynamically by _decode_jpeg. In other words, the height
            # and width of image is unknown at compile-time.
            image = tf.image.decode_jpeg(image_buffer, channels=3)

            # After this point, all image pixels reside in [0,1)
            # until the very end, when they're rescaled to (-1, 1).  The various
            # adjust_* ops all require this range for dtype float.
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image

    def _aspect_preserving_resize(self, image, target_smaller_side):
        """"Resize image such that the smaller size has size
        ``target_smaller_sider`` while preserving the aspect ratio.

        Args:
            image (tf.Tensor): Tensor containing the image to resize.
            target_smaller_side (int): Target size for the smaller side in pixel.

        Returns:
            tf.Tensor: The resized image, with the same aspect ratio as the input.

        """

        shape = tf.shape(image)
        height = tf.to_float(shape[0])
        width = tf.to_float(shape[1])
        smaller_side = tf.reduce_min(shape[0:2])
        scale = tf.divide(target_smaller_side, tf.to_float(smaller_side))
        new_height = tf.to_int32(tf.round(scale * height))
        new_width = tf.to_int32(tf.round(scale * width))
        resized_image = tf.image.resize_images(image, [new_height, new_width])

        return resized_image

    def _color_distortion(self, image, scope=None):
        """Distort the color of the image.

        Args:
          image (tf.Tensor): Tensor containing single image.
          scope (str): Optional scope for name_scope.

        Returns:
          tf.Tensor: The color-distorted image.
        """
        with tf.name_scope(
            values=[image], name=scope, default_name='distort_color'):
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

            # The random_* ops do not necessarily clamp.
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image
