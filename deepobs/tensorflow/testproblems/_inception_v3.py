# -*- coding: utf-8 -*-

import tensorflow as tf


def _inception_v3(x, training, weight_decay):
    def conv2d_BN(inputs, filters, kernel_size, strides, padding, training):
        """Creates a convolutional layer, followed by a batch normalization layer
        and a ReLU activation.

        Args:
            inputs (tf.Tensor): Input tensor to the layer.
            filters (int): Number of filters for the conv layer.
                No default specified.
            kernel_size (int): Size of the conv filter. No default specified.
            strides (int): Stride for the convolutions. No default specified.
            padding (str): Padding of the convolutional layers. Can be ``SAME``
                or ``VALID``. No default specified.
            training (tf.bool): Switch to determine if we are in training
                (or evaluation) mode.

        Returns:
            tf.Tenors: Output after the conv and batch norm layer.

        """
        # We use the fixed set up described in the github implementation
        # https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py
        # which uses the non-default batch norm momentum of 0.9997
        # with tf.variable_scope("conv2d_BN"):
        x = tf.layers.conv2d(
            inputs,
            filters,
            kernel_size,
            strides,
            padding,
            activation=None,
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        x = tf.layers.batch_normalization(
            x, momentum=0.9997, training=training)
        x = tf.nn.relu(x)
        return x

    def inception_block5(inputs, variant, training, name="inception_block5"):
        """Defines the Inception block 5.

        Args:
            inputs (tf.Tensor): Input to the inception block.
            variant (str): Describes which variant of the inception block 5 to
                use. Can be ``a`` or ``b``.
            training (tf.bool): Switch to determine if we are in training
                (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block5``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Switch between the two versions
        if variant == 'a':
            num_filters = 32
        elif variant == 'b':
            num_filters = 64
        else:
            raise ValueError('requested variant of the inception block not known')
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_1x1 = conv2d_BN(inputs, 64, 1, (1, 1), 'SAME', training)
            with tf.variable_scope("branch1"):
                branch1_pool = tf.layers.average_pooling2d(inputs, 3, (1, 1), 'SAME')
                branch1_1x1 = conv2d_BN(branch1_pool, num_filters, 1, (1, 1), 'SAME', training)
            with tf.variable_scope("branch2"):
                branch2_1x1 = conv2d_BN(inputs, 48, 1, (1, 1), 'SAME', training)
                branch2_5x5 = conv2d_BN(branch2_1x1, 64, 5, (1, 1), 'SAME', training)
            with tf.variable_scope("branch3"):
                branch3_1x1 = conv2d_BN(inputs, 64, 1, (1, 1), 'SAME', training)
                branch3_3x3_0 = conv2d_BN(branch3_1x1, 96, 3, (1, 1), 'SAME', training)
                branch3_3x3_1 = conv2d_BN(branch3_3x3_0, 96, 3, (1, 1), 'SAME', training)
            output = tf.concat([branch0_1x1, branch1_1x1, branch2_5x5, branch3_3x3_1], 3)
        return output
    def inception_block10(inputs, training, name="inception_block10"):
        """Defines the Inception block 10.

        Args:
            inputs (tf.Tensor): Input to the inception block.
            training (tf.bool): Switch to determine if we are in training
                (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block5``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_pool = tf.layers.max_pooling2d(inputs, 3, (2, 2), 'VALID')
            with tf.variable_scope("branch1"):
                branch1_3x3 = conv2d_BN(inputs, 384, 3, (2, 2), 'VALID', training)
            with tf.variable_scope("branch2"):
                branch2_1x1 = conv2d_BN(inputs, 64, 1, (1, 1), 'SAME', training)
                branch2_3x3_0 = conv2d_BN(branch2_1x1, 96, 3, (1, 1), 'SAME', training)
                branch2_3x3_1 = conv2d_BN(branch2_3x3_0, 96, 3, (2, 2), 'VALID', training)
            output = tf.concat([branch0_pool, branch1_3x3, branch2_3x3_1], 3)
        return output

    def inception_block6(inputs, variant, training, name="inception_block6"):
        """Defines the Inception block 6.

        Args:
            inputs (tf.Tensor): Input to the inception block.
            variant (str): Describes which variant of the inception block 6 to use.
                Can be ``a``, ``b`` or ``c``.
            training (bool): Switch to determine if we are in training
                (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block5``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Switch between the two versions
        if variant == 'a':
            num_filters = 128
        elif variant == 'b':
            num_filters = 160
        elif variant == 'c':
            num_filters = 192
        else:
            raise ValueError('requested variant of the inception block not known')
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_1x1 = conv2d_BN(inputs, 192, 1, (1, 1), 'SAME', training)
            with tf.variable_scope("branch1"):
                branch1_pool = tf.layers.average_pooling2d(inputs, 3, (1, 1), 'SAME')
                branch1_1x1 = conv2d_BN(branch1_pool, 192, 1, (1, 1), 'SAME', training)
            with tf.variable_scope("branch2"):
                branch2_1x1 = conv2d_BN(inputs, num_filters, 1, (1, 1), 'SAME', training)
                branch2_1x7 = conv2d_BN(branch2_1x1, num_filters, [1, 7], (1, 1), 'SAME', training)
                branch2_7x1 = conv2d_BN(branch2_1x7, 192, [7, 1], (1, 1), 'SAME', training)
            with tf.variable_scope("branch3"):
                branch3_1x1 = conv2d_BN(inputs, num_filters, 1, (1, 1), 'SAME', training)
                branch3_7x1_0 = conv2d_BN(branch3_1x1, num_filters, [7, 1], (1, 1), 'SAME', training)
                branch3_1x7_0 = conv2d_BN(branch3_7x1_0, num_filters, [1, 7], (1, 1), 'SAME', training)
                branch3_7x1_1 = conv2d_BN(branch3_1x7_0, num_filters, [7, 1], (1, 1), 'SAME', training)
                branch3_1x7_1 = conv2d_BN(branch3_7x1_1, 192, [1, 7], (1, 1), 'SAME', training)
            output = tf.concat([branch0_1x1, branch1_1x1, branch2_7x1, branch3_1x7_1], 3)
        return output

    def inception_blockD(inputs, training, name="inception_blockD"):
        """Defines the Inception block D.

        Args:
            inputs (tf.Tensor): Input to the inception block.
            training (bool): Switch to determine if we are in training
                (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block5``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_pool = tf.layers.max_pooling2d(inputs, 3, (2, 2), 'VALID')
            with tf.variable_scope("branch1"):
                branch1_1x1 = conv2d_BN(inputs, 192, 1, (1, 1), 'SAME', training)
                branch1_1x7 = conv2d_BN(branch1_1x1, 192, [1, 7], (1, 1), 'SAME', training)
                branch1_7x1 = conv2d_BN(branch1_1x7, 192, [7, 1], (1, 1), 'SAME', training)
                branch1_3x3 = conv2d_BN(branch1_7x1, 192, 3, (2, 2), 'VALID', training)
            with tf.variable_scope("branch2"):
                branch2_1x1 = conv2d_BN(inputs, 192, 1, (1, 1), 'SAME', training)
                branch2_3x3 = conv2d_BN(branch2_1x1, 320, 3, (2, 2), 'VALID', training)
            output = tf.concat([branch0_pool, branch1_3x3, branch2_3x3], 3)
        return output

    def inception_block7(inputs, training, name="inception_block7"):
        """Defines the Inception block 7.

        Args:
            inputs (tf.Tensor): Input to the inception block.
            training (bool): Switch to determine if we are in training
                (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block5``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_1x1 = conv2d_BN(inputs, 320, 1, (1, 1), 'SAME', training)
            with tf.variable_scope("branch1"):
                branch1_1x1 = conv2d_BN(inputs, 384, 1, (1, 1), 'SAME', training)
                branch1_1x3 = conv2d_BN(branch1_1x1, 384, [1, 3], (1, 1), 'SAME', training)
                branch1_3x1 = conv2d_BN(branch1_1x1, 384, [3, 1], (1, 1), 'SAME', training)
                branch1_concat = tf.concat([branch1_1x3, branch1_3x1], 3)
            with tf.variable_scope("branch2"):
                branch2_1x1 = conv2d_BN(inputs, 448, 1, (1, 1), 'SAME', training)
                branch2_3x3 = conv2d_BN(branch2_1x1, 384, 3, (1, 1), 'SAME', training)
                branch2_1x3 = conv2d_BN(branch2_3x3, 384, [1, 3], (1, 1), 'SAME', training)
                branch2_3x1 = conv2d_BN(branch2_3x3, 384, [3, 1], (1, 1), 'SAME', training)
                branch2_concat = tf.concat([branch2_1x3, branch2_3x1], 3)
            with tf.variable_scope("branch3"):
                branch3_pool = tf.layers.average_pooling2d(inputs, 3, (1, 1), 'SAME')
                branch3_1x1 = conv2d_BN(branch3_pool, 192, 1, (1, 1), 'SAME', training)
            output = tf.concat([branch0_1x1, branch1_concat, branch2_concat, branch3_1x1], 3)
        return output

    num_classes = 1001 # since we have a class for 'background'

    # we resize to 299x299 again, as this is the input size for inception-v3.
    # Our dataset returns always the cropped 224x224 versions
    x = tf.image.resize_images(x, [299, 299])

    with tf.variable_scope("stem"):
        x = conv2d_BN(x, 32, 3, (2, 2), 'VALID', training)
        x = conv2d_BN(x, 32, 3, (1, 1), 'VALID', training)
        x = conv2d_BN(x, 64, 3, (1, 1), 'SAME', training)
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

        x = conv2d_BN(x, 80, 1, (1, 1), 'VALID', training)
        x = conv2d_BN(x, 192, 3, (1, 1), 'VALID', training)
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

    x = inception_block5(x, 'a', training, "inception_block5a")
    x = inception_block5(x, 'b', training, "inception_block5b_0")
    x = inception_block5(x, 'b', training, "inception_block5b_1")

    x = inception_block10(x, training)

    x = inception_block6(x, 'a', training, "inception_block6a")
    x = inception_block6(x, 'b', training, "inception_block6b_0")
    x = inception_block6(x, 'b', training, "inception_block6b_1")
    x = inception_block6(x, 'c', training, "inception_block6c")

    # Auxiliary head
    with tf.variable_scope("aux_head"):
        x_aux = tf.layers.average_pooling2d(x, 5, (3, 3), 'VALID')
        x_aux = conv2d_BN(x_aux, 128, 1, (1, 1), 'SAME', training)
        x_aux = conv2d_BN(x_aux, 768, 5, (1, 1), 'VALID', training)
        x_aux = tf.contrib.layers.flatten(x_aux)
        aux_linear_outputs = tf.layers.dense(x_aux, num_classes, None, True)

    x = inception_blockD(x, training, "inception_blockD")

    x = inception_block7(x, training, "inception_block7_0")
    x = inception_block7(x, training, "inception_block7_1")

    with tf.variable_scope("output"):
        x = tf.layers.average_pooling2d(x, 8, (1, 1), 'VALID')

        x = tf.layers.dropout(x, 0.8, training)
        x = tf.contrib.layers.flatten(x)

        linear_outputs = tf.layers.dense(x, num_classes, None, True)

    return linear_outputs, aux_linear_outputs
