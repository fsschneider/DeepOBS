# -*- coding: utf-8 -*-
"""
Inception-v3 for ImageNet, presented in https://arxiv.org/pdf/1512.00567.pdf.
There are many changes from the paper to the "official" Tensorflow implementation https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py
as well as the model.txt that can be found in the sources of the original paper (https://arxiv.org/e-print/1512.00567v3).
We chose to implement the version from Tensorflow (with possibly some minor changes)
In the paper, they trained it the following way:
 - 100 Epochs
 - Batch size 32
 - RMSPRop with a decay of 0.9 and eps=1.0
 - Initial learning rate 0.045
 - Decay evey two epochs with exponential rate of 0.94
 - Gradient clipping with threshold 2.0
"""

import tensorflow as tf
import imagenet_input


class set_up:
    """Class providing the functionality for the Inception v3 architecture on `ImagNet`.

    Details about the architecture can be found in the `original paper`_.

    There are many changes from the paper to the "official" `TensorFlow implementation`_ as well as the model.txt that can be found in the sources of the `original paper`_.
    We chose to implement the version from Tensorflow (with possibly some minor changes)

    Args:
        batch_size (int): Batch size of the data points. Defaults to ``128``.
        weight_decay (float): Weight decay factor. In this model weight decay is applied to the weights, but not the biases. Defaults to ``4e-5``.

    Attributes:
        data_loading (deepobs.data_loading): Data loading class for `ImageNet`, :class:`.imagenet_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase.

    .. _original paper: https://arxiv.org/abs/1512.00567
    .. _TensorFlow implementation: https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py
    """
    def __init__(self, batch_size=128, weight_decay=4e-5):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. Defaults to ``128``.
            weight_decay (float): Weight decay factor. In this model weight decay is applied to the weights, but not the biases. Defaults to ``4e-5``.

        """
        self.data_loading = imagenet_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        """Returns the losses and the accuray of the model.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        return self.losses, self.accuracy

    def set_up(self, weight_decay):
        """Sets up the test problem.

        Args:
            weight_decay (float): Weight decay factor, which is only applied to the weights and not the biases.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        X, y, phase = self.data_loading.load()
        num_classes = 1000

        print "X", X.get_shape()

        # we resize to 299x299 again, as this is the input size for inception-v3. The switchable_iterator returns always the cropped 224x224 versions
        padd_X = tf.image.resize_images(X, size=[299, 299])
        print "padd_X", padd_X.get_shape()

        training = tf.cond(tf.equal(phase, tf.constant("train")), lambda: True, lambda: False)

        conv0 = self.conv2d_BN(inputs=padd_X, filters=32, kernel_size=3, strides=(2, 2), padding='VALID', training=training, name="conv0")
        print "conv0", conv0.get_shape()
        conv1 = self.conv2d_BN(inputs=conv0, filters=32, kernel_size=3, strides=(1, 1), padding='VALID', training=training, name="conv1")
        print "conv1", conv1.get_shape()
        conv2 = self.conv2d_BN(inputs=conv1, filters=64, kernel_size=3, strides=(1, 1), padding='SAME', training=training, name="conv2")
        print "conv2", conv2.get_shape()
        pool0 = tf.nn.max_pool(value=conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool0")
        print "pool0", pool0.get_shape()

        conv3 = self.conv2d_BN(inputs=pool0, filters=80, kernel_size=1, strides=(1, 1), padding='VALID', training=training, name="conv3")
        print "conv3", conv3.get_shape()
        conv4 = self.conv2d_BN(inputs=conv3, filters=192, kernel_size=3, strides=(1, 1), padding='VALID', training=training, name="conv4")
        print "conv4", conv4.get_shape()
        pool1 = tf.nn.max_pool(value=conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")
        print "pool1", pool1.get_shape()

        inception_5a = self.inception_block5(input_layer=pool1, variant='a', training=training, name="inception_5a")
        print "inception_5a", inception_5a.get_shape()
        inception_5b_0 = self.inception_block5(input_layer=inception_5a, variant='b', training=training, name="inception_5b_0")
        print "inception_5b_0", inception_5b_0.get_shape()
        inception_5b_1 = self.inception_block5(input_layer=inception_5b_0, variant='b', training=training, name="inception_5b_1")
        print "inception_5b_1", inception_5b_1.get_shape()

        inception_10 = self.inception_block10(input_layer=inception_5b_1, training=training, name="inception_10")
        print "inception_10", inception_10.get_shape()

        inception_6a = self.inception_block6(input_layer=inception_10, variant='a', training=training, name="inception_6a")
        print "inception_6a", inception_6a.get_shape()
        inception_6b_0 = self.inception_block6(input_layer=inception_6a, variant='b', training=training, name="inception_6b_0")
        print "inception_6b_0", inception_6b_0.get_shape()
        inception_6b_1 = self.inception_block6(input_layer=inception_6b_0, variant='b', training=training, name="inception_6b_1")
        print "inception_6b_1", inception_6b_1.get_shape()
        inception_6c = self.inception_block6(input_layer=inception_6b_1, variant='c', training=training, name="inception_6c")
        print "inception_6c", inception_6c.get_shape()

        # Auxiliary head
        aux_pool = tf.layers.average_pooling2d(inputs=inception_6c, pool_size=5, strides=(3, 3), padding='VALID', name="aux_pool")
        print "** aux_pool", aux_pool.get_shape()
        aux_conv0 = self.conv2d_BN(inputs=aux_pool, filters=128, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="aux_conv0")
        print "** aux_conv0", aux_conv0.get_shape()
        aux_conv1 = self.conv2d_BN(inputs=aux_conv0, filters=768, kernel_size=5, strides=(1, 1), padding='VALID', training=training, name="aux_conv1")
        print "** aux_conv1", aux_conv1.get_shape()
        aux_conv1_flat = tf.contrib.layers.flatten(inputs=aux_conv1)
        print "** aux_conv1_flat", aux_conv1_flat.get_shape()
        aux_logits = tf.layers.dense(inputs=aux_conv1_flat, units=num_classes, activation=None, use_bias=True, name="aux_logits")
        print "** aux_logits", aux_logits.get_shape()
        aux_losses = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=aux_logits, weights=0.4, label_smoothing=0.1)
        print "** aux_losses", aux_losses.get_shape()
        #

        inception_D = self.inception_blockD(input_layer=inception_6c, training=training, name="eption_D")
        print "inception_D", inception_D.get_shape()

        inception_7_0 = self.inception_block7(input_layer=inception_D, training=training, name="inception_7_0")
        print "inception_7_0", inception_7_0.get_shape()
        inception_7_1 = self.inception_block7(input_layer=inception_7_0, training=training, name="inception_7_1")
        print "inception_7_1", inception_7_1.get_shape()

        pool2 = tf.layers.average_pooling2d(inputs=inception_7_1, pool_size=8, strides=(1, 1), padding='VALID', name="pool2")
        print "pool2", pool2.get_shape()

        drop = tf.layers.dropout(pool2, rate=0.8, training=training, name="drop")
        print "drop", drop.get_shape()
        drop_flat = tf.contrib.layers.flatten(inputs=drop)
        print "drop_flat", drop_flat.get_shape()

        logits = tf.layers.dense(inputs=drop_flat, units=num_classes, activation=None, use_bias=True, name="logits")
        print "logits", logits.get_shape()

        main_losses = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, label_smoothing=0.1)
        print "main_losses", main_losses.get_shape()

        y_pred = tf.argmax(logits, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Add main_loss and aux_loss if we are training
        losses = tf.cond(training, lambda: tf.add(main_losses, aux_losses), lambda: tf.add(main_losses, 0.0), name="losses")
        print "losses", losses.get_shape()

        # Add weight decay to the weight variables, but not to the biases (and not to batch norm)
        [tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay * tf.nn.l2_loss(w)) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]

        return losses, accuracy

    def conv2d_BN(self, inputs, filters, kernel_size, strides, padding, training, name="conv2d_BN"):
        """Creates a convolutional layer, followed by a batch normalization layer and a ReLU activation.

        Args:
            inputs (tf.Tensor): Input tensor to the layer.
            filters (int): Number of filters for the conv layer. No default value specified.
            kernel_size (int): Size of the conv filter. No default value specified.
            strides (int): Stride for the convolutions. No default value specified.
            padding (str): Padding of the convolutional layers. Can be ``SAME`` or ``VALID``. No default value specified.
            training (bool): Switch to determine if we are in training (or evaluation) mode.
            name (str): Name of the layer. Defaults to ``conv2d_BN``.

        Returns:
            tf.Tenors: Output after the conv and batch norm layer.

        """
        # We use the fixed set up described in the github implementation https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py which uses the non-default batch norm momentum of 0.9997
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, use_bias=False, name="conv")
            bn = tf.layers.batch_normalization(conv, momentum=0.9997, training=training, name="BN")
            relu = tf.nn.relu(bn, name="relu")
        return relu

    def inception_block5(self, input_layer, variant, training, name="inception_block5"):
        """Defines the Inception block 5.

        Args:
            input_layer (tf.Tensor): Input to the inception block.
            variant (str): Describes which variant of the inception block 5 to use. Can be ``a`` or ``b``.
            training (bool): Switch to determine if we are in training (or evaluation) mode.
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
                branch0_1x1 = self.conv2d_BN(inputs=input_layer, filters=64, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch0_1x1")
            with tf.variable_scope("branch1"):
                branch1_pool = tf.layers.average_pooling2d(inputs=input_layer, pool_size=3, strides=(1, 1), padding='SAME', name="branch1_pool")
                branch1_1x1 = self.conv2d_BN(inputs=branch1_pool, filters=num_filters, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch1_1x1")
            with tf.variable_scope("branch2"):
                branch2_1x1 = self.conv2d_BN(inputs=input_layer, filters=48, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch2_1x1")
                branch2_5x5 = self.conv2d_BN(inputs=branch2_1x1, filters=64, kernel_size=5, strides=(1, 1), padding='SAME', training=training, name="branch2_5x5")
            with tf.variable_scope("branch3"):
                branch3_1x1 = self.conv2d_BN(inputs=input_layer, filters=64, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch1_1x1")
                branch3_3x3_0 = self.conv2d_BN(inputs=branch3_1x1, filters=96, kernel_size=3, strides=(1, 1), padding='SAME', training=training, name="branch3_3x3_0")
                branch3_3x3_1 = self.conv2d_BN(inputs=branch3_3x3_0, filters=96, kernel_size=3, strides=(1, 1), padding='SAME', training=training, name="branch3_3x3_1")
            output = tf.concat(axis=3, values=[branch0_1x1, branch1_1x1, branch2_5x5, branch3_3x3_1])
        return output

    def inception_block10(self, input_layer, training, name="inception_block10"):
        """Defines the Inception block 10.

        Args:
            input_layer (tf.Tensor): Input to the inception block.
            training (bool): Switch to determine if we are in training (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block10``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_pool = tf.layers.max_pooling2d(inputs=input_layer, pool_size=3, strides=(2, 2), padding='VALID', name="branch0_pool")
            with tf.variable_scope("branch1"):
                branch1_3x3 = self.conv2d_BN(inputs=input_layer, filters=384, kernel_size=3, strides=(2, 2), padding='VALID', training=training, name="branch1_3x3")
            with tf.variable_scope("branch2"):
                branch2_1x1 = self.conv2d_BN(inputs=input_layer, filters=64, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch2_1x1")
                branch2_3x3_0 = self.conv2d_BN(inputs=branch2_1x1, filters=96, kernel_size=3, strides=(1, 1), padding='SAME', training=training, name="branch2_3x3_0")
                branch2_3x3_1 = self.conv2d_BN(inputs=branch2_3x3_0, filters=96, kernel_size=3, strides=(2, 2), padding='VALID', training=training, name="branch2_3x3_1")
            output = tf.concat(axis=3, values=[branch0_pool, branch1_3x3, branch2_3x3_1])
        return output

    def inception_block6(self, input_layer, variant, training, name="inception_block6"):
        """Defines the Inception block 6.

        Args:
            input_layer (tf.Tensor): Input to the inception block.
            variant (str): Describes which variant of the inception block 6 to use. Can be ``a``, ``b`` or ``c``.
            training (bool): Switch to determine if we are in training (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block6``.

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
                branch0_1x1 = self.conv2d_BN(inputs=input_layer, filters=192, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch0_1x1")
            with tf.variable_scope("branch1"):
                branch1_pool = tf.layers.average_pooling2d(inputs=input_layer, pool_size=3, strides=(1, 1), padding='SAME', name="branch1_pool")
                branch1_1x1 = self.conv2d_BN(inputs=branch1_pool, filters=192, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch1_1x1")
            with tf.variable_scope("branch2"):
                branch2_1x1 = self.conv2d_BN(inputs=input_layer, filters=num_filters, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch2_1x1")
                branch2_1x7 = self.conv2d_BN(inputs=branch2_1x1, filters=num_filters, kernel_size=[1, 7], strides=(1, 1), padding='SAME', training=training, name="branch2_1x7")
                branch2_7x1 = self.conv2d_BN(inputs=branch2_1x7, filters=192, kernel_size=[7, 1], strides=(1, 1), padding='SAME', training=training, name="branch2_7x1")
            with tf.variable_scope("branch3"):
                branch3_1x1 = self.conv2d_BN(inputs=input_layer, filters=128, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch3_1x1")
                branch3_7x1_0 = self.conv2d_BN(inputs=branch3_1x1, filters=num_filters, kernel_size=[7, 1], strides=(1, 1), padding='SAME', training=training, name="branch3_7x1_0")
                branch3_1x7_0 = self.conv2d_BN(inputs=branch3_7x1_0, filters=num_filters, kernel_size=[1, 7], strides=(1, 1), padding='SAME', training=training, name="branch3_1x7_0")
                branch3_7x1_1 = self.conv2d_BN(inputs=branch3_1x7_0, filters=num_filters, kernel_size=[7, 1], strides=(1, 1), padding='SAME', training=training, name="branch3_7x1_1")
                branch3_1x7_1 = self.conv2d_BN(inputs=branch3_7x1_1, filters=192, kernel_size=[1, 7], strides=(1, 1), padding='SAME', training=training, name="branch3_1x7_1")
            output = tf.concat(axis=3, values=[branch0_1x1, branch1_1x1, branch2_7x1, branch3_1x7_1])
        return output

    def inception_blockD(self, input_layer, training, name="inception_blockD"):
        """Defines the Inception block D.

        Args:
            input_layer (tf.Tensor): Input to the inception block.
            training (bool): Switch to determine if we are in training (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_blockD``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_pool = tf.layers.max_pooling2d(inputs=input_layer, pool_size=3, strides=(2, 2), padding='VALID', name="branch0_pool")
            with tf.variable_scope("branch1"):
                branch1_1x1 = self.conv2d_BN(inputs=input_layer, filters=192, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch1_1x1")
                branch1_1x7 = self.conv2d_BN(inputs=branch1_1x1, filters=192, kernel_size=[1, 7], strides=(1, 1), padding='SAME', training=training, name="branch1_1x7")
                branch1_7x1 = self.conv2d_BN(inputs=branch1_1x7, filters=192, kernel_size=[7, 1], strides=(1, 1), padding='SAME', training=training, name="branch1_7x1")
                branch1_3x3 = self.conv2d_BN(inputs=branch1_7x1, filters=192, kernel_size=3, strides=(2, 2), padding='VALID', training=training, name="branch1_3x3")
            with tf.variable_scope("branch2"):
                branch2_1x1 = self.conv2d_BN(inputs=input_layer, filters=192, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch2_1x1")
                branch2_3x3 = self.conv2d_BN(inputs=branch2_1x1, filters=320, kernel_size=3, strides=(2, 2), padding='VALID', training=training, name="branch2_3x3")
            output = tf.concat(axis=3, values=[branch0_pool, branch1_3x3, branch2_3x3])
        return output

    def inception_block7(self, input_layer, training, name="inception_block7"):
        """Defines the Inception block 7.

        Args:
            input_layer (tf.Tensor): Input to the inception block.
            training (bool): Switch to determine if we are in training (or evaluation) mode.
            name (str): Name of the block. Defaults to ``inception_block7``.

        Returns:
            tf.Tenors: Output after the inception block.

        """
        # Build block
        with tf.variable_scope(name):
            with tf.variable_scope("branch0"):
                branch0_1x1 = self.conv2d_BN(inputs=input_layer, filters=320, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch0_1x1")
            with tf.variable_scope("branch1"):
                branch1_1x1 = self.conv2d_BN(inputs=input_layer, filters=384, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch1_1x1")
                branch1_1x3 = self.conv2d_BN(inputs=branch1_1x1, filters=384, kernel_size=[1, 3], strides=(1, 1), padding='SAME', training=training, name="branch1_1x3")
                branch1_3x1 = self.conv2d_BN(inputs=branch1_1x1, filters=384, kernel_size=[3, 1], strides=(1, 1), padding='SAME', training=training, name="branch1_3x1")
                branch1_concat = tf.concat(axis=3, values=[branch1_1x3, branch1_3x1])
            with tf.variable_scope("branch2"):
                branch2_1x1 = self.conv2d_BN(inputs=input_layer, filters=448, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch2_1x1")
                branch2_3x3 = self.conv2d_BN(inputs=branch2_1x1, filters=384, kernel_size=3, strides=(1, 1), padding='SAME', training=training, name="branch2_3x3")
                branch2_1x3 = self.conv2d_BN(inputs=branch2_3x3, filters=384, kernel_size=[1, 3], strides=(1, 1), padding='SAME', training=training, name="branch2_1x3")
                branch2_3x1 = self.conv2d_BN(inputs=branch2_3x3, filters=384, kernel_size=[3, 1], strides=(1, 1), padding='SAME', training=training, name="branch2_3x1")
                branch2_concat = tf.concat(axis=3, values=[branch2_1x3, branch2_3x1])
            with tf.variable_scope("branch3"):
                branch3_pool = tf.layers.average_pooling2d(inputs=input_layer, pool_size=3, strides=(1, 1), padding='SAME', name="branch3_pool")
                branch3_1x1 = self.conv2d_BN(inputs=branch3_pool, filters=192, kernel_size=1, strides=(1, 1), padding='SAME', training=training, name="branch3_1x1")
            output = tf.concat(axis=3, values=[branch0_1x1, branch1_concat, branch2_concat, branch3_1x1])
        return output
