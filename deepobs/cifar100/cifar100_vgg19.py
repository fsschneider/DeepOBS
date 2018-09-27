# -*- coding: utf-8 -*-
"""
VGG 19 architecture adapted for Cifar-100. The input image will be padded to fit the 224x224 size of the original VGG network and 100 output classes are used.
"""

import tensorflow as tf
import cifar100_input


class set_up:
    def __init__(self, batch_size=128, weight_decay=5e-4):
        self.data_loading = cifar100_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, weight_decay):
        X, y, phase = self.data_loading.load()
        num_classes = 100

        print "X", X.get_shape()

        cond_keep_prob = tf.cond(tf.equal(phase, tf.constant("train")), lambda: tf.constant(0.5), lambda: tf.constant(1.0))

        # for now padd to 224x224 image size for VGG
        padd_X = tf.image.resize_images(X, size=[224, 224])
        print "padd_X", padd_X.get_shape()

        # conv1_1
        conv1_1 = tf.layers.conv2d(inputs=padd_X, filters=64, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv1_1")
        print "conv1_1", conv1_1.get_shape()
        # conv1_2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv1_2")
        print "conv1_2", conv1_2.get_shape()
        # pool1
        pool1 = tf.nn.max_pool(value=conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")
        print "pool1", pool1.get_shape()

        # conv2_1
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv2_1")
        print "conv2_1", conv2_1.get_shape()
        # conv2_2
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv2_2")
        print "conv2_2", conv2_2.get_shape()
        # pool2
        pool2 = tf.nn.max_pool(value=conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")
        print "pool2", pool2.get_shape()

        # conv3_1
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv3_1")
        print "conv3_1", conv3_1.get_shape()
        # conv3_2
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv3_2")
        print "conv3_2", conv3_2.get_shape()
        # conv3_3
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv3_3")
        print "conv3_3", conv3_3.get_shape()
        # conv3_4
        conv3_4 = tf.layers.conv2d(inputs=conv3_3, filters=256, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv3_4")
        print "conv3_4", conv3_4.get_shape()
        # pool3
        pool3 = tf.nn.max_pool(value=conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")
        print "pool3", pool3.get_shape()

        # conv4_1
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv4_1")
        print "conv4_1", conv4_1.get_shape()
        # conv4_2
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv4_2")
        print "conv4_2", conv4_2.get_shape()
        # conv4_3
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv4_3")
        print "conv4_3", conv4_3.get_shape()
        # conv4_4
        conv4_4 = tf.layers.conv2d(inputs=conv4_3, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv4_4")
        print "conv4_4", conv4_4.get_shape()
        # pool4
        pool4 = tf.nn.max_pool(value=conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")
        print "pool4", pool4.get_shape()

        # conv5_1
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv5_1")
        print "conv5_1", conv5_1.get_shape()
        # conv5_2
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv5_2")
        print "conv5_2", conv5_2.get_shape()
        # conv5_3
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv5_3")
        print "conv5_3", conv5_3.get_shape()
        # conv5_4
        conv5_4 = tf.layers.conv2d(inputs=conv5_3, filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation=tf.nn.relu, name="conv5_4")
        print "conv5_4", conv5_4.get_shape()
        # pool5
        pool5 = tf.nn.max_pool(value=conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")
        print "pool5", pool5.get_shape()
        # flatten pool5
        pool5_flat = tf.contrib.layers.flatten(inputs=pool5)
        print "pool5_flat", pool5_flat.get_shape()

        # fc_1
        fc1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu, use_bias=True, name="fc1")
        fc1_drop = tf.nn.dropout(fc1, keep_prob=cond_keep_prob)
        print "fc1", fc1.get_shape()
        print "fc1_drop", fc1_drop.get_shape()
        # fc_2
        fc2 = tf.layers.dense(inputs=fc1_drop, units=4096, activation=tf.nn.relu, use_bias=True, name="fc2")
        fc2_drop = tf.nn.dropout(fc2, keep_prob=cond_keep_prob)
        print "fc2", fc2.get_shape()
        print "fc2_drop", fc2_drop.get_shape()
        # fc_3
        fc3 = tf.layers.dense(inputs=fc2, units=num_classes,
                              activation=None, use_bias=True, name="fc3")
        print "fc3", fc3.get_shape()

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc3)

        # Add weight decay to the weight variables, but not to the biases
        [tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay * tf.nn.l2_loss(w)) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]

        y_pred = tf.argmax(fc3, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy
