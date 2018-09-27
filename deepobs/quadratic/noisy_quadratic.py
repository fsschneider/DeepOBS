# -*- coding: utf-8 -*-
"""
Simple N-Dimensional Noisy Quadratic Problem

        0.5* (theta - x)^T * Q * (theta - x)

where x is normally distributed with mean 0.0 and sigma given by the noise_level (default is 3.0).
"""

import tensorflow as tf
import numpy as np
import quadratic_input


class set_up:
    def __init__(self, batch_size=128, size=1000, Q=np.concatenate((np.random.uniform(0., 1., 90), np.random.uniform(30., 60., 10)), axis=0), noise_level=3.0, weight_decay=None):
        self.data_loading = quadratic_input.data_loading(batch_size=batch_size, dim=Q.shape[0], train_size=size, noise_level=noise_level)
        self.losses, self.accuracy = self.set_up(Q=Q, weight_decay=weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, Q, weight_decay=None):
        if weight_decay is not None:
            print "WARNING: Weight decay is non-zero but no weight decay is used for this model."
        Q = np.array(Q)
        if len(Q.shape) == 1:
            Q = np.diag(Q)
            # rotate it randomly
            R = self.random_rotation(Q.shape[0])
            Q = np.matmul(np.transpose(R), np.matmul(Q, R))
        dim = Q.shape[0]
        X, y, phase = self.data_loading.load()
        print "X", X.get_shape()

        Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        theta = tf.get_variable("theta", shape=(1, dim), initializer=tf.constant_initializer(1.0))

        losses = 0.5 * tf.matmul(tf.subtract(theta, X), tf.matmul(Q, tf.transpose(tf.subtract(theta, X))))

        # There is no accuracy here but keep it, so code can be reused
        accuracy = tf.zeros([1, 1], tf.float32)

        return losses, accuracy

    def random_rotation(self, D):
        # produces a rotation matrix R in SO(D) (the special orthogonal
        # group SO(D), or orthogonal matrices with unit determinant, drawn
        # uniformly from the Haar measure.
        #
        # The algorithm used is the subgroup algorithm as originally proposed by
        #
        # P. Diaconis & M. Shahshahani, "The subgroup algorithm for generating
        # uniform random variables". Probability in the Engineering and
        # Informational Sciences 1: 15?32 (1987)
        assert D >= 2
        D = int(D)  # make sure that the dimension is an integer

        # induction start: uniform draw from D=2 Haar measure
        t = 2 * np.pi * np.random.uniform()
        R = [[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]

        for d in range(2, D):
            v = np.random.normal(size=(d + 1, 1))
            # draw on S_d the unit sphere
            v = np.divide(v, np.sqrt(np.transpose(v).dot(v)))
            e = np.concatenate((np.array([[1.0]]), np.zeros((d, 1))), axis=0)
            # random coset location of SO(d-1) in SO(d)
            x = np.divide((e - v), (np.sqrt(np.transpose(e - v).dot(e - v))))

            D = np.vstack([np.hstack([[[1.0]], np.zeros((1, d))]),
                           np.hstack([np.zeros((d, 1)), R])])
            R = D - 2 * np.outer(x, np.transpose(x).dot(D))
        # return negative to fix determinant
        return np.negative(R)
