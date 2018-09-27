"""
MLP on MNIST.
"""

import tensorflow as tf
import mnist_input


class set_up:
    def __init__(self, batch_size, weight_decay=None):
        self.data_loading = mnist_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, weight_decay):
        if weight_decay is not None:
            print("WARNING: Weight decay is non-zero but no weight decay is used for this model.")
        X, y, phase = self.data_loading.load()
        print("X", X.get_shape())

        X_flat = tf.reshape(X, [-1, 784])
        print("X_flat", X_flat.get_shape())

        W1 = self.weight_variable("W1", [784, 1000], init_stddev=3e-2)
        b1 = self.bias_variable("b1", [1000], init_val=.00)
        h1 = tf.nn.relu(tf.matmul(X_flat, W1) + b1)

        W2 = self.weight_variable("W2", [1000, 500], init_stddev=3e-2)
        b2 = self.bias_variable("b2", [500], init_val=.00)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        W3 = self.weight_variable("W3", [500, 100], init_stddev=3e-2)
        b3 = self.bias_variable("b3", [100], init_val=.00)
        h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

        W4 = self.weight_variable("W4", [100, 10], init_stddev=3e-2)
        b4 = self.bias_variable("b4", [10], init_val=.00)
        linear_outputs = tf.matmul(h3, W4) + b4

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        correct_prediction = tf.equal(
            tf.argmax(linear_outputs, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy

    def weight_variable(self, name, shape, init_stddev):
        initial = tf.truncated_normal_initializer(stddev=init_stddev)
        return tf.get_variable(name, shape, initializer=initial)

    def bias_variable(self, name, shape, init_val):
        initial = tf.constant_initializer(init_val)
        return tf.get_variable(name, shape, initializer=initial)
