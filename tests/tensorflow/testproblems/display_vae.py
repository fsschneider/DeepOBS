# -*- coding: utf-8 -*-
"""Script to visualize generated VAE images from DeepOBS."""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


def generate(sess, sampled_z, grid_size=5):
    """Function to generate images using the decoder.

    Args:
        sess (tf.Session): A TensorFlow session.
        sampled_z (tf.Variable): Sampled ``z`` with dimensions ``latent size``
            times ``number of examples``.
        grid_size (int): Will display grid_size**2 number of generated images.

    """
    z = tf.get_default_graph().get_tensor_by_name("encoder/z:0")

    dec = tf.get_default_graph().get_tensor_by_name("decoder/decoder_op:0")
    imgs = sess.run(dec, feed_dict={z: sampled_z})
    imgs = [
        np.reshape(imgs[i], [28, 28]) for i in range(grid_size * grid_size)
    ]

    fig = plt.figure()
    for i in range(grid_size * grid_size):
        axis = fig.add_subplot(grid_size, grid_size, i + 1)
        axis.imshow(imgs[i], cmap='gray')
        axis.axis("off")
    return fig


def display_images(testproblem_cls, grid_size=5, num_epochs=1):
    """Display images from a DeepOBS data set.

  Args:
    testproblem_cls: The DeepOBS VAE testproblem class.
    grid_size (int): Will display grid_size**2 number of generated images.
  """
    tf.reset_default_graph()

    tf.set_random_seed(42)
    np.random.seed(42)
    sampled_z = [
        np.random.normal(0, 1, 8) for _ in range(grid_size * grid_size)
    ]

    testprob = testproblem_cls(batch_size=grid_size * grid_size)
    testprob.set_up()
    loss = tf.reduce_mean(testprob.losses) + testprob.regularizer
    train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(testprob.train_init_op)

    # Epoch 0
    fig = generate(sess, sampled_z, grid_size=grid_size)
    fig.suptitle(testproblem_cls.__name__ + " epoch 0")
    # fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.show()
    # Train Loop
    for i in range(num_epochs):
        while True:
            try:
                sess.run(train_step)
            except tf.errors.OutOfRangeError:
                break
        fig = generate(sess, sampled_z, grid_size=grid_size)
        fig.suptitle(testproblem_cls.__name__ + " epoch " + str(i+1))
        # fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.show()

if __name__ == "__main__":
    display_images(testproblems.mnist_vae)

    display_images(testproblems.fmnist_vae)

    plt.show()
