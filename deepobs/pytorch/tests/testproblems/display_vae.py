# -*- coding: utf-8 -*-
"""Script to visualize generated VAE images from DeepOBS."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems

def generate(testprob, sampled_z, grid_size=5):
    """Function to generate images using the decoder.

    Args:
        sess (tf.Session): A TensorFlow session.
        sampled_z (tf.Variable): Sampled ``z`` with dimensions ``latent size``
            times ``number of examples``.
        grid_size (int): Will display grid_size**2 number of generated images.

    """

    # batch the sampled_zs according to grid size and make it compatible with testproblem
    batched_z = np.zeros((grid_size*grid_size,8))
    for i in range(len(sampled_z)):
        batched_z[i,:] = sampled_z[i]
    batched_z = torch.from_numpy(batched_z)
    batched_z = batched_z.to('cuda', dtype = torch.float32)

    imgs = testprob.net.decode(batched_z)
    imgs = np.squeeze(imgs.cpu().detach().numpy())

    fig = plt.figure()
    for i in range(grid_size * grid_size):
        axis = fig.add_subplot(grid_size, grid_size, i + 1)
        axis.imshow(imgs[i], cmap='gray')
        axis.axis("off")
    return fig


def display_images(testproblem_cls, grid_size=5, num_epochs=4):
    """Display images from a DeepOBS data set.

  Args:
    testproblem_cls: The DeepOBS VAE testproblem class.
    grid_size (int): Will display grid_size**2 number of generated images.
  """

    torch.manual_seed(42)
    np.random.seed(42)
    sampled_z = [
        np.random.normal(0, 1, 8) for _ in range(grid_size * grid_size)
    ]

    testprob = testproblem_cls(batch_size=grid_size * grid_size)
    testprob.set_up()

    testprob.train_init_op()
    # Epoch 0
    fig = generate(testprob, sampled_z, grid_size=grid_size)
    fig.suptitle(testproblem_cls.__name__ + " epoch 0")
    # fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.show()
    # Train Loop

    opt = torch.optim.Adam(testprob.net.parameters(), lr = 0.0005)
    for i in range(num_epochs):
        while True:
            try:
                opt.zero_grad()
                batch_loss, _ = testprob.get_batch_loss_and_accuracy()

                # if the testproblem has a regularization, add the regularization loss.
                # TODO the regularization loss is added to every batch loss! correct?
                if hasattr(testprob, 'get_regularization_loss'):
                    regularizer_loss = testprob.get_regularization_loss()
                    batch_loss += regularizer_loss
                batch_loss.backward()
                opt.step()
            except StopIteration:
                break

        fig = generate(testprob, sampled_z, grid_size=grid_size)

        fig.suptitle(testproblem_cls.__name__ + " epoch " + str(i+1))
        # fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.show()

if __name__ == "__main__":
#    display_images(testproblems.mnist_vae)

    display_images(testproblems.fmnist_vae)

    plt.show()
