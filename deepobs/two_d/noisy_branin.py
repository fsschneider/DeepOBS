# -*- coding: utf-8 -*-
"""
Simple 2D Noisy Branin Loss Function:

        (v - 5.1/(4*pi**2) u**2 + 5/pi u - 6)**2 * y + 10*(1-1/(8*pi))*cos(u) *x + 10

where x and y are normally distributed with mean 1.0 and sigma given by the noise_level (default is 3.0).
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import two_d_input


class set_up:
    """Simple 2D Noisy Branin Loss Function:

    :math:`(v - 5.1/(4 \cdot \pi^2) u^2 + 5/ \pi u - 6)^2 \cdot y + 10 \cdot (1-1/(8 \cdot \pi)) \cdot \cos(u) \cdot x + 10`

    where X and Y are normally distributed with mean 1.0 and sigma given by the noise_level.

    Args:
        batch_size (int): Batch size of the data points. Defaults to ``128``.
        size (int): Size of the training set. Defaults to ``1000``.
        noise_level (float): Noise level of the training set. All training points are sampled from a gaussian distribution with the noise level as the standard deviation. Defaults to ``6.0``.
        starting_point (list): Coordinates of the starting point of the optimization process. Defaults to ``[2.5, 12.5]``.
        weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

    Attributes:
        data_loading (deepobs.data_loading): Data loading class for 2D functions, :class:`.two_d_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model. As there is no accuracy when the loss function is given directly, we set it to ``0``.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase.

    """
    def __init__(self, batch_size=128, size=1000, noise_level=6, starting_point=[2.5, 12.5], weight_decay=None):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. Defaults to ``128``.
            size (int): Size of the training set. Defaults to ``1000``.
            noise_level (float): Noise level of the training set. All training points are sampled from a gaussian distribution with the noise level as the standard deviation. Defaults to ``6``.
            starting_point (list): Coordinates of the starting point of the optimization process. Defaults to ``[2.5, 12.5]``.
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

        """
        self.data_loading = two_d_input.data_loading(batch_size=batch_size, train_size=size, noise_level=noise_level)
        self.losses, self.accuracy = self.set_up(starting_point=starting_point, weight_decay=weight_decay)

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

    def set_up(self, starting_point=[2.5, 12.5], weight_decay=None):
        """Sets up the test problem.

        Args:
            starting_point (list): Coordinates of the starting point of the optimization process. Defaults to ``[2.5, 12.5]``.
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        if weight_decay is not None:
            print "WARNING: Weight decay is non-zero but no weight decay is used for this model."
        X, y, phase = self.data_loading.load()

        # Set model weights
        u = tf.get_variable("weight", shape=(), initializer=tf.constant_initializer(starting_point[0]))
        v = tf.get_variable("bias", shape=(), initializer=tf.constant_initializer(starting_point[1]))

        # Define some constants.
        a = 1.
        b = 5.1 / (4. * np.pi ** 2)
        c = 5 / np.pi
        r = 6.
        s = 10.
        t = 1 / (8. * np.pi)

        losses = a * (v - b * u ** 2 + c * u - r) ** 2 * y + s * (1 - t) * tf.cos(u) * X + s

        # There is no accuracy here but keep it, so code can be reused
        accuracy = tf.zeros([1, 1], tf.float32)

        return losses, accuracy

    def branin(self, u, v):
        """Deterministic version of Branin function.

        Args:
            u (float): Coordinate of the first parameter.
            v (float): Coordinate of the second parameter.

        Returns:
            float: Function value of the deterministic Branin function at ``(u,v)``.

        """
        # Define some constants.
        a = 1.
        b = 5.1 / (4. * np.pi ** 2)
        c = 5 / np.pi
        r = 6.
        s = 10.
        t = 1 / (8. * np.pi)
        return a * (v - b * u ** 2 + c * u - r) ** 2 + s * (1 - t) * np.cos(u) + s

    def plot_run(self, u_history, v_history, loss_history):
        """Plot the history of weights (u, v) and the corresponding loss of an optimizer. Plot the Deterministic Branin as well.

        Args:
            u_history (list): List of ``u`` values (first parameter) as passed by the optimizer.
            v_history (list): List of ``v`` values (second parameter) as passed by the optimizer.
            loss_history (list): List of ``loss`` values as passed by the optimizer. Could be train or test loss.

        Returns:
            fig: Plot with the deterministic Branin and the optimizers trajectory.

        """
        # meshgrid
        traj_box = [[min(u_history), max(u_history)], [
            min(v_history), max(v_history)]]
        plot_limits = [[min(-5.0, traj_box[0][0]), max(10, traj_box[0][1])],
                       [min(0, traj_box[1][0]), max(15, traj_box[1][1])]]
        u_linsp = np.linspace(plot_limits[0][0], plot_limits[0][1], 20)
        v_linsp = np.linspace(plot_limits[1][0], plot_limits[1][1], 20)
        u_MG, v_MG = np.meshgrid(u_linsp, v_linsp)
        zs = np.array([self.branin(u, v)
                       for u, v in zip(np.ravel(u_MG), np.ravel(v_MG))])
        Z = zs.reshape(u_MG.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(u_MG, v_MG, Z, rstride=1, cstride=1,
                        color='b', alpha=0.2, linewidth=0)
        ax.contour(u_MG, v_MG, Z, 20, alpha=0.5, offset=0, stride=30)

        # Highlight Minimum
        ax.scatter(-np.pi, 12.275, 0.397887, 'r*', marker='*', s=200)
        ax.scatter(np.pi, 2.275, 0.397887, 'r*', marker='*', s=200)
        ax.scatter(9.42478, 2.475, 0.397887, 'r*', marker='*', s=200)

        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('Loss')
        ax.view_init(elev=30., azim=30)

        ax.plot(u_history, v_history, loss_history, markerfacecolor='r',
                markeredgecolor='r', marker='.', markersize=1)
        ax.plot(u_history, v_history, markerfacecolor='r',
                markeredgecolor='r', marker='.', markersize=2)
        return fig

    def anim_run(self, u_history, v_history, loss_history, name="Optimizer Trajectory"):
        """Animate the history of weights (u, v) and the corresponding loss of an optimizer. Plot the deterministic Branin as well.

        Args:
            u_history (list): List of ``u`` values (first parameter) as passed by the optimizer.
            v_history (list): List of ``v`` values (second parameter) as passed by the optimizer.
            loss_history (list): List of ``loss`` values as passed by the optimizer. Could be train or test loss.
            name (str): Name of the optimizer. Defaults to "Optimizer Trajectory".

        Returns:
            matplotlib.animation.FuncAnimation: Animation object showing the optimizer's trajectory per iteration.

        """
        # meshgrid
        traj_box = [[min(u_history), max(u_history)], [
            min(v_history), max(v_history)]]
        plot_limits = [[min(-5.0, traj_box[0][0]), max(10, traj_box[0][1])],
                       [min(0, traj_box[1][0]), max(15, traj_box[1][1])]]
        u_linsp = np.linspace(plot_limits[0][0], plot_limits[0][1], 20)
        v_linsp = np.linspace(plot_limits[1][0], plot_limits[1][1], 20)
        u_MG, v_MG = np.meshgrid(u_linsp, v_linsp)
        zs = np.array([self.branin(u, v)
                       for u, v in zip(np.ravel(u_MG), np.ravel(v_MG))])
        Z = zs.reshape(u_MG.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(u_MG, v_MG, Z, rstride=1, cstride=1,
                        color='b', alpha=0.2, linewidth=0)
        ax.contour(u_MG, v_MG, Z, 20, alpha=0.5, offset=0, stride=30)
        # Highlight Minimum
        ax.scatter(-np.pi, 12.275, 0.397887, 'r*', marker='*', s=200)
        ax.scatter(np.pi, 2.275, 0.397887, 'r*', marker='*', s=200)
        ax.scatter(9.42478, 2.475, 0.397887, 'r*', marker='*', s=200)
        # Create animation
        line, = ax.plot([], [], [], 'r-', label=name, lw=1.5)
        point, = ax.plot([], [], [], 'bo')
        display_value = ax.text(2., 2., 27.5, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            display_value.set_text('')

            return line, point, display_value

        def animate(i):
            # Animate line
            line.set_data(u_history[:i], v_history[:i])
            line.set_3d_properties(loss_history[:i])

            # Animate points
            point.set_data(u_history[i], v_history[i])
            point.set_3d_properties(loss_history[i])

            # Animate display value
            display_value.set_text('Current Loss = ' + str(
                loss_history[i]) + ' at iteration: ' + str(i) + ' / ' + str(len(loss_history) - 1))

            return line, point, display_value

        ax.legend(loc=1)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(u_history), interval=120,
                                       repeat_delay=60, blit=True)

        return anim
