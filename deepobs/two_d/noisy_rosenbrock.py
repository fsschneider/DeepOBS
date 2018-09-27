# -*- coding: utf-8 -*-
"""
Simple 2D Noisy Rosenbrock Loss Function:

        (1 - u)**2 + 100 * x * (v - u**2)**2

where x is normally distributed with mean 1.0 and sigma given by the noise_level (default is 3.0).
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import two_d_input


class set_up:
    def __init__(self, batch_size=128, size=1000, noise_level=3, starting_point=[-0.5, 1.5], weight_decay=None):
        self.data_loading = two_d_input.data_loading(batch_size=batch_size, train_size=size, noise_level=noise_level)
        self.losses, self.accuracy = self.set_up(starting_point=starting_point, weight_decay=weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, starting_point=[-4.5, 4.5], weight_decay=None):
        if weight_decay is not None:
            print "WARNING: Weight decay is non-zero but no weight decay is used for this model."
        X, y, phase = self.data_loading.load()

        # Set model weights
        u = tf.get_variable("weight", shape=(), initializer=tf.constant_initializer(starting_point[0]))
        v = tf.get_variable("bias", shape=(), initializer=tf.constant_initializer(starting_point[1]))

        losses = tf.add(tf.pow(tf.subtract(1.0, u), 2.0), tf.multiply(tf.multiply(100.0, X), tf.pow(tf.subtract(v, tf.pow(u, 2.0)), 2.0)), 'losses')

        # There is no accuracy here but keep it, so code can be reused
        accuracy = tf.zeros([1, 1], tf.float32)

        return losses, accuracy

    def rosenbrock(self, u, v):
        """Deterministic version of rosenbrock"""
        return (1 - u)**2 + 100 * (v - u**2)**2

    def plot_run(self, u_history, v_history, loss_history):
        """Plot the history of weights (u, v) and the corresponding loss of a optimizer. Plot the Deterministic Rosenbrock as well."""
        # meshgrid
        traj_box = [[min(u_history), max(u_history)], [min(v_history), max(v_history)]]
        plot_limits = [[min(-1.5, traj_box[0][0]), max(1.5, traj_box[0][1])], [min(-1.0, traj_box[1][0]), max(2.0, traj_box[1][1])]]
        u_linsp = np.linspace(plot_limits[0][0], plot_limits[0][1], 20)
        v_linsp = np.linspace(plot_limits[1][0], plot_limits[1][1], 20)
        u_MG, v_MG = np.meshgrid(u_linsp, v_linsp)
        zs = np.array([self.rosenbrock(u, v) for u, v in zip(np.ravel(u_MG), np.ravel(v_MG))])
        Z = zs.reshape(u_MG.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(u_MG, v_MG, Z, rstride=1, cstride=1, color='b', alpha=0.2, linewidth=0)
        ax.contour(u_MG, v_MG, Z, 20, alpha=0.5, offset=0, stride=30)

        # Highlight Minimum
        ax.scatter(1.0, 1.0, 0.0, 'r*', marker='*', s=200)

        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('Loss')
        ax.view_init(elev=30., azim=30)

        ax.plot(u_history, v_history, loss_history, markerfacecolor='r', markeredgecolor='r', marker='.', markersize=1)
        ax.plot(u_history, v_history, markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)
        return fig

    def anim_run(self, u_history, v_history, loss_history, name="Optimizer Trajectory"):
        """Animate the history of weights (u, v) and the corresponding loss of a optimizer. Plot the Deterministic Rosenbrock as well."""
        # meshgrid
        traj_box = [[min(u_history), max(u_history)], [min(v_history), max(v_history)]]
        plot_limits = [[min(-1.5, traj_box[0][0]), max(1.5, traj_box[0][1])], [min(-1.0, traj_box[1][0]), max(2.0, traj_box[1][1])]]
        u_linsp = np.linspace(plot_limits[0][0], plot_limits[0][1], 20)
        v_linsp = np.linspace(plot_limits[1][0], plot_limits[1][1], 20)
        u_MG, v_MG = np.meshgrid(u_linsp, v_linsp)
        zs = np.array([self.rosenbrock(u, v) for u, v in zip(np.ravel(u_MG), np.ravel(v_MG))])
        Z = zs.reshape(u_MG.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(u_MG, v_MG, Z, rstride=1, cstride=1, color='b', alpha=0.2, linewidth=0)
        ax.contour(u_MG, v_MG, Z, 20, alpha=0.5, offset=0, stride=30)
        # Highlight Minimum
        ax.scatter(1.0, 1.0, 0.0, 'r*', marker='*', s=200)
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
            display_value.set_text('Current Loss = ' + str(loss_history[i]) + ' at iteration: ' + str(i) + ' / ' + str(len(loss_history) - 1))

            return line, point, display_value

        ax.legend(loc=1)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(u_history), interval=120,
                                       repeat_delay=60, blit=True)

        return anim
