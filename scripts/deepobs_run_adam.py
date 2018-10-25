#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run AdamOptimizer on a deepobs test problem.

Runs tensorflow's built-in Adam optimizer on a test problem from the deepobs package.

Usage:

    python run_adam.py <dataset>.<problem> --all_the_command_line_arguments

Execute python run_adam.py --help to see a description for the command line
arguments.
"""
from __future__ import print_function

import tensorflow as tf
import argparse
import importlib
import time
import os
import numpy as np

import deepobs


# ------- Parse Command Line Arguments ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Adam Optimizer on a DeepOBS test problem.")
    parser.add_argument("test_problem",
                        help="Name of the test_problem (e.g. 'cifar10.cifar10_3c3d'")
    parser.add_argument("--data_dir",
                        help="Path to the base data dir. If not set, deepobs uses its default.")
    parser.add_argument("--bs", "--batch_size", required=True, type=int,
                        help="The batch size (positive integer).")
    parser.add_argument("--wd", "--weight_decay", type=float,
                        help="Factor used for the weight_deacy. If non given, the default weight decay for this model is used. Note, not all models have weight decay, and this value will be ignored in this case.")

    # Optimizer hyperparams other than learning rate
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="The beta1 parameter of Adam.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="The beta2 parameter of Adam.")
    parser.add_argument("--epsilon", type=float, default=1e-8,
                        help="The epsilon parameter of Adam.")

    # Learning rate, either constant or schedule
    parser.add_argument("--lr", required=True, type=float,
                        help="Initial learning rate (positive float) to use. To set a learning rate *schedule*, use '--lr_sched_epochs' and '--lr_sched_values' additionally.")
    parser.add_argument("--lr_sched_epochs", nargs="+", type=int,
                        help="One or more epoch numbers (positive integers) that mark learning rate changes, e.g., '--lr_sched_epochs 2 4 5' to change the learning rate after 2, 4 and 5 epochs. The corresponding factors of each change have to be passed via '--lr_sched_factors'.")
    parser.add_argument("--lr_sched_factors", nargs="+", type=float,
                        help="Learning rate factors relative to the initial learning rate at the epochs defined with --lr_sched_epochs, e.g. '--lr_sched_factors= 0.1 0.01 0.001' would reduce the initial learning rate at the defined epochs by a factor of ten.")

    # Number of epochs and checkpoint interval
    parser.add_argument("-N", "--num_epochs", required=True, type=int,
                        help="Total number of training epochs.")
    parser.add_argument("-C", "--checkpoint_epochs", default=1, type=int,
                        help="Interval of training epochs at which to evaluate on the test set and on a larger chunk of the training set.")

    # Random seed
    parser.add_argument("-r", "--random_seed", type=int, default=42,
                        help="An integer to set as tensorflow's random seed.")

    # Logging
    parser.add_argument("--nologs", action="store_const", const=True,
                        default=False, help="Add this flag to switch off tensorflow logging.")
    parser.add_argument("--train_log_interval", type=int, default=10,
                        help="The interval of iterations at which the mini-batch training loss is logged. Set to 1 to log every training step. Default is 10.")
    parser.add_argument("--print_train_iter", action="store_const",
                        const=True, default=False,
                        help="Add this flag to print training loss to stdout at each logged training step.")
    parser.add_argument("--saveto", default="results",
                        help="Folder for saving the results file and the tensorboard logs. If not specified, defaults to 'results'. Within that directory, results and logs will be saved to a subdirectory named after the test problem. Directories will be created if they do not already exist.")
    parser.add_argument("--pickle", action="store_const", const=True,
                        default=False, help="Add this flag to switch on logging in a pickle file.")
    parser.add_argument("--no_time", action="store_const", const=True,
                        default=False, help="Add this flag to switch off timing. Otherwise, the script will report the number of iterations necessary to reach the predefined convergence performance (test accuracy) on the given problem.")
    # Run Name
    parser.add_argument("--run_name", default="",
                        help="Give the run a name describing it. This name will be added to the automatically created name.")

    return parser


def read_args():
    parser = parse_args()
    args = parser.parse_args()
    return args
# -----------------------------------------------------------------------------


def main(test_problem, bs, lr, N, beta1=0.9, beta2=0.999, epsilon=1e-8, data_dir=None, wd=None, lr_sched_epochs=None, lr_sched_factors=None, checkpoint_epochs=1, random_seed=42, nologs=False, train_log_interval=10, print_train_iter=False, saveto="results", pickle=False, no_time=False, run_name=""):
    # Put all input arguments back into an args variable, so I can use it as before (without the main function)
    args = argparse.Namespace(**locals())
    # Create a uniquely identfying name for this experiment
    name = str(args.run_name)
    # If no name given or last character is '/' don't include two dashes
    if name != "" and name[-1] != '/':
        name += "__"
    name += args.test_problem.split(".")[-1]
    name += "__adam"
    name += "__bs_" + str(args.bs)
    name += "__beta1_" + deepobs.run_utils.float2str(args.beta1)
    name += "__beta2_" + deepobs.run_utils.float2str(args.beta2)
    name += "__eps_" + deepobs.run_utils.float2str(args.epsilon)
    if args.lr_sched_epochs is None:
        name += "__lr_" + deepobs.run_utils.float2str(args.lr)
    else:
        # If learning rate schedule is used
        name += "__lr_sched_" + deepobs.run_utils.float2str(args.lr)
        for step, val in zip(args.lr_sched_epochs, args.lr_sched_factors):
            name += "_" + str(step) + "_" + deepobs.run_utils.float2str(val * args.lr)
    name += "__seed_" + str(args.random_seed)
    name += "__" + time.strftime("%Y-%m-%d-%H-%M-%S")

    # Set the log/results directory (create if it does not exist)
    logdir = os.path.join(args.saveto, args.test_problem.split(".")[-1])
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Set the data directory path of deepobs
    if args.data_dir is not None:
        deepobs.dataset_utils.set_data_dir(args.data_dir)

    # Set up test problem
    test_problem = importlib.import_module("deepobs." + args.test_problem)
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)  # Set random seed Tensorflow
    np.random.seed(args.random_seed)
    # use weight decay if given, otherwise use the problem dependant default
    if args.wd is not None:
        set_up_test_problem = test_problem.set_up(batch_size=args.bs, weight_decay=args.wd)
    else:
        set_up_test_problem = test_problem.set_up(batch_size=args.bs)
    losses, accuracy = set_up_test_problem.get()
    loss = tf.reduce_mean(losses)

    # If there are terms in the REGULARIZATION_LOSSES collection, add them to the loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if regularization_losses:
        loss = loss + tf.add_n(regularization_losses)

    # Learning rate schedule; constant or schedule
    global_step = tf.Variable(0, trainable=False)
    lr = tf.Variable(args.lr, trainable=False)
    learning_rate_sched = deepobs.run_utils.make_learning_rate_schedule(args)

    # Set up optimizer, updating all variables in TRAINABLE_VARIABLES collection,
    # with a dependency on performing all operation in the collection UPDATE_OPS
    opt = tf.train.AdamOptimizer(lr, beta1=args.beta1, beta2=args.beta2,
                                 epsilon=args.epsilon)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        step = opt.minimize(loss, global_step=global_step)

    # Tensorboard summaries
    if not args.nologs:
        # per iteration
        train_loss_summary = tf.summary.scalar("training/training_loss", loss,
                                               collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
        # per epoch
        learning_rate_summary = tf.summary.scalar("hyperparams/learning_rate", lr, collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])
        batch_size_summary = tf.summary.scalar("hyperparams/batch_size", args.bs, collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])
        beta1_summary = tf.summary.scalar("hyperparams/beta1", args.beta1, collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])
        beta2_summary = tf.summary.scalar("hyperparams/beta2", args.beta2, collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])

        per_iteration_summaries = tf.summary.merge_all(key="per_iteration")
        per_epoch_summaries = tf.summary.merge_all(key="per_epoch")
        summary_writer = tf.summary.FileWriter(
            os.path.join(logdir, name))

    # Timing
    if not args.no_time:
        convergence_iterations = 0  # Set convergence iteration initially to zero
        convergence_reached = False
        convergence_performance, use_accuracy = deepobs.run_utils.convergence_performance(args)

    # ------- start of train looop --------
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for n in range(args.num_epochs + 1):

        # Evaluate if we hit the checkpoint epoch (and in the last step)
        if n % args.checkpoint_epochs == 0 or n == args.num_epochs:

            print("********************************")
            print("CHECKPOINT (", n, "of", args.num_epochs, "epochs )")

            # Evaluate on training data
            sess.run(set_up_test_problem.train_eval_init_op)
            train_loss_, train_acc_ = 0.0, 0.0
            num_eval_iters = 0
            while True:
                try:
                    l_, a_ = sess.run([loss, accuracy])
                    train_loss_ += l_
                    train_acc_ += a_
                    num_eval_iters += 1
                except tf.errors.OutOfRangeError:
                    train_loss_ /= float(num_eval_iters)
                    train_acc_ /= float(num_eval_iters)
                    break

            # Evaluate on test data
            sess.run(set_up_test_problem.test_init_op)
            test_loss_, test_acc_ = 0.0, 0.0
            num_eval_iters = 0
            while True:
                try:
                    l_, a_ = sess.run([loss, accuracy])
                    test_loss_ += l_
                    test_acc_ += a_
                    num_eval_iters += 1
                except tf.errors.OutOfRangeError:
                    test_loss_ /= float(num_eval_iters)
                    test_acc_ /= float(num_eval_iters)
                    break

            # If time option is true, track progress towards convergence performance
            if not args.no_time:
                # If current performance is larger than the predefined convergence performance save number of epochs
                if use_accuracy:
                    performance_measure = test_acc_
                    convergence = performance_measure > convergence_performance
                else:
                    performance_measure = test_loss_
                    convergence = performance_measure < convergence_performance
                if convergence and not convergence_reached:
                    convergence_reached = True
                    convergence_iterations = n
                    print("** Reached convergence performance **")

            # Log results to tensorflow summaries
            if not args.nologs:
                summary = tf.Summary()
                summary.value.add(tag="checkpoint/checkpoint_train_loss",
                                  simple_value=train_loss_)
                summary.value.add(tag="checkpoint/checkpoint_train_acc",
                                  simple_value=train_acc_)
                summary.value.add(tag="checkpoint/checkpoint_test_loss",
                                  simple_value=test_loss_)
                summary.value.add(tag="checkpoint/checkpoint_test_acc",
                                  simple_value=test_acc_)
                if not args.no_time:
                    summary.value.add(tag="time/percentage_convergence_performance", simple_value=(performance_measure / convergence_performance))
                    summary.value.add(tag="time/convergence_iterations", simple_value=convergence_iterations)
                summary_writer.add_summary(summary, n)
                summary_writer.flush()

            print("TRAIN: loss", train_loss_, "acc", train_acc_)
            print("TEST: loss", test_loss_, "acc", test_acc_)
            print("********************************")

            # Per epoch summary
            per_epoch_summary_ = sess.run(per_epoch_summaries)
            summary_writer.add_summary(per_epoch_summary_, n)
            summary_writer.flush()

            # Break from train loop after the last round of evaluation
            if n == args.num_epochs:
                break

        # Training Step
        sess.run(set_up_test_problem.train_init_op)
        # Change learning rate
        sess.run(tf.assign(lr, learning_rate_sched[n]))
        while True:
            try:
                # Training step, with logging if we hit the train_log_interval
                if sess.run(global_step) % args.train_log_interval != 0:
                    _ = sess.run(step)
                else:  # if n%args.train_log_interval==0:
                    if not args.nologs:
                        _, loss_, per_iter_summary_ = sess.run([step, loss, per_iteration_summaries])
                        summary_writer.add_summary(per_iter_summary_, sess.run(global_step))
                    else:
                        _, loss_ = sess.run([step, loss])
                    if args.print_train_iter:
                        print("Epoch", n, ", Step", sess.run(global_step), ": loss", loss_)
            except tf.errors.OutOfRangeError:
                break

    sess.close()
    # ------- end of train looop --------

    # Dump to pickle
    if args.pickle and not args.nologs:
        deepobs.run_utils.dump2pickle(args, summary_writer, name)


if __name__ == '__main__':
    main(**vars(read_args()))
