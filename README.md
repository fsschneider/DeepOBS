# DeepOBS - A Deep Learning Optimizer Benchmark Suite

DeepOBS is a benchmarking suite that drastically simplifies, automates and improves the evaluation of deep learning optimizers.

It can evaluate the performance of new optimizers on a variety of **real-world test problems** and automatically compare them with **realistic baselines**.

The full documentation is available on readthedocs: https://deepobs-iclr.readthedocs.io/

The paper describing DeepOBS is currently under review for ICLR 2019:
https://openreview.net/forum?id=rJg6ssC5Y7

##  Quick Start Guide

### Install Deep OBS
	pip install git+https://github.com/anonymousICLR2019submitter/DeepOBS.git

### Download the data
	deepobs_prepare_data.sh

This will automatically download, sort and prepare all the datasets (except ImageNet). It can take a while, as it will download roughly 1 GB.
(If you already have the data, you could skip this step and always tell Deep OBS where the data is instead.)
The data is now in a folder called 'data_deepobs'.

You are now ready to run different optimizers on different test problems, you can try for example

	deepobs_run_sgd.py mnist.mnist_mlp --num_epochs=2 --lr=1e-1 --bs=128 --nologs

to run SGD on a simple multi-layer perceptron (with a learning rate of 1e-1 and a batch size of 128 for 4 epochs without keeping logs).

Of course, the real value of a benchmark lies in evaluating new optimizers:

### Download and edit a run script
You can download a template run script from there

https://github.com/anonymousICLR2019submitter/DeepOBS/blob/master/scripts/deepobs_run_sgd.py

Now you have a deepobs_run_script.py script in your folder. In order to run your optimizer, you need to change a few things in this script.
The script takes take of the training, evaluation and logging.
Let's assume that we want to benchmark the RMSProp optimizer. Then we only have to change Line 129 from

	opt = tf.train.GradientDescentOptimizer(lr)

   to

	opt = tf.train.RMSPropOptimizer(lr)

Usually the hyperparameters of the optimizers need to be included as well, but for now let's only take the learning rate as a hyperparameter for RMSProp (and if you want change all the 'sgd's in the comments to 'rmsprop'). Let's name this run script now deepobs_run_rmsprop.py

### Run your optimizer
   You can now run your optimizer on a test problem. Let's try it on a noisy quadratic problem:

	python deepobs_run_rmsprop.py quadratic.noisy_quadratic --num_epochs=100 --lr=1e-1 --bs=128 --pickle --run_name=RMSProp_1e-1/

   (we can repeat this a couple of times with different random seeds. This way, we will get a measure of uncertainty in the benchmark plots)

	python deepobs_run_rmsprop.py quadratic.noisy_quadratic --num_epochs=100 --lr=1e-1 --bs=128 --pickle --run_name=RMSProp_1e-1/ --random_seed=43
	python deepobs_run_rmsprop.py quadratic.noisy_quadratic --num_epochs=100 --lr=1e-1 --bs=128 --pickle --run_name=RMSProp_1e-1/ --random_seed=44

   You can monitor the training in real-time using Tensorboard

    tensorboard --logdir=results

   For this example, we will run the above code again, but with a different learning rate. We will call this "second optimizer" RMRProp_1e-2

	python deepobs_run_rmsprop.py quadratic.noisy_quadratic --num_epochs=100 --lr=1e-2 --bs=128 --pickle --run_name=RMSProp_1e-2/
	python deepobs_run_rmsprop.py quadratic.noisy_quadratic --num_epochs=100 --lr=1e-2 --bs=128 --pickle --run_name=RMSProp_1e-2/ --random_seed=43
	python deepobs_run_rmsprop.py quadratic.noisy_quadratic --num_epochs=100 --lr=1e-2 --bs=128 --pickle --run_name=RMSProp_1e-2/ --random_seed=44

   If you want to you can quickly run both optimizers on another problem

	python deepobs_run_rmsprop.py mnist.mnist_mlp --num_epochs=5 --lr=1e-1 --bs=128 --pickle --run_name=RMSProp_1e-1/
	python deepobs_run_rmsprop.py mnist.mnist_mlp --num_epochs=5 --lr=1e-1 --bs=128 --pickle --run_name=RMSProp_1e-1/ --random_seed=43
	python deepobs_run_rmsprop.py mnist.mnist_mlp --num_epochs=5 --lr=1e-1 --bs=128 --pickle --run_name=RMSProp_1e-1/ --random_seed=44

	python deepobs_run_rmsprop.py mnist.mnist_mlp --num_epochs=5 --lr=1e-2 --bs=128 --pickle --run_name=RMSProp_1e-2/
	python deepobs_run_rmsprop.py mnist.mnist_mlp --num_epochs=5 --lr=1e-2 --bs=128 --pickle --run_name=RMSProp_1e-2/ --random_seed=43
	python deepobs_run_rmsprop.py mnist.mnist_mlp --num_epochs=5 --lr=1e-2 --bs=128 --pickle --run_name=RMSProp_1e-2/ --random_seed=44


### Plot Results
   Now we can plot the results of those two "new" optimizers "RMSProp_1e-1" and "RMSProp_1e-2". Since the performance is always relative, we automatically plot the performance against the most popular optimizers (SGD, Momentum, Adam) with the best settings we found after tuning their hyperparameters. Try out:

	deepobs_plot_results.py --results_dir=results --log

   which shows you the learning curves (loss and accuracy for both test and train dataset, but in the case of optimizing a quadratic, there is no accuracy) on a logarithmic plot.
   Additionally it will print out a table summarizing the performances over all test problems (here we only have one or two).
   If you add the option --saveto=save_dir the plots and a color coded table are saved as .png and ready-to-include .tex-files!

### Estimate runtime overhead
   You can estimate the runtime overhead of the new optimizers compared to SGD like this:

	deepobs_estimate_runtime.py deepobs_run_rmsprop.py --optimizer_arguments=--lr=1e-2
   It will return an estimate of the overhead of the new optimizer compared to SGD. In our case it should be quite close to 1.0, as RMSProp costs roughly the same as SGD.
