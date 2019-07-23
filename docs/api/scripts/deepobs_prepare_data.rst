============
Prepare Data
============

A convenience script to download all data sets for DeepOBS and preprocess them
so they are ready to be used with DeepOBS.

.. NOTE::
  Currently there is no data downloading and preprocessing mechanic implemented
  for `ImageNet`. Downloading the `ImageNet` data set requires an account and
  can take a lot of time to download. Additionally, it requires quite a large
  amount of memory. The best way currently is to download and preprocess the
  `ImageNet` data set separately if needed and move it into the DeepOBS data
  folder.

The file will create a set of folders of the following structure:

| data_deepobs
| ├── cifar10
| │   ├── data_batch_1.bin
| │   ├── data_batch_2.bin
| │   └── ...
| ├── cifar100
| │   ├── train.bin
| │   ├── test.bin
| │   └── ...
| ├── fmnist
| │   ├── t10k-images-idx3-ubyte.gz
| │   ├── t10k-labels-idx1-ubyte.gz
| │   └── ...
| ├── mnist
| │   ├── t10k-images-idx3-ubyte.gz
| │   ├── t10k-labels-idx1-ubyte.gz
| │   └── ...
| ├── svhn
| │   ├── data_batch_0.bin
| │   ├── data_batch_1.bin
| │   └── ...
| ├── tolstoi
| │   ├── train.npy
| │   ├── test.npy
| │   └── ...
| ├── imagenet
| │   ├── train-00000-of-01024
| │   ├── ...
| │   ├── validation-00000-of-00128
| │   └── ...

DeepOBS expects a structure like this, so if you already have (most of the) the
data sets already, you still need to bring it into this order.

**Usage:**

  .. code:: python

    usage: deepobs_prepare_data.sh [--data_dir=DIR] [--skip SKIP] [--only ONLY]

Named Arguments
===============

+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| -d --data_dir | Path where the data sets should be saved. Defaults to the current folder.                                                                                                                                                                                          |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| -s --skip     | Defines which data sets should be skipped. Argument needs to be one of the following ``mnist``, ``fmnist``, ``cifar10``, ``cifar100``, ``svhn``, ``imagenet``, ``tolstoi``. You can use the ``--skip`` argument multiple times.                                    |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| -o --only     | Specify if only a single data set should be downloaded. Argument needs to be one of the following ``mnist``, ``fmnist``, ``cifar10``, ``cifar100``, ``svhn``, ``imagenet``, ``tolstoi``. This overwrites the ``--skip`` argument and should can only be used once. |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
