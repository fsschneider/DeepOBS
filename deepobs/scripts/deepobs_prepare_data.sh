#!/usr/bin/env bash

# Default options
DATA_FOLDER=data_deepobs
prepare_cifar10=true
prepare_cifar100=true
prepare_imagenet=true
prepare_fmnist=true
prepare_mnist=true
prepare_svhn=true
prepare_tolstoi=true

# Read variables
for i in "$@"
do
case $i in
    -d=*|--data_dir=*)
    DATA_FOLDER="${i#*=}"
    shift # past argument=value
    ;;
    -s=cifar10|--skip=cifar10)
    prepare_cifar10=false
    shift # past argument=value
    ;;
    -s=cifar100|--skip=cifar100)
    prepare_cifar100=false
    shift # past argument=value
    ;;
    -s=imagenet|--skip=imagenet)
    prepare_imagenet=false
    shift # past argument=value
    ;;
    -s=fmnist|--skip=fmnist)
    prepare_fmnist=false
    shift # past argument=value
    ;;
    -s=mnist|--skip=mnist)
    prepare_mnist=false
    shift # past argument=value
    ;;
    -s=svhn|--skip=svhn)
    prepare_svhn=false
    shift # past argument=value
    ;;
    -s=tolstoi|--skip=tolstoi)
    prepare_tolstoi=false
    shift # past argument=value
    ;;
    -o=anki|--only=anki)
    prepare_cifar10=false
    prepare_cifar100=false
    prepare_imagenet=false
    prepare_fmnist=false
    prepare_mnist=false
    prepare_svhn=false
    prepare_tolstoi=false
    break
    shift # past argument=value
    ;;
    -o=cifar10|--only=cifar10)
    prepare_cifar10=true
    prepare_cifar100=false
    prepare_imagenet=false
    prepare_fmnist=false
    prepare_mnist=false
    prepare_svhn=false
    prepare_tolstoi=false
    break
    shift # past argument=value
    ;;
    -o=cifar100|--only=cifar100)
    prepare_cifar10=false
    prepare_cifar100=true
    prepare_imagenet=false
    prepare_fmnist=false
    prepare_mnist=false
    prepare_svhn=false
    prepare_tolstoi=false
    break
    shift # past argument=value
    ;;
    -o=imagenet|--only=imagenet)
    prepare_cifar10=false
    prepare_cifar100=false
    prepare_imagenet=true
    prepare_fmnist=false
    prepare_mnist=false
    prepare_svhn=false
    prepare_tolstoi=false
    break
    shift # past argument=value
    ;;
    -o=fmnist|--only=fmnist)
    prepare_cifar10=false
    prepare_cifar100=false
    prepare_imagenet=false
    prepare_fmnist=true
    prepare_mnist=false
    prepare_svhn=false
    prepare_tolstoi=false
    break
    shift # past argument=value
    ;;
    -o=mnist|--only=mnist)
    prepare_cifar10=false
    prepare_cifar100=false
    prepare_imagenet=false
    prepare_fmnist=false
    prepare_mnist=true
    prepare_svhn=false
    prepare_tolstoi=false
    break
    shift # past argument=value
    ;;
    -o=svhn|--only=svhn)
    prepare_cifar10=false
    prepare_cifar100=false
    prepare_imagenet=false
    prepare_fmnist=false
    prepare_mnist=false
    prepare_svhn=true
    prepare_tolstoi=false
    break
    shift # past argument=value
    ;;
    -o=tolstoi|--only=tolstoi)
    prepare_cifar10=false
    prepare_cifar100=false
    prepare_imagenet=false
    prepare_fmnist=false
    prepare_mnist=false
    prepare_svhn=false
    prepare_tolstoi=true
    break
    shift # past argument=value
    ;;
esac
done

# Create all folders
echo "*** Creating Folders..."
mkdir -p $DATA_FOLDER
mkdir -p $DATA_FOLDER/cifar-10
mkdir -p $DATA_FOLDER/cifar-100
mkdir -p $DATA_FOLDER/imagenet
mkdir -p $DATA_FOLDER/fmnist
mkdir -p $DATA_FOLDER/mnist
mkdir -p $DATA_FOLDER/svhn
mkdir -p $DATA_FOLDER/tolstoi

# Cifar-10
if [ "$prepare_cifar10" = "true" ]
  then echo "*** Preparing Cifar-10..."
  echo "Downloading Cifar-10 Binary..."
  wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -O $DATA_FOLDER/cifar-10/cifar-10-binary.tar.gz
  echo "Extracting..."
  tar -xvf $DATA_FOLDER/cifar-10/cifar-10-binary.tar.gz -C $DATA_FOLDER/cifar-10/ --strip 1
  rm -f $DATA_FOLDER/cifar-10/cifar-10-binary.tar.gz
else echo "*** Skipping Cifar-10..."
fi

# Cifar-100
if [ "$prepare_cifar100" = "true" ]
  then echo "*** Preparing Cifar-100..."
  echo "Downloading Cifar-100 Binary..."
  wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -O $DATA_FOLDER/cifar-100/cifar-100-binary.tar.gz
  echo "Extracting..."
  tar -xvf $DATA_FOLDER/cifar-100/cifar-100-binary.tar.gz -C $DATA_FOLDER/cifar-100/ --strip 1
  rm -f $DATA_FOLDER/cifar-100/cifar-100-binary.tar.gz
else echo "*** Skipping Cifar-100..."
fi

# ImageNet
if [ "$prepare_imagenet" = "true" ]
  then echo "*** Preparing ImageNet..."
  echo "CURRENTLY NOT IMPLEMENTED!!!!"
else echo "*** Skipping ImageNet..."
fi

# Fashion-MNIST
if [ "$prepare_fmnist" = "true" ]
  then echo "*** Preparing Fashion-MNIST..."
  echo "Downloading Train Images..."
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -O $DATA_FOLDER/fmnist/train-images-idx3-ubyte.gz
  echo "Downloading Train Labels..."
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -O $DATA_FOLDER/fmnist/train-labels-idx1-ubyte.gz
  echo "Downloading Test Images..."
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -O $DATA_FOLDER/fmnist/t10k-images-idx3-ubyte.gz
  echo "Downloading Test Labels..."
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -O $DATA_FOLDER/fmnist/t10k-labels-idx1-ubyte.gz
else echo "*** Skipping Fashion-MNIST..."
fi


# MNIST
if [ "$prepare_mnist" = "true" ]
  then echo "*** Preparing MNIST..."
  echo "Downloading Train Images..."
  wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O $DATA_FOLDER/mnist/train-images-idx3-ubyte.gz
  echo "Downloading Train Labels..."
  wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O $DATA_FOLDER/mnist/train-labels-idx1-ubyte.gz
  echo "Downloading Test Images..."
  wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O $DATA_FOLDER/mnist/t10k-images-idx3-ubyte.gz
  echo "Downloading Test Labels..."
  wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O $DATA_FOLDER/mnist/t10k-labels-idx1-ubyte.gz
else echo "*** Skipping MNIST..."
fi

# SVHN
if [ "$prepare_svhn" = "true" ]
  then echo "*** Preparing SVHN..."
  echo "Downloading Train..."
  wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat -O $DATA_FOLDER/svhn/train_32x32.mat
  echo "Downloading Test..."
  wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -O $DATA_FOLDER/svhn/test_32x32.mat
  # echo "Downloading Extra..."
  # wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat -O $DATA_FOLDER/svhn/extra_32x32.mat
  echo "Preprocessing SVHN..."
  python -c "import deepobs.scripts._svhn_preprocess as _svhn_preprocess; _svhn_preprocess.preprocess(file_path='$DATA_FOLDER/svhn/', save_path='$DATA_FOLDER/svhn/')"
  rm -f $DATA_FOLDER/svhn/train_32x32.mat
  rm -f $DATA_FOLDER/svhn/test_32x32.mat
  # rm -f $DATA_FOLDER/svhn/extra_32x32.mat
else echo "*** Skipping SVHN..."
fi

# Tolstoi
if [ "$prepare_tolstoi" = "true" ]
  then echo "*** Preparing Tolstoi..."
  echo "Downloading Tolstoi..."
  # wget http://www.gutenberg.org/files/2600/2600-0.txt -O $DATA_FOLDER/tolstoi/input.txt
  # alternative link, since gutenberg is currently geoblocked from germany
  wget https://github.com/AllenDowney/fluent-python-notebooks/raw/master/attic/sequences/war-and-peace.txt -O $DATA_FOLDER/tolstoi/input.txt
  # remove some lines (index, gutenberg-stuff)
  # sed -i '1,35d' $DATA_FOLDER/tolstoi/input.txt
  # sed -i '5,806d' $DATA_FOLDER/tolstoi/input.txt
  # alternative version
  sed -i '1,29d' $DATA_FOLDER/tolstoi/input.txt
  sed -i '5,775d' $DATA_FOLDER/tolstoi/input.txt
  sed -i '63850,64207d' $DATA_FOLDER/tolstoi/input.txt
  echo "Preprocessing Tolstoi..."
  # remove old files if they exist
  rm -f $DATA_FOLDER/tolstoi/test.npy
  rm -f $DATA_FOLDER/tolstoi/train.npy
  rm -f $DATA_FOLDER/tolstoi/vocab.pkl
  python -c "import deepobs.scripts._tolstoi_preprocess as _tolstoi_preprocess;_tolstoi_preprocess.preprocess(file_path='$DATA_FOLDER/tolstoi/', encoding='utf-8', test_size=0.2)"
else echo "*** Skipping Tolstoi..."
fi


echo "*** Done."
