#!/usr/bin/env bash

# Default options
DATA_FOLDER=baselines_deepobs

# Read variables
for i in "$@"
do
case $i in
    -d=*|--data_dir=*)
    DATA_FOLDER="${i#*=}"
    shift # past argument=value
    ;;
esac
done

# Create folder
mkdir -p $DATA_FOLDER

echo "Downloading Baselines..."
wget https://github.com/fsschneider/DeepOBS_Baselines/archive/v1.1.tar.gz -O $DATA_FOLDER/baselines.tar.gz
echo "Extracting Baselines..."
tar -xvf $DATA_FOLDER/baselines.tar.gz -C $DATA_FOLDER/ --strip-components=1
rm -f $DATA_FOLDER/baselines.tar.gz
