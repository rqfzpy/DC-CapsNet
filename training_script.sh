#!/bin/bash

# Scripts to train
scripts=("dc_capsnet.py" "capsnet.py" "cnn.py")

# Dataset list
datasets=("MNIST" "F-MNIST" "Kuzushiji-MNIST")
# datasets=("SVHN" "CIFAR10" "IDRID" "ISIC")
# datasets=("MNIST" "F-MNIST" "Kuzushiji-MNIST" "SVHN" "CIFAR10" "SMALLNORB" "AFFNIST")

# Number of groups
groups=(32)
# groups=(16 32 64)

# Training parameters
epochs=300
batch_size=256
test_batch_size=1000
lr=0.0001
routing_iterations=3

for script in "${scripts[@]}"; do
  for dataset in "${datasets[@]}"; do
    if [[ "$script" == "capsnet.py" ]]; then
      echo "==== Running $script, Dataset: $dataset ===="
      python "$script" \
        --dataset "$dataset" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --test-batch-size "$test_batch_size" \
        --lr "$lr" \
        --routing_iterations "$routing_iterations" \
        --log-interval 10 \
        --with_reconstruction
    elif [[ "$script" == "lenet.py" ]]; then
      echo "==== Running $script, Dataset: $dataset ===="
      python "$script" \
        --dataset "$dataset" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --test-batch-size "$test_batch_size" \
        --lr "$lr" \
        --log-interval 10
    else
      for group in "${groups[@]}"; do
        echo "==== Running $script, Dataset: $dataset, Group: $group ===="
        python "$script" \
          --dataset "$dataset" \
          --group_num "$group" \
          --epochs "$epochs" \
          --batch-size "$batch_size" \
          --test-batch-size "$test_batch_size" \
          --lr "$lr" \
          --routing_iterations "$routing_iterations" \
          --log-interval 10
      done
    fi
  done
done