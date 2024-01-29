#!/bin/bash

resnet_cifar100='5fbcde870ba8462d9cc1874d21024700'
resnet_cifar10='149b43f9c3414f889e9f768b79d729c2'
resnet_mufac='1bcdd3b016d14404ab22c476184bff75'
resnet_pneumoniamnist='c64c2793582b444f9ba6e9b321465346'

thrsholds=(-1 -0.25 -0.5 -0.75 0 0.25 0.5 0.75 1)
retrain_id=$resnet_cifar10
epochs=15


for zap_thresh in "${thrsholds[@]}"; do
    python unlearn.py --run_id "$retrain_id" --epochs $epochs --mu_method "zap_lrp" --zap_thresh $zap_thresh
done
