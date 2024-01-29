#!/bin/bash

#TODO: neggrad advanced and relabel advanced
methods=('finetune' 'neggrad' 'relabel' 'boundary' 'unsir' 'scrub')

resnet_cifar10='149b43f9c3414f889e9f768b79d729c2'
resnet_cifar100='5fbcde870ba8462d9cc1874d21024700'
resnet_mufac='1bcdd3b016d14404ab22c476184bff75'
resnet_pneumoniamnist='c64c2793582b444f9ba6e9b321465346'

retrain_id=$resnet_mufac
epochs=3

for method in "${methods[@]}"; do
    python unlearn.py --run_id "$retrain_id" --epochs $epochs --mu_method "$method" 
done
