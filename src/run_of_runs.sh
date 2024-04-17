#!/bin/bash

# ./run_experiments_forget_class.sh "unlearn" 4 "rocket" 
./run_experiments_forget_class.sh "unlearn" 5 "rocket" 
# ./run_experiments_forget_class.sh "unlearn" 1 "beaver" 
# ./run_experiments_forget_class.sh "unlearn" 2 "beaver" 
# ./run_experiments_forget_class.sh "unlearn" 3 "beaver" 
# ./run_experiments_forget_class.sh "unlearn" 4 "beaver" 
# ./run_experiments_forget_class.sh "unlearn" 5 "beaver" 

# ./run_experiments_forget_instances.sh "unlearn" 5  
## Comment out everything up until vit_muFac
./run_experiments_forget_instances.sh "unlearn" 4  

## Copy-paste the corresponding run IDs
# ./run_max_entropy_ablation.sh  3 "mucac" "resnet"
# ./run_max_entropy_ablation.sh  4 "mucac" "resnet"
# ./run_max_entropy_ablation.sh  5 "mucac" "resnet"
###
# ./run_max_entropy_ablation.sh  1 "pm" "resnet"
# ./run_max_entropy_ablation.sh  2 "pm" "resnet"
# ./run_max_entropy_ablation.sh  3 "pm" "resnet"
# ./run_max_entropy_ablation.sh  4 "pm" "resnet"
# ./run_max_entropy_ablation.sh  5 "pm" "resnet"
###
# ./run_max_entropy_ablation.sh  1 "cifar10" "vit"
# ./run_max_entropy_ablation.sh  2 "cifar10" "vit"
# ./run_max_entropy_ablation.sh  3 "cifar10" "vit"
# ./run_max_entropy_ablation.sh  4 "cifar10" "vit"
# ./run_max_entropy_ablation.sh  5 "cifar10" "vit"
###
# ./run_max_entropy_ablation.sh  1 "cifar100" "vit"
# ./run_max_entropy_ablation.sh  2 "cifar100" "vit"
# ./run_max_entropy_ablation.sh  3 "cifar100" "vit"
# ./run_max_entropy_ablation.sh  4 "cifar100" "vit"
# ./run_max_entropy_ablation.sh  5 "cifar100" "vit"
###
# ./run_max_entropy_ablation.sh  1 "mufac" "vit"
# ./run_max_entropy_ablation.sh  2 "mufac" "vit"
# ./run_max_entropy_ablation.sh  3 "mufac" "vit"
# ./run_max_entropy_ablation.sh  4 "mufac" "vit"
# ./run_max_entropy_ablation.sh  5 "mufac" "vit"
###
# ./run_max_entropy_ablation.sh  1 "mucac" "vit"
# ./run_max_entropy_ablation.sh  2 "mucac" "vit"
# ./run_max_entropy_ablation.sh  3 "mucac" "vit"
# ./run_max_entropy_ablation.sh  4 "mucac" "vit"
# ./run_max_entropy_ablation.sh  5 "mucac" "vit"
###
# ./run_max_entropy_ablation.sh  1 "pm" "vit"
# ./run_max_entropy_ablation.sh  2 "pm" "vit"
# ./run_max_entropy_ablation.sh  3 "pm" "vit"
# ./run_max_entropy_ablation.sh  4 "pm" "vit"
# ./run_max_entropy_ablation.sh  5 "pm" "vit"
###
# ./run_max_entropy_ablation.sh  1 "rocket" "resnet"
# ./run_max_entropy_ablation.sh  2 "rocket" "resnet"
# ./run_max_entropy_ablation.sh  3 "rocket" "resnet"
# ./run_max_entropy_ablation.sh  4 "rocket" "resnet"
# ./run_max_entropy_ablation.sh  5 "rocket" "resnet"
###
# ./run_max_entropy_ablation.sh  1 "rocket" "vit"
# ./run_max_entropy_ablation.sh  2 "rocket" "vit"
# ./run_max_entropy_ablation.sh  3 "rocket" "vit"
# ./run_max_entropy_ablation.sh  4 "rocket" "vit"
# ./run_max_entropy_ablation.sh  5 "rocket" "vit"
###
# ./run_max_entropy_ablation.sh  1 "beaver" "resnet"
# ./run_max_entropy_ablation.sh  2 "beaver" "resnet"
# ./run_max_entropy_ablation.sh  3 "beaver" "resnet"
# ./run_max_entropy_ablation.sh  4 "beaver" "resnet"
# ./run_max_entropy_ablation.sh  5 "beaver" "resnet"
###
# ./run_max_entropy_ablation.sh  1 "beaver" "vit"
# ./run_max_entropy_ablation.sh  2 "beaver" "vit"
# ./run_max_entropy_ablation.sh  3 "beaver" "vit"
# ./run_max_entropy_ablation.sh  4 "beaver" "vit"
# ./run_max_entropy_ablation.sh  5 "beaver" "vit"