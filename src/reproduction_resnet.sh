#! /bin/bash

###########################
# # ResNet-18 on CIFAR-10
###########################
# python train.py --seed 3407 --cudnn slow --dataset cifar-10 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false --class_to_forget none
# python retrain.py --epochs 10 --registered_model resnet18-cifar-10-3407-original
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method our --lr 1e-4 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-3407-retrained --method unsir

# ###########################
# # # ResNet-18 frog
# ###########################
# python retrain.py --registered_model resnet18-cifar-10-3407-original --is_class_unlearning True --class_to_forget frog
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method our --lr 1e-4 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-frog-3407-retrained --method unsir

# ###########################
# # # ResNet-18 horse
# ###########################
# python retrain.py --registered_model resnet18-cifar-10-3407-original --is_class_unlearning True --class_to_forget horse
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method our --lr 1e-4 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-3407-retrained --method unsir


# ###########################
# # # ResNet-18 on CIFAR-100
# ###########################
# python train.py --seed 3407 --cudnn slow --dataset cifar-100 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false 
# python retrain.py --registered_model resnet18-cifar-100-3407-original --is_class_unlearning False
python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method our --lr 1e-4 --weight_decay  0
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method our --lr 5e-5
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-3407-retrained --method unsir
 
###########################
# # ResNet-18 rocket
###########################
# python retrain.py --registered_model resnet18-cifar-100-3407-original --is_class_unlearning True --class_to_forget rocket
python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method our --lr 1e-4 --weight_decay  0
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method our --lr 5e-5
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method finetune
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-3407-retrained --method unsir

# ###########################
# # # ResNet-18 beaver
# ###########################
# python retrain.py --registered_model resnet18-cifar-100-3407-original --is_class_unlearning True --class_to_forget beaver
python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method our --lr 1e-4 --weight_decay  0
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method our --lr 5e-5
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-3407-retrained --method unsir

