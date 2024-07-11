#! /bin/bash

# How to reproduce the results for the other seeds as well
# Experiments run on i7-12700K CPU, 32GB RAM, RTX 4080 16GB
# SEED: 13


###########################
# ResNet-18 on CIFAR-10
###########################
# python train.py --seed 13 --cudnn slow --dataset cifar-10 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false --class_to_forget none
# python retrain.py --registered_model resnet18-cifar-10-13-original --is_class_unlearning False
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method our --lr 1e-4 --weight_decay 0.01
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method unsir
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-13-retrained --method scrub

###########################
# ResNet-18 cat
###########################
# python retrain.py --registered_model resnet18-cifar-10-13-original --is_class_unlearning True --class_to_forget cat
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method our --lr 1e-4 --weight_decay 0.001
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method unsir
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-13-retrained --method ssd

###########################
# ResNet-18 horse
###########################
# python retrain.py --registered_model resnet18-cifar-10-13-original --is_class_unlearning True --class_to_forget horse
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method our --lr 1e-4 --weight_decay 0.001
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method unsir
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-13-retrained --method ssd


###########################
# ResNet-18 on CIFAR-100
###########################
# python train.py --seed 13 --cudnn slow --dataset cifar-100 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false 
# python retrain.py --registered_model resnet18-cifar-100-13-original --is_class_unlearning False
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method our --lr 1e-4 --weight_decay 0.01
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method finetune
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-13-retrained --method unsir
 
##########################
# ResNet-18 rocket
##########################
# python retrain.py --registered_model resnet18-cifar-100-13-original --is_class_unlearning True --class_to_forget rocket
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method our --lr 1e-4 --weight_decay 0.001 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method finetune
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method unsir
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-13-retrained --method ssd

###########################
# ResNet-18 beaver
###########################
# python retrain.py --registered_model resnet18-cifar-100-13-original --is_class_unlearning True --class_to_forget beaver
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method our --lr 1e-4 --weight_decay 0.001 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method finetune 
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method unsir
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-13-retrained --method ssd

##########################
# ResNet-18 on MUFAC
##########################
# python train.py --seed 13 --cudnn slow --dataset mufac --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false --class_to_forget none
# python retrain.py --epochs 10 --registered_model resnet18-mufac-13-original --is_class_unlearning False
python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method our --lr 1e-5 --weight_decay 0.1 
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method bad-teacher
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method finetune
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method neggrad
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method amnesiac
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method boundary
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method scrub
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method ssd
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-13-retrained --method unsir


#####################################################################################################
############################################## V i T ################################################
#####################################################################################################

###########################
# # ViT on CIFAR-10
###########################
# python train.py --seed 13 --cudnn slow --dataset cifar-10 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10 --is_class_unlearning False
# python retrain.py --registered_model vit-cifar-10-13-original --is_class_unlearning False 
python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.01
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method amnesiac 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method bad-teacher 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method finetune
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-13-retrained --batch_size 32 --method unsir

###########################
# ViT cat
###########################
# python retrain.py --registered_model vit-cifar-10-13-original --is_class_unlearning True --class_to_forget cat
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method finetune
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-13-retrained --batch_size 32 --method unsir

###########################
# ViT horse
###########################
# python retrain.py --registered_model vit-cifar-10-13-original --is_class_unlearning True --class_to_forget horse
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method finetune
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-13-retrained --batch_size 32 --method unsir

###########################
# ViT on CIFAR-100
###########################
# python train.py --seed 13 --cudnn slow --dataset cifar-100 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10
# python retrain.py --registered_model vit-cifar-100-13-original --is_class_unlearning False 
python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.01
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method finetune
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-13-retrained --batch_size 32 --method unsir

###########################
# ViT rocket
###########################
# python retrain.py --registered_model vit-cifar-100-13-original --is_class_unlearning True --class_to_forget rocket
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method finetune
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-13-retrained --batch_size 32 --method unsir

###########################
# ViT beaver
###########################
# python retrain.py --registered_model vit-cifar-100-13-original --is_class_unlearning True --class_to_forget beaver
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method finetune
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method unsir
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-13-retrained --batch_size 32 --method scrub

###########################
# ViT on MUFAC
###########################
# python train.py --seed 13 --cudnn slow --dataset mufac --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10 --is_class_unlearning False
# python retrain.py --registered_model vit-mufac-13-original --is_class_unlearning False 
python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.01
# python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 32 --method finetune
# python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 32 --method unsir
# python unlearn.py --epochs 3 --registered_model vit-mufac-13-retrained --batch_size 16 --method amnesiac