#! /bin/bash

# How to reproduce the results of the paper regarding ResNet-18.
# Experiments run on i7-12700K CPU, 32GB RAM, RTX 4080 16GB.

###########################
# # ViT on CIFAR-10
###########################
# python train.py --seed 3407 --cudnn slow --dataset cifar-10 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10
# python retrain.py --registered_model vit-cifar-10-3407-original --is_class_unlearning False 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method finetune 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method neggrad 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method amnesiac 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method bad-teacher 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method boundary 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method scrub 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method ssd 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-3407-retrained --batch_size 32 --method unsir 

###########################
# # ViT cat
###########################
# python retrain.py --registered_model vit-cifar-10-3407-original --is_class_unlearning True --class_to_forget cat
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method finetune 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-3407-retrained --batch_size 32 --method unsir

###########################
# # ViT horse
###########################
# python retrain.py --registered_model vit-cifar-10-3407-original --is_class_unlearning True --class_to_forget horse
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method finetune 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-3407-retrained --batch_size 32 --method unsir

# ###########################
# # ViT on CIFAR-100
# ###########################
# python train.py --seed 3407 --cudnn slow --dataset cifar-100 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10
# python retrain.py --registered_model vit-cifar-100-3407-original --is_class_unlearning False 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method finetune 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method neggrad 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method amnesiac 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method bad-teacher 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method boundary 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method scrub 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method ssd 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-3407-retrained --batch_size 32 --method unsir 

###########################
# # ViT rocket
###########################
# python retrain.py --registered_model vit-cifar-100-3407-original --is_class_unlearning True --class_to_forget rocket
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method finetune 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-3407-retrained --batch_size 32 --method unsir

###########################
# # ViT beaver
###########################
# python retrain.py --registered_model vit-cifar-100-3407-original --is_class_unlearning True --class_to_forget beaver
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method our --lr 1e-5 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method finetune 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method neggrad
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method amnesiac
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method bad-teacher
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method boundary
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method scrub
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method ssd
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-3407-retrained --batch_size 3 --method unsir


###########################
# # ViT on MUFAC
###########################
# python train.py --seed 3407 --cudnn slow --dataset mufac --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10 --is_class_unlearning False
# python retrain.py --registered_model vit-mufac-3407-original --is_class_unlearning False 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.001
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method finetune 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method neggrad 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method amnesiac 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method bad-teacher 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method boundary 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method scrub 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 32 --method ssd 
# python unlearn.py --epochs 3 --registered_model vit-mufac-3407-retrained --batch_size 16 --method unsir 