#!/bin/bash
SEED=3407

if [ "$1" = "train" ]; then
    # ResNet-18 & CIFAR-10
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset cifar-10 \
        --model resnet18 \
        --batch_size 128 \
        --epochs 150 \
        --loss cross_entropy \
        --optimizer sgd \
        --lr 0.1 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler True \
        --warmup_epochs 30 \
        --is_early_stop True \
        --patience 50 

    # ResNet-18 & CIFAR-100
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset cifar-100 \
        --model resnet18 \
        --batch_size 128 \
        --epochs 150 \
        --loss cross_entropy \
        --optimizer sgd \
        --lr 0.1 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler True \
        --warmup_epochs 30 \
        --is_early_stop True \
        --patience 50 
    
    # ResNet-18 & MUFAC
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mufac \
        --model resnet18 \
        --batch_size 128 \
        --epochs 150 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 50 

    # ResNet-18 & MUCAC
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mucac \
        --model resnet18 \
        --batch_size 128 \
        --epochs 150 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 50 

    # ResNet-18 & PneumoniaMNIST
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mucac \
        --model resnet18 \
        --batch_size 128 \
        --epochs 150 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 50 

    # ViT & CIFAR-10
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset cifar-10 \
        --model vit \
        --batch_size 64 \
        --epochs 30 \
        --loss cross_entropy \
        --optimizer sgd \
        --lr 0.0001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop False

    # ViT & CIFAR-100
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset cifar-100 \
        --model vit \
        --batch_size 64 \
        --epochs 30 \
        --loss cross_entropy \
        --optimizer sgd \
        --lr 0.0001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop False
    
    # ViT & MUFAC
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mufac \
        --model vit \
        --batch_size 64 \
        --epochs 30 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.0001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop False
    
    # ViT & MUCAC
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mucac \
        --model vit \
        --batch_size 64 \
        --epochs 30 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.0001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop False
    
    # ViT & PneumoniaMNIST
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset pneumoniamnist \
        --model vit \
        --batch_size 64 \
        --epochs 30 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.0001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop False
