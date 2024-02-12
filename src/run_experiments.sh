#!/bin/bash
if [ "$2" = 1 ]; then
    SEED=3407
    original_id_resnet_cifar10="d8f87780c6844d9eb6d1864f9221405c"
    original_id_resnet_cifar100="cae03f8c7e154373a65feb8130121422"
    original_id_resnet_mufac="3092bba899db4a6c87a6b9f5ceedb021"
    # original_id_resnet_mucac=
    # original_id_resnet_pneumoniamnist=
    original_id_vit_cifar10="05e77d7a2510463595f20730c845b49c"
    original_id_vit_cifar100="ffc8273633d743bc832419ff8b0de988"
    original_id_vit_mufac="7913a5ed7f5c4734a1ad01cad921b49a"
    # original_id_vit_mucac=
    # original_id_vit_pneumoniamnist=
elif [ "$2" = 2 ]; then
    SEED=1703
    original_id_resnet_cifar10="62a12e95a5d84152b4e9071ce5aa2e15"
    original_id_resnet_cifar100="e0cbe23525974122a34ec913eeb170e3"
    original_id_resnet_mufac="4716a9d2792f48af94903de6658a7b59"
    # original_id_resnet_mucac=
    # original_id_resnet_pneumoniamnist=
    original_id_vit_cifar10="1f52187408844bcfb2efecca0c520d93"
    # original_id_vit_cifar100=
    # original_id_vit_mufac=
    # original_id_vit_mucac=
    # original_id_vit_pneumoniamnist=
elif [ "$2" = 3 ]; then
    SEED=851
    # original_id_resnet_cifar10=
    # original_id_resnet_cifar100=
    # original_id_resnet_mufac=
    # original_id_resnet_mucac=
    # original_id_resnet_pneumoniamnist=
    # original_id_vit_cifar10=
    # original_id_vit_cifar100=
    # original_id_vit_mufac=
    # original_id_vit_mucac=
    # original_id_vit_pneumoniamnist=
else
    echo "Invalid seed. Try 1,2,3 // {1: 3407, 2: 1703, 3: 851}."
    exit
fi

#################################
############ TRAIN ##############
#################################

if [ "$1" = "train" ]; then

    ###############################
    ######### ResNet-18 ###########
    ###############################
    # # ResNet-18 & CIFAR-10
    # echo "Training ResNet-18 on CIFAR-10"
    # python train.py \
    #     --seed $SEED \
    #     --cudnn slow \
    #     --dataset cifar-10 \
    #     --model resnet18 \
    #     --batch_size 128 \
    #     --epochs 150 \
    #     --loss cross_entropy \
    #     --optimizer sgd \
    #     --lr 0.1 \
    #     --momentum 0.9 \
    #     --weight_decay 0.0005 \
    #     --is_lr_scheduler True \
    #     --warmup_epochs 30 \
    #     --is_early_stop True \
    #     --patience 50 

    # # ResNet-18 & CIFAR-100
    # echo "Training ResNet-18 on CIFAR-100"
    # python train.py \
    #     --seed $SEED \
    #     --cudnn slow \
    #     --dataset cifar-100 \
    #     --model resnet18 \
    #     --batch_size 128 \
    #     --epochs 150 \
    #     --loss cross_entropy \
    #     --optimizer sgd \
    #     --lr 0.1 \
    #     --momentum 0.9 \
    #     --weight_decay 0.0005 \
    #     --is_lr_scheduler True \
    #     --warmup_epochs 30 \
    #     --is_early_stop True \
    #     --patience 50 
    
    # # ResNet-18 & MUFAC
    # echo "Training ResNet-18 on MUFAC"
    # python train.py \
    #     --seed $SEED \
    #     --cudnn slow \
    #     --dataset mufac \
    #     --model resnet18 \
    #     --batch_size 128 \
    #     --epochs 150 \
    #     --loss weighted_cross_entropy \
    #     --optimizer sgd \
    #     --lr 0.001 \
    #     --momentum 0.9 \
    #     --weight_decay 0.0005 \
    #     --is_lr_scheduler False \
    #     --is_early_stop True \
    #     --patience 50 

    # ResNet-18 & MUCAC
    echo "Training ResNet-18 on MUCAC"
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mucac \
        --model resnet18 \
        --batch_size 128 \
        --epochs 150 \
        --loss cross_entropy \
        --optimizer sgd \
        --lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 50 

    # ResNet-18 & PneumoniaMNIST
    echo "Training ResNet-18 on PneumoniaMNIST"
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset pneumoniamnist \
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

    ###############################
    ############# ViT #############
    ###############################

    # # ViT & CIFAR-10
    # echo "Training ViT on CIFAR-10"
    # python train.py \
    #     --seed $SEED \
    #     --cudnn slow \
    #     --dataset cifar-10 \
    #     --model vit \
    #     --batch_size 64 \
    #     --epochs 30 \
    #     --loss cross_entropy \
    #     --optimizer sgd \
    #     --lr 0.0001 \
    #     --momentum 0.9 \
    #     --weight_decay 0.0005 \
    #     --is_lr_scheduler False \
    #     --is_early_stop True \
    #     --patience 10

    # # ViT & CIFAR-100
    # echo "Training ViT on CIFAR-100"
    # python train.py \
    #     --seed $SEED \
    #     --cudnn slow \
    #     --dataset cifar-100 \
    #     --model vit \
    #     --batch_size 64 \
    #     --epochs 30 \
    #     --loss cross_entropy \
    #     --optimizer sgd \
    #     --lr 0.0001 \
    #     --momentum 0.9 \
    #     --weight_decay 0.0005 \
    #     --is_lr_scheduler False \
    #     --is_early_stop True \
    #     --patience 10
    
    # # ViT & MUFAC
    # echo "Training ViT on MUFAC"
    # python train.py \
    #     --seed $SEED \
    #     --cudnn slow \
    #     --dataset mufac \
    #     --model vit \
    #     --batch_size 32 \
    #     --epochs 30 \
    #     --loss weighted_cross_entropy \
    #     --optimizer sgd \
    #     --lr 0.0001 \
    #     --momentum 0.9 \
    #     --weight_decay 0.0005 \
    #     --is_lr_scheduler False \
    #     --is_early_stop True \
    #     --patience 10
    
    # ViT & MUCAC
    echo "Training ViT on MUCAC"
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mucac \
        --model vit \
        --batch_size 32 \
        --epochs 30 \
        --loss cross_entropy \
        --optimizer sgd \
        --lr 0.0001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 10
    
    # ViT & PneumoniaMNIST
    echo "Training ViT on PneumoniaMNIST"
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset pneumoniamnist \
        --model vit \
        --batch_size 32 \
        --epochs 30 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.0001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 10
fi