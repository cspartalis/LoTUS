#!/bin/bash
if [ "$2" = 1 ]; then
    SEED=3407

    original_id_resnet_cifar10="d8f87780c6844d9eb6d1864f9221405c"
    original_id_resnet_cifar100="cae03f8c7e154373a65feb8130121422"
    original_id_resnet_mufac="3092bba899db4a6c87a6b9f5ceedb021"
    original_id_resnet_mucac="7ff615b07b2f4a1189fdde7c20458003"
    original_id_resnet_pneumoniamnist="83940119ef6346a39577fceca3215f8e"
    original_id_vit_cifar10="05e77d7a2510463595f20730c845b49c"
    original_id_vit_cifar100="ffc8273633d743bc832419ff8b0de988"
    original_id_vit_mufac="7913a5ed7f5c4734a1ad01cad921b49a"
    original_id_vit_mucac="81d8454e88a148ed9d096377d1ed4cd7"
    original_id_vit_pneumoniamnist="74d76adeabd746808a49f904adaa873a"

    retrained_id_resnet_cifar10="7d82db407d834469911144093aa3950e"
    retrained_id_resnet_cifar100=""
    retrained_id_resnet_mufac="07d632c9ce5a4d5f951d52dad225fdd1"
    retrained_id_resnet_mucac=""
    retrained_id_resnet_pneumoniamnist="00c81dadd70b45ba9a8663b9067735d3"
    retrained_id_vit_cifar10="7f57ced503c94a9aab42dd3f27392736"
    retrained_id_vit_cifar100="26ee164ccba6486ea53e018efec00db9"
    retrained_id_vit_mufac=""
    retrained_id_vit_mucac="6abe09d9d85c4b7aa04e11f0dd6482ec"
    retrained_id_vit_pneumoniamnist="9e3dafb74c47466a8392cd97b01194a3"

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
    # ResNet-18 & CIFAR-10
    echo "Training ResNet-18 on CIFAR-10"
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
    echo "Training ResNet-18 on CIFAR-100"
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
    echo "Training ResNet-18 on MUFAC"
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

    # ViT & CIFAR-10
    echo "Training ViT on CIFAR-10"
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
        --is_early_stop True \
        --patience 10

    # ViT & CIFAR-100
    echo "Training ViT on CIFAR-100"
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
        --is_early_stop True \
        --patience 10

    # ViT & MUFAC
    echo "Training ViT on MUFAC"
    python train.py \
        --seed $SEED \
        --cudnn slow \
        --dataset mufac \
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

#################################
########### RETRAIN #############
#################################
elif [ "$1" = "retrain" ]; then

    ###############################
    ######### ResNet-18 ###########
    ###############################
    # ResNet-18 & CIFAR-10
    echo "Retraining ResNet-18 on CIFAR-10"
    python retrain.py \
        --run_id $original_id_resnet_cifar10 \
        --cudnn slow

    # ResNet-18 & CIFAR-100
    echo "Retraining ResNet-18 on CIFAR-100"
    python retrain.py \
        --run_id $original_id_resnet_cifar100 \
        --cudnn slow

    # ResNet-18 & MUFAC
    echo "Retraining ResNet-18 on MUFAC"
    python retrain.py \
        --run_id $original_id_resnet_mufac \
        --cudnn slow

    # ResNet-18 & MUCAC
    echo "Retraining ResNet-18 on MUCAC"
    python retrain.py \
        --run_id $original_id_resnet_mucac \
        --cudnn slow

    # ResNet-18 & PneumoniaMNIST
    echo "Retraining ResNet-18 on PneumoniaMNIST"
    python retrain.py \
        --run_id $original_id_resnet_pneumoniamnist \
        --cudnn slow

    ###############################
    ############# ViT #############
    ###############################

    # ViT & CIFAR-10
    echo "Retraining ViT on CIFAR-10"
    python retrain.py \
        --run_id $original_id_vit_cifar10 \
        --cudnn slow

    # ViT & CIFAR-100
    echo "Retraining ViT on CIFAR-100"
    python retrain.py \
        --run_id $original_id_vit_cifar100 \
        --cudnn slow

    # ViT & MUFAC
    echo "Retraining ViT on MUFAC"
    python retrain.py \
        --run_id $original_id_vit_mufac \
        --cudnn slow

    # ViT & MUCAC
    echo "Retraining ViT on MUCAC"
    python retrain.py \
        --run_id $original_id_vit_mucac \
        --cudnn slow

    # ViT & PneumoniaMNIST
    echo "Retraining ViT on PneumoniaMNIST"
    python retrain.py \
        --run_id $original_id_vit_pneumoniamnist \
        --cudnn slow

#################################
########### UNLEARN #############
#################################
elif [ "$1" = "unlearn" ]; then

    thrsholds=(-1 -0.25 -0.5 -0.75 0 0.25 0.5 0.75 1)

    ###############################
    #### ResNet-18 & CIFAR-10 #####
    ###############################

    # echo "Unlearning ResNet-18 on CIFAR-10"
    # echo "=== finetune ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method finetune --epochs 15
    # echo "=== neggrad ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method neggrad --epochs 15
    # echo "=== neggrad_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method neggrad_advanced --epochs 15
    # echo "=== relabel ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method relabel --epochs 15
    # echo "=== relabel_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method relabel_advanced --epochs 15
    # echo "=== boundary ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method boundary --epochs 15
    # echo "=== unsir ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method unsir --epochs 1
    # echo "=== scrub ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method scrub --epochs 15
    # echo "=== ssd ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method ssd --epochs 15
    # echo "=== blindspot ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method blindspot --epochs 15

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_lrp_ce --epochs 15 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_fim_ce --epochs 15 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_lrp_kl --epochs 15 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_kl ==="
    #     python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_fim_kl --epochs 15 --rel_thresh $rel_thresh
    # done

    ###############################
    #### ResNet-18 & CIFAR-100 ####
    ###############################
    
    echo "Unlearning ResNet-18 on CIFAR-100"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method finetune --epochs 15
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method neggrad --epochs 15
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method neggrad_advanced --epochs 15
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method relabel --epochs 15
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method relabel_advanced --epochs 15
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method boundary --epochs 15
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method unsir --epochs 1
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method scrub --epochs 15
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method ssd --epochs 15
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method blindspot --epochs 15

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_lrp_ce --epochs 15 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_fim_ce --epochs 15 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_lrp_kl --epochs 15 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_kl ==="
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_fim_kl --epochs 15 --rel_thresh $rel_thresh
    done

fi


 