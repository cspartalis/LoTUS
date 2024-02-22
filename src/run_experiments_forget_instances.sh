#!/bin/bash
if [ "$2" = 1 ]; then
    SEED=3407

    original_id_resnet_cifar10="9c9aedb8d36d49469952118f6cbb8188"
    original_id_resnet_cifar100="29def4b723f240a997d3cc66db6cbb0d"
    original_id_resnet_mufac="4b3874797ed143ad9cdfe4583e98d4f7"
    original_id_resnet_mucac="70cc89fa36f54e66bdb483a8c5dc6f9b"
    original_id_resnet_pneumoniamnist="78e40a58e0ca47afb1a69bdb97e1bc86"
    original_id_vit_cifar10="94ef8d8e0e644ed085df04f796042992"
    original_id_vit_cifar100="970f57399d4a4347893a23e2f28e6a6a"
    original_id_vit_mufac="4e855dffc5274695b388889e80a8d018"
    original_id_vit_mucac="47019c12923d4709bd8ceb984edd4b08"
    original_id_vit_pneumoniamnist="b05882b5cdf54b3eb7f8d61c916e436d"

    retrained_id_resnet_cifar10="51fb67619ea7422887d9aa27fdc7083a"
    retrained_id_resnet_cifar100="336951c7d04e4f7a84a7607bca1bb251"
    retrained_id_resnet_mufac="32d4ddd99d8a4e78b7e3dba1282a9d4a"
    retrained_id_resnet_mucac="71e3953dac484489833665adb8e775e2"
    retrained_id_resnet_pneumoniamnist="d97de61da5b5457494114a39cba84590"
    retrained_id_vit_cifar10="b48e15b08a124170a8e4a8ca98e94a1e"
    retrained_id_vit_cifar100="5c53b23715aa4388bbc1a6bb4bd3809c"
    retrained_id_vit_mufac="63cbefb7025f409d9780fc394a74f2af"
    retrained_id_vit_mucac="54021076df0d4664bda9bd0c7239aade"
    retrained_id_vit_pneumoniamnist="9b9967c050054a24aea7d4ef5729d2f4"

elif [ "$2" = 2 ]; then
    SEED=1703

    original_id_resnet_cifar10=""
    original_id_resnet_cifar100=""
    original_id_resnet_mufac=""
    original_id_resnet_mucac=""
    original_id_resnet_pneumoniamnist=""
    original_id_vit_cifar10=""
    original_id_vit_cifar100=""
    original_id_vit_mufac=""
    original_id_vit_mucac=""
    original_id_vit_pneumoniamnist=""

    retrained_id_resnet_cifar10=""
    retrained_id_resnet_cifar100=""
    retrained_id_resnet_mufac=""
    retrained_id_resnet_mucac=""
    retrained_id_resnet_pneumoniamnist=""
    retrained_id_vit_cifar10=""
    retrained_id_vit_cifar100=""
    retrained_id_vit_mufac=""
    retrained_id_vit_mucac=""
    retrained_id_vit_pneumoniamnist=""
elif [ "$2" = 3 ]; then
    SEED=851

    original_id_resnet_cifar10=""
    original_id_resnet_cifar100=""
    original_id_resnet_mufac=""
    original_id_resnet_mucac=""
    original_id_resnet_pneumoniamnist=""
    original_id_vit_cifar10=""
    original_id_vit_cifar100=""
    original_id_vit_mufac=""
    original_id_vit_mucac=""
    original_id_vit_pneumoniamnist=""

    retrained_id_resnet_cifar10=""
    retrained_id_resnet_cifar100=""
    retrained_id_resnet_mufac=""
    retrained_id_resnet_mucac=""
    retrained_id_resnet_pneumoniamnist=""
    retrained_id_vit_cifar10=""
    retrained_id_vit_cifar100=""
    retrained_id_vit_mufac=""
    retrained_id_vit_mucac=""
    retrained_id_vit_pneumoniamnist=""
else
    echo "Invalid arguments!"
    echo "$1 should be one of the following: train, retrain, unlearn"
    echo "$2 should be one of the following: 1, 2, 3 (for seeds 3407, 1703, 851 respectively)"
    echo "Example: ./run_experiments_forget_instances.sh train 1"
    echo "Exiting..."
    exit
fi

################################################################################################
###################################### T R A I N ###############################################
################################################################################################

if [ "$1" = "train" ]; then

    ###############################
    ######### ResNet-18 ###########
    ###############################
    # ResNet-18 & CIFAR-10
    echo "*** Training ResNet-18 on CIFAR-10"
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
    echo "*** Training ResNet-18 on CIFAR-100"
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
    echo "*** Training ResNet-18 on MUFAC"
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
    echo "*** Training ResNet-18 on PneumoniaMNIST"
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
    echo "*** Training ViT on CIFAR-10"
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
    echo "*** Training ViT on CIFAR-100"
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
    echo "*** Training ViT on MUFAC"
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
    echo "*** Training ViT on MUCAC"
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
    echo "*** Training ViT on PneumoniaMNIST"
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

################################################################################################
######################################## R E T R A I N #########################################
################################################################################################
elif [ "$1" = "retrain" ]; then

    ###############################
    ######### ResNet-18 ###########
    ###############################
    # ResNet-18 & CIFAR-10
    echo "*** Retraining ResNet-18 on CIFAR-10"
    python retrain.py \
        --run_id $original_id_resnet_cifar10 \
        --cudnn slow

    # ResNet-18 & CIFAR-100
    echo "*** Retraining ResNet-18 on CIFAR-100"
    python retrain.py \
        --run_id $original_id_resnet_cifar100 \
        --cudnn slow

    # ResNet-18 & MUFAC
    echo "*** Retraining ResNet-18 on MUFAC"
    python retrain.py \
        --run_id $original_id_resnet_mufac \
        --cudnn slow

    # ResNet-18 & MUCAC
    echo "*** Retraining ResNet-18 on MUCAC"
    python retrain.py \
        --run_id $original_id_resnet_mucac \
        --cudnn slow

    # ResNet-18 & PneumoniaMNIST
    echo "*** Retraining ResNet-18 on PneumoniaMNIST"
    python retrain.py \
        --run_id $original_id_resnet_pneumoniamnist \
        --cudnn slow

    ###############################
    ############# ViT #############
    ###############################

    # ViT & CIFAR-10
    echo "*** Retraining ViT on CIFAR-10"
    python retrain.py \
        --run_id $original_id_vit_cifar10 \
        --cudnn slow

    # ViT & CIFAR-100
    echo "*** Retraining ViT on CIFAR-100"
    python retrain.py \
        --run_id $original_id_vit_cifar100 \
        --cudnn slow

    # ViT & MUFAC
    echo "*** Retraining ViT on MUFAC"
    python retrain.py \
        --run_id $original_id_vit_mufac \
        --cudnn slow

    # ViT & MUCAC
    echo "*** Retraining ViT on MUCAC"
    python retrain.py \
        --run_id $original_id_vit_mucac \
        --cudnn slow

    # ViT & PneumoniaMNIST
    echo "*** Retraining ViT on PneumoniaMNIST"
    python retrain.py \
        --run_id $original_id_vit_pneumoniamnist \
        --cudnn slow

################################################################################################
##################################### U N L E A R N ############################################
################################################################################################
elif [ "$1" = "unlearn" ]; then

    thrsholds=(-1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25)

    # ###############################
    # #### ResNet-18 & CIFAR-10 #####
    # ###############################

    # echo "*** Unlearning ResNet-18 on CIFAR-10"
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
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_fim_kl --epochs 15 --rel_thresh $rel_thresh
    # done

    # ###############################
    # #### ResNet-18 & CIFAR-100 ####
    # ###############################

    # echo "*** Unlearning ResNet-18 on CIFAR-100"
    # echo "=== finetune ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method finetune --epochs 15
    # echo "=== neggrad ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method neggrad --epochs 15
    # echo "=== neggrad_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method neggrad_advanced --epochs 15
    # echo "=== relabel ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method relabel --epochs 15
    # echo "=== relabel_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method relabel_advanced --epochs 15
    # echo "=== boundary ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method boundary --epochs 15
    # echo "=== unsir ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method unsir --epochs 1
    # echo "=== scrub ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method scrub --epochs 15
    # echo "=== ssd ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method ssd --epochs 15
    # echo "=== blindspot ==="
    # python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method blindspot --epochs 15

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_lrp_ce --epochs 15 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_fim_ce --epochs 15 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_lrp_kl --epochs 15 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_fim_kl --epochs 15 --rel_thresh $rel_thresh
    # done

#     ###############################
#     ###### ResNet-18 & MUFAC ######
#     ###############################
#     echo "*** Unlearning ResNet-18 on MUFAC"
#     echo "=== finetune ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method finetune --epochs 3
#     echo "=== neggrad ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method neggrad --epochs 3
#     echo "=== neggrad_advanced ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method neggrad_advanced --epochs 3
#     echo "=== relabel ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method relabel --epochs 3
#     echo "=== relabel_advanced ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method relabel_advanced --epochs 3
#     echo "=== boundary ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method boundary --epochs 3
#     echo "=== unsir ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method unsir --epochs 1
#     echo "=== scrub ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method scrub --epochs 3
#     echo "=== ssd ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method ssd --epochs 3
#     echo "=== blindspot ==="
#     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method blindspot --epochs 3

#     for rel_thresh in "${thrsholds[@]}"; do
#         echo "=== our_lrp_ce ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
#     done

#     for rel_thresh in "${thrsholds[@]}"; do
#         echo "=== our_fim_ce ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
#     done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    # done

    ###############################
    ###### ResNet-18 & MUCAC ######
    ###############################
    echo "*** Unlearning ResNet-18 on MUCAC"
    # echo "=== finetune ==="
    # python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method finetune --epochs 3
    # echo "=== neggrad ==="
    # python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method neggrad --epochs 3
    # echo "=== neggrad_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method neggrad_advanced --epochs 3
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method relabel --epochs 3
    # relabel_advanced is the same as relabel, because it is a binary classification problem
    # echo "=== relabel_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method relabel_advanced --epochs 3
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method boundary --epochs 3
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method unsir --epochs 1
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method scrub --epochs 3
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method ssd --epochs 3
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method blindspot --epochs 3

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    done

    # ################################
    # ## ResNet-18 & PneumoniaMNIST ##
    # ################################
    # echo "*** Unlearning ResNet-18 on PneumoniaMNIST"
    # echo "=== finetune ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method finetune --epochs 3
    # echo "=== neggrad ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method neggrad --epochs 3
    # echo "=== neggrad_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method neggrad_advanced --epochs 3
    # echo "=== relabel ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method relabel --epochs 3
    # # # relabel_advanced is the same as relabel, because it is a binary classification problem
    # # # echo "=== relabel_advanced ==="
    # # # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method relabel_advanced --epochs 3
    # echo "=== boundary ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method boundary --epochs 3
    # echo "=== unsir ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method unsir --epochs 1
    # echo "=== scrub ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method scrub --epochs 3
    # echo "=== ssd ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method ssd --epochs 3
    # echo "=== blindspot ==="
    # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method blindspot --epochs 3

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    # done

    ############################################################################################

    ###############################
    ######## ViT & CIFAR-10 #######
    ###############################
    echo "*** Unlearning ViT on CIFAR-10"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method finetune --epochs 3
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method neggrad --epochs 3
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method neggrad_advanced --epochs 3
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method relabel --epochs 3
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method relabel_advanced --epochs 3
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method boundary --epochs 3
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method unsir --epochs 1
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method scrub --epochs 3
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method ssd --epochs 3
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method blindspot --epochs 3

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    done

    ###############################
    ####### ViT & CIFAR-100 #######
    ###############################

    echo "*** Unlearning ViT on CIFAR-100"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method finetune --epochs 3
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method neggrad --epochs 3
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method neggrad_advanced --epochs 3
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method relabel --epochs 3
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method relabel_advanced --epochs 3
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method boundary --epochs 3
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method unsir --epochs 1
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method scrub --epochs 3
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method ssd --epochs 3
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method blindspot --epochs 3

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_kl ==="
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    done

    ###############################
    ######### ViT & MUFAC #########
    ###############################
    echo "*** Unlearning ViT on MUFAC"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method finetune --epochs 3
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method neggrad --epochs 3
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method neggrad_advanced --epochs 3
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method relabel --epochs 3
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method relabel_advanced --epochs 3
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method boundary --epochs 3
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method unsir --epochs 1
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method scrub --epochs 3
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method ssd --epochs 3
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method blindspot --epochs 3

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    done

    # ###############################
    # ######### ViT & MUCAC #########
    # ###############################
    # echo "*** Unlearning ViT on MUCAC"
    # echo "=== finetune ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method finetune --epochs 3
    # echo "=== neggrad ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method neggrad --epochs 3
    # echo "=== neggrad_advanced ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method neggrad_advanced --epochs 3
    # echo "=== relabel ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method relabel --epochs 3
    # # relabel_advanced is the same as relabel, because it is a binary classification problem
    # # echo "=== relabel_advanced ==="
    # # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method relabel_advanced --epochs 3
    # echo "=== boundary ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method boundary --epochs 3
    # echo "=== unsir ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method unsir --epochs 1
    # echo "=== scrub ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method scrub --epochs 3
    # echo "=== ssd ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method ssd --epochs 3
    # echo "=== blindspot ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method blindspot --epochs 3

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_ce ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_lrp_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    # done

    # for rel_thresh in "${thrsholds[@]}"; do
    #     echo "=== our_fim_kl ==="
    #     echo $rel_thresh
    #     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    # done

    ################################
    ##### ViT & PneumoniaMNIST #####
    ################################
    echo "*** Unlearning ViT on PneumoniaMNIST"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method finetune --epochs 3
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method neggrad --epochs 3
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method neggrad_advanced --epochs 3
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method relabel --epochs 3
    # # relabel_advanced is the same as relabel, because it is a binary classification problem
    # # echo "=== relabel_advanced ==="
    # # python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method relabel_advanced --epochs 3
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method boundary --epochs 3
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method unsir --epochs 1
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method scrub --epochs 3
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method ssd --epochs 3
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method blindspot --epochs 3

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh
    done

    for rel_thresh in "${thrsholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh
    done

else
    echo "Invalid argument: $1 should be 'train', 'retrain' or 'unlearn'"
fi
