#!/bin/bash
class_to_forget="rocket"
if [ "$2" = 1 ]; then
    seed=3407
    # Original ID
    original_id_resnet_rocket="e22bccf864ba441f9c4a3ddb5140b71b"
    original_id_vit_rocket="6638937d9c6c464383a0a4a0cb18e572"
    # Retrained ID
    retrained_id_resnet_rocket="1f8492ae39354050b5b6b302b025eab6"
    retrained_id_vit_rocket="fe4376708d994c82b875b6472e3171a7"
elif [ "$2" = 2 ]; then
    seed=1703
    # Original ID
    original_id_resnet_rocket="6a0e7e70cc6f43098ba276df0c9e52f0"
    original_id_vit_rocket="c914b1ade68b419daa7bd3bfcf2ba36a"
    # Retrained ID
    retrained_id_resnet_rocket="3b3851168ffe402d9e0e1479afbead95"
    retrained_id_vit_rocket="455a2cd1b26547dea3bb51684a2cd253"
elif [ "$2" = 3 ]; then
    seed=851
    # Original ID
    original_id_resnet_rocket="e100dc6864c745aa994dac07ec81ed0b"
    original_id_vit_rocket="670095e07aa24a0090d3b2a018a050ce"
    # Retrained ID
    retrained_id_resnet_rocket="27929698000b4863a433ce73a572a50c"
    retrained_id_vit_rocket="3052bf6fd5d841ac92d721081f7eecf4"
elif [ "$2" = 4 ]; then
    seed=425
    # Original ID
    original_id_resnet_rocket=""
    original_id_vit_rocket=""
    # Retrained ID
    retrained_id_resnet_rocket=""
    retrained_id_vit_rocket=""
elif [ "$2" = 5 ]; then
    seed=212
    # Original ID
    original_id_resnet_rocket=""
    original_id_vit_rocket=""
    # Retrained ID
    retrained_id_resnet_rocket=""
    retrained_id_vit_rocket=""
else
    echo "Invalid arguments!"
    echo "$1 should be one of the following: train, retrain, unlearn"
    echo "$2 should be one of the following: 1, 2, 3 (for seeds 3407, 1703, 851 respectively)"
    echo "Example: ./run_experiments_forget_instances.sh train 1"
    echo "Exiting..."
    exit
fi


#################################
########## T R A I N ############
#################################

if [ "$1" = "train" ]; then
    # ResNet-18 & CIFAR-100
    echo "Training ResNet-18 on CIFAR-100"
    python train.py \
        --seed $seed \
        --cudnn slow \
        --dataset cifar-100 \
        --model resnet18 \
        --batch_size 512 \
        --epochs 150 \
        --loss cross_entropy \
        --optimizer sgd \
        --lr 0.1 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler True \
        --warmup_epochs 30 \
        --is_early_stop True \
        --patience 50 \
        --is_class_unlearning True \
        --class_to_forget $class_to_forget

    # ViT & CIFAR-100
    echo "Training ViT on CIFAR-100"
    python train.py \
        --seed $seed \
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
        --patience 10 \
        --is_class_unlearning True \
        --class_to_forget $class_to_forget
elif [ "$1" = "retrain" ]; then
    # ResNet-18 & CIFAR-100
    echo "Retraining ResNet-18 on CIFAR-100"
    python retrain.py \
    --run_id $original_id_resnet_rocket \
    --cudnn slow \

    # ViT & CIFAR-100
    echo "Retraining ViT on CIFAR-100"
    python retrain.py \
    --run_id $original_id_vit_rocket \
    --cudnn slow \

elif [ "$1" = "unlearn" ]; then
    thresholds=(-1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1)

    echo "*** Unlearning ResNet-18 on CIFAR-100"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method finetune --epochs 15 --batch_size 128
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method neggrad --epochs 15 --batch_size 128
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method neggrad_advanced --epochs 15 --batch_size 128
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method relabel --epochs 15 --batch_size 128
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method relabel_advanced --epochs 15 --batch_size 128
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method boundary --epochs 15 --batch_size 128
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method unsir --epochs 15 --batch_size 128
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method scrub --epochs 15 --batch_size 128
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method ssd --epochs 15 --batch_size 128
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method blindspot --epochs 15 --batch_size 128

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method our_lrp_ce --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method our_fim_ce --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method our_lrp_kl --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method our_fim_kl --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    echo "*** Unlearning ViT on CIFAR-100"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
    sleep 100
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
    sleep 100
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
    sleep 100
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
    sleep 100
    # timely expensive: 16h and no results yet
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
    sleep 100
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
    sleep 100
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
    sleep 100
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16
    sleep 100

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

else
    echo "Invalid argument: $1 should be 'train', 'retrain' or 'unlearn'"
fi

