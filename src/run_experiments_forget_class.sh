#!/bin/bash
class_to_forget="$3"

if [ "$class_to_forget" != "rocket" ] && [ "$class_to_forget" != "beaver" ]; then
    echo "Invalid class to forget!"
    echo "$3 should be either 'rocket' or 'beaver'"
    echo "Exiting..."
    exit
fi

if [ "$2" = 1 ]; then
    seed=3407
    # Original ID Rocket
    original_id_resnet_rocket="e22bccf864ba441f9c4a3ddb5140b71b"
    original_id_vit_rocket="6638937d9c6c464383a0a4a0cb18e572"

    # Original ID Beaver
    original_id_resnet_beaver="b12fd6e1aa754bf38fdf2571568c378d"
    original_id_vit_beaver="629d826c2add45ed822d272db41c60c7"

    # Retrained ID Rocket
    retrained_id_resnet_rocket="1f8492ae39354050b5b6b302b025eab6"
    retrained_id_vit_rocket="fe4376708d994c82b875b6472e3171a7"

    # Retrained ID Beaver
    retrained_id_resnet_beaver="65bee0844ca741a4a98cacb16b495ebc"
    retrained_id_vit_beaver="7b337654d24742339ee738e1d93d22a2"
elif [ "$2" = 2 ]; then
    seed=1703
    # Original ID Rocket
    original_id_resnet_rocket="6a0e7e70cc6f43098ba276df0c9e52f0"
    original_id_vit_rocket="c914b1ade68b419daa7bd3bfcf2ba36a"

    # Original ID Beaver
    original_id_resnet_beaver="a6b5f31cc212473e8e081efba560748e"
    original_id_vit_beaver="50f71937d21e4cd4a803b1afa44d6561"

    # Retrained ID Rocket
    retrained_id_resnet_rocket="3b3851168ffe402d9e0e1479afbead95"
    retrained_id_vit_rocket="455a2cd1b26547dea3bb51684a2cd253"

    # Retrained ID Beaver
    retrained_id_resnet_beaver="f4a1b588321b443ba52f6ee053555d46"
    retrained_id_vit_beaver="ab98cd54fca044d9a41ea5d53976357b"

elif [ "$2" = 3 ]; then
    seed=851
    # Original ID Rocket
    original_id_resnet_rocket="e100dc6864c745aa994dac07ec81ed0b"
    original_id_vit_rocket="670095e07aa24a0090d3b2a018a050ce"

    # Original ID Beaver
    original_id_resnet_beaver="f25ee7038da14450b6397dcd289a68b1"
    original_id_vit_beaver="d56d8ccf53c84113a4ccba2a6d86bfcc"

    # Retrained ID Rocket
    retrained_id_resnet_rocket="27929698000b4863a433ce73a572a50c"
    retrained_id_vit_rocket="3052bf6fd5d841ac92d721081f7eecf4"

    # Retrained ID Beaver
    retrained_id_resnet_beaver="d1d3eccd44814c32aecab260aff0e3c8"
    retrained_id_vit_beaver="a442940c4cc34ddcb88620c10d761c85"

elif [ "$2" = 4 ]; then
    seed=425
    # Original ID Rocket
    original_id_resnet_rocket="efaca81004364fbda853bdcd7496d81d"
    original_id_vit_rocket="0cd76c4c9bf54d928b1dc78aacf04721"

    # Original ID Beaver
    original_id_resnet_beaver="c3c2e7182daf45bdb52eac3af6d82e71"
    original_id_vit_beaver="10f7eed1fb1947058e860dd082b9b318"

    # Retrained ID Rocket
    retrained_id_resnet_rocket="c2ba5de81c8d480b96fd4dc7db4779d9"
    retrained_id_vit_rocket="3bc808c7ea454fbf8c6b9aa329a3af48"

    # Retrained ID Beaver
    retrained_id_resnet_beaver="4aece3a397a740c1a1af51bf4942401c"
    retrained_id_vit_beaver="4a6f566f1076412cb56fda0ec1384187"

elif [ "$2" = 5 ]; then
    seed=212
    # Original ID Rocket
    original_id_resnet_rocket="a9fdafbf88bc4410bd63706972dd8f05"
    original_id_vit_rocket="38b25ae0b9694f28a577076f7e9d8a55"

    # Original ID Beaver
    original_id_resnet_beaver="404a860ccd5346f89d447e070d985fd1"
    original_id_vit_beaver="2d5d247853cf4e669af0e543831e3c9d"

    # Retrained ID Rocket
    retrained_id_resnet_rocket="02982b24f04440849f4c001060ce6969"
    retrained_id_vit_rocket="95018175c32a4553b49587824067312d"

    # Retrained ID Beaver
    retrained_id_resnet_beaver="3ca3841f0a0b4823b352667a83b6088f"
    retrained_id_vit_beaver="87dd025e141e4e488e1794b29614c800"

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
    if [ "$3" = "rocket" ]; then
        # ResNet-18 & CIFAR-100
        echo "Retraining ResNet-18 on CIFAR-100"
        python retrain.py \
            --run_id $original_id_resnet_rocket \
            --cudnn slow

        # ViT & CIFAR-100
        echo "Retraining ViT on CIFAR-100"
        python retrain.py \
            --run_id $original_id_vit_rocket \
            --cudnn slow

    elif [ "$3" = "beaver" ]; then
        # # ResNet-18 & CIFAR-100
        # echo "Retraining ResNet-18 on CIFAR-100"
        # python retrain.py \
        #     --run_id $original_id_resnet_beaver \
        #     --cudnn slow

        # ViT & CIFAR-100
        echo "Retraining ViT on CIFAR-100"
        python retrain.py \
            --run_id $original_id_vit_beaver \
            --cudnn slow \

    else
        echo "Invalid argument: $3 should be 'rocket' or 'beaver'"
    fi

elif [ "$1" = "unlearn" ]; then
    if [ "$3" = "rocket" ]; then
        # echo "*** Unlearning ResNet-18 on CIFAR-100"
        # echo "=== finetune ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method finetune --epochs 15 --batch_size 128
        # echo "=== neggrad ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method neggrad --epochs 15 --batch_size 128
        # echo "=== neggrad_advanced ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method neggrad_advanced --epochs 15 --batch_size 128
        # echo "=== relabel ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method relabel --epochs 15 --batch_size 128
        # echo "=== relabel_advanced ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method relabel_advanced --epochs 15 --batch_size 128
        # echo "=== boundary ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method boundary --epochs 15 --batch_size 128
        # echo "=== unsir ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method unsir --epochs 15 --batch_size 128
        # echo "=== scrub ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method scrub --epochs 15 --batch_size 128
        # echo "=== ssd ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method ssd --epochs 15 --batch_size 128
        # echo "=== blindspot ==="
        # python unlearn.py --run_id $retrained_id_resnet_rocket --cudnn slow --mu_method blindspot --epochs 15 --batch_size 128


        echo "*** Unlearning ViT on CIFAR-100"
        echo "=== finetune ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
        echo "=== neggrad ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
        echo "=== neggrad_advanced ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
        echo "=== relabel ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
        echo "=== relabel_advanced ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
        echo "=== boundary ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
        echo "=== unsir ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
        echo "=== scrub ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
        echo "=== ssd ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
        echo "=== blindspot ==="
        python unlearn.py --run_id $retrained_id_vit_rocket --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16


    elif [ "$3" = "beaver" ]; then
        echo "*** Unlearning ResNet-18 on CIFAR-100"
        echo "=== finetune ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method finetune --epochs 15 --batch_size 128
        echo "=== neggrad ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method neggrad --epochs 15 --batch_size 128
        echo "=== neggrad_advanced ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method neggrad_advanced --epochs 15 --batch_size 128
        echo "=== relabel ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method relabel --epochs 15 --batch_size 128
        echo "=== relabel_advanced ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method relabel_advanced --epochs 15 --batch_size 128
        echo "=== boundary ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method boundary --epochs 15 --batch_size 128
        echo "=== unsir ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method unsir --epochs 15 --batch_size 128
        echo "=== scrub ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method scrub --epochs 15 --batch_size 128
        echo "=== ssd ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method ssd --epochs 15 --batch_size 128
        echo "=== blindspot ==="
        python unlearn.py --run_id $retrained_id_resnet_beaver --cudnn slow --mu_method blindspot --epochs 15 --batch_size 128


        echo "*** Unlearning ViT on CIFAR-100"
        echo "=== finetune ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
        echo "=== neggrad ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
        echo "=== neggrad_advanced ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
        echo "=== relabel ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
        echo "=== relabel_advanced ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
        echo "=== boundary ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
        echo "=== unsir ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
        echo "=== scrub ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
        echo "=== ssd ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
        echo "=== blindspot ==="
        python unlearn.py --run_id $retrained_id_vit_beaver --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16

    else
        echo "Invalid argument: $3 should be 'rocket' or 'beaver'"
    fi

else
    echo "Invalid argument: $1 should be 'train', 'retrain' or 'unlearn'"

fi
