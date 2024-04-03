
#!/bin/bash
if [ "$2" = 1 ]; then
    seed=3407

    # ORIGINAL MODEL
    original_id_resnet_cifar10="65499bef2a244808956ff55c5a85310c"
    original_id_resnet_cifar100="c10f2e7abb0d4f23949df8cd8f71beee"
    original_id_resnet_mufac="f9265b5ed07545228d6f2c770a03cbdd"
    original_id_resnet_mucac="059fdaca8a3a46178f03f28881998bc8"
    original_id_resnet_pneumoniamnist="729fc5f5355e40399d3cd57a44c33b5d"

    original_id_vit_cifar10="4df80ced6a614303baba6af60f1a4305"
    original_id_vit_cifar100="6681f2b3c9ef409e9194692898e7be6d"
    original_id_vit_mufac="3107d92e9b244023896ea01dffe05d10"
    original_id_vit_mucac="f5c4ec3cc6b14ecba1520b0de8d6ef8c"
    original_id_vit_pneumoniamnist="8595862f5e7d449396aad9e6dd695814"

    # RETRAINED MODEL
    retrained_id_resnet_cifar10="86d35af47edf4a568f261a3bdb7e0bc1"
    retrained_id_resnet_cifar100="11b414abec0045f9aa0547f9b2d90ae9"
    retrained_id_resnet_mufac="54970d1e563942b586e86dc162bf8e66"
    retrained_id_resnet_mucac="008ec2c6bbc74c62a846817e1e8ad886"
    retrained_id_resnet_pneumoniamnist="057ee25c09d14aca8d6c4cce886667b6"

    retrained_id_vit_cifar10="63af8e821c15432ab356349b5f37e6ee"
    retrained_id_vit_cifar100="16c33aadf7d2498ca6a32a8418c24841"
    retrained_id_vit_mufac="251567d1abc0432fa67ffbaccca949d0"
    retrained_id_vit_mucac="ca234c336fdd4cb3bb2c8e57c14b1999"
    retrained_id_vit_pneumoniamnist="29a3415d1af54fa8b50129ba4a323152"

elif [ "$2" = 2 ]; then
    seed=1703

    original_id_resnet_cifar10="437c57ab3b444ae081657bcfccbcd0c3"
    original_id_resnet_cifar100="2aa6ebe9f6424863aaa2d56d961b3ab4"
    original_id_resnet_mufac="b5963418198b41cfad1c15e24fc184f6"
    original_id_resnet_mucac="855234be765e4c9eab3919eb8ab5ce67"
    original_id_resnet_pneumoniamnist="d62e8c2527d94c51ad69164a6c4977de"

    original_id_vit_cifar10="32f2224ae5654e678275acc8d1c4525a"
    original_id_vit_cifar100="838fbca787c8472fabb898e3efe28909"
    original_id_vit_mufac="aa9deb9d921b4230ac58a2b8b8e43e95"
    original_id_vit_mucac="f633887ec1d9458b8967449bb552da87"
    original_id_vit_pneumoniamnist="b1e21749543e497dbb0f43596894aa41"

    retrained_id_resnet_cifar10="8dd8fb624b5a4fabb1f2782ef51d8c56"
    retrained_id_resnet_cifar100="a8972f256d714c88bbae209e4d8fd395"
    retrained_id_resnet_mufac="ff144aecb35c48e7ba47abfef832654b"
    retrained_id_resnet_mucac="d831371ca6054c6487af92da473b41e5"
    retrained_id_resnet_pneumoniamnist="4f4f177c478c4d678655a2d00e3fc6ad"

    retrained_id_vit_cifar10="852a888a180b4485b52287d7a518a114"
    retrained_id_vit_cifar100="bac990a7540b49b7af3737601db881ef"
    retrained_id_vit_mufac="2c7779c7799f4609a6e99b274db65824"
    retrained_id_vit_mucac="c62c9bafbc854e76bf469ae6dc9c204f"
    retrained_id_vit_pneumoniamnist="9ee3c5374050486db45731b569b8c728"
elif [ "$2" = 3 ]; then
    seed=851

    original_id_resnet_cifar10="f7215bc5a0844eee9bd3412b521d458c"
    original_id_resnet_cifar100="ac1414570d4f4461a8cddf3e0d87c849"
    original_id_resnet_mufac="51a853fc291646e3a4631d6e37deacab"
    original_id_resnet_mucac="3cb70a1a8224472b895e2fca061d7eb8"
    original_id_resnet_pneumoniamnist="aafb8d9daea841858f74c39a10709dec"

    original_id_vit_cifar10="c24d6ba8438740f680c60967543fb449"
    original_id_vit_cifar100="10de5f391cf54247ab9abdd040760c1e"
    original_id_vit_mufac="47ca4985cddf4d298eea1bd02c06af18"
    original_id_vit_mucac="f84387a1773f4492b4e4f6839440768d"
    original_id_vit_pneumoniamnist="7ef4d5aecc144df6a9b5cd61c2c6d568"

    retrained_id_resnet_cifar10="bdb4893da46b4575866076d45bb043db"
    retrained_id_resnet_cifar100="7e4d4088c7cd4a56a1096c95b68bbdf6"
    retrained_id_resnet_mufac="f0989baac8514550b9bfe85869e95e64"
    retrained_id_resnet_mucac="9efe0758e9294566a696b15baf66f3f0"
    retrained_id_resnet_pneumoniamnist="c7c79e3fbc484f3c8eb1cefeaeaae993"

    retrained_id_vit_cifar10="0b841d78456048959a0fe121488f5542"
    retrained_id_vit_cifar100="8232266a606a4fa19fb2f5d883c2776b"
    retrained_id_vit_mufac="c858860fd4474ddeb9f87667f32c437b"
    retrained_id_vit_mucac="97909b8bdf35444e8a520374dc3796a6"
    retrained_id_vit_pneumoniamnist="203914933c80470a9ce8915408f2ba1b"
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
        --seed $seed \
        --cudnn slow \
        --dataset cifar-10 \
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
        --patience 50

    # ResNet-18 & CIFAR-100
    echo "*** Training ResNet-18 on CIFAR-100"
    python train.py \
        --seed $seed \
        --cudnn slow \
        --dataset cifar-100 \
        --model resnet18 \
        --batch_size 512\
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
        --seed $seed \
        --cudnn slow \
        --dataset mufac \
        --model resnet18 \
        --batch_size 512\
        --epochs 30 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 10

    # ResNet-18 & MUCAC
    echo "Training ResNet-18 on MUCAC"
    python train.py \
        --seed $seed \
        --cudnn slow \
        --dataset mucac \
        --model resnet18 \
        --batch_size 512\
        --epochs 30 \
        --loss bce_with_logits \
        --optimizer sgd \
        --lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 10

    # ResNet-18 & PneumoniaMNIST
    echo "*** Training ResNet-18 on PneumoniaMNIST"
    python train.py \
        --seed $seed \
        --cudnn slow \
        --dataset pneumoniamnist \
        --model resnet18 \
        --batch_size 512\
        --epochs 30 \
        --loss weighted_cross_entropy \
        --optimizer sgd \
        --lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --is_lr_scheduler False \
        --is_early_stop True \
        --patience 10

    ###############################
    ############# ViT #############
    ###############################

    # ViT & CIFAR-10
    echo "*** Training ViT on CIFAR-10"
    python train.py \
        --seed $seed \
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
        --patience 10

    # ViT & MUFAC
    echo "*** Training ViT on MUFAC"
    python train.py \
        --seed $seed \
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
        --is_early_stop True \
        --patience 10

    # ViT & MUCAC
    echo "*** Training ViT on MUCAC"
    python train.py \
        --seed $seed \
        --cudnn slow \
        --dataset mucac \
        --model vit \
        --batch_size 64 \
        --epochs 30 \
        --loss bce_with_logits \
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
        --seed $seed \
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

    thresholds=(-1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1)

    ###############################
    #### ResNet-18 & CIFAR-10 #####
    ###############################

    echo "*** Unlearning ResNet-18 on CIFAR-10"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method finetune --epochs 15 --batch_size 128
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method neggrad --epochs 15 --batch_size 128
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method neggrad_advanced --epochs 15 --batch_size 128
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method relabel --epochs 15 --batch_size 128
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method relabel_advanced --epochs 15 --batch_size 128
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method boundary --epochs 15 --batch_size 128
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method unsir --epochs 15 --batch_size 128
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method scrub --epochs 15 --batch_size 128
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method ssd --epochs 15 --batch_size 128
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method blindspot --epochs 15 --batch_size 128

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_lrp_ce --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_fim_ce --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_lrp_kl --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar10 --cudnn slow --mu_method our_fim_kl --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    ###############################
    #### ResNet-18 & CIFAR-100 ####
    ###############################

    echo "*** Unlearning ResNet-18 on CIFAR-100"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method finetune --epochs 15 --batch_size 128
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method neggrad --epochs 15 --batch_size 128
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method neggrad_advanced --epochs 15 --batch_size 128
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method relabel --epochs 15 --batch_size 128
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method relabel_advanced --epochs 15 --batch_size 128
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method boundary --epochs 15 --batch_size 128
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method unsir --epochs 15 --batch_size 128
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method scrub --epochs 15 --batch_size 128
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method ssd --epochs 15 --batch_size 128
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method blindspot --epochs 15 --batch_size 128

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_lrp_ce --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_fim_ce --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_lrp_kl --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_cifar100 --cudnn slow --mu_method our_fim_kl --epochs 15 --rel_thresh $rel_thresh --batch_size 128
    done

    ###############################
    ###### ResNet-18 & MUFAC ######
    ###############################
    echo "*** Unlearning ResNet-18 on MUFAC"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method finetune --epochs 3 --batch_size 128
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method neggrad --epochs 3 --batch_size 128
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 128
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method relabel --epochs 3 --batch_size 128
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 128
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method boundary --epochs 3 --batch_size 128
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method unsir --epochs 3 --batch_size 128
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method scrub --epochs 3 --batch_size 128
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method ssd --epochs 3 --batch_size 128
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method blindspot --epochs 3 --batch_size 128

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mufac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    ###############################
    ###### ResNet-18 & MUCAC ######
    ###############################
    echo "*** Unlearning ResNet-18 on MUCAC"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method finetune --epochs 3 --batch_size 128
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method neggrad --epochs 3 --batch_size 128
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 128
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method relabel --epochs 3 --batch_size 128
    # relabel_advanced is the same as relabel, because it is a binary classification problem
    # echo "=== relabel_advanced ==="
    # python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 128
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method boundary --epochs 3 --batch_size 128
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method unsir --epochs 3 --batch_size 128
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method scrub --epochs 3 --batch_size 128
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method ssd --epochs 3 --batch_size 128
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method blindspot --epochs 3 --batch_size 128

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_mucac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    ################################
    ## ResNet-18 & PneumoniaMNIST ##
    ################################
    echo "*** Unlearning ResNet-18 on PneumoniaMNIST"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method finetune --epochs 3 --batch_size 128
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method neggrad --epochs 3 --batch_size 128
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 128
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method relabel --epochs 3 --batch_size 128
    # # relabel_advanced is the same as relabel, because it is a binary classification problem
    # # echo "=== relabel_advanced ==="
    # # python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 128
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method boundary --epochs 3 --batch_size 128
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method unsir --epochs 3 --batch_size 128
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method scrub --epochs 3 --batch_size 128
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method ssd --epochs 3 --batch_size 128
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method blindspot --epochs 3 --batch_size 128

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_resnet_pneumoniamnist --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 128
    done

    ############################################################################################

    ###############################
    ######## ViT & CIFAR-10 #######
    ###############################
    echo "*** Unlearning ViT on CIFAR-10"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
    sleep 100
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
    sleep 100
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
    sleep 100
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
    sleep 100
    # timely expensive: 16h and no results yet
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
    sleep 100
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
    sleep 100
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
    sleep 100
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16
    sleep 100

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar10 --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
        sleep 100
    done

    ###############################
    ####### ViT & CIFAR-100 #######
    ###############################

    echo "*** Unlearning ViT on CIFAR-100"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        python unlearn.py --run_id $retrained_id_vit_cifar100 --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    ###############################
    ######### ViT & MUFAC #########
    ###############################
    echo "*** Unlearning ViT on MUFAC"
    echo "=== finetune ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
    echo "=== neggrad ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
    echo "=== neggrad_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
    echo "=== relabel ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
    echo "=== relabel_advanced ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
    echo "=== boundary ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
    echo "=== unsir ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
    echo "=== scrub ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
    echo "=== ssd ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
    echo "=== blindspot ==="
    python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_ce ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_lrp_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    for rel_thresh in "${thresholds[@]}"; do
        echo "=== our_fim_kl ==="
        echo $rel_thresh
        python unlearn.py --run_id $retrained_id_vit_mufac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
    done

    # ###############################
    # ######### ViT & MUCAC #########
    # ###############################
    # echo "*** Unlearning ViT on MUCAC"
    # echo "=== finetune ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
    # echo "=== neggrad ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
    # echo "=== neggrad_advanced ==="
    # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
#     echo "=== relabel ==="
#     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
#     # ## relabel_advanced is the same as relabel, because it is a binary classification problem
#     # # echo "=== relabel_advanced ==="
#     # # python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
#     echo "=== boundary ==="
#     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
#     echo "=== unsir ==="
#     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
#     echo "=== scrub ==="
#     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
#     echo "=== ssd ==="
#     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
#     echo "=== blindspot ==="
#     python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_lrp_ce ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_fim_ce ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_lrp_kl ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_fim_kl ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_mucac --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

#     ################################
#     ##### ViT & PneumoniaMNIST #####
#     ################################
#     echo "*** Unlearning ViT on PneumoniaMNIST"
#     echo "=== finetune ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method finetune --epochs 3 --batch_size 16
#     echo "=== neggrad ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method neggrad --epochs 3 --batch_size 16
#     echo "=== neggrad_advanced ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method neggrad_advanced --epochs 3 --batch_size 16
#     echo "=== relabel ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method relabel --epochs 3 --batch_size 16
#     # # relabel_advanced is the same as relabel, because it is a binary classification problem
#     # # echo "=== relabel_advanced ==="
#     # # python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method relabel_advanced --epochs 3 --batch_size 16
#     echo "=== boundary ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method boundary --epochs 3 --batch_size 16
#     echo "=== unsir ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method unsir --epochs 3 --batch_size 16
#     echo "=== scrub ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method scrub --epochs 3 --batch_size 16
#     echo "=== ssd ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method ssd --epochs 3 --batch_size 16
#     echo "=== blindspot ==="
#     python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method blindspot --epochs 3 --batch_size 16

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_lrp_ce ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_lrp_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_fim_ce ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_fim_ce --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_lrp_kl ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_lrp_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

#     for rel_thresh in "${thresholds[@]}"; do
#         echo "=== our_fim_kl ==="
#         echo $rel_thresh
#         python unlearn.py --run_id $retrained_id_vit_pneumoniamnist --cudnn slow --mu_method our_fim_kl --epochs 3 --rel_thresh $rel_thresh --batch_size 16
#     done

# else
#     echo "Invalid argument: $1 should be 'train', 'retrain' or 'unlearn'"
fi
