#! /bin/bash

# How to reproduce the results of the paper regarding ResNet-18.
# Experiments run on i7-12700K CPU, 32GB RAM, RTX 4080 16GB.
# https://arxiv.org/abs/2109.08203 --> Torch_manual_seed($seed) is all you need! ;). Even it seems magic, this guy made a lot of effort to come up with this number.
# The seeds we used are $seed, 13,12

# Default seed value
default_seed=3407

# Check if an argument is provided
if [ -z "$1" ]; then
  seed=$default_seed
else
  seed=$1
fi

baselines=(finetune neggrad amnesiac bad-teacher scrub ssd unsir)
###########################
# # ResNet-18 on CIFAR-10
###########################
# python train.py --seed $seed --cudnn slow --dataset cifar-10 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false --class_to_forget none
# python retrain.py --registered_model resnet18-cifar-10-$seed-original --is_class_unlearning False
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-$seed-retrained --method our --lr 1e-4 --weight_decay 0.01
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-$seed-retrained --method $baseline
# done

# ###########################
# # # ResNet-18 cat
# ###########################
# python retrain.py --registered_model resnet18-cifar-10-$seed-original --is_class_unlearning True --class_to_forget cat
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-$seed-retrained --method our --lr 1e-4 --weight_decay 0.001
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-cat-$seed-retrained --method $baseline
# done

# ###########################
# # # ResNet-18 horse
# ###########################
# python retrain.py --registered_model resnet18-cifar-10-$seed-original --is_class_unlearning True --class_to_forget horse
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-$seed-retrained --method our --lr 1e-4 --weight_decay 0.001
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-horse-$seed-retrained --method $baseline
# done


# ###########################
# # # ResNet-18 on CIFAR-100
# ###########################
# python train.py --seed $seed --cudnn slow --dataset cifar-100 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false 
# python retrain.py --registered_model resnet18-cifar-100-$seed-original --is_class_unlearning False
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-$seed-retrained --method our --lr 1e-4 --weight_decay 0.01
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-$seed-retrained --method $baseline
# done
 
###########################
# # ResNet-18 rocket
###########################
# python retrain.py --registered_model resnet18-cifar-100-$seed-original --is_class_unlearning True --class_to_forget rocket
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-$seed-retrained --method our --lr 1e-4 --weight_decay 0.001 
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-rocket-$seed-retrained --method $baseline
# done

# ###########################
# # # ResNet-18 beaver
# ###########################
# python retrain.py --registered_model resnet18-cifar-100-$seed-original --is_class_unlearning True --class_to_forget beaver
# python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-$seed-retrained --method our --lr 1e-4 --weight_decay 0.001 
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-beaver-$seed-retrained --method $baseline
# done

###########################
# # ResNet-18 on MUFAC
###########################
python train.py --seed $seed --cudnn slow --dataset mufac --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false --class_to_forget none
python retrain.py --registered_model resnet18-mufac-$seed-original --is_class_unlearning False
# python unlearn.py --epochs 10 --registered_model resnet18-mufac-$seed-retrained --method our --lr 1e-3 --weight_decay 0.01
python unlearn.py --epochs 10 --registered_model resnet18-mufac-$seed-retrained --method our --lr 1e-4 --weight_decay 0.01
python unlearn.py --epochs 10 --registered_model resnet18-mufac-$seed-retrained --method our --lr 1e-5 --weight_decay 0.01 
for baseline in ${baselines[@]}; do
    echo "Running $baseline"
    python unlearn.py --epochs 10 --registered_model resnet18-mufac-$seed-retrained --method $baseline
done

###########################
# ViT on CIFAR-10
###########################
# python train.py --seed $seed --cudnn slow --dataset cifar-10 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10
# python retrain.py --registered_model vit-cifar-10-$seed-original --is_class_unlearning False 
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-$seed-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.01
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 3 --registered_model vit-cifar-10-$seed-retrained --batch_size 32 --method $baseline
# done

###########################
# ViT cat
###########################
# python retrain.py --registered_model vit-cifar-10-$seed-original --is_class_unlearning True --class_to_forget cat
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-$seed-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 3 --registered_model vit-cifar-10-cat-$seed-retrained --batch_size 32 --method $baseline
# done

###########################
# ViT horse
###########################
# python retrain.py --registered_model vit-cifar-10-$seed-original --is_class_unlearning True --class_to_forget horse
# python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-$seed-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 3 --registered_model vit-cifar-10-horse-$seed-retrained --batch_size 32 --method $baseline
# done

###########################
# ViT on CIFAR-100
###########################
# python train.py --seed $seed --cudnn slow --dataset cifar-100 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10
# python retrain.py --registered_model vit-cifar-100-$seed-original --is_class_unlearning False 
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-$seed-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.01
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 3 --registered_model vit-cifar-100-$seed-retrained --batch_size 32 --method $baseline
# done

###########################
# ViT rocket
###########################
# python retrain.py --registered_model vit-cifar-100-$seed-original --is_class_unlearning True --class_to_forget rocket
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-$seed-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 3 --registered_model vit-cifar-100-rocket-$seed-retrained --batch_size 32 --method $baseline
# done

###########################
# ViT beaver
###########################
# python retrain.py --registered_model vit-cifar-100-$seed-original --is_class_unlearning True --class_to_forget beaver
# python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-$seed-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.001
# for baseline in ${baselines[@]}; do
#     echo "Running $baseline"
#     python unlearn.py --epochs 3 --registered_model vit-cifar-100-beaver-$seed-retrained --batch_size 32 --method $baseline
# done


###########################
# ViT on MUFAC
###########################
python train.py --seed $seed --cudnn slow --dataset mufac --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10 --is_class_unlearning False
python retrain.py --registered_model vit-mufac-$seed-original --is_class_unlearning False 
python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 32 --method our --lr 1e-5 --weight_decay 0.01
python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 32 --method our --lr 1e-6 --weight_decay 0.01
python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 32 --method our --lr 1e-7 --weight_decay 0.01
for baseline in ${baselines[@]}; do
    if [ $baseline != "unsir" ]; then
        echo "Running $baseline"
        python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 32 --method $baseline
    fi
done
python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 16 --method unsir 