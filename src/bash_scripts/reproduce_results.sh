# /bin/bash

# How to reproduce the results of the paper regarding ResNet-18.
# Experiments run on i7-12700K CPU, 32GB RAM, RTX 4080 16GB.
# Experiments on ImageNet were run on A6000 48GB.
# The seeds we used are 3407, 13,12
# https://arxiv.org/abs/2109.08203 --> Torch_manual_seed(3407) is all you need! 

cd ..
seeds=(3407 13 12)
baselines=(finetune neggrad relabel badT scrub ssd unsir)

# ###############################################
# ResNet-18 on CIFAR-10 
# ###############################################
for seed in ${seeds[@]}; do
  python train.py --seed $seed --dataset cifar-10 --model resnet18 --batch_size 512 --epochs 150 --lr 0.1 --is_lr_scheduler True --is_early_stop True 
  python retrain.py --registered_model resnet18-cifar-10-$seed-original --is_class_unlearning False
  python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-$seed-retrained --method our --alpha 2 
  for baseline in ${baselines[@]}; do
      echo "Running $baseline"
      python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-$seed-retrained --method $baseline
  done
done

# ###############################################
# ResNet-18 on CIFAR-100 
# ###############################################
for seed in ${seeds[@]}; do
  python train.py --seed $seed --dataset cifar-100 --model resnet18 --batch_size 512 --epochs 150 --lr 0.1 --is_lr_scheduler True --is_early_stop True 
  python retrain.py --registered_model resnet18-cifar-100-$seed-original --is_class_unlearning False
  python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-$seed-retrained --method our --alpha 2 
  for baseline in ${baselines[@]}; do
      echo "Running $baseline"
      python unlearn.py --epochs 10 --registered_model resnet18-cifar-100-$seed-retrained --method $baseline
  done
done
 
# #############################################
# ResNet-18 on MUFAC
# #############################################
for seed in ${seeds[@]}; do
  python train.py --seed $seed --dataset mufac --model resnet18 --batch_size 512 --epochs 150 --lr 0.1  --is_lr_scheduler True --is_early_stop True
  python retrain.py --registered_model resnet18-mufac-$seed-original --is_class_unlearning False
  python unlearn.py --epochs 10 --registered_model resnet18-mufac-$seed-retrained --method our --alpha 2
  for baseline in ${baselines[@]}; do
      echo "Running $baseline"
      if [ $baseline == "unsir" ]; then
        python unlearn.py --epochs 10 --registered_model resnet18-mufac-$seed-retrained --method $baseline --batch_size 64 
      else
        python unlearn.py --epochs 10 --registered_model resnet18-mufac-$seed-retrained --method $baseline
      fi
  done
done


###############################################
# ViT on CIFAR-10 
###############################################
for seed in ${seeds[@]}; do
  python train.py --seed $seed --dataset cifar-10 --model vit --batch_size 64 --epochs 30 --is_early_stop True --patience 10
  python retrain.py --registered_model vit-cifar-10-$seed-original --is_class_unlearning False 
  python unlearn.py --epochs 3 --registered_model vit-cifar-10-$seed-retrained --batch_size 32 --method our --lr 1e-6 --alpha 2 
  for baseline in ${baselines[@]}; do
      echo "Running $baseline"
      if [ $baseline == "unsir" ]; then
        python unlearn.py --epochs 3 --registered_model vit-cifar-10-$seed-retrained --batch_size 16 --method $baseline 
      else
        python unlearn.py --epochs 3 --registered_model vit-cifar-10-$seed-retrained --batch_size 32 --method $baseline
      fi
  done
done


# #############################################
# ViT on CIFAR-100 
# #############################################
for seed in ${seeds[@]}; do
  python train.py --seed $seed --dataset cifar-100 --model vit --batch_size 64 --epochs 30  --is_early_stop True --patience 10
  python retrain.py --registered_model vit-cifar-100-$seed-original --is_class_unlearning False 
  python unlearn.py --epochs 3 --registered_model vit-cifar-100-$seed-retrained --batch_size 32 --method our --lr 1e-6 --alpha 2 
  for baseline in ${baselines[@]}; do
      echo "Running $baseline"
      if [ $baseline == "unsir" ]; then
        python unlearn.py --epochs 3 --registered_model vit-cifar-100-$seed-retrained --batch_size 16 --method $baseline
      else
        python unlearn.py --epochs 3 --registered_model vit-cifar-100-$seed-retrained --batch_size 32 --method $baseline
      fi
  done
done


# #############################################
# ViT on MUFAC 
# #############################################
for seed in ${seeds[@]}; do
  python train.py --seed $seed --dataset mufac --model vit --batch_size 64 --epochs 30 --is_early_stop True --patience 10
  python retrain.py --registered_model vit-mufac-$seed-original --is_class_unlearning False 
  python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 32 --method our --lr 1e-6 --seed $seed --alpha 2 
  for baseline in ${baselines[@]}; do
      echo "Running $baseline"
      if [ $baseline == "unsir" ]; then
        python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 16 --method $baseline
      else
        python unlearn.py --epochs 3 --registered_model vit-mufac-$seed-retrained --batch_size 32 --method $baseline
      fi	
  done
done

# #############################################
# ViT on ImageNet1k
# #############################################
baselines=(finetune neggrad relabel badT scrub unsir) # ssd is not here
for seed in ${seeds[@]}; do
  python original_imagenet.py --seed $seed --dataset imagenet --model vit --batch_size 4096 --epochs 0 
  python unlearn_imagenet.py --epochs 3 --registered_model vit-imagenet-$seed-original --batch_size 256 --method our --lr 1e-6 --alpha 2
  for baseline in ${baselines[@]}; do
      echo "Running $baseline"
      if [ $baseline == "unsir" ]; then
        python unlearn_imagenet.py --epochs 3 --registered_model vit-imagenet-$seed-original --batch_size 64 --method $baseline
      else
        python unlearn_imagenet.py --epochs 3 --registered_model vit-imagenet-$seed-original --batch_size 256 --method $baseline
      fi
  done
done

# Comment: UNSIR has large space complexity, so when the image size is big
# or the params of the model are large, CUDA out of memory error may occur.
# In that case, you can decrease the batch size of UNSIR.
