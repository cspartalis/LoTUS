#! /bin/bash

###########################
# # ResNet-18 on CIFAR-10
###########################
# python train.py --seed 3407 --cudnn slow --dataset cifar-10 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false --class_to_forget none
# python retrain.py --cudnn slow --registered_model resnet18-cifar-10-3407-original
python unlearn.py --method finetune --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method amnesiac --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method boundary --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method unsir --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method scrub --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method ssd --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method bad-teacher --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method neggrad --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
python unlearn.py --method neggrad_advanced --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 
# python unlearn.py --method our --epochs 10 --batch_size 128 --subset_size 0.3 --is_zapping True --cudnn slow --registered_model resnet18-cifar-10-3407-retrained 

# ###########################
# # # ResNet-18 cat
# ###########################
# python retrain.py --registered_model resnet18-cifar-10-3407-original --is_class_unlearning True --class_to_forget cat
python unlearn.py --method finetune --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained
python unlearn.py --method amnesiac --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained
python unlearn.py --method boundary --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained
python unlearn.py --method unsir --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained 
python unlearn.py --method scrub --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained 
python unlearn.py --method ssd --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained   
python unlearn.py --method bad-teacher --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained 
python unlearn.py --method neggrad --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained 
python unlearn.py --method neggrad_advanced --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-cat-3407-retrained 

# ###########################
# # # ResNet-18 automobile
# ###########################
# python retrain.py --registered_model resnet18-cifar-10-3407-original --is_class_unlearning True --class_to_forget automobile
# python unlearn.py --method finetune --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained
# python unlearn.py --method amnesiac --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained
# python unlearn.py --method boundary --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained
# python unlearn.py --method unsir --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained 
# python unlearn.py --method scrub --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained 
# python unlearn.py --method ssd --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained   
# python unlearn.py --method bad-teacher --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained 
# python unlearn.py --method neggrad --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained 
# python unlearn.py --method neggrad_advanced --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-10-automobile-3407-retrained 

# ###########################
# # # ResNet-18 on CIFAR-100
# ###########################
# # python train.py --seed 3407 --cudnn slow --dataset cifar-100 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false 
# python retrain.py --registered_model resnet18-cifar-100-3407-original --is_class_unlearning False
python unlearn.py --method finetune --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method amnesiac --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method boundary --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method unsir --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method scrub --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method ssd --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method bad-teacher --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method neggrad --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
python unlearn.py --method neggrad_advanced --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 
# python unlearn.py --method our --epochs 5 --batch_size 128 --subset_size 0.3 --is_zapping True --cudnn slow --registered_model resnet18-cifar-100-3407-retrained 

# ###########################
# # # ResNet-18 rocket
# ###########################
# python retrain.py --registered_model resnet18-cifar-100-3407-original --is_class_unlearning True --class_to_forget rocket
python unlearn.py --method finetune --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method amnesiac --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method boundary --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method unsir --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method scrub --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method ssd --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method bad-teacher --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method neggrad --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 
python unlearn.py --method neggrad_advanced --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-rocket-3407-retrained 

# ###########################
# # # ResNet-18 beaver
# ###########################
# python retrain.py --registered_model resnet18-cifar-10-3407-original --is_class_unlearning True --class_to_forget beaver
python unlearn.py --method finetune --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method amnesiac --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method boundary --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method unsir --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method scrub --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method ssd --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method bad-teacher --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method neggrad --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 
python unlearn.py --method neggrad_advanced --epochs 10 --batch_size 128 --cudnn slow --registered_model resnet18-cifar-100-beaver-3407-retrained 