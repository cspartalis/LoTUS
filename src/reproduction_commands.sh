#! /bin/bash

# # ResNet-18 on CIFAR-10
# python train.py --seed 3407 --cudnn slow --dataset cifar-10 --model resnet18 --batch_size 512 --epochs 150 --loss cross_entropy --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --is_lr_scheduler true --warmup_epochs 30 --is_early_stop true --patience 50 --is_class_unlearning false --class_to_forget none
# python retrain.py --cudnn slow --run_id 5225cf95aeaf40ccaeb5b955af2bd742
python unlearn.py --method our --epochs 5 --batch_size 128 --subset_size 0.3 --is_zapping True --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8
python unlearn.py --method finetune --epochs 5 --batch_size 128 --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8
python unlearn.py --method amnesiac --epochs 5 --batch_size 128 --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8
python unlearn.py --method boundary --epochs 5 --batch_size 128 --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8
python unlearn.py --method unsir --epochs 5 --batch_size 128 --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8
python unlearn.py --method scrub --epochs 5 --batch_size 128 --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8
python unlearn.py --method ssd --epochs 5 --batch_size 128 --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8
python unlearn.py --method bad-teacher --epochs 5 --batch_size 128 --cudnn slow --run_id 4933a16e6e594a849a80a1aebb0132a8