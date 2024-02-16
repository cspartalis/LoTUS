SEED=3407
CLASS_TO_FORGET="rocket"

original_id_resnet_rocket=""
original_id_vit_rocket=""
retrained_id_resnet_rocket=""
retrained_id_vit_rocket=""

#################################
########## T R A I N ############
#################################
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
    --patience 50 \
    --is_class_unlearning True \
    --class_to_forget $CLASS_TO_FORGET

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
    --patience 10 \
    --is_class_unlearning True \
    --class_to_forget $CLASS_TO_FORGET

