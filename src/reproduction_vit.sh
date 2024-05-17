###########################
# ViT on CIFAR-10
###########################
# python train.py --seed 3407 --cudnn slow --dataset cifar-10 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10
# python retrain.py --registered_model vit-cifar-10-3407-original --is_class_unlearning False 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method our --lr 1e-5
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method finetune 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method neggrad 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method amnesiac 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method bad-teacher 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method boundary 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method scrub 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method ssd 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-3407-retrained --batch_size 16 --method unsir 

# ###########################
# # # ViT frog
# ###########################
python retrain.py --registered_model vit-cifar-10-3407-original --is_class_unlearning True --class_to_forget frog
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method our --lr 1e-5
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method finetune 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method amnesiac
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method neggrad
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method bad-teacher
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method boundary
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method scrub
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method ssd
python unlearn.py --epochs 10 --registered_model vit-cifar-10-frog-3407-retrained --method unsir

# ###########################
# # # ResNet-18 horse
# ###########################
python retrain.py --registered_model vit-cifar-10-3407-original --is_class_unlearning True --class_to_forget horse
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method our --lr 1e-5
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method finetune 
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method amnesiac
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method neggrad
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method bad-teacher
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method boundary
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method scrub
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method ssd
python unlearn.py --epochs 10 --registered_model vit-cifar-10-horse-3407-retrained --method unsir

###########################
# ViT on CIFAR-100
###########################
# python train.py --seed 3407 --cudnn slow --dataset cifar-100 --model vit --batch_size 64 --epochs 30 --loss cross_entropy --optimizer sgd --lr 0.0001 --momentum 0.9 --weight_decay 0.0005 --is_lr_scheduler False --is_early_stop True --patience 10
python retrain.py --registered_model vit-cifar-100-3407-original --is_class_unlearning False 
python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method our --lr 1e-4
python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method our --lr 5e-5
python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method our --lr 1e-5
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method finetune 
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method neggrad 
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method amnesiac 
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method bad-teacher 
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method boundary 
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method scrub 
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method ssd 
# python unlearn.py --epochs 10 --registered_model vit-cifar-100-3407-retrained --batch_size 16 --method unsir 