seeds=(3407 13 12)
temperatures=(16 32 64 128 512 1024 2048)

for seed in ${seeds[@]}; do
    for temp in ${temperatures[@]}; do
        python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-$seed-retrained --method our --lr 1e-4 --weight_decay 0.01 --maxT $temp
    done

    python unlearn.py --epochs 10 --registered_model resnet18-cifar-10-$seed-retrained --method our --lr 1e-4 --weight_decay 0.01 --minT 2048 --maxT 2049
done
