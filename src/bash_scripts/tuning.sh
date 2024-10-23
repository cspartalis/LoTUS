# !/bin/bash
# Descirption: This script is used to find the best alpha value.
# We perform a grid search in the cifar-10 dataset for seed 13 and then we apply the best alpha in every setting. 
# Cross-validation cannot be applied, thus we adopt the tuning approach of Foster et al. in the SSD paper.
cd ..
alphas=(32 64 128 256 512 1024)
models=(vit)
datasets=(cifar-10 cifar-100 mufac)
seeds=(3407 13 12)

for model in ${models[@]}; do
    if [[ $model == "resnet18" ]]; then
        epochs=10
        batch_size=128
        lr=1e-4
    else
        epochs=3
        batch_size=32
        lr=1e-6
    fi

    for dataset in ${datasets[@]};do
        for alpha in ${alphas[@]}; do
            for seed in ${seeds[@]}; do
                dict=$(python unlearn.py --epochs $epochs --batch_size $batch_size --registered_model $model-$dataset-$seed-retrained --method our --lr=$lr --alpha=$alpha)
                echo $dict
            done
        done
    done
done
