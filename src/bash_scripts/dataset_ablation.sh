# !/bin/bash
# Descirption: This script is used to run the experiments for few-shot retention.
# Evaluating the Retain and Forget performance when only a subset of the retain set is available.
subset_sizes=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
datasets=(cifar-10 cifar-100)
seeds=(3407 13 12)
cd ..

for dataset in ${datasets[@]}; do
    if [ $dataset == "cifar-10" ]; then
        alpha=8
    else
        alpha=2
    fi
    for ss in ${subset_sizes[@]}; do
        for seed in ${seeds[@]}; do
            echo "subset_size: $ss dataset: $dataset seed: $seed"
            dict=$(python unlearn.py --epochs 10 --batch_size 128 --registered_model resnet18-$dataset-$seed-retrained --method our_dataset_ablation --alpha=$alpha --subset_size=$ss)
        done
    done
done