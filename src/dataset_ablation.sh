subset_sizes=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
datasets=(cifar-10 cifar-100 mufac)
seeds=(3407 13 12)

for dataset in ${datasets[@]}; do
    if [[ $dataset == "cifar-10" ]]; then
        alpha=512
    else
        alpha=32
    fi
    for ss in ${subset_sizes[@]}; do
        for seed in ${seeds[@]}; do
            echo "subset_size: $ss dataset: $dataset seed: $seed"
            dict=$(python unlearn.py --epochs 10 --batch_size 128 --registered_model resnet18-$dataset-$seed-retrained --method our --lr=1e-4 --alpha=$alpha --beta=0 --subset_size=$ss)
        done
    done
done