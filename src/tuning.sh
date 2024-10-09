
seeds=(3407 13 12)
datasets=(cifar-10 cifar-100 mufac cifar-10-cat cifar-10-horse cifar-100-rocket cifar-100-beaver)
alphas=(2 4 8 16 32 64 128 256 512 1024)
models=(resnet18 vit)

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

    for dataset in ${datasets[@]}; do
        if [[ $dataset == "cifar-10" || $dataset == "cifar-100" || $dataset == "mufac" || $dataset == "imagenet" ]]; then
            beta=0
        else
            beta=1
        fi
        for alpha in ${alphas[@]}; do
            for seed in ${seeds[@]}; do
                echo "$model-$dataset-$seed-retrained"
                dict=$(python unlearn.py --epochs $epochs --batch_size $batch_size --registered_model $model-$dataset-$seed-retrained --method our --lr=$lr --alpha=$alpha --beta=$beta)
                echo "Seed: $seed, Dataset: $dataset, Alpha: $alpha"
                echo $dict
            done
        done
    done
done
