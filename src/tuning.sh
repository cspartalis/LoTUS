
alphas=(2 4 8 16 32 64 128)
betas=(0 1 2)
# seeds=(12 13 3407)
# datasets=(cifar-10 cifar-100 mufac cifar-10-cat cifar-10-horse cifar-100-rocket cifar-100-beaver)
# models=(resnet18 vit)
datasets=(cifar-10)
seeds=(12)
models=(resnet18)

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
        for alpha in ${alphas[@]}; do
            for beta in ${betas[@]}; do
                for seed in ${seeds[@]}; do
                    echo "$model-$dataset-$seed-retrained"
                    dict=$(python unlearn.py --epochs $epochs --batch_size $batch_size --registered_model $model-$dataset-$seed-retrained --method our --lr=$lr --alpha=$alpha --beta=$beta)
                    echo "Seed: $seed, Dataset: $dataset, Alpha: $alpha"

                    if [[ $mia == "1.0" ]]; then
                        break
                    fi
                done
            done
        done
    done
done

        # if [[ $dataset == "cifar-10" || $dataset == "cifar-100" || $dataset == "mufac" || $dataset == "imagenet" ]]; then
        #     beta=0
        # else
        #     beta=1
        # fi

