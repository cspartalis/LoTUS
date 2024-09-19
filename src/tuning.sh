
# seeds=(12 13 3407)
# datasets=(cifar-10 cifar-100 mufac cifar-10-cat cifar-10-horse cifar-100-rocket cifar-100-beaver)
seed=(12)
datasets=(cifar-10)
model=resnet18

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
    for maxT in ${maxTemps[@]}; do
        for seed in ${seeds[@]}; do
            mia=$(python unlearn.py --epochs $epochs --batch_size $batch_size --registered_model $model-$dataset-$seed-retrained --method our --lr $lr --maxT $maxT --beta=$beta)
            echo "Seed: $seed, Dataset: $dataset, MaxT: $maxT, MIA: $mia"

            # if [[ $mia == "1.0" ]]; then
            #     break
            # fi
        done
    done
done
