seeds=(12 13 3407)
maxTemps=(2 4 8 16 32 64 128 256 512)
# datasets=(cifar-10 cifar-100)
datasets=(cifar-10-cat cifar-10-horse cifar-100-rocket cifar-100-rocket)
model=vit
beta=1
epochs=3
batch_size=32
lr=1e-6

for dataset in ${datasets[@]}; do
    for maxT in ${maxTemps[@]}; do
        for seed in ${seeds[@]}; do
            mia=$(python unlearn.py --epochs $epochs --batch_size $batch_size --registered_model $model-$dataset-$seed-retrained --method our --lr $lr --maxT $maxT --beta=1)
            echo "Seed: $seed, Dataset: $dataset, MaxT: $maxT, MIA: $mia"

            if [[ $mia == "1.0" ]]; then
                break
            fi
        done
    done
done
