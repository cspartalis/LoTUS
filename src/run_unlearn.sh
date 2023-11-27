basic_arguments="--model resnet18 --dataset cifar-10 --epochs 10 --is_lr_scheduler False --is_early_stop False --run_id 149b43f9c3414f889e9f768b79d729c2 --cudnn benchmark"
baseline_arguments="--lr 0.01"
boundary_arguments="--lr 0.00001 --weight_decay 0"

if [ "$1" = "baselines" ]; then
    echo "Finetune"
    python unlearn.py --mu_method finetune $basic_arguments $baseline_arguments

    echo "NegGrad"
    python unlearn.py --mu_method neggrad $basic_arguments $baseline_arguments

    echo "Relabel"
    python unlearn.py --mu_method relabel $basic_arguments $baseline_arguments

    echo "Boundary"
    python unlearn.py --mu_method boundary $basic_arguments $boundary_arguments
fi

if [ "$1" = "zapping" ]; then
    for threshold in $(seq 0 0.1 1); do
        threshold=$(echo "$threshold" | tr ',' '.')
        python unlearn.py --mu_method zapping $basic_arguments $baseline_arguments --zap_threshold $threshold --is_diff_grads False
    done
    for threshold in $(seq 0 0.1 1); do
        threshold=$(echo "$threshold" | tr ',' '.')
        python unlearn.py --mu_method zapping $basic_arguments $baseline_arguments --zap_threshold $threshold --is_diff_grads True
    done
fi
