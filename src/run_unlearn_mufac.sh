#!/bin/bash

methods=('finetune' 'neggrad' 'relabel' 'boundary' 'unsir' 'scrub')

for method in "${methods[@]}"; do
    python unlearn.py --run_id 1bcdd3b016d14404ab22c476184bff75 --epochs 3 --mu_method "$method"
done
