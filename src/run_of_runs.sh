#!/bin/bash

echo "ViT MUCAC"
./run_max_entropy_ablation.sh  1 "mucac" "vit"
./run_max_entropy_ablation.sh  2 "mucac" "vit"
./run_max_entropy_ablation.sh  3 "mucac" "vit"
./run_max_entropy_ablation.sh  4 "mucac" "vit"
./run_max_entropy_ablation.sh  5 "mucac" "vit"

echo "ViT Rocket"
./run_max_entropy_ablation.sh  1 "rocket" "vit"
./run_max_entropy_ablation.sh  2 "rocket" "vit"
./run_max_entropy_ablation.sh  3 "rocket" "vit"
./run_max_entropy_ablation.sh  4 "rocket" "vit"
./run_max_entropy_ablation.sh  5 "rocket" "vit"

echo "ViT Beaver"
./run_max_entropy_ablation.sh  1 "beaver" "vit"
./run_max_entropy_ablation.sh  2 "beaver" "vit"
./run_max_entropy_ablation.sh  3 "beaver" "vit"
./run_max_entropy_ablation.sh  4 "beaver" "vit"
./run_max_entropy_ablation.sh  5 "beaver" "vit"