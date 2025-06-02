#! /bin/bash

for seed in {2..49}; do
    echo "Calculating errors for seed $seed"
    python run/error.py \
        --num-inference-steps 50 \
        --guidance-scale 7.5 \
        --seed $seed
done