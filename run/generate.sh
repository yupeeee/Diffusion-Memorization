#! /bin/bash

for seed in {0..49}; do
    echo "Generating images/latents/noise_preds for seed $seed"
    python run/generate.py \
        --num-inference-steps 50 \
        --guidance-scale 7.5 \
        --seed $seed \
        --batch-size 16 \
        --num-workers 4 \
        --devices "auto"
done