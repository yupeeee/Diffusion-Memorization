#! /bin/bash

for seed in {0..0}; do
    echo "Extracting UNet features for seed $seed"
    python run/unet_features.py \
        --num-inference-steps 50 \
        --guidance-scale 7.5 \
        --seed $seed
done