#!/bin/bash

# Define the exposure values
exposures=(0.2 0.5 0.8)

# Loop through each exposure value
for exposure in "${exposures[@]}"; do
    # Define the output directory based on the exposure value
    output_dir="output/$exposure"

    # Execute the command with the current exposure value
    CUDA_VISIBLE_DEVICES=5 ../miniconda3/envs/img2img-turbo/bin/python src/inference.py \
    --model_path "../unice_checkpoints/exposure_old.pkl" \
    --input_dir /local/mnt/workspace/ruodcui/code/adaptive_3dlut/data/BAID512/input/ \
    --output_dir $output_dir \
    --prompt "exposure control" \
    --exposure $exposure
done