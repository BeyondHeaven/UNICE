#!/bin/bash

# Define the ev values
ev_values=(-2 +2)

# Read the model paths from a txt file
model_list_file="./model_list.txt"

# Loop through each line in the model list file
while IFS= read -r model_pkl; do
    # Loop through each ev value
    for ev in "${ev_values[@]}"; do
        # Define the exposure value (fixed to 0.5)
        exposure=0.5

        # Define the output directory based on the model, ev, and exposure value
        output_dir="MEfivek/${model_pkl}/${ev}/${exposure}"

        # Execute the command with the current model, ev, and exposure value
        CUDA_VISIBLE_DEVICES=5 ../miniconda3/envs/img2img-turbo/bin/python src/inference.py \
        --model_path "output/pix2pix_turbo/exposure/checkpoints/${model_pkl}" \
        --input_dir ./data/multi-exposure512/fivek/${ev} \
        --output_dir "${output_dir}" \
        --prompt "exposure control" \
        --exposure "${exposure}" \
        --filelist ./data/multi-exposure512/fivek/test.txt

    done
done < "$model_list_file"
