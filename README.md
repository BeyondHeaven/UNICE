# UNICE

This is the **exposure control** branch. For the **fusion** functionality, please switch to the `fusion` branch.

## 📦 Environment Setup

To set up the environment, use the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

## 🚀 Training

To train the model, run the following command:

```bash
CUDA_VISIBLE_DEVICES=1 ../miniconda3/envs/img2img-turbo/bin/python src/train_pix2pix_turbo.py \
  --pretrained_model_name_or_path="stabilityai/sd-turbo" \
  --output_dir="output/pix2pix_turbo/exposure" \
  --dataset_folder="data/exposure" \
  --resolution=512 \
  --train_batch_size=2 \
  --enable_xformers_memory_efficient_attention \
  --viz_freq 50 \
  --report_to "wandb" \
  --tracker_project_name "pix2pix_turbo_exposure"
```

> **Note:**
> On a Tesla A100 40GB GPU:
> - Batch size 1 requires ~19561MiB
> - Batch size 2 requires ~34853MiB

## 🧪 Testing

To test the model with different exposure values, use the following script:

```bash
#!/bin/bash

# Define the exposure values
exposures=(0.2 0.5 0.8)

# Loop through each exposure value
for exposure in "${exposures[@]}"; do
    output_dir="output/$exposure"

    CUDA_VISIBLE_DEVICES=5 ../miniconda3/envs/img2img-turbo/bin/python src/inference.py \
    --model_path "checkpoints/exposure_old.pkl" \
    --input_dir /local/mnt/workspace/ruodcui/code/adaptive_3dlut/data/BAID512/input/ \
    --output_dir $output_dir \
    --prompt "exposure control" \
    --exposure $exposure
done
```

## 🙏 Acknowledgements

This project borrows code from img2img-turbo. We sincerely thank the authors for their contributions to the community.
