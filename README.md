# UNICE
Code for the paper "UNICE: Training A Universal Image Contrast Enhancer"

This repository contains the **exposure control** branch.
For the **fusion** functionality, please switch to the `fusion` branch.

## 🌟 Overview
The core idea of this method is to use a multi-exposure fusion sequence as supervision signals, generate a sequence from a single 8-bit image, and then perform multi-exposure fusion.
<img src="img/method_cmp.png" alt="Comparison with previous methods" width="600">

## 🚀 Training

To set up the environment, use the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

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

> GPU Memory requirements:
> On a Tesla A100 40GB GPU:
> - Batch size 1 requires ~19561MiB
> - Batch size 2 requires ~34853MiB

## 🧪 Testing

🔗 **Pre-trained weights** are available at [huggingface.](https://huggingface.co/lahaina/unice/tree/main/checkpoints)

To test the model with different exposure values, use the following script:

```bash
#!/bin/bash

# Define the exposure value
exposure=0.5
output_dir="output/$exposure"

CUDA_VISIBLE_DEVICES=5 ../miniconda3/envs/img2img-turbo/bin/python src/inference.py \
--model_path "checkpoints/exposure.pkl" \
--input_dir /local/mnt/workspace/ruodcui/code/adaptive_3dlut/data/BAID512/input/ \
--output_dir $output_dir \
--prompt "exposure control" \
--exposure $exposure

```

## 🙏 Acknowledgements

This project borrows code from img2img-turbo. We sincerely thank the authors for their contributions to the community.