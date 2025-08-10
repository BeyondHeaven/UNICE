# UNICE - Fusion Branch

## 📦 Environment Setup

To set up the environment, use the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

## 🚀 Training

To train the model on the fusion dataset, run the following command:

```bash
CUDA_VISIBLE_DEVICES=2 ../miniconda3/envs/img2img-turbo/bin/python src/train_pix2pix_turbo.py \
  --pretrained_model_name_or_path="stabilityai/sd-turbo" \
  --output_dir="output/pix2pix_turbo/fusion_all" \
  --dataset_folder="data/fusion_all" \
  --resolution=512 \
  --train_batch_size=2 \
  --enable_xformers_memory_efficient_attention \
  --viz_freq 50 \
  --report_to "wandb" \
  --tracker_project_name "pix2pix_turbo_fusion_all"
```

> **Note:**
> On a Tesla A100 40GB GPU:
> - Batch size 1 requires ~19545MiB
> - Batch size 2 requires ~34857MiB

### 📁 Dataset Structure

The directory structure for `data/fusion_all` should be organized as follows:

```
data/fusion_all
├── fake_HDR
│   ├── fivek
│   │   ├── 0
│   │   │   ├── 0.2
│   │   │   ├── 0.5
│   │   │   └── 0.8
│   │   └── other exposure
│   └── other sub-datasets
└── GT
    ├── fivek
    ├── HDRP
    ├── PASCAL-RAW
    ├── PPR10K
    └── RAISE
```

## 🧪 Testing

To test the model, use the following command:

```bash
CUDA_VISIBLE_DEVICES=3 ../miniconda3/envs/img2img-turbo/bin/python src/inference.py \
  --model_path ./unice_checkpoints/fusion.pkl \
  --input_dir ./fake_HDR \
  --output_dir fusion_output \
  --prompt "exposure control"
```
The fake_HDR directory contains several subfolders, each representing images with different exposure levels.
You can configure these subfolders in `src/inference.py` by modifying the following line:
```python
EXPOSURE_FOLDERS = ['0.2', '0.5', '0.8', 'input']
```
Using more exposure levels generally leads to better results, but it will also increase computational cost.

## 🙏 Acknowledgements

This project borrows code from img2img-turbo. We sincerely thank the authors for their contributions to the community.
