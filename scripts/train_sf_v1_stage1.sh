#!/bin/bash
# CineBrain-SF v1 — Stage I Training (Branch Pretraining)
# Loss: L_align + L_slow + L_fast
# 4-GPU distributed training on ts3 gpu2

export CUDA_VISIBLE_DEVICES=0,2,3,4

# Subject to train (change as needed)
SUB=${1:-05}

echo "============================================"
echo "CineBrain-SF v1 Stage I Training"
echo "Subject: sub-${SUB}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

torchrun --standalone --nproc_per_node=4 train_video_fmri.py \
    --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v1_train_stage1.yaml \
    --seed 42
