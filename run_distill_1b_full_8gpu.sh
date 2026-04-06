#!/bin/bash
# Stage 1B: Full dataset distillation training (8 GPUs on gpu4)
# EEG learns to match fMRI features via MSE distillation
# Slow branch frozen, Fast branch trainable

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600

# Activate conda env
source /public/home/maoyaoxin/anaconda3/bin/activate cinebrain

cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain

torchrun --standalone --nproc_per_node=8 train_video_fmri.py     --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v1_distill_1b_full.yaml     --seed 42
