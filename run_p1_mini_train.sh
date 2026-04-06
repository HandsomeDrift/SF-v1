#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=3600
source /public/home/maoyaoxin/anaconda3/bin/activate cinebrain
cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain
torchrun --standalone --nproc_per_node=4 train_video_fmri.py     --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/exp_p1_mini_train.yaml     --seed 42
