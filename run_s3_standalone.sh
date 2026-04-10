#!/bin/bash
# Stage 3 standalone: uses fixed code (alpha floor + cross-attn zero-init)
# Run AFTER Stage 2 completes and auto-pipeline is killed
set -euo pipefail

PYTHON=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
export CUDA_HOME=/usr/local/cuda-12.4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain

NGPU=${NGPU:-6}
MODEL_YAML=configs/sf_v1/cinebrain_sf_v1_model.yaml
TMPDIR=configs/sf_v1/_pipeline_tmp
LOG_DIR=logs
mkdir -p $TMPDIR $LOG_DIR

# Auto-detect latest Stage 2 checkpoint
S2_CKPT=$(ls -td ckpts_5b/phase1v2_s2_*/ 2>/dev/null | head -1)
if [ -z "$S2_CKPT" ]; then echo "[ERROR] S2 checkpoint not found!"; exit 1; fi
S2_CKPT=${S2_CKPT%/}
echo "[Pipeline] Using S2 checkpoint: $S2_CKPT"

TIMESTAMP=$(date +%m-%d-%H-%M)
S3_NAME=phase1v2_s3_${TIMESTAMP}

cat > $TMPDIR/s3_standalone.yaml <<EOF
args:
  experiment_name: $S3_NAME
  load: ${S2_CKPT}
  train_iters: 3000
  eval_interval: 500
  eval_iters: 1
  save_interval: 1000
  skip_eval_sampling: true
  train_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/sub-0005_train_va.json
  valid_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/sub-0005_test_va.json
deepspeed:
  gradient_clipping: 1.0
model:
  sparse_attn_drop: 0.3
  flow_codebook_k: 64
  reset_gate_net: true
EOF

echo "============================================"
echo "  STAGE 3: Joint Fine-tuning (3000 iter, ${NGPU} GPU)"
echo "  Load: $S2_CKPT"
echo "  Fixes: alpha floor 0.05 + cross-attn zero-init"
echo "  Timestamp: $TIMESTAMP"
echo "============================================"

NCCL_TIMEOUT=3600 $PYTHON -m torch.distributed.run --standalone --nproc_per_node=$NGPU \
    train_video_fmri.py \
    --base $MODEL_YAML configs/sf_v1/sf_v1_stage3_joint.yaml $TMPDIR/s3_standalone.yaml \
    --seed 42 \
    2>&1 | tee $LOG_DIR/stage3_standalone.log

S3_CKPT=$(ls -td ckpts_5b/${S3_NAME}-*/ 2>/dev/null | head -1)
if [ -z "$S3_CKPT" ]; then echo "[ERROR] S3 checkpoint not found!"; exit 1; fi

echo "============================================"
echo "  Stage 3 Complete!"
echo "  Checkpoint: $S3_CKPT"
echo "============================================"
