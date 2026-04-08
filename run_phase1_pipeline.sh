#!/bin/bash
# ================================================================
# CineBrain-SF v1 Phase 1+2 — Automated 3-Stage Training Pipeline
# ================================================================
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_phase1_pipeline.sh [mini|full]
#
# Default: mini (mini500 dataset, reduced iters ~3-4h total)
# "full": full dataset training (~3 days)
# ================================================================

set -e

PYTHON=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
export CUDA_HOME=/usr/local/cuda-12.4

cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain

MODE=${1:-mini}
TIMESTAMP=$(date +%m-%d-%H-%M)
TMPDIR=configs/sf_v1/_pipeline_tmp
mkdir -p $TMPDIR

if [ "$MODE" = "mini" ]; then
    TRAIN_JSON=sub-0005_train_va_mini500.json
    VALID_JSON=sub-0005_test_va_mini50.json
    S1_ITERS=500; S2_ITERS=300; S3_ITERS=500
    EVAL_INT=100; SAVE_INT=9999
    echo "[Pipeline] mini mode: 500/300/500 iters"
else
    TRAIN_JSON=sub-0005_train_va.json
    VALID_JSON=sub-0005_test_va.json
    S1_ITERS=3000; S2_ITERS=2000; S3_ITERS=3000
    EVAL_INT=500; SAVE_INT=1000
    echo "[Pipeline] full mode: 3000/2000/3000 iters"
fi

MODEL_YAML=configs/sf_v1/cinebrain_sf_v1_model.yaml
S1_BASE_LOAD=ckpts_5b/sf_v1_distill_1b_full-04-02-19-00
LOG_DIR=logs/pipeline_${MODE}_${TIMESTAMP}
mkdir -p $LOG_DIR

# ---- Helper: find latest checkpoint dir for an experiment ----
find_ckpt() {
    local name=$1
    local ckpt=$(ls -td ckpts_5b/${name}-*/ 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "[ERROR] Checkpoint not found for $name" >&2
        exit 1
    fi
    echo "${ckpt%/}"
}

# ================================================================
#  STAGE 1: Branch Pre-training
# ================================================================
S1_NAME=phase1_s1_${MODE}_${TIMESTAMP}

cat > $TMPDIR/s1_override.yaml <<EOF
args:
  experiment_name: $S1_NAME
  load: $S1_BASE_LOAD
  train_iters: $S1_ITERS
  eval_interval: $EVAL_INT
  save_interval: $SAVE_INT
  train_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/$TRAIN_JSON
  valid_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/$VALID_JSON
EOF

echo "============================================"
echo "  STAGE 1: Branch Pre-training ($S1_ITERS iters)"
echo "============================================"

$PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 \
    train_video_fmri.py \
    --base $MODEL_YAML configs/sf_v1/sf_v1_phase1_train.yaml $TMPDIR/s1_override.yaml \
    --seed 42 \
    2>&1 | tee $LOG_DIR/stage1.log

S1_CKPT=$(find_ckpt $S1_NAME)
echo "[Pipeline] Stage 1 done: $S1_CKPT"

# ================================================================
#  STAGE 2: Fusion Training
# ================================================================
S2_NAME=phase1_s2_${MODE}_${TIMESTAMP}

cat > $TMPDIR/s2_override.yaml <<EOF
args:
  experiment_name: $S2_NAME
  load: $S1_CKPT
  train_iters: $S2_ITERS
  eval_interval: $EVAL_INT
  save_interval: $SAVE_INT
  train_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/$TRAIN_JSON
  valid_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/$VALID_JSON
deepspeed:
  gradient_clipping: 1.0
model:
  sparse_attn_drop: 0.3
  flow_codebook_k: 64
EOF

echo "============================================"
echo "  STAGE 2: Fusion Training ($S2_ITERS iters)"
echo "============================================"

$PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 \
    train_video_fmri.py \
    --base $MODEL_YAML configs/sf_v1/sf_v1_stage2_fusion.yaml $TMPDIR/s2_override.yaml \
    --seed 42 \
    2>&1 | tee $LOG_DIR/stage2.log

S2_CKPT=$(find_ckpt $S2_NAME)
echo "[Pipeline] Stage 2 done: $S2_CKPT"

# ================================================================
#  STAGE 3: Joint Fine-tuning
# ================================================================
S3_NAME=phase1_s3_${MODE}_${TIMESTAMP}

cat > $TMPDIR/s3_override.yaml <<EOF
args:
  experiment_name: $S3_NAME
  load: $S2_CKPT
  train_iters: $S3_ITERS
  eval_interval: $EVAL_INT
  save_interval: $SAVE_INT
  train_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/$TRAIN_JSON
  valid_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/$VALID_JSON
model:
  sparse_attn_drop: 0.3
  flow_codebook_k: 64
EOF

echo "============================================"
echo "  STAGE 3: Joint Fine-tuning ($S3_ITERS iters)"
echo "============================================"

$PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 \
    train_video_fmri.py \
    --base $MODEL_YAML configs/sf_v1/sf_v1_stage3_joint.yaml $TMPDIR/s3_override.yaml \
    --seed 42 \
    2>&1 | tee $LOG_DIR/stage3.log

S3_CKPT=$(find_ckpt $S3_NAME)

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "  Stage 1: $S1_CKPT"
echo "  Stage 2: $S2_CKPT"
echo "  Stage 3: $S3_CKPT"
echo "  Logs:    $LOG_DIR/"
echo "============================================"

# Clean up temp configs
rm -rf $TMPDIR
