#!/bin/bash
# Usage: bash inference.sh [SUB_ID]
# Example: bash inference.sh 05   (default: 05)

SUB_ID="${1:-05}"

export CUDA_VISIBLE_DEVICES=0

# Resolve dataset root from local_config.yaml
DATASET_ROOT=$(python -c "from local_config import get_paths; print(get_paths()['dataset_root'])")

python sample_brain_va.py \
    --base configs/cogvideox_5b_lora_brain_va.yaml configs/infer_brain_va_5b_sub${SUB_ID}.yaml \
    --seed 42 \
    --jsonpath "${DATASET_ROOT}/sub-00${SUB_ID}_test_va.json"
