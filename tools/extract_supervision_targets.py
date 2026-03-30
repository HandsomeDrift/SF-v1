"""
Offline extraction of SF v1 supervision targets from stimulus videos.

Extracts per-clip targets using SigLIP2 and saves as .npy files:
  - gt_keyframe_embed.npy  (1152,)  SigLIP image embed of middle frame
  - gt_text_embed.npy      (1152,)  SigLIP text embed of caption
  - gt_dynamics_embed.npy  (1152,)  mean of adjacent frame difference features
  - gt_motion_embed.npy    (1152,)  feature of (last_frame - first_frame)
  - gt_tc_embed.npy        (1152,)  mean adjacent-frame cosine similarity vector

Usage:
    python tools/extract_supervision_targets.py \
        --dataset_root /public/home/maoyaoxin/xxt/datasets \
        --output_dir /public/home/maoyaoxin/xxt/datasets/sf_targets \
        --batch_size 32 \
        --num_workers 4

Outputs:
    {output_dir}/{clip_id}/gt_keyframe_embed.npy
    {output_dir}/{clip_id}/gt_text_embed.npy
    {output_dir}/{clip_id}/gt_dynamics_embed.npy
    {output_dir}/{clip_id}/gt_motion_embed.npy
    {output_dir}/{clip_id}/gt_tc_embed.npy
"""
import argparse
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import decord
from transformers import AutoModel, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root dir containing clips/ and captions JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for supervision targets")
    parser.add_argument("--siglip_path", type=str, default=None,
                        help="Path to SigLIP2 model (default: from local_config)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for SigLIP inference")
    parser.add_argument("--num_frames", type=int, default=33,
                        help="Number of frames per clip")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start clip index (for parallel extraction)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End clip index (-1 = all)")
    return parser.parse_args()


def load_video_frames(video_path, num_frames=33):
    """Load video frames as numpy array (T, H, W, 3) uint8."""
    vr = decord.VideoReader(video_path)
    indices = np.arange(0, min(num_frames, len(vr)), 1)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
    return frames


def load_captions(dataset_root):
    """Load clip_id -> caption mapping from all train JSON files."""
    import json
    captions = {}
    # Try the dedicated captions file first
    cap_file = os.path.join(dataset_root, "captions-qwen-2.5-vl-7b.json")
    if os.path.exists(cap_file):
        with open(cap_file) as f:
            cap_data = json.load(f)
        if isinstance(cap_data, dict):
            return cap_data
        elif isinstance(cap_data, list):
            for item in cap_data:
                if "video" in item and "text" in item:
                    clip_id = os.path.splitext(os.path.basename(item["video"]))[0]
                    captions[clip_id] = item["text"]
            if captions:
                return captions

    # Fallback: collect from all train JSON files
    for json_path in sorted(glob.glob(os.path.join(dataset_root, "sub-*_train_va.json"))):
        with open(json_path) as f:
            data = json.load(f)
        for item in data:
            clip_id = os.path.splitext(os.path.basename(item["video"]))[0]
            if clip_id not in captions:
                captions[clip_id] = item["text"]

    # Also collect from test JSONs for completeness
    for json_path in sorted(glob.glob(os.path.join(dataset_root, "sub-*_test_va.json"))):
        with open(json_path) as f:
            data = json.load(f)
        for item in data:
            clip_id = os.path.splitext(os.path.basename(item["video"]))[0]
            if clip_id not in captions:
                captions[clip_id] = item["text"]

    return captions


@torch.no_grad()
def extract_targets_for_clip(
    frames_np, caption, siglip_model, processor, device="cuda"
):
    """
    Extract all 5 supervision targets for a single clip.

    Args:
        frames_np: (T, H, W, 3) uint8 numpy array
        caption: str text caption
        siglip_model: loaded SigLIP2 model
        processor: SigLIP2 processor

    Returns:
        dict of numpy arrays
    """
    T = frames_np.shape[0]

    # --- Process all frames through SigLIP ---
    # SigLIP expects PIL images or pixel_values
    from PIL import Image
    pil_frames = [Image.fromarray(frames_np[i]) for i in range(T)]

    # Process in batches to avoid OOM
    all_embeds = []
    batch_size = 16
    for i in range(0, T, batch_size):
        batch_frames = pil_frames[i:i+batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.amp.autocast("cuda", enabled=False):
            outputs = siglip_model.get_image_features(**inputs)
        all_embeds.append(outputs.cpu())
    frame_embeds = torch.cat(all_embeds, dim=0)  # (T, 1152)
    frame_embeds = F.normalize(frame_embeds, dim=-1)

    # --- gt_keyframe_embed: middle frame ---
    mid_idx = T // 2
    gt_keyframe_embed = frame_embeds[mid_idx].numpy()  # (1152,)

    # --- gt_text_embed: caption ---
    text_inputs = processor(text=caption, return_tensors="pt", padding=True, truncation=True, max_length=64)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}
    with torch.amp.autocast("cuda", enabled=False):
        gt_text_embed = siglip_model.get_text_features(**text_inputs)
    gt_text_embed = F.normalize(gt_text_embed, dim=-1).cpu().squeeze(0).numpy()  # (1152,)

    # --- gt_dynamics_embed: mean of adjacent frame differences ---
    if T > 1:
        frame_diffs = frame_embeds[1:] - frame_embeds[:-1]  # (T-1, 1152)
        gt_dynamics_embed = frame_diffs.mean(dim=0).numpy()  # (1152,)
    else:
        gt_dynamics_embed = np.zeros(1152, dtype=np.float32)

    # --- gt_motion_embed: last_frame - first_frame feature ---
    gt_motion_embed = (frame_embeds[-1] - frame_embeds[0]).numpy()  # (1152,)

    # --- gt_tc_embed: adjacent frame cosine similarity encoding ---
    if T > 1:
        # cosine similarity between adjacent frames → encode as weighted mean
        cos_sims = F.cosine_similarity(frame_embeds[:-1], frame_embeds[1:], dim=-1)  # (T-1,)
        # Weight each frame embed by its similarity to the next
        weights = cos_sims.unsqueeze(-1)  # (T-1, 1)
        gt_tc_embed = (weights * frame_embeds[:-1]).mean(dim=0).numpy()  # (1152,)
    else:
        gt_tc_embed = frame_embeds[0].numpy()

    return {
        "gt_keyframe_embed": gt_keyframe_embed.astype(np.float32),
        "gt_text_embed": gt_text_embed.astype(np.float32),
        "gt_dynamics_embed": gt_dynamics_embed.astype(np.float32),
        "gt_motion_embed": gt_motion_embed.astype(np.float32),
        "gt_tc_embed": gt_tc_embed.astype(np.float32),
    }


def main():
    args = parse_args()
    decord.bridge.set_bridge("native")

    # Load SigLIP2
    if args.siglip_path is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from local_config import get_paths
        siglip_path = get_paths()["siglip2"]
    else:
        siglip_path = args.siglip_path

    print(f"Loading SigLIP2 from {siglip_path}...")
    siglip_model = AutoModel.from_pretrained(siglip_path).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(siglip_path)
    print("SigLIP2 loaded.")

    # Load captions
    captions = load_captions(args.dataset_root)
    print(f"Loaded {len(captions)} captions.")

    # Find all video clips
    clips_dir = os.path.join(args.dataset_root, "clips")
    clip_files = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))
    print(f"Found {len(clip_files)} video clips.")

    # Slice if needed
    end_idx = args.end_idx if args.end_idx > 0 else len(clip_files)
    clip_files = clip_files[args.start_idx:end_idx]
    print(f"Processing clips [{args.start_idx}:{end_idx}] ({len(clip_files)} clips)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each clip
    skipped = 0
    for clip_path in tqdm(clip_files, desc="Extracting targets"):
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]
        out_dir = os.path.join(args.output_dir, clip_id)

        # Skip if already extracted
        if os.path.exists(os.path.join(out_dir, "gt_tc_embed.npy")):
            skipped += 1
            continue

        # Get caption
        caption = captions.get(clip_id, "A video clip.")

        # Load frames
        try:
            frames = load_video_frames(clip_path, args.num_frames)
        except Exception as e:
            print(f"Error loading {clip_path}: {e}")
            continue

        # Extract targets
        try:
            targets = extract_targets_for_clip(
                frames, caption, siglip_model, processor, args.device
            )
        except Exception as e:
            print(f"Error extracting targets for {clip_id}: {e}")
            continue

        # Save
        os.makedirs(out_dir, exist_ok=True)
        for name, arr in targets.items():
            np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    print(f"Done! Skipped {skipped} already-extracted clips.")
    print(f"Targets saved to {args.output_dir}")


if __name__ == "__main__":
    main()
