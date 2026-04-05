"""
P1 Temporal Target Extraction for CineBrain-SF v1 (Slow-Fast).

Extracts per-clip temporal supervision targets and appends them to existing
shard files in supervision_cache/version_v1/shards/:

  - temporal_frame_embs : (N_clips, 9, 1152)  SigLIP2 embeddings at 9 uniformly sampled frames
  - flow_mag_traj       : (N_clips, 9)         per-frame motion proxy via pixel-level frame differencing

Frame sampling: 9 frames from 33-frame clips at indices [0, 4, 8, 12, 16, 20, 24, 28, 32].

Step 1b diagnostics are printed at the end:
  - Intra-clip delta variance (z_t - z_1)
  - Delta norm distribution for high-dynamic vs low-dynamic clips
  - Spearman correlation between delta norm and flow_mag_traj mean

Usage:
    python tools/extract_temporal_targets.py --gpu 0 --split both
    python tools/extract_temporal_targets.py --gpu 2 --split train

Author: auto-generated for SF-v1
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_INDICES = [0, 4, 8, 12, 16, 20, 24, 28, 32]
T_OUT = len(FRAME_INDICES)  # 9
SHARD_DIR = "/public/home/maoyaoxin/xxt/datasets/supervision_cache/version_v1/shards"
SIGLIP_DIM = 1152


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extract temporal supervision targets")
    p.add_argument("--gpu", type=int, default=0, help="GPU device index")
    p.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which dataset split to process",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for SigLIP inference (frames per forward pass)",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip shards that already contain temporal_frame_embs",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_clip_list(split):
    """Load clip list from JSON. Returns list of (clip_id, video_path) sorted by id."""
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from local_config import get_paths

    dataset_root = get_paths()["dataset_root"]

    splits = []
    if split in ("train", "both"):
        splits.append("train")
    if split in ("test", "both"):
        splits.append("test")

    clips = []
    seen = set()
    for s in splits:
        json_path = os.path.join(dataset_root, "sub-0005_{}_va.json".format(s))
        with open(json_path) as f:
            data = json.load(f)
        for item in data:
            vpath = item["video"]
            cid = int(os.path.splitext(os.path.basename(vpath))[0])
            if cid not in seen:
                clips.append((cid, vpath))
                seen.add(cid)
    clips.sort(key=lambda x: x[0])
    return clips


def load_video_frames(video_path, indices=None):
    """Load specific frames from a video. Returns torch.Tensor (T, H, W, 3) uint8."""
    if indices is None:
        indices = FRAME_INDICES
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    n_frames = len(vr)
    safe_indices = [min(i, n_frames - 1) for i in indices]
    frames = vr.get_batch(safe_indices)  # (T, H, W, 3) torch uint8
    return frames


# ---------------------------------------------------------------------------
# SigLIP model
# ---------------------------------------------------------------------------
def load_siglip(device):
    """Load SigLIP2 vision model and image processor."""
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from local_config import get_paths
    from transformers import AutoModel, SiglipImageProcessor

    siglip_path = get_paths()["siglip2"]
    print("Loading SigLIP2 from: {}".format(siglip_path))

    model = AutoModel.from_pretrained(siglip_path, torch_dtype=torch.float16)
    model = model.vision_model  # only need image encoder
    model = model.to(device).eval()

    proc = SiglipImageProcessor(
        do_resize=True,
        size={"height": 384, "width": 384},
        resample=3,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    print("SigLIP2 loaded on {}, dtype=float16".format(device))
    return model, proc


@torch.no_grad()
def extract_siglip_embeddings(frames, model, processor, device, batch_size=8):
    """
    Extract SigLIP image embeddings for a set of frames.

    Args:
        frames: torch.Tensor (T, H, W, 3) uint8
    Returns:
        embeddings: torch.Tensor (T, 1152) float32
    """
    T = frames.shape[0]
    frames_np = frames.cpu().numpy() if frames.is_cuda else frames.numpy()

    # Run processor on all frames at once
    inputs = processor(
        images=[frames_np[i] for i in range(T)], return_tensors="pt"
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16)

    all_embs = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch_pv = pixel_values[start:end]
        outputs = model(pixel_values=batch_pv)
        emb = outputs.pooler_output.float()  # (B, 1152)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)  # (T, 1152)


# ---------------------------------------------------------------------------
# Flow magnitude trajectory (simplified pixel-diff proxy)
# ---------------------------------------------------------------------------
def compute_flow_mag_traj(frames):
    """
    Compute per-frame motion proxy via pixel-level frame differencing.

    Args:
        frames: torch.Tensor (T, H, W, 3) uint8
    Returns:
        flow_mag_traj: torch.Tensor (T,) float32
    """
    frames_float = frames.float() / 255.0  # (T, H, W, 3)
    diffs = (frames_float[1:] - frames_float[:-1]).abs().mean(dim=(1, 2, 3))  # (T-1,)
    flow_mag_traj = torch.cat([torch.zeros(1), diffs])  # (T,)
    return flow_mag_traj


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------
def get_all_shard_files():
    """Return sorted list of all shard file paths."""
    return sorted(glob.glob(os.path.join(SHARD_DIR, "clips_*.pt")))


def parse_shard_range(shard_path):
    """Parse (start, end) clip IDs from shard filename."""
    base = os.path.splitext(os.path.basename(shard_path))[0]
    parts = base.split("_")
    return int(parts[1]), int(parts[2])


# ---------------------------------------------------------------------------
# Diagnostics (Step 1b)
# ---------------------------------------------------------------------------
def run_diagnostics(all_temporal_embs, all_flow_trajs):
    """Print Step 1b delta distribution diagnostics."""
    print("\n" + "=" * 60)
    print("STEP 1b: TEMPORAL DELTA DIAGNOSTICS")
    print("=" * 60)

    N = len(all_temporal_embs)
    print("Analyzing {} clips...\n".format(N))

    all_embs_t = torch.stack(all_temporal_embs)   # (N, 9, 1152)
    all_flows_t = torch.stack(all_flow_trajs)     # (N, 9)

    # ---- 1. Intra-clip delta variance: delta_t = z_t - z_1 ----
    z_1 = all_embs_t[:, 0:1, :]                   # (N, 1, 1152)
    deltas = all_embs_t[:, 1:, :] - z_1            # (N, 8, 1152)
    delta_norms = deltas.norm(dim=-1)               # (N, 8)
    per_clip_var = delta_norms.var(dim=1)           # (N,)

    print("--- Intra-clip delta (z_t - z_1) norm variance ---")
    print("  Mean variance:   {:.4f}".format(per_clip_var.mean().item()))
    print("  Median variance: {:.4f}".format(per_clip_var.median().item()))
    print("  Std of variance: {:.4f}".format(per_clip_var.std().item()))
    print("  Min variance:    {:.4f}".format(per_clip_var.min().item()))
    print("  Max variance:    {:.4f}".format(per_clip_var.max().item()))
    print()

    # Mean delta norm per time step
    mean_per_t = delta_norms.mean(dim=0)  # (8,)
    print("--- Mean delta norm per time step (t=2..9) ---")
    for t_idx, frame_idx in enumerate(FRAME_INDICES[1:]):
        print("  t={} (frame {:2d}): {:.4f}".format(t_idx + 2, frame_idx, mean_per_t[t_idx].item()))
    print()

    # ---- 2. High-dynamic vs Low-dynamic ----
    mean_flow = all_flows_t.mean(dim=1)  # (N,)
    med_flow = mean_flow.median().item()

    high_mask = mean_flow > med_flow
    low_mask = ~high_mask

    high_dn = delta_norms[high_mask].mean(dim=1)  # (N_high,)
    low_dn = delta_norms[low_mask].mean(dim=1)    # (N_low,)

    print("--- High-dynamic vs Low-dynamic clips (median flow = {:.6f}) ---".format(med_flow))
    print("  High-dynamic ({} clips):".format(high_mask.sum().item()))
    print("    Mean delta norm: {:.4f}".format(high_dn.mean().item()))
    print("    Std delta norm:  {:.4f}".format(high_dn.std().item()))
    print("  Low-dynamic ({} clips):".format(low_mask.sum().item()))
    print("    Mean delta norm: {:.4f}".format(low_dn.mean().item()))
    print("    Std delta norm:  {:.4f}".format(low_dn.std().item()))
    ratio = high_dn.mean().item() / (low_dn.mean().item() + 1e-8)
    print("  Ratio (high/low):  {:.4f}".format(ratio))
    print()

    # ---- 3. Spearman correlation ----
    clip_mean_dn = delta_norms.mean(dim=1).numpy()  # (N,)
    clip_mean_fl = mean_flow.numpy()                  # (N,)
    rho, pval = spearmanr(clip_mean_dn, clip_mean_fl)
    print("--- Spearman correlation: delta_norm vs flow_mag_traj_mean ---")
    print("  rho  = {:.4f}".format(rho))
    print("  p    = {:.2e}".format(pval))
    print()

    # Percentile distribution
    all_flat = delta_norms.flatten().numpy()
    print("--- Delta norm percentile distribution (all clips, all time steps) ---")
    for p in [10, 25, 50, 75, 90]:
        print("  P{:02d}: {:.4f}".format(p, np.percentile(all_flat, p)))
    print()

    print("Done. Temporal targets have been appended to all shard files.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("P1 Temporal Target Extraction")
    print("=" * 60)
    print("  Device:   {}".format(device))
    print("  Split:    {}".format(args.split))
    print("  Indices:  {}".format(FRAME_INDICES))
    print("  T_out:    {}".format(T_OUT))
    print("  Shard dir: {}".format(SHARD_DIR))
    print()

    # Load clip list
    clips = load_clip_list(args.split)
    print("Total clips to process: {}".format(len(clips)))

    clip_id_to_video = {cid: vpath for cid, vpath in clips}

    # Load SigLIP
    model, processor = load_siglip(device)

    # Get shard files
    shard_files = get_all_shard_files()
    print("Found {} shard files".format(len(shard_files)))
    print()

    total_processed = 0
    total_skipped = 0
    total_missing = 0
    total_errors = 0

    # For diagnostics
    diag_embs = []
    diag_flows = []

    t0 = time.time()

    for shard_path in shard_files:
        shard_start, shard_end = parse_shard_range(shard_path)
        shard_name = os.path.basename(shard_path)
        print("\n" + "=" * 60)
        print("Processing shard: {} (clips {}-{})".format(shard_name, shard_start, shard_end))
        print("=" * 60)

        shard_data = torch.load(shard_path, map_location="cpu")
        clip_ids = shard_data["clip_ids"]
        n_clips = len(clip_ids)

        # Skip if already done
        if args.skip_existing and "temporal_frame_embs" in shard_data:
            print("  -> Already has temporal_frame_embs, skipping.")
            total_skipped += n_clips
            continue

        # Pre-allocate
        temporal_embs = torch.zeros(n_clips, T_OUT, SIGLIP_DIM, dtype=torch.float32)
        flow_trajs = torch.zeros(n_clips, T_OUT, dtype=torch.float32)
        valid_mask = torch.zeros(n_clips, dtype=torch.bool)

        pbar = tqdm(enumerate(clip_ids), total=n_clips, desc="  " + shard_name, ncols=88)
        for idx, cid in pbar:
            if cid not in clip_id_to_video:
                total_skipped += 1
                continue

            video_path = clip_id_to_video[cid]
            if not os.path.exists(video_path):
                total_missing += 1
                tqdm.write("  WARN: missing video {}".format(video_path))
                continue

            try:
                frames = load_video_frames(video_path)  # (9, H, W, 3)

                embs = extract_siglip_embeddings(
                    frames, model, processor, device, args.batch_size
                )  # (9, 1152)
                temporal_embs[idx] = embs

                flow_traj = compute_flow_mag_traj(frames)  # (9,)
                flow_trajs[idx] = flow_traj

                valid_mask[idx] = True
                total_processed += 1

                diag_embs.append(embs)
                diag_flows.append(flow_traj)

            except Exception as e:
                total_errors += 1
                tqdm.write("  ERROR clip {}: {}".format(cid, e))
                continue

        # Append to shard (preserve all existing keys)
        shard_data["temporal_frame_embs"] = temporal_embs
        shard_data["flow_mag_traj"] = flow_trajs
        shard_data["temporal_valid_mask"] = valid_mask
        shard_data["_temporal_meta"] = {
            "frame_sampling_indices": FRAME_INDICES,
            "T_out": T_OUT,
            "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "siglip_dim": SIGLIP_DIM,
            "method": "pixel_diff_motion_proxy",
        }

        torch.save(shard_data, shard_path)
        n_valid = valid_mask.sum().item()
        print("  -> Saved: {}/{} clips with temporal targets".format(n_valid, n_clips))

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("  Processed:     {}".format(total_processed))
    print("  Skipped:       {}".format(total_skipped))
    print("  Missing video: {}".format(total_missing))
    print("  Errors:        {}".format(total_errors))
    print("  Elapsed:       {:.1f}s ({:.1f}min)".format(elapsed, elapsed / 60))

    # Run diagnostics if we have enough data
    if len(diag_embs) >= 10:
        run_diagnostics(diag_embs, diag_flows)
    else:
        print("\nNot enough clips ({}) for diagnostics (need >= 10).".format(len(diag_embs)))


if __name__ == "__main__":
    main()
