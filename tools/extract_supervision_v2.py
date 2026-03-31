"""
CineBrain-SF v1 — Offline Supervision Target Extraction (v2)

Extracts per-clip targets and saves as sharded .pt files per episode range.
Complies with the offline supervision target specification document.

Targets extracted:
  Phase 1 (SigLIP): keyframe_img_emb, scene_text_emb  (reuse existing)
  Phase 2 (VAE):    keyframe_vae_latent, structure_latent
  Phase 3 (RAFT):   flow_mag, flow_token, ofs_score, dyn_label

Usage:
    # Phase 1: SigLIP targets (keyframe + text embeddings)
    python tools/extract_supervision_v2.py --phase siglip \
        --dataset_root /public/home/maoyaoxin/xxt/datasets \
        --output_dir /public/home/maoyaoxin/xxt/datasets/supervision_cache/version_v1 \
        --device cuda

    # Phase 2: VAE targets (keyframe latent + structure latent)
    python tools/extract_supervision_v2.py --phase vae \
        --dataset_root /public/home/maoyaoxin/xxt/datasets \
        --output_dir /public/home/maoyaoxin/xxt/datasets/supervision_cache/version_v1 \
        --device cuda

    # Phase 3: RAFT optical flow targets
    python tools/extract_supervision_v2.py --phase raft \
        --dataset_root /public/home/maoyaoxin/xxt/datasets \
        --output_dir /public/home/maoyaoxin/xxt/datasets/supervision_cache/version_v1 \
        --device cuda

    # All phases at once
    python tools/extract_supervision_v2.py --phase all \
        --dataset_root /public/home/maoyaoxin/xxt/datasets \
        --output_dir /public/home/maoyaoxin/xxt/datasets/supervision_cache/version_v1 \
        --device cuda

Output structure:
    supervision_cache/version_v1/
      metadata/
        extraction_config.yaml
        version_info.json
      shards/
        clips_0000_0999.pt    # clip 000000-000999
        clips_1000_1999.pt    # clip 001000-001999
        ...
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, default="all", choices=["siglip", "vae", "raft", "all"])
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--shard_size", type=int, default=1000, help="Clips per shard file")
    p.add_argument("--num_frames", type=int, default=33)
    p.add_argument("--video_size", type=int, nargs=2, default=[480, 720])
    p.add_argument("--keyframe_index", type=int, default=0, help="0=first frame (spec default)")
    p.add_argument("--start_shard", type=int, default=0)
    p.add_argument("--end_shard", type=int, default=-1)
    p.add_argument("--force_rebuild", action="store_true")
    p.add_argument("--raft_iters", type=int, default=20, help="RAFT iteration count")
    p.add_argument("--flow_patch_size", type=int, default=16, help="Patch size for flow tokenization")
    p.add_argument("--ofs_percentile", type=float, default=50.0, help="Percentile for fast/slow threshold")
    return p.parse_args()


# ============================================================
# Video loading
# ============================================================
def load_clip_frames(video_path, num_frames=33):
    """Load video frames as uint8 numpy (T, H, W, 3)."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    indices = list(range(min(num_frames, len(vr))))
    frames = vr.get_batch(indices).asnumpy()
    return frames


def load_captions(dataset_root):
    """Load clip_id (int) -> caption text."""
    captions = {}
    for pattern in ["sub-*_train_va.json", "sub-*_test_va.json"]:
        for jp in sorted(glob.glob(os.path.join(dataset_root, pattern))):
            for item in json.load(open(jp)):
                cid = int(os.path.splitext(os.path.basename(item["video"]))[0])
                if cid not in captions:
                    captions[cid] = item["text"]
    return captions


def get_all_clip_paths(dataset_root):
    """Return sorted list of (clip_id_int, video_path)."""
    clips_dir = os.path.join(dataset_root, "clips")
    paths = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))
    result = []
    for p in paths:
        cid = int(os.path.splitext(os.path.basename(p))[0])
        result.append((cid, p))
    return result


# ============================================================
# Phase 1: SigLIP
# ============================================================
@torch.no_grad()
def extract_siglip_shard(clip_list, captions, siglip_model, processor, args):
    """Extract keyframe_img_emb and scene_text_emb for a shard of clips."""
    from PIL import Image
    keyframe_img_embs = []
    scene_text_embs = []
    clip_ids = []

    for cid, vpath in tqdm(clip_list, desc="SigLIP", leave=False):
        frames = load_clip_frames(vpath, args.num_frames)
        kf = Image.fromarray(frames[args.keyframe_index])

        # Image embed
        img_inputs = processor(images=kf, return_tensors="pt", padding=True)
        img_inputs = {k: v.to(args.device) for k, v in img_inputs.items() if isinstance(v, torch.Tensor)}
        img_emb = siglip_model.get_image_features(**img_inputs)
        img_emb = F.normalize(img_emb, dim=-1).cpu().squeeze(0)

        # Text embed
        caption = captions.get(cid, "A video clip.")
        txt_inputs = processor(text=caption, return_tensors="pt", padding=True, truncation=True, max_length=64)
        txt_inputs = {k: v.to(args.device) for k, v in txt_inputs.items() if isinstance(v, torch.Tensor)}
        txt_emb = siglip_model.get_text_features(**txt_inputs)
        txt_emb = F.normalize(txt_emb, dim=-1).cpu().squeeze(0)

        keyframe_img_embs.append(img_emb)
        scene_text_embs.append(txt_emb)
        clip_ids.append(cid)

    return {
        "clip_ids": clip_ids,
        "keyframe_img_emb": torch.stack(keyframe_img_embs),
        "scene_text_emb": torch.stack(scene_text_embs),
    }


# ============================================================
# Phase 2: VAE
# ============================================================
@torch.no_grad()
def extract_vae_shard(clip_list, vae_model, scale_factor, args):
    """Extract keyframe_vae_latent and structure_latent for a shard."""
    from torchvision.transforms.functional import resize, center_crop

    keyframe_vae_latents = []
    structure_latents = []
    clip_ids = []

    for cid, vpath in tqdm(clip_list, desc="VAE", leave=False):
        frames = load_clip_frames(vpath, args.num_frames)

        # Keyframe: first frame
        kf = torch.from_numpy(frames[args.keyframe_index]).permute(2, 0, 1).float()  # (3, H, W)
        kf = resize(kf, [args.video_size[0], args.video_size[1]], antialias=True)
        kf = (kf / 127.5 - 1.0).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W) for 3D VAE

        kf = kf.to(args.device)
        with torch.amp.autocast("cuda"):
            kf_latent = vae_model.encode(kf)  # returns tensor directly, already sampled
            kf_latent = kf_latent * scale_factor  # (1, C, 1, h, w)
        kf_latent = kf_latent.cpu().squeeze(0).squeeze(1)  # (C, h, w)

        # Structure latent = same as keyframe VAE latent (spec section 2.3)
        # They share the same latent but semantically serve different purposes
        keyframe_vae_latents.append(kf_latent)
        structure_latents.append(kf_latent.clone())
        clip_ids.append(cid)

    return {
        "clip_ids": clip_ids,
        "keyframe_vae_latent": torch.stack(keyframe_vae_latents),
        "structure_latent": torch.stack(structure_latents),
    }


# ============================================================
# Phase 3: RAFT optical flow
# ============================================================
@torch.no_grad()
def extract_raft_shard(clip_list, raft_model, args):
    """Extract flow_mag, flow_token, ofs_score for a shard."""
    flow_mags = []
    flow_tokens = []
    ofs_scores = []
    clip_ids = []

    for cid, vpath in tqdm(clip_list, desc="RAFT", leave=False):
        frames = load_clip_frames(vpath, args.num_frames)
        T = frames.shape[0]

        if T < 2:
            # Degenerate: single frame clip
            flow_mags.append(torch.tensor(0.0))
            flow_tokens.append(torch.zeros(1))
            ofs_scores.append(torch.tensor(0.0))
            clip_ids.append(cid)
            continue

        # Convert to tensor (T, 3, H, W) float, resize to 520x960 for RAFT
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
        # RAFT expects specific input size, resize to multiples of 8
        h, w = 520, 960
        frames_t = F.interpolate(frames_t, size=(h, w), mode="bilinear", align_corners=False)

        # Process adjacent frame pairs in batches
        all_flow_mags = []
        batch_size = 8
        for i in range(0, T - 1, batch_size):
            end = min(i + batch_size, T - 1)
            img1 = frames_t[i:end].to(args.device)
            img2 = frames_t[i+1:end+1].to(args.device)
            flows = raft_model(img1, img2, num_flow_updates=args.raft_iters)
            flow = flows[-1]  # (B, 2, H, W) last iteration
            mag = torch.norm(flow, dim=1)  # (B, H, W)
            all_flow_mags.append(mag.cpu())

        all_flow_mags = torch.cat(all_flow_mags, dim=0)  # (T-1, H, W)

        # flow_mag: mean magnitude per frame pair, then mean across pairs
        per_pair_mag = all_flow_mags.mean(dim=(1, 2))  # (T-1,)
        flow_mag_mean = per_pair_mag.mean()

        # flow_token: patch-pooled mean flow magnitude (coarse spatial summary)
        # Average across time first, then patch-pool
        mean_flow_map = all_flow_mags.mean(dim=0)  # (H, W)
        ph, pw = args.flow_patch_size, args.flow_patch_size
        nh, nw = h // ph, w // pw
        patches = mean_flow_map[:nh*ph, :nw*pw].reshape(nh, ph, nw, pw).mean(dim=(1, 3))  # (nh, nw)
        flow_token = patches.flatten()  # (nh*nw,)

        # ofs_score: optical flow stability = std of per-pair flow magnitudes
        # High std → temporally unstable motion; Low std → smooth/consistent motion
        ofs = per_pair_mag.std() if len(per_pair_mag) > 1 else torch.tensor(0.0)

        flow_mags.append(flow_mag_mean)
        flow_tokens.append(flow_token)
        ofs_scores.append(ofs)
        clip_ids.append(cid)

    # Compute dyn_label based on OFS percentile threshold
    ofs_tensor = torch.tensor([s.item() for s in ofs_scores])
    threshold = torch.quantile(ofs_tensor, args.ofs_percentile / 100.0)
    dyn_labels = (ofs_tensor >= threshold).long()  # 0=slow, 1=fast

    return {
        "clip_ids": clip_ids,
        "flow_mag": torch.stack(flow_mags) if flow_mags else torch.tensor([]),
        "flow_token": torch.stack(flow_tokens) if flow_tokens else torch.tensor([]),
        "ofs_score": ofs_tensor,
        "dyn_label": dyn_labels,
        "ofs_threshold": threshold.item(),
    }


# ============================================================
# Model loading helpers
# ============================================================
def load_siglip(device):
    from local_config import get_paths
    from transformers import AutoModel, AutoProcessor
    path = get_paths()["siglip2"]
    model = AutoModel.from_pretrained(path).to(device).eval()
    processor = AutoProcessor.from_pretrained(path)
    return model, processor


def load_vae(device):
    from local_config import get_paths
    from omegaconf import OmegaConf

    from vae_modules.autoencoder import VideoAutoencoderInferenceWrapper

    vae_path = get_paths()["vae"]
    vae_config = OmegaConf.create({
        "cp_size": 1,  # single-process context parallel
        "ckpt_path": vae_path,
        "ignore_keys": ["loss"],
        "loss_config": {"target": "torch.nn.Identity"},
        "regularizer_config": {"target": "vae_modules.regularizers.DiagonalGaussianRegularizer"},
        "encoder_config": {
            "target": "vae_modules.cp_enc_dec.ContextParallelEncoder3D",
            "params": {
                "double_z": True, "z_channels": 16, "resolution": 256,
                "in_channels": 3, "out_ch": 3, "ch": 128,
                "ch_mult": [1, 2, 2, 4], "attn_resolutions": [],
                "num_res_blocks": 3, "dropout": 0.0, "gather_norm": True,
            }
        },
        "decoder_config": {
            "target": "vae_modules.cp_enc_dec.ContextParallelDecoder3D",
            "params": {
                "double_z": True, "z_channels": 16, "resolution": 256,
                "in_channels": 3, "out_ch": 3, "ch": 128,
                "ch_mult": [1, 2, 2, 4], "attn_resolutions": [],
                "num_res_blocks": 3, "dropout": 0.0, "gather_norm": False,
            }
        },
    })
    model = VideoAutoencoderInferenceWrapper(**vae_config)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    scale_factor = 0.7
    return model, scale_factor


def load_raft(device):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ============================================================
# Metadata
# ============================================================
def save_metadata(args, total_clips, shard_count):
    import yaml as _yaml
    meta_dir = os.path.join(args.output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    version_info = {
        "target_version": "v1",
        "keyframe_rule": f"frame_index={args.keyframe_index} (0=first frame)",
        "image_encoder_name": "SigLIP2-so400m-patch14-384",
        "text_encoder_name": "SigLIP2-so400m-patch14-384 (text head)",
        "vae_name": "CogVideoX-5B 3D-VAE",
        "flow_model_name": "RAFT-Large (torchvision)",
        "ofs_threshold_rule": f"percentile={args.ofs_percentile}% within shard",
        "video_fps_used": 8,
        "clip_seconds": 4,
        "num_frames": args.num_frames,
        "image_resolution": args.video_size,
        "flow_patch_size": args.flow_patch_size,
        "raft_iterations": args.raft_iters,
        "shard_size": args.shard_size,
        "total_clips": total_clips,
        "total_shards": shard_count,
        "creation_time": datetime.now().isoformat(),
    }
    with open(os.path.join(meta_dir, "version_info.json"), "w") as f:
        json.dump(version_info, f, indent=2)

    config = {
        "phase": args.phase,
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "device": args.device,
        "shard_size": args.shard_size,
        "keyframe_index": args.keyframe_index,
        "num_frames": args.num_frames,
        "video_size": args.video_size,
        "raft_iters": args.raft_iters,
        "flow_patch_size": args.flow_patch_size,
        "ofs_percentile": args.ofs_percentile,
    }
    with open(os.path.join(meta_dir, "extraction_config.yaml"), "w") as f:
        _yaml.dump(config, f, default_flow_style=False)

    print(f"Metadata saved to {meta_dir}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "shards"), exist_ok=True)

    # Initialize context parallel for standalone VAE use
    if args.phase in ("vae", "all"):
        # Init torch.distributed (needed by some VAE internals)
        if not torch.distributed.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(29500 + os.getpid() % 1000)
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

        # Initialize context parallel in BOTH modules that define it
        cp_group = torch.distributed.new_group([0])

        import vae_modules.utils as _vae_utils
        if not _vae_utils.is_context_parallel_initialized():
            _vae_utils._CONTEXT_PARALLEL_SIZE = 1
            _vae_utils._CONTEXT_PARALLEL_GROUP = cp_group

        import sgm.util as _sgm_util
        if not _sgm_util.is_context_parallel_initialized():
            _sgm_util._CONTEXT_PARALLEL_SIZE = 1
            _sgm_util._CONTEXT_PARALLEL_GROUP = cp_group

        print("Initialized context parallel for standalone VAE")

    # Discover all clips
    all_clips = get_all_clip_paths(args.dataset_root)
    total_clips = len(all_clips)
    print(f"Found {total_clips} clips")

    # Compute shards
    num_shards = math.ceil(total_clips / args.shard_size)
    end_shard = args.end_shard if args.end_shard > 0 else num_shards
    print(f"Total shards: {num_shards}, processing [{args.start_shard}:{end_shard})")

    # Load captions (needed for SigLIP)
    captions = load_captions(args.dataset_root) if args.phase in ("siglip", "all") else {}

    # Load models per phase
    siglip_model, siglip_proc = (None, None)
    vae_model, scale_factor = (None, None)
    raft_model = None

    phases = [args.phase] if args.phase != "all" else ["siglip", "vae", "raft"]

    for phase in phases:
        print(f"\n{'='*60}")
        print(f"Phase: {phase.upper()}")
        print(f"{'='*60}")

        # Lazy load models
        if phase == "siglip" and siglip_model is None:
            print("Loading SigLIP2...")
            siglip_model, siglip_proc = load_siglip(args.device)
        elif phase == "vae" and vae_model is None:
            print("Loading CogVideoX VAE...")
            vae_model, scale_factor = load_vae(args.device)
        elif phase == "raft" and raft_model is None:
            print("Loading RAFT-Large...")
            raft_model = load_raft(args.device)

        for shard_idx in range(args.start_shard, end_shard):
            start = shard_idx * args.shard_size
            end = min(start + args.shard_size, total_clips)
            shard_clips = all_clips[start:end]
            shard_name = f"clips_{start:04d}_{end-1:04d}"
            shard_path = os.path.join(args.output_dir, "shards", f"{shard_name}.pt")

            # Check if shard already has this phase's data
            if os.path.exists(shard_path) and not args.force_rebuild:
                existing = torch.load(shard_path, map_location="cpu", weights_only=False)
                if phase == "siglip" and "keyframe_img_emb" in existing:
                    print(f"  Shard {shard_name}: SigLIP already done, skip")
                    continue
                if phase == "vae" and "keyframe_vae_latent" in existing:
                    print(f"  Shard {shard_name}: VAE already done, skip")
                    continue
                if phase == "raft" and "flow_mag" in existing:
                    print(f"  Shard {shard_name}: RAFT already done, skip")
                    continue

            print(f"  Processing shard {shard_name} ({len(shard_clips)} clips)...")

            # Extract
            if phase == "siglip":
                data = extract_siglip_shard(shard_clips, captions, siglip_model, siglip_proc, args)
            elif phase == "vae":
                data = extract_vae_shard(shard_clips, vae_model, scale_factor, args)
            elif phase == "raft":
                data = extract_raft_shard(shard_clips, raft_model, args)

            # Merge with existing shard if present
            if os.path.exists(shard_path):
                existing = torch.load(shard_path, map_location="cpu", weights_only=False)
                existing.update(data)
                data = existing

            torch.save(data, shard_path)
            print(f"  Saved {shard_path}")

        # Free model memory between phases
        if phase == "siglip":
            del siglip_model, siglip_proc
            siglip_model, siglip_proc = None, None
        elif phase == "vae":
            del vae_model
            vae_model = None
        elif phase == "raft":
            del raft_model
            raft_model = None
        torch.cuda.empty_cache()

    # Save metadata
    save_metadata(args, total_clips, num_shards)
    print(f"\nDone! All targets saved to {args.output_dir}")


if __name__ == "__main__":
    main()
