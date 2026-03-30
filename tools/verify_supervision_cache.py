"""
Verify supervision cache completeness, shapes, and report storage stats.

Usage:
    python tools/verify_supervision_cache.py \
        --cache_dir /public/home/maoyaoxin/xxt/datasets/supervision_cache/version_v1 \
        --total_clips 8100
"""
import argparse
import glob
import os
import json
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--total_clips", type=int, default=8100)
    args = p.parse_args()

    shards_dir = os.path.join(args.cache_dir, "shards")
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "*.pt")))

    print(f"Cache dir: {args.cache_dir}")
    print(f"Shard files found: {len(shard_files)}")
    print()

    # Collect stats
    total_clips_found = 0
    all_keys = set()
    shape_report = {}
    total_size = 0
    missing_fields = []

    expected_fields = [
        "clip_ids", "keyframe_img_emb", "scene_text_emb",
        "keyframe_vae_latent", "structure_latent",
        "flow_mag", "flow_token", "ofs_score", "dyn_label",
    ]

    for sf in shard_files:
        fsize = os.path.getsize(sf)
        total_size += fsize
        data = torch.load(sf, map_location="cpu", weights_only=False)
        n = len(data.get("clip_ids", []))
        total_clips_found += n
        all_keys.update(data.keys())

        for field in expected_fields:
            if field not in data:
                missing_fields.append((os.path.basename(sf), field))
            elif isinstance(data[field], torch.Tensor) and field not in shape_report:
                shape_report[field] = list(data[field].shape)

        if "clip_ids" in data and isinstance(data["clip_ids"], list) and n > 0:
            if "clip_ids" not in shape_report:
                shape_report["clip_ids"] = f"list[{n}]"

    print("=" * 60)
    print("COMPLETENESS CHECK")
    print("=" * 60)
    print(f"  Expected clips: {args.total_clips}")
    print(f"  Found clips:    {total_clips_found}")
    print(f"  Status:         {'PASS' if total_clips_found >= args.total_clips else 'FAIL'}")
    print()

    print("=" * 60)
    print("FIELD PRESENCE")
    print("=" * 60)
    for field in expected_fields:
        present = field in all_keys
        print(f"  {field:30s} {'PRESENT' if present else 'MISSING'}")
    print()

    if missing_fields:
        print("  Missing fields per shard:")
        for shard, field in missing_fields[:20]:
            print(f"    {shard}: {field}")
        if len(missing_fields) > 20:
            print(f"    ... and {len(missing_fields) - 20} more")
        print()

    print("=" * 60)
    print("SHAPE REPORT")
    print("=" * 60)
    for field, shape in sorted(shape_report.items()):
        print(f"  {field:30s} {shape}")
    print()

    print("=" * 60)
    print("STORAGE REPORT")
    print("=" * 60)
    print(f"  Total shards:    {len(shard_files)}")
    print(f"  Total size:      {total_size / 1e6:.1f} MB")
    print(f"  Avg shard size:  {total_size / max(len(shard_files), 1) / 1e6:.1f} MB")
    print()

    # Metadata check
    meta_dir = os.path.join(args.cache_dir, "metadata")
    print("=" * 60)
    print("METADATA CHECK")
    print("=" * 60)
    for fname in ["version_info.json", "extraction_config.yaml"]:
        fpath = os.path.join(meta_dir, fname)
        exists = os.path.exists(fpath)
        print(f"  {fname:35s} {'PRESENT' if exists else 'MISSING'}")
        if exists and fname.endswith(".json"):
            with open(fpath) as f:
                info = json.load(f)
            for k, v in info.items():
                print(f"    {k}: {v}")
    print()

    # Spot check
    if shard_files:
        print("=" * 60)
        print("SPOT CHECK (first shard, first clip)")
        print("=" * 60)
        data = torch.load(shard_files[0], map_location="cpu", weights_only=False)
        cid = data["clip_ids"][0] if "clip_ids" in data else "?"
        print(f"  clip_id: {cid}")
        for k, v in data.items():
            if k == "clip_ids":
                continue
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                sample = v[0]
                print(f"  {k}: shape={list(sample.shape)}, dtype={sample.dtype}, "
                      f"min={sample.min().item():.4f}, max={sample.max().item():.4f}, "
                      f"mean={sample.float().mean().item():.4f}")
            elif isinstance(v, (int, float)):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
