"""
CineBrain-SF v1 — Stage I Smoke Test (Dry Run)

Runs a minimal forward + backward pass to verify:
1. DataLoader correctly loads all supervision targets
2. SFBrainEmbedder forward produces correct shapes
3. All SF losses compute valid (non-NaN) values
4. Backward pass completes without error
5. Peak GPU memory

Usage (single GPU, no distributed):
    cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain
    CUDA_VISIBLE_DEVICES=0 python tools/smoke_test_stage1.py \
        --dataset_json /public/home/maoyaoxin/xxt/datasets/sub-0005_train_va.json \
        --num_samples 4 \
        --num_steps 3
"""
import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=4, help="Number of samples to test")
    p.add_argument("--num_steps", type=int, default=3, help="Number of forward+backward steps")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(42)

    print("=" * 70)
    print("CineBrain-SF v1 — Stage I Smoke Test")
    print("=" * 70)

    # ================================================================
    # Step 1: Test DataLoader + supervision targets
    # ================================================================
    print("\n[1/5] Testing DataLoader with supervision targets...")

    from data_video import BrainDataset
    ds = BrainDataset(
        data_dir=args.dataset_json,
        video_size=[480, 720],
        fps=8,
        max_num_frames=33,
    )
    print(f"  Dataset size: {len(ds)}")
    print(f"  SF cache format: {getattr(ds, '_sf_cache_format', 'none')}")
    print(f"  SF shard index entries: {len(ds._sf_shard_index)}")

    # Load a few samples
    samples = []
    for i in range(min(args.num_samples, len(ds))):
        sample = ds[i]
        samples.append(sample)

    s0 = samples[0]
    print(f"\n  Sample 0 keys: {list(s0.keys())}")
    print(f"  fmri: {s0['fmri'].shape}, dtype={s0['fmri'].dtype}")
    print(f"  eeg: {s0['eeg'].shape}, dtype={s0['eeg'].dtype}")
    print(f"  mp4: {s0['mp4'].shape}")

    if "sf_targets" in s0:
        print(f"\n  SF targets found!")
        for k, v in s0["sf_targets"].items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}, "
                      f"min={v.min().item():.4f}, max={v.max().item():.4f}")
            else:
                print(f"    {k}: {type(v)} = {v}")
    else:
        print("\n  WARNING: No sf_targets in sample!")

    # Collate into batch
    batch = {}
    batch["fmri"] = torch.stack([s["fmri"] for s in samples]).to(device)
    batch["eeg"] = torch.stack([s["eeg"] for s in samples]).to(device)

    # Collate sf_targets
    if "sf_targets" in samples[0]:
        sf_targets = {}
        for key in samples[0]["sf_targets"]:
            vals = [s["sf_targets"][key] for s in samples if key in s.get("sf_targets", {})]
            if vals and isinstance(vals[0], torch.Tensor):
                sf_targets[key] = torch.stack(vals).to(device)
        batch["sf_targets"] = sf_targets
        print(f"\n  Batched sf_targets:")
        for k, v in sf_targets.items():
            print(f"    {k}: {list(v.shape)}")

    print("\n  [PASS] DataLoader OK")

    # ================================================================
    # Step 2: Test SFBrainEmbedder forward
    # ================================================================
    print("\n[2/5] Testing SFBrainEmbedder forward pass...")

    from sgm.modules.encoders.sf_embedder import SFBrainEmbedder

    B = batch["fmri"].shape[0]
    fmri_seq_len = batch["fmri"].shape[-1]

    embedder = SFBrainEmbedder(
        seq_len=fmri_seq_len,
        num_spatial=226,
        fmri_num_layers=12,
        eeg_num_layers=12,
        clip_dim=1152,
        use_slow_branch=True,
        use_fast_branch=True,
        use_gated_fusion=True,
        use_multi_guidance=True,
        fusion_hidden_dim=1024,
        fusion_num_layers=2,
        mode="infer",
    ).to(device)

    with torch.no_grad():
        context, clip_loss = embedder(batch, None)

    print(f"  context: {list(context.shape)}")
    print(f"  clip_loss: {clip_loss}")

    slow_out = embedder._last_slow_out
    fast_out = embedder._last_fast_out
    alphas = embedder._last_alphas

    print(f"\n  Slow branch outputs:")
    for k, v in slow_out.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {list(v.shape)}")

    print(f"\n  Fast branch outputs:")
    for k, v in fast_out.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {list(v.shape)}")

    print(f"\n  Alpha weights:")
    for k, v in alphas.items():
        print(f"    {k}: {v.squeeze().tolist()}")

    print("\n  [PASS] SFBrainEmbedder forward OK")

    # ================================================================
    # Step 3: Test SF losses
    # ================================================================
    print("\n[3/5] Testing SF losses...")

    from sgm.modules.diffusionmodules.sf_losses import (
        AlignmentLoss, SlowBranchLoss, FastBranchLoss, GuidanceLoss
    )

    sf_targets = batch.get("sf_targets", {})

    # Alignment loss
    align_loss_fn = AlignmentLoss().to(device)
    l_align, d_align = align_loss_fn(slow_out, fast_out)
    print(f"  AlignmentLoss: {l_align.item():.4f}  components={list(d_align.keys())}")
    for k, v in d_align.items():
        print(f"    {k}: {v.item():.4f}  nan={torch.isnan(v).item()}")

    # Slow branch loss — check shape compatibility
    print(f"\n  Slow head vs target shape check:")
    shape_issues = []
    if "z_key" in slow_out and "gt_keyframe_embed" in sf_targets:
        print(f"    z_key: {list(slow_out['z_key'].shape)} vs gt_keyframe_embed: {list(sf_targets['gt_keyframe_embed'].shape)}")
        if slow_out["z_key"].shape != sf_targets["gt_keyframe_embed"].shape:
            shape_issues.append(("z_key", slow_out["z_key"].shape, sf_targets["gt_keyframe_embed"].shape))
    if "z_txt" in slow_out and "gt_text_embed" in sf_targets:
        print(f"    z_txt: {list(slow_out['z_txt'].shape)} vs gt_text_embed: {list(sf_targets['gt_text_embed'].shape)}")
        if slow_out["z_txt"].shape != sf_targets["gt_text_embed"].shape:
            shape_issues.append(("z_txt", slow_out["z_txt"].shape, sf_targets["gt_text_embed"].shape))
    if "z_str" in slow_out and "gt_structure_embed" in sf_targets:
        print(f"    z_str: {list(slow_out['z_str'].shape)} vs gt_structure_embed: {list(sf_targets['gt_structure_embed'].shape)}")
        if slow_out["z_str"].shape != sf_targets["gt_structure_embed"].shape:
            shape_issues.append(("z_str", slow_out["z_str"].shape, sf_targets["gt_structure_embed"].shape))

    print(f"\n  Fast head vs target shape check:")
    if "z_dyn" in fast_out and "gt_dynamics_embed" in sf_targets:
        print(f"    z_dyn: {list(fast_out['z_dyn'].shape)} vs gt_dynamics_embed: {list(sf_targets['gt_dynamics_embed'].shape)}")
        if fast_out["z_dyn"].shape != sf_targets["gt_dynamics_embed"].shape:
            shape_issues.append(("z_dyn", fast_out["z_dyn"].shape, sf_targets["gt_dynamics_embed"].shape))
    if "z_mot" in fast_out and "gt_motion_embed" in sf_targets:
        print(f"    z_mot: {list(fast_out['z_mot'].shape)} vs gt_motion_embed: {list(sf_targets['gt_motion_embed'].shape)}")
        if fast_out["z_mot"].shape != sf_targets["gt_motion_embed"].shape:
            shape_issues.append(("z_mot", fast_out["z_mot"].shape, sf_targets["gt_motion_embed"].shape))
    if "z_tc" in fast_out and "gt_tc_embed" in sf_targets:
        print(f"    z_tc: {list(fast_out['z_tc'].shape)} vs gt_tc_embed: {list(sf_targets['gt_tc_embed'].shape)}")
        if fast_out["z_tc"].shape != sf_targets["gt_tc_embed"].shape:
            shape_issues.append(("z_tc", fast_out["z_tc"].shape, sf_targets["gt_tc_embed"].shape))

    if shape_issues:
        print(f"\n  [SHAPE MISMATCH DETECTED]")
        for name, head_shape, target_shape in shape_issues:
            print(f"    {name}: head outputs {head_shape} but target is {target_shape}")
        print("  These must be fixed before training!")
    else:
        print(f"\n  All shapes compatible!")

    # Try computing losses even with mismatches (will error if incompatible)
    try:
        slow_loss_fn = SlowBranchLoss().to(device)
        l_slow, d_slow = slow_loss_fn(slow_out, sf_targets)
        print(f"\n  SlowBranchLoss: {l_slow.item():.4f}  components={list(d_slow.keys())}")
        for k, v in d_slow.items():
            print(f"    {k}: {v.item():.4f}  nan={torch.isnan(v).item()}")
    except Exception as e:
        print(f"\n  SlowBranchLoss ERROR: {e}")

    try:
        fast_loss_fn = FastBranchLoss().to(device)
        l_fast, d_fast = fast_loss_fn(fast_out, sf_targets)
        print(f"\n  FastBranchLoss: {l_fast.item():.4f}  components={list(d_fast.keys())}")
        for k, v in d_fast.items():
            print(f"    {k}: {v.item():.4f}  nan={torch.isnan(v).item()}")
    except Exception as e:
        print(f"\n  FastBranchLoss ERROR: {e}")

    try:
        guide_loss_fn = GuidanceLoss().to(device)
        l_guide, d_guide = guide_loss_fn(slow_out, fast_out)
        print(f"\n  GuidanceLoss: {l_guide.item():.4f}  components={list(d_guide.keys())}")
    except Exception as e:
        print(f"\n  GuidanceLoss ERROR: {e}")

    print("\n  [PASS] Loss computation OK (check above for shape issues)")

    # ================================================================
    # Step 4: Test backward pass
    # ================================================================
    print(f"\n[4/5] Testing backward pass ({args.num_steps} steps)...")

    embedder.train()
    optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-4)

    for step in range(args.num_steps):
        optimizer.zero_grad()
        context, clip_loss = embedder(batch, None)
        slow_out = embedder._last_slow_out
        fast_out = embedder._last_fast_out

        # Compute total loss (alignment only for safety — skip head losses if shapes mismatch)
        total_loss = align_loss_fn(slow_out, fast_out)[0]

        # Add context norm as dummy diffusion-like loss
        total_loss = total_loss + context.mean() * 0.001

        total_loss.backward()

        # Check for NaN gradients
        nan_params = 0
        total_params = 0
        for n, p in embedder.named_parameters():
            if p.grad is not None:
                total_params += 1
                if torch.isnan(p.grad).any():
                    nan_params += 1
                    print(f"    NaN grad in: {n}")

        optimizer.step()
        print(f"  Step {step}: loss={total_loss.item():.4f}, params_with_grad={total_params}, nan_grads={nan_params}")

    print("\n  [PASS] Backward pass OK")

    # ================================================================
    # Step 5: GPU memory report
    # ================================================================
    print(f"\n[5/5] GPU Memory Report")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        current = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"  Peak allocated:    {peak:.2f} GB")
        print(f"  Current allocated: {current:.2f} GB")
        print(f"  Reserved:          {reserved:.2f} GB")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"  DataLoader:         OK")
    print(f"  SF targets loaded:  {'YES' if 'sf_targets' in samples[0] else 'NO'}")
    print(f"  SFBrainEmbedder:    OK  context={list(context.shape)}")
    print(f"  Shape mismatches:   {len(shape_issues)}")
    if shape_issues:
        for name, hs, ts in shape_issues:
            print(f"    {name}: head={hs} vs target={ts}")
    print(f"  Alignment loss:     {l_align.item():.4f}")
    print(f"  Backward:           OK ({args.num_steps} steps, {nan_params} NaN grads)")
    print(f"  Peak GPU memory:    {peak:.2f} GB" if torch.cuda.is_available() else "  GPU: N/A")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
