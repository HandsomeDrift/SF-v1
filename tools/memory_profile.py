"""
CineBrain-SF v1 — GPU Memory Profiling (3 configs)

Config A: Baseline CineSync (BrainmbedderCLIP only)
Config B: SF v1 branches + fusion + guidance (no decoder)
Config C: Full model (SF branches + CogVideoX-5B DiT)

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/memory_profile.py \
        --dataset_json /public/home/maoyaoxin/xxt/datasets/sub-0005_train_va.json \
        --batch_size 1
"""
import argparse
import gc
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_dit", action="store_true", help="Skip Config C (DiT) if it OOMs")
    return p.parse_args()


def reset_memory(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def report_memory(device, label):
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    current = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    print(f"  [{label}]")
    print(f"    Peak allocated:    {peak:.2f} GB")
    print(f"    Current allocated: {current:.2f} GB")
    print(f"    Reserved:          {reserved:.2f} GB")
    return peak


def load_batch(dataset_json, batch_size, device):
    """Load a small batch for profiling."""
    from data_video import BrainDataset
    ds = BrainDataset(data_dir=dataset_json, video_size=[480, 720], fps=8, max_num_frames=33)
    samples = [ds[i] for i in range(batch_size)]

    batch = {
        "fmri": torch.stack([s["fmri"] for s in samples]).to(device),
        "eeg": torch.stack([s["eeg"] for s in samples]).to(device),
        "mp4": torch.stack([s["mp4"] for s in samples]).to(device),
    }
    if "sf_targets" in samples[0]:
        sf = {}
        for k in samples[0]["sf_targets"]:
            vals = [s["sf_targets"][k] for s in samples]
            if isinstance(vals[0], torch.Tensor):
                sf[k] = torch.stack(vals).to(device)
        batch["sf_targets"] = sf

    return batch, ds


def main():
    args = parse_args()
    device = args.device
    B = args.batch_size

    print("=" * 70)
    print(f"CineBrain-SF v1 — GPU Memory Profiling (batch_size={B})")
    print("=" * 70)

    # Load batch once
    print("\nLoading data...")
    batch, ds = load_batch(args.dataset_json, B, device)
    fmri_seq_len = batch["fmri"].shape[-1]
    print(f"  fMRI seq_len: {fmri_seq_len}")
    print(f"  Batch loaded to GPU")

    results = {}

    # ================================================================
    # Config A: Baseline CineSync
    # ================================================================
    print(f"\n{'='*70}")
    print("Config A: Baseline CineSync (BrainmbedderCLIP)")
    print(f"{'='*70}")

    reset_memory(device)
    from sgm.modules.encoders.modules import BrainmbedderCLIP

    baseline = BrainmbedderCLIP(
        seq_len=fmri_seq_len, num_spatial=226, num_layers=12
    ).to(device)

    n_params_a = sum(p.numel() for p in baseline.parameters()) / 1e6
    print(f"  Parameters: {n_params_a:.1f}M")
    report_memory(device, "after model load")

    with torch.no_grad():
        ctx_a, _ = baseline(batch, None)
    peak_a_fwd = report_memory(device, "after forward")

    # backward
    baseline.train()
    ctx_a, _ = baseline(batch, None)
    loss_a = ctx_a.mean()
    loss_a.backward()
    peak_a_bwd = report_memory(device, "after backward")

    results["A_baseline"] = {"params_M": n_params_a, "fwd_peak_GB": peak_a_fwd, "bwd_peak_GB": peak_a_bwd}

    del baseline, ctx_a, loss_a
    reset_memory(device)

    # ================================================================
    # Config B: SF v1 branches only (no DiT decoder)
    # ================================================================
    print(f"\n{'='*70}")
    print("Config B: SF v1 (SlowBranch + FastBranch + Fusion + Guidance)")
    print(f"{'='*70}")

    reset_memory(device)
    from sgm.modules.encoders.sf_embedder import SFBrainEmbedder

    sf_emb = SFBrainEmbedder(
        seq_len=fmri_seq_len, num_spatial=226,
        fmri_num_layers=12, eeg_num_layers=12, clip_dim=1152,
        use_slow_branch=True, use_fast_branch=True,
        use_gated_fusion=True, use_multi_guidance=True,
        fusion_hidden_dim=2048, fusion_num_layers=4,
        mode="infer",
    ).to(device)

    n_params_b = sum(p.numel() for p in sf_emb.parameters()) / 1e6
    print(f"  Parameters: {n_params_b:.1f}M")
    print(f"  Delta vs baseline: +{n_params_b - n_params_a:.1f}M")
    report_memory(device, "after model load")

    with torch.no_grad():
        ctx_b, _ = sf_emb(batch, None)
    peak_b_fwd = report_memory(device, "after forward")

    sf_emb.train()
    ctx_b, _ = sf_emb(batch, None)
    loss_b = ctx_b.mean()
    loss_b.backward()
    peak_b_bwd = report_memory(device, "after backward")

    results["B_sf_branches"] = {"params_M": n_params_b, "fwd_peak_GB": peak_b_fwd, "bwd_peak_GB": peak_b_bwd}

    del sf_emb, ctx_b, loss_b
    reset_memory(device)

    # ================================================================
    # Config C: Full model (SF + DiT decoder with LoRA)
    # ================================================================
    if not args.skip_dit:
        print(f"\n{'='*70}")
        print("Config C: Full Model (SF + CogVideoX-5B DiT + LoRA r=128)")
        print(f"{'='*70}")

        reset_memory(device)

        try:
            # Load DiT model standalone
            from sat.model.base_model import BaseModel
            import dit_video_concat_fmri as dit_module

            # Create DiT with same config as training
            from argparse import Namespace
            from omegaconf import OmegaConf

            transformer_args = Namespace(
                checkpoint_activations=True,
                vocab_size=1,
                max_sequence_length=64,
                layernorm_order="pre",
                skip_init=True,
                model_parallel_size=1,
                is_decoder=False,
            )

            module_configs = {
                "pos_embed_config": OmegaConf.create({
                    "target": "dit_video_concat_fmri.Rotary3DPositionEmbeddingMixin",
                    "params": {"hidden_size_head": 64, "text_length": 226},
                }),
                "patch_embed_config": OmegaConf.create({
                    "target": "dit_video_concat_fmri.ImagePatchEmbeddingMixin",
                    "params": {"text_hidden_size": 4096},
                }),
                "adaln_layer_config": OmegaConf.create({
                    "target": "dit_video_concat_fmri.AdaLNMixin",
                    "params": {"qk_ln": True},
                }),
                "final_layer_config": OmegaConf.create({
                    "target": "dit_video_concat_fmri.FinalLayerMixin",
                }),
                # Skip LoRA for profiling to isolate DiT base cost
                # "lora_config": OmegaConf.create({...}),
            }

            dit = dit_module.DiffusionTransformer(
                time_embed_dim=512,
                elementwise_affine=True,
                num_frames=49,
                time_compressed_rate=4,
                latent_width=90,
                latent_height=60,
                num_layers=42,
                patch_size=2,
                in_channels=16,
                out_channels=16,
                hidden_size=3072,
                adm_in_channels=256,
                num_attention_heads=48,
                transformer_args=transformer_args,
                modules=module_configs,
            ).to(device)

            n_params_dit = sum(p.numel() for p in dit.parameters()) / 1e6
            print(f"  DiT parameters: {n_params_dit:.1f}M")
            report_memory(device, "after DiT load")

            # Now load SF embedder too
            sf_emb2 = SFBrainEmbedder(
                seq_len=fmri_seq_len, num_spatial=226,
                fmri_num_layers=12, eeg_num_layers=12, clip_dim=1152,
                use_slow_branch=True, use_fast_branch=True,
                use_gated_fusion=True, use_multi_guidance=True,
                fusion_hidden_dim=2048, fusion_num_layers=4,
                mode="infer",
            ).to(device)

            n_params_c = n_params_dit + sum(p.numel() for p in sf_emb2.parameters()) / 1e6
            print(f"  Total parameters: {n_params_c:.1f}M")
            report_memory(device, "after SF + DiT load")

            # Forward: SF embedder → DiT
            with torch.no_grad():
                ctx_c, _ = sf_emb2(batch, None)
                # DiT forward needs: x (video latent), timesteps, context
                T_latent = 9  # 33 frames / time_compressed_rate=4 ≈ 9
                x = torch.randn(B, T_latent, 16, 60, 90, device=device)
                timesteps = torch.randint(0, 1000, (B,), device=device)
                # DiT expects context as kwargs
                dit_out = dit(x, timesteps=timesteps, context=ctx_c, y=None)

            peak_c_fwd = report_memory(device, "after full forward")

            results["C_full"] = {"params_M": n_params_c, "fwd_peak_GB": peak_c_fwd, "bwd_peak_GB": -1}
            print(f"\n  (Skipping backward for Config C to avoid OOM)")

            del dit, sf_emb2, ctx_c, x, dit_out

        except Exception as e:
            print(f"  Config C FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["C_full"] = {"params_M": -1, "fwd_peak_GB": -1, "bwd_peak_GB": -1, "error": str(e)}

        reset_memory(device)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("MEMORY PROFILING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'Params':>10s} {'Fwd Peak':>12s} {'Bwd Peak':>12s}")
    print("-" * 60)
    for name, r in results.items():
        params = f"{r['params_M']:.0f}M" if r['params_M'] > 0 else "ERR"
        fwd = f"{r['fwd_peak_GB']:.2f} GB" if r['fwd_peak_GB'] > 0 else "ERR"
        bwd = f"{r['bwd_peak_GB']:.2f} GB" if r['bwd_peak_GB'] > 0 else "N/A"
        print(f"  {name:<23s} {params:>10s} {fwd:>12s} {bwd:>12s}")

    print(f"\n  A800 80GB budget: 80.00 GB")
    if "C_full" in results and results["C_full"]["fwd_peak_GB"] > 0:
        headroom = 80.0 - results["C_full"]["fwd_peak_GB"]
        print(f"  Headroom (fwd only):  {headroom:.2f} GB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
