#!/usr/bin/env python
"""
Verify: Can gradient pass through a frozen network back to GatedFusion?

Uses a simple frozen proxy network instead of full DiT to test the principle.
Also tests with context directly to confirm GatedFusion is reachable.

Usage:
    cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain
    CUDA_VISIBLE_DEVICES=0 python tools/verify_gradient_passthrough.py \
        --ckpt ckpts_5b/sf_v1_p1_full_v2-04-03-13-44/3000/mp_rank_00_model_states.pt
"""
import os, sys, yaml, argparse, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-config", default="configs/sf_v1/cinebrain_sf_v1_model.yaml")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


class FrozenProxyNetwork(nn.Module):
    """Simulates DiT cross-attention: consumes context via linear projection.
    All params frozen, but gradient should still flow through to context."""
    def __init__(self, context_dim=4096, hidden_dim=1024):
        super().__init__()
        self.cross_attn_proj = nn.Linear(context_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Freeze all params
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, context):
        # Simulate: context → cross-attn → output
        h = self.cross_attn_proj(context.float())  # (B, 226, hidden)
        h = F.gelu(h)
        h = self.out_proj(h)
        return h.mean(dim=(1, 2))  # scalar-ish output for loss


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("GRADIENT PASSTHROUGH VERIFICATION")
    print("=" * 60)

    # ─── Build SFBrainEmbedder ───
    from sgm.modules.encoders.sf_embedder import SFBrainEmbedder
    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)
    emb_cfg = cfg["model"]["conditioner_config"]["params"]["emb_models"][0]["params"]
    embedder = SFBrainEmbedder(**emb_cfg).to(device).to(torch.bfloat16)
    embedder.mode = "infer"

    # ─── Load checkpoint ───
    print("\nLoading checkpoint...")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = state.get("module", state)
    emb_prefix = "conditioner.embedders.0."
    emb_state = {k[len(emb_prefix):]: v for k, v in model_state.items() if k.startswith(emb_prefix)}
    missing, unexpected = embedder.load_state_dict(emb_state, strict=False)
    print(f"  Loaded {len(emb_state)} keys")
    del state, model_state  # free memory

    # ─── Configure freeze ───
    print("\nConfiguring freeze strategy...")

    # Freeze branches
    for name, p in embedder.named_parameters():
        if any(x in name for x in ["slow_branch", "fast_branch", "fmri_encoder", "eeg_encoder"]):
            p.requires_grad_(False)

    # Ensure GatedFusion + MultiGuidanceAdapter are trainable
    fusion_count = 0
    guidance_count = 0
    for name, p in embedder.named_parameters():
        if "gated_fusion" in name:
            p.requires_grad_(True)
            fusion_count += 1
        if "guidance_adapter" in name:
            p.requires_grad_(True)
            guidance_count += 1

    trainable = sum(p.numel() for p in embedder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in embedder.parameters())
    print(f"  GatedFusion params: {fusion_count}")
    print(f"  MultiGuidanceAdapter params: {guidance_count}")
    print(f"  Trainable: {trainable:,} / {total:,}")

    # Build proxy frozen network
    proxy = FrozenProxyNetwork(context_dim=4096).to(device)
    print(f"  Proxy network: {sum(p.numel() for p in proxy.parameters()):,} params (all frozen)")

    # ─── Build test batch ───
    print("\nLoading test sample...")
    from data_video import BrainDataset
    from local_config import get_paths
    paths = get_paths()
    ds = BrainDataset(
        data_dir=os.path.join(paths["dataset_root"], "sub-0005_test_va.json"),
        video_size=[480, 720], fps=8, max_num_frames=33, skip_frms_num=0,
        sf_target_keys=["keyframe_img_emb", "scene_text_emb", "temporal_frame_embs", "flow_mag_traj"],
    )
    item = ds[0]
    batch = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device).to(torch.bfloat16) if v.is_floating_point() else v.unsqueeze(0).to(device)
        elif isinstance(v, dict):
            batch[k] = {dk: dv.unsqueeze(0).to(device).to(torch.bfloat16) if dv.is_floating_point() else dv.unsqueeze(0).to(device)
                        for dk, dv in v.items() if isinstance(dv, torch.Tensor)}
        else:
            batch[k] = v

    # ═══════════════════════════════════════════════════════
    # TEST 1: Direct loss on context (baseline — should always work)
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("TEST 1: Direct MSE loss on context")
    print("─" * 60)

    embedder.zero_grad()
    context, _ = embedder(batch, siglip_model=None)
    print(f"  context: shape={context.shape}, requires_grad={context.requires_grad}")

    target = torch.randn_like(context)
    loss1 = F.mse_loss(context.float(), target.float())
    loss1.backward()

    check_grads("GatedFusion", embedder.gated_fusion)
    check_grads("MultiGuidanceAdapter", embedder.guidance_adapter)

    # ═══════════════════════════════════════════════════════
    # TEST 2: Loss through frozen proxy (simulates frozen DiT)
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("TEST 2: Loss through FROZEN proxy network (simulates DiT)")
    print("─" * 60)

    embedder.zero_grad()
    context2, _ = embedder(batch, siglip_model=None)

    # Pass context through frozen proxy
    proxy_output = proxy(context2)
    loss2 = proxy_output.float().pow(2).mean()  # simple scalar loss
    print(f"  proxy_output: shape={proxy_output.shape}")
    print(f"  loss2 = {loss2.item():.6f}")

    loss2.backward()

    check_grads("GatedFusion", embedder.gated_fusion)
    check_grads("MultiGuidanceAdapter", embedder.guidance_adapter)

    # Verify frozen modules
    print("\n" + "─" * 60)
    print("FROZEN MODULE VERIFICATION")
    print("─" * 60)

    branch_has_grad = False
    for name, p in embedder.named_parameters():
        if any(x in name for x in ["slow_branch", "fast_branch"]) and p.grad is not None:
            if p.grad.abs().sum() > 0:
                branch_has_grad = True
                break

    proxy_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in proxy.parameters())

    print(f"  Branches have gradient: {'✗ YES (bad!)' if branch_has_grad else '✓ No (correct)'}")
    print(f"  Proxy has gradient:     {'✗ YES (bad!)' if proxy_has_grad else '✓ No (correct)'}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


def check_grads(name, module):
    """Check and report gradient statistics for a module."""
    grads = []
    no_grad_count = 0
    for pname, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None:
            norm = p.grad.float().norm().item()
            mean = p.grad.float().abs().mean().item()
            grads.append((pname, norm, mean))
        else:
            no_grad_count += 1

    if grads:
        total_norm = sum(g[1] ** 2 for g in grads) ** 0.5
        print(f"\n  [{name}] ✓ {len(grads)} params have gradients (total norm: {total_norm:.2e})")
        for pname, norm, mean in grads[:5]:
            print(f"    {pname:45s}  norm={norm:.2e}  mean_abs={mean:.2e}")
        if len(grads) > 5:
            print(f"    ... and {len(grads) - 5} more")
    else:
        print(f"\n  [{name}] ✗ FAIL: 0 params have gradients! ({no_grad_count} trainable without grad)")


if __name__ == "__main__":
    main()
