#!/usr/bin/env python
"""
P1 Temporal Dynamics Evaluation — 4 Assessment Criteria (HANDOFF.md §7.2)

Loads a P1 checkpoint (SFBrainEmbedder only), runs validation set forward pass,
and evaluates:
  1. Temporal learnability:  valid L_temporal_delta, per-frame cosine sim
  2. Flow trajectory:        Pearson correlation predicted vs GT flow_mag_traj
  3. Fast/Slow independence: cosine similarity distribution (should be < P0 baseline)
  4. Gating behavior:        alpha_mot vs clip dynamics correlation

Usage:
    cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate_p1.py \
        --ckpt ckpts_5b/sf_v1_p1_full_v2-04-03-13-44/3000/mp_rank_00_model_states.pt \
        --data-json sub-0005_test_va.json \
        --output eval_results/p1_v2_iter3000.json
"""
import os, sys, json, argparse, yaml
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to mp_rank_00_model_states.pt")
    p.add_argument("--data-json", default="sub-0005_test_va.json",
                   help="Validation json (relative to dataset_root)")
    p.add_argument("--model-config", default="configs/sf_v1/cinebrain_sf_v1_model.yaml")
    p.add_argument("--output", default="eval_results/p1_eval.json")
    p.add_argument("--max-samples", type=int, default=0, help="0=all samples")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build_embedder(model_config_path, device):
    """Build SFBrainEmbedder from config (no SAT/DeepSpeed needed)."""
    from sgm.modules.encoders.sf_embedder import SFBrainEmbedder

    with open(model_config_path) as f:
        cfg = yaml.safe_load(f)
    emb_cfg = cfg["model"]["conditioner_config"]["params"]["emb_models"][0]["params"]
    embedder = SFBrainEmbedder(**emb_cfg)
    embedder.to(device).to(torch.bfloat16)
    embedder.mode = "infer"  # skip CLIP loss
    embedder.eval()
    return embedder


def load_checkpoint(embedder, ckpt_path, device):
    """Load DeepSpeed checkpoint into SFBrainEmbedder."""
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # DeepSpeed wraps in "module" key
    model_state = state.get("module", state)

    # Filter to only SFBrainEmbedder keys (prefix: model.diffusion_model... or conditioner...)
    # The embedder is nested under the full model. Try direct load first.
    # In SAT, the conditioner embedder keys look like:
    #   model.conditioner.embedders.0.slow_branch.xxx
    # We need to strip prefix to match SFBrainEmbedder's state_dict keys
    prefix = "model.conditioner.embedders.0."
    embedder_state = {}
    for k, v in model_state.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            embedder_state[new_k] = v

    if not embedder_state:
        # Maybe the state_dict is already flat
        print("  No prefix match, trying alternative prefixes...")
        for alt_prefix in ["conditioner.embedders.0.", "embedders.0.", ""]:
            embedder_state = {k[len(alt_prefix):]: v for k, v in model_state.items()
                              if k.startswith(alt_prefix)} if alt_prefix else model_state
            if embedder_state:
                print(f"  Using prefix: '{alt_prefix}'")
                break

    missing, unexpected = embedder.load_state_dict(embedder_state, strict=False)
    print(f"  Loaded {len(embedder_state)} keys")
    if missing:
        # Filter out SigLIP keys (expected missing - not loaded in eval)
        real_missing = [k for k in missing if not k.startswith("siglip")]
        if real_missing:
            print(f"  WARNING: {len(real_missing)} missing keys: {real_missing[:5]}...")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")
    return embedder


def build_dataset(data_json, model_config_path):
    """Build BrainDataset for validation."""
    from data_video import BrainDataset
    from local_config import get_paths
    paths = get_paths()

    data_path = os.path.join(paths["dataset_root"], data_json)
    ds = BrainDataset(
        data_dir=data_path,
        video_size=[480, 720],
        fps=8,
        max_num_frames=33,
        skip_frms_num=0,
        sf_target_keys=["keyframe_img_emb", "scene_text_emb", "flow_token_pca",
                        "dyn_class_3", "motion_dir_8", "ofs_log_zscore",
                        "temporal_frame_embs", "flow_mag_traj"],
    )
    print(f"Validation dataset: {len(ds)} samples from {data_json}")
    return ds


def prepare_batch(item, device):
    """Convert single dataset item to batched tensors on device."""
    batch = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                batch[k] = v.unsqueeze(0).to(device).to(torch.bfloat16)
            else:
                batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, dict):
            batch[k] = {}
            for dk, dv in v.items():
                if isinstance(dv, torch.Tensor):
                    if dv.is_floating_point():
                        batch[k][dk] = dv.unsqueeze(0).to(device).to(torch.bfloat16)
                    else:
                        batch[k][dk] = dv.unsqueeze(0).to(device)
        else:
            batch[k] = v
    return batch


@torch.no_grad()
def run_evaluation(embedder, dataset, device, max_samples=0):
    """Run forward pass on all samples and collect metrics."""
    n = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))

    # Accumulators
    all_flow_pred = []
    all_flow_gt = []
    all_cos_fast_slow = []
    all_alpha_mot = []
    all_alpha_key = []
    all_alpha_txt = []
    all_alpha_brain = []
    all_dyn_labels = []  # derived from flow_mag_traj mean
    all_temp_delta_losses = []
    all_temp_abs_losses = []
    all_per_frame_cos = []  # per-frame cosine sim between pred and gt delta

    flow_mag_means = []  # to compute median for 2-class split

    # First pass: collect flow_mag means for median split
    print("Pass 1: collecting flow_mag statistics...")
    for i in range(n):
        item = dataset[i]
        tgt = item.get("sf_targets", {})
        if "gt_flow_mag_traj" in tgt:
            flow_mag_means.append(tgt["gt_flow_mag_traj"].float().mean().item())
    if flow_mag_means:
        flow_mag_median = np.median(flow_mag_means)
        print(f"  flow_mag_traj mean median: {flow_mag_median:.4f} (for 2-class split)")
    else:
        flow_mag_median = 0.0
        print("  WARNING: no flow_mag_traj found in targets")

    # Second pass: forward + metrics
    print(f"Pass 2: running forward pass on {n} samples...")
    for i in range(n):
        if i % 50 == 0:
            print(f"  [{i}/{n}]")
        item = dataset[i]
        batch = prepare_batch(item, device)

        try:
            context, _ = embedder(batch, siglip_model=None)
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue

        fast_out = embedder._last_fast_out
        slow_out = embedder._last_slow_out
        alphas = embedder._last_alphas
        targets = batch.get("sf_targets", {})

        # --- Metric 1: Temporal learnability ---
        if "temporal_tokens" in fast_out and "gt_temporal_frame_embs" in targets:
            pred = fast_out["temporal_tokens"].float()       # (1, T, 1152)
            gt = targets["gt_temporal_frame_embs"].float()   # (1, T, D)
            T = min(pred.shape[1], gt.shape[1])
            pred_t, gt_t = pred[:, :T], gt[:, :T]

            # Delta
            pred_delta = pred_t - pred_t[:, :1]
            gt_delta = gt_t - gt_t[:, :1]
            l_delta = F.mse_loss(pred_delta, gt_delta).item()
            l_abs = F.mse_loss(pred_t, gt_t).item()
            all_temp_delta_losses.append(l_delta)
            all_temp_abs_losses.append(l_abs)

            # Per-frame cosine sim (delta, skip t=0 which is always zero)
            for t in range(1, T):
                cos = F.cosine_similarity(pred_delta[:, t], gt_delta[:, t], dim=-1).item()
                all_per_frame_cos.append((t, cos))

        # --- Metric 2: Flow trajectory correlation ---
        if "flow_traj_pred" in fast_out and "gt_flow_mag_traj" in targets:
            fp = fast_out["flow_traj_pred"].float().cpu().squeeze(0)  # (T,)
            fg = targets["gt_flow_mag_traj"].float().cpu().squeeze(0)  # (T,)
            T = min(len(fp), len(fg))
            all_flow_pred.append(fp[:T].numpy())
            all_flow_gt.append(fg[:T].numpy())

        # --- Metric 3: Fast/Slow independence ---
        if "fast_feat" in fast_out and "slow_feat" in slow_out:
            ff = fast_out["fast_feat"].float()    # (1, 226, 2048)
            sf = slow_out["slow_feat"].float()    # (1, 226, 2048)
            # Mean pool over spatial dim, then cosine sim
            ff_mean = ff.mean(dim=1)  # (1, 2048)
            sf_mean = sf.mean(dim=1)  # (1, 2048)
            cos = F.cosine_similarity(ff_mean, sf_mean, dim=-1).item()
            all_cos_fast_slow.append(cos)

        # --- Metric 4: Gating behavior ---
        if alphas is not None:
            all_alpha_mot.append(alphas["alpha_mot"].float().item())
            all_alpha_key.append(alphas["alpha_key"].float().item())
            all_alpha_txt.append(alphas["alpha_txt"].float().item())
            all_alpha_brain.append(alphas["alpha_brain"].float().item())

            # Derive 2-class dyn label from flow_mag
            if "gt_flow_mag_traj" in targets:
                fmm = targets["gt_flow_mag_traj"].float().mean().item()
                all_dyn_labels.append(1 if fmm >= flow_mag_median else 0)

    return {
        "flow_pred": all_flow_pred,
        "flow_gt": all_flow_gt,
        "cos_fast_slow": all_cos_fast_slow,
        "alpha_mot": all_alpha_mot,
        "alpha_key": all_alpha_key,
        "alpha_txt": all_alpha_txt,
        "alpha_brain": all_alpha_brain,
        "dyn_labels": all_dyn_labels,
        "temp_delta_losses": all_temp_delta_losses,
        "temp_abs_losses": all_temp_abs_losses,
        "per_frame_cos": all_per_frame_cos,
        "flow_mag_median": flow_mag_median,
        "n_samples": n,
    }


def compute_report(data):
    """Compute final metrics from accumulated data."""
    report = {"n_samples": data["n_samples"]}

    # --- 1. Temporal learnability ---
    if data["temp_delta_losses"]:
        report["valid_L_temp_delta_mean"] = float(np.mean(data["temp_delta_losses"]))
        report["valid_L_temp_delta_std"] = float(np.std(data["temp_delta_losses"]))
        report["valid_L_temp_abs_mean"] = float(np.mean(data["temp_abs_losses"]))

    if data["per_frame_cos"]:
        # Group by frame index
        from collections import defaultdict
        frame_cos = defaultdict(list)
        for t, cos in data["per_frame_cos"]:
            frame_cos[t].append(cos)
        report["per_frame_cos_mean"] = {t: float(np.mean(v)) for t, v in sorted(frame_cos.items())}

    # --- 2. Flow trajectory Pearson ---
    if data["flow_pred"] and data["flow_gt"]:
        # Flatten all samples for global correlation
        pred_flat = np.concatenate(data["flow_pred"])
        gt_flat = np.concatenate(data["flow_gt"])
        r, p = stats.pearsonr(pred_flat, gt_flat)
        report["flow_traj_pearson_r"] = float(r)
        report["flow_traj_pearson_p"] = float(p)
        report["flow_traj_PASS"] = r > 0.3

        # Per-sample Spearman (within-clip temporal ranking)
        spearman_rs = []
        for fp, fg in zip(data["flow_pred"], data["flow_gt"]):
            if len(fp) > 2:
                rho, _ = stats.spearmanr(fp, fg)
                if not np.isnan(rho):
                    spearman_rs.append(rho)
        if spearman_rs:
            report["flow_traj_spearman_mean"] = float(np.mean(spearman_rs))
            report["flow_traj_spearman_std"] = float(np.std(spearman_rs))

    # --- 3. Fast/Slow independence ---
    if data["cos_fast_slow"]:
        cos_arr = np.array(data["cos_fast_slow"])
        report["fast_slow_cosine_mean"] = float(np.mean(cos_arr))
        report["fast_slow_cosine_std"] = float(np.std(cos_arr))
        report["fast_slow_cosine_median"] = float(np.median(cos_arr))
        # Lower is better (more independent). P0 baseline should be ~0.9+
        report["fast_slow_independence_note"] = "Lower = more independent. P0 baseline ~0.9+"

    # --- 4. Gating behavior ---
    if data["alpha_mot"] and data["dyn_labels"]:
        alpha_mot = np.array(data["alpha_mot"])
        dyn = np.array(data["dyn_labels"])

        # Split by dynamics
        high_dyn = alpha_mot[dyn == 1]
        low_dyn = alpha_mot[dyn == 0]

        report["gating_alpha_mot_mean"] = float(np.mean(alpha_mot))
        report["gating_alpha_mot_high_dyn_mean"] = float(np.mean(high_dyn)) if len(high_dyn) > 0 else None
        report["gating_alpha_mot_low_dyn_mean"] = float(np.mean(low_dyn)) if len(low_dyn) > 0 else None

        # Spearman correlation between alpha_mot and flow_mag mean
        if len(alpha_mot) > 10:
            rho, p = stats.spearmanr(alpha_mot, dyn)
            report["gating_alpha_mot_dyn_spearman"] = float(rho)
            report["gating_alpha_mot_dyn_p"] = float(p)

        # Also report other alphas
        report["gating_alpha_key_mean"] = float(np.mean(data["alpha_key"]))
        report["gating_alpha_txt_mean"] = float(np.mean(data["alpha_txt"]))
        report["gating_alpha_brain_mean"] = float(np.mean(data["alpha_brain"]))

    # --- Summary ---
    checks = []
    checks.append(("1_temporal_learnability",
                    report.get("valid_L_temp_delta_mean", 999) < 0.1,
                    "valid L_temp_delta should decrease"))
    checks.append(("2_flow_traj_pearson",
                    report.get("flow_traj_PASS", False),
                    "Pearson r > 0.3"))
    checks.append(("3_fast_slow_independence",
                    report.get("fast_slow_cosine_mean", 1.0) < 0.85,
                    "cosine < 0.85 (lower = better)"))
    checks.append(("4_gating_behavior",
                    report.get("gating_alpha_mot_dyn_spearman", 0) > 0.05,
                    "alpha_mot correlates with dynamics"))

    report["checks"] = {name: {"pass": passed, "note": note} for name, passed, note in checks}
    report["checks_passed"] = sum(1 for _, passed, _ in checks if passed)
    report["checks_total"] = len(checks)

    return report


def print_report(report):
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 70)
    print("P1 TEMPORAL DYNAMICS EVALUATION REPORT")
    print("=" * 70)

    print(f"\nSamples evaluated: {report['n_samples']}")

    print("\n--- 1. Temporal Learnability ---")
    print(f"  Valid L_temp_delta (mean): {report.get('valid_L_temp_delta_mean', 'N/A'):.6f}")
    print(f"  Valid L_temp_abs   (mean): {report.get('valid_L_temp_abs_mean', 'N/A'):.6f}")
    if "per_frame_cos_mean" in report:
        print("  Per-frame cosine sim (pred_delta vs gt_delta):")
        for t, v in report["per_frame_cos_mean"].items():
            print(f"    t={t}: {v:.4f}")

    print("\n--- 2. Flow Trajectory Quality ---")
    print(f"  Pearson r:  {report.get('flow_traj_pearson_r', 'N/A'):.4f}  "
          f"(p={report.get('flow_traj_pearson_p', 'N/A'):.2e})")
    print(f"  Spearman ρ: {report.get('flow_traj_spearman_mean', 'N/A'):.4f} "
          f"± {report.get('flow_traj_spearman_std', 'N/A'):.4f}")
    print(f"  PASS (r>0.3): {report.get('flow_traj_PASS', 'N/A')}")

    print("\n--- 3. Fast/Slow Independence ---")
    print(f"  Cosine similarity: {report.get('fast_slow_cosine_mean', 'N/A'):.4f} "
          f"± {report.get('fast_slow_cosine_std', 'N/A'):.4f}")
    print(f"  Median: {report.get('fast_slow_cosine_median', 'N/A'):.4f}")
    print(f"  Note: {report.get('fast_slow_independence_note', '')}")

    print("\n--- 4. Gating Behavior ---")
    print(f"  alpha_mot overall:  {report.get('gating_alpha_mot_mean', 'N/A'):.4f}")
    print(f"  alpha_mot high_dyn: {report.get('gating_alpha_mot_high_dyn_mean', 'N/A')}")
    print(f"  alpha_mot low_dyn:  {report.get('gating_alpha_mot_low_dyn_mean', 'N/A')}")
    print(f"  Spearman(alpha_mot, dyn): {report.get('gating_alpha_mot_dyn_spearman', 'N/A')}")
    print(f"  Other alphas — key: {report.get('gating_alpha_key_mean', 'N/A'):.4f}, "
          f"txt: {report.get('gating_alpha_txt_mean', 'N/A'):.4f}, "
          f"brain: {report.get('gating_alpha_brain_mean', 'N/A'):.4f}")

    print("\n--- SUMMARY ---")
    for name, info in report.get("checks", {}).items():
        status = "✓ PASS" if info["pass"] else "✗ FAIL"
        print(f"  [{status}] {name}: {info['note']}")
    print(f"\n  Result: {report['checks_passed']}/{report['checks_total']} checks passed")
    print("=" * 70)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Build model
    embedder = build_embedder(args.model_config, device)

    # Load checkpoint
    embedder = load_checkpoint(embedder, args.ckpt, device)

    # Build dataset
    dataset = build_dataset(args.data_json, args.model_config)

    # Run evaluation
    raw_data = run_evaluation(embedder, dataset, device, max_samples=args.max_samples)

    # Compute report
    report = compute_report(raw_data)

    # Print
    print_report(report)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
