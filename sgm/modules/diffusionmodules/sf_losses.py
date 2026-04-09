"""CineBrain-SF v1 loss functions: alignment, slow, fast (distillation), guidance.
All losses are bf16-safe: use input tensor dtype/device for initialization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """5-way cross-modal alignment loss with MoCo queue for bs=1 support.

    P1-4: When queue is provided, uses queue-augmented InfoNCE instead of
    in-batch-only contrastive (which degenerates to 0 at bs=1).
    """
    def __init__(self, lambda_fv=1.0, lambda_ft=1.0, lambda_ev=1.0, lambda_et=1.0, lambda_fe=0.5):
        super().__init__()
        self.lambdas = {"fv": lambda_fv, "ft": lambda_ft, "ev": lambda_ev, "et": lambda_et, "fe": lambda_fe}
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)

    def contrastive(self, feat_a, feat_b, feat_b_queue=None):
        """InfoNCE with optional queue negatives.
        feat_a: (B, D) anchors
        feat_b: (B, D) positive targets
        feat_b_queue: (K, D) queue of past targets (optional)
        """
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)

        if feat_b_queue is not None and feat_b_queue.shape[0] > 0:
            # Queue-augmented: positives + queue negatives
            feat_b_all = torch.cat([feat_b, feat_b_queue.to(feat_b.device)], dim=0)  # (B+K, D)
            logits = scale * feat_a @ feat_b_all.T  # (B, B+K)
            labels = torch.arange(feat_a.shape[0], device=feat_a.device)  # positive = first B
            return F.cross_entropy(logits, labels)
        else:
            # Original in-batch contrastive (broken at bs=1)
            logits_ab = scale * feat_a @ feat_b.T
            labels = torch.arange(feat_a.shape[0], device=feat_a.device)
            return (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ab.T, labels)) / 2

    def forward(self, slow_out, fast_out, video_embed=None, text_embed=None, queues=None):
        """
        queues: optional dict with keys 'fmri', 'eeg', 'video', 'text', each (K, 1152)
        """
        fmri_cls = slow_out["fmri_cls"]
        eeg_cls = fast_out["eeg_cls"]
        losses = {}
        total = fmri_cls.new_tensor(0.0)

        vq = queues.get("video") if queues else None
        tq = queues.get("text") if queues else None
        eq = queues.get("eeg") if queues else None
        fq = queues.get("fmri") if queues else None

        if video_embed is not None:
            losses["L_fv"] = self.contrastive(fmri_cls, video_embed, vq)
            losses["L_ev"] = self.contrastive(eeg_cls, video_embed, vq)
            total = total + self.lambdas["fv"] * losses["L_fv"] + self.lambdas["ev"] * losses["L_ev"]
        if text_embed is not None:
            losses["L_ft"] = self.contrastive(fmri_cls, text_embed, tq)
            losses["L_et"] = self.contrastive(eeg_cls, text_embed, tq)
            total = total + self.lambdas["ft"] * losses["L_ft"] + self.lambdas["et"] * losses["L_et"]

        losses["L_fe"] = self.contrastive(fmri_cls, eeg_cls, eq)
        total = total + self.lambdas["fe"] * losses["L_fe"]
        return total, losses


class SlowBranchLoss(nn.Module):
    """Supervision for slow branch heads."""
    def __init__(self, lambda_key=1.0, lambda_txt=1.0, lambda_str=1.0):
        super().__init__()
        self.lambda_key = lambda_key
        self.lambda_txt = lambda_txt
        self.lambda_str = lambda_str

    def forward(self, slow_out, targets):
        _ref = next(iter(slow_out.values()))
        losses = {}
        total = _ref.new_tensor(0.0)

        if "z_key" in slow_out and "gt_keyframe_embed" in targets:
            losses["L_key"] = F.mse_loss(slow_out["z_key"], targets["gt_keyframe_embed"])
            total = total + self.lambda_key * losses["L_key"]
        if "z_txt" in slow_out and "gt_text_embed" in targets:
            cos_sim = F.cosine_similarity(slow_out["z_txt"], targets["gt_text_embed"], dim=-1).mean()
            losses["L_txt"] = (cos_sim.new_tensor(1.0) - cos_sim)
            total = total + self.lambda_txt * losses["L_txt"]
        if "z_str" in slow_out and "gt_structure_embed" in targets:
            z_str = slow_out["z_str"]
            gt_str = targets["gt_structure_embed"]
            if gt_str.shape != z_str.shape:
                gt_str = gt_str.reshape_as(z_str)
            losses["L_str"] = F.mse_loss(z_str, gt_str)
            total = total + self.lambda_str * losses["L_str"]

        return total, losses


class FastBranchDistillLoss(nn.Module):
    """Distillation loss for fast branch: EEG learns to match fMRI features.

    Two-level distillation:
    1. CLS-level: MSE(eeg_cls_proj, fmri_cls.detach()) - global semantics
    2. Spatial-level: MSE(eeg_pooled_proj, fmri_pooled.detach()) - spatial features

    P1 extension (when targets provided):
    3. L_temporal_delta: MSE(predicted delta, gt delta) - frame-to-frame changes
    4. L_temporal_abs: MSE(predicted abs, gt abs temporal frame embs)
    5. L_flow_traj: MSE(predicted flow traj, gt flow mag traj)
    6. L_dyn: CrossEntropy(dyn_logits, dyn_label_2class) - coarse dynamics

    This replaces the old FastBranchLoss (classification on RAFT optical flow).
    """
    def __init__(self, lambda_cls=1.0, lambda_spatial=1.0,
                 lambda_temporal_delta=1.0, lambda_temporal_abs=0.2,
                 lambda_flow_traj=0.3, lambda_dyn=0.1, lambda_struct=0.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_spatial = lambda_spatial
        # P1 loss weights
        self.lambda_temporal_delta = lambda_temporal_delta
        self.lambda_temporal_abs = lambda_temporal_abs
        self.lambda_flow_traj = lambda_flow_traj
        self.lambda_dyn = lambda_dyn
        # Stage 3: structural similarity loss (DynaMind-inspired)
        self.lambda_struct = lambda_struct

    def forward(self, fast_out, slow_out, targets=None):
        """
        Args:
            fast_out: dict from FastBranch with eeg_cls_proj, eeg_pooled_proj,
                      and optionally temporal_tokens, global_dyn_token, flow_traj_pred, dyn_logits
            slow_out: dict from SlowBranch with fmri_cls, fmri_spatial (teacher, detached)
            targets: optional dict with gt_temporal_frame_embs, gt_flow_mag_traj, gt_dyn_label_2class
        Returns:
            total: scalar loss
            losses: dict of individual losses
        """
        _ref = fast_out["eeg_cls"]
        losses = {}
        total = _ref.new_tensor(0.0)

        # CLS-level distillation: (B, 1152) vs (B, 1152)
        if "eeg_cls_proj" in fast_out and "fmri_cls" in slow_out:
            eeg_cls = fast_out["eeg_cls_proj"]
            fmri_cls = slow_out["fmri_cls"].detach()
            losses["L_distill_cls"] = F.mse_loss(eeg_cls, fmri_cls)
            total = total + self.lambda_cls * losses["L_distill_cls"]

        # Spatial-level distillation: (B, 2048) vs (B, 2048)
        # fMRI spatial features pooled to match EEG pooled features
        if "eeg_pooled_proj" in fast_out and "fmri_spatial" in slow_out:
            eeg_pooled = fast_out["eeg_pooled_proj"]
            fmri_spatial = slow_out["fmri_spatial"].detach()  # (B, 226, 2048)
            fmri_pooled = fmri_spatial.mean(dim=1)  # (B, 2048)
            losses["L_distill_spatial"] = F.mse_loss(eeg_pooled, fmri_pooled)
            total = total + self.lambda_spatial * losses["L_distill_spatial"]

        # --- P1: Temporal dynamics losses (only when temporal_tokens present) ---
        if targets is None:
            targets = {}

        if "temporal_tokens" in fast_out:
            temporal_tokens = fast_out["temporal_tokens"]  # (B, T, 1152)

            # L_temporal_delta: MSE between predicted delta and GT delta
            # delta = frame_t - frame_1 (relative to first frame)
            if "gt_temporal_frame_embs" in targets:
                gt_frame_embs = targets["gt_temporal_frame_embs"]  # (B, T, D)
                # Ensure T dimensions match (truncate to min)
                T_pred = temporal_tokens.shape[1]
                T_gt = gt_frame_embs.shape[1]
                T = min(T_pred, T_gt)

                gt_frame_embs = gt_frame_embs[:, :T, :]
                pred_tokens = temporal_tokens[:, :T, :]

                # Delta: relative to first frame
                gt_delta = gt_frame_embs - gt_frame_embs[:, :1, :]  # (B, T, D)
                pred_delta = pred_tokens - pred_tokens[:, :1, :]    # (B, T, D)
                losses["L_temporal_delta"] = F.mse_loss(pred_delta, gt_delta)
                total = total + self.lambda_temporal_delta * losses["L_temporal_delta"]

                # L_temporal_abs: MSE between predicted and GT absolute frame embeddings
                losses["L_temporal_abs"] = F.mse_loss(pred_tokens, gt_frame_embs)
                total = total + self.lambda_temporal_abs * losses["L_temporal_abs"]

            # L_flow_traj: regression MSE or codebook classification CE
            if "flow_traj_pred" in fast_out and "gt_flow_mag_traj" in targets:
                flow_pred = fast_out["flow_traj_pred"]
                flow_gt = targets["gt_flow_mag_traj"]        # (B, T) continuous values

                if flow_pred.ndim == 3:
                    # P1-3 Codebook mode: flow_pred is (B, T, K) logits
                    K = flow_pred.shape[-1]
                    T = min(flow_pred.shape[1], flow_gt.shape[1])
                    # Quantize GT to nearest codebook bin (uniform bins in [0, max_val])
                    flow_gt_clip = flow_gt[:, :T].clamp(0)
                    max_val = flow_gt_clip.max().clamp(min=1e-6)
                    bin_indices = (flow_gt_clip / max_val * (K - 1)).long().clamp(0, K - 1)
                    losses["L_flow_traj"] = F.cross_entropy(
                        flow_pred[:, :T].reshape(-1, K), bin_indices.reshape(-1)
                    )
                else:
                    # Regression mode: flow_pred is (B, T) scalars
                    T = min(flow_pred.shape[1], flow_gt.shape[1])
                    losses["L_flow_traj"] = F.mse_loss(flow_pred[:, :T], flow_gt[:, :T])

                total = total + self.lambda_flow_traj * losses["L_flow_traj"]

        # L_struct: Structural similarity loss (DynaMind-inspired)
        # Matches inter-frame relationship structure, not per-frame values
        if self.lambda_struct > 0 and "temporal_tokens" in fast_out and "gt_temporal_frame_embs" in targets:
            pred_t = fast_out["temporal_tokens"]
            gt_t = targets["gt_temporal_frame_embs"]
            T = min(pred_t.shape[1], gt_t.shape[1])
            pred_norm = F.normalize(pred_t[:, :T].float(), dim=-1)
            gt_norm = F.normalize(gt_t[:, :T].float(), dim=-1)
            S_pred = torch.bmm(pred_norm, pred_norm.transpose(1, 2))  # (B, T, T)
            S_gt = torch.bmm(gt_norm, gt_norm.transpose(1, 2))        # (B, T, T)
            # Only compute on off-diagonal (diagonal is always 1.0 vs 1.0 = 0, wastes gradient)
            mask = ~torch.eye(T, device=S_pred.device, dtype=torch.bool).unsqueeze(0)
            losses["L_struct"] = F.mse_loss(S_pred[mask], S_gt[mask])
            total = total + self.lambda_struct * losses["L_struct"]

        # L_dyn: Coarse dynamics classification (2-class)
        if "dyn_logits" in fast_out and "gt_dyn_label_2class" in targets:
            dyn_logits = fast_out["dyn_logits"]          # (B, 2)
            dyn_labels = targets["gt_dyn_label_2class"]  # (B,) long
            losses["L_dyn"] = F.cross_entropy(dyn_logits, dyn_labels.long())
            total = total + self.lambda_dyn * losses["L_dyn"]

        return total, losses


class GuidanceLoss(nn.Module):
    """Guidance consistency: head predictions should match stimulus embeddings."""
    def __init__(self, lambda_gk=0.5, lambda_gt=0.5, lambda_gm=0.5):
        super().__init__()
        self.lambda_gk = lambda_gk
        self.lambda_gt = lambda_gt
        self.lambda_gm = lambda_gm

    def forward(self, slow_out, fast_out, targets, video_embed=None, text_embed=None):
        _ref = next(iter(slow_out.values()))
        losses = {}
        total = _ref.new_tensor(0.0)

        if "z_key" in slow_out and video_embed is not None:
            cos_sim = F.cosine_similarity(slow_out["z_key"], video_embed, dim=-1).mean()
            losses["L_gk"] = cos_sim.new_tensor(1.0) - cos_sim
            total = total + self.lambda_gk * losses["L_gk"]
        if "z_txt" in slow_out and text_embed is not None:
            cos_sim = F.cosine_similarity(slow_out["z_txt"], text_embed, dim=-1).mean()
            losses["L_gt"] = cos_sim.new_tensor(1.0) - cos_sim
            total = total + self.lambda_gt * losses["L_gt"]

        # P2-3: L_gm — motion guidance consistency
        # Encourage motion feature energy to track flow magnitude
        if self.lambda_gm > 0 and "eeg_pooled_proj" in fast_out and "gt_flow_mag_traj" in targets:
            mot_energy = fast_out["eeg_pooled_proj"].norm(dim=-1)  # (B,)
            flow_energy = targets["gt_flow_mag_traj"].mean(dim=-1)  # (B,)
            # Scale mot_energy to match flow_energy range via detached ratio
            scale = (flow_energy.detach().mean() / (mot_energy.detach().mean() + 1e-8)).clamp(0.01, 100)
            losses["L_gm"] = F.mse_loss(mot_energy * scale, flow_energy)
            total = total + self.lambda_gm * losses["L_gm"]

        return total, losses


class AuxAlignmentLoss(nn.Module):
    """InfoNCE: pull eeg_cls toward fmri_cls (detached) for curriculum training."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_cls, fmri_cls):
        fmri_cls = fmri_cls.detach()
        eeg_norm = F.normalize(eeg_cls, dim=-1)
        fmri_norm = F.normalize(fmri_cls, dim=-1)
        logits = eeg_norm @ fmri_norm.T / self.temperature
        labels = torch.arange(eeg_cls.shape[0], device=eeg_cls.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
