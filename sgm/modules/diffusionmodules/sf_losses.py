"""CineBrain-SF v1 loss functions: alignment, slow, fast, guidance.
All losses are bf16-safe: use input tensor dtype/device for initialization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """5-way cross-modal alignment loss (extends existing CLIP loss)."""
    def __init__(self, lambda_fv=1.0, lambda_ft=1.0, lambda_ev=1.0, lambda_et=1.0, lambda_fe=0.5):
        super().__init__()
        self.lambdas = {"fv": lambda_fv, "ft": lambda_ft, "ev": lambda_ev, "et": lambda_et, "fe": lambda_fe}
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)

    def contrastive(self, feat_a, feat_b):
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_ab = scale * feat_a @ feat_b.T
        labels = torch.arange(feat_a.shape[0], device=feat_a.device)
        return (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ab.T, labels)) / 2

    def forward(self, slow_out, fast_out, video_embed=None, text_embed=None):
        fmri_cls = slow_out["fmri_cls"]
        eeg_cls = fast_out["eeg_cls"]
        losses = {}
        total = fmri_cls.new_tensor(0.0)

        if video_embed is not None:
            losses["L_fv"] = self.contrastive(fmri_cls, video_embed)
            losses["L_ev"] = self.contrastive(eeg_cls, video_embed)
            total = total + self.lambdas["fv"] * losses["L_fv"] + self.lambdas["ev"] * losses["L_ev"]
        if text_embed is not None:
            losses["L_ft"] = self.contrastive(fmri_cls, text_embed)
            losses["L_et"] = self.contrastive(eeg_cls, text_embed)
            total = total + self.lambdas["ft"] * losses["L_ft"] + self.lambdas["et"] * losses["L_et"]

        losses["L_fe"] = self.contrastive(fmri_cls, eeg_cls)
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


class FastBranchLoss(nn.Module):
    """Supervision for fast branch heads (v3 targets)."""
    def __init__(self, lambda_dyn=1.0, lambda_mot=1.0, lambda_tc=0.5, lambda_dir=0.5):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_mot = lambda_mot
        self.lambda_tc = lambda_tc
        self.lambda_dir = lambda_dir

    def forward(self, fast_out, targets):
        _ref = next(iter(fast_out.values()))
        losses = {}
        total = _ref.new_tensor(0.0)

        # Dynamics: CrossEntropy (3-class: slow/mid/fast)
        if "z_dyn" in fast_out and "gt_dynamics_class" in targets:
            logits = fast_out["z_dyn"]  # (B, 3)
            labels = targets["gt_dynamics_class"].long()
            losses["L_dyn"] = F.cross_entropy(logits, labels)
            total = total + self.lambda_dyn * losses["L_dyn"]

        # Motion: cosine + MSE hybrid (PCA 128-dim)
        if "z_mot" in fast_out and "gt_motion_embed" in targets:
            pred = fast_out["z_mot"]  # (B, 128)
            gt = targets["gt_motion_embed"]
            mse = F.mse_loss(pred, gt)
            cos = (1.0 - F.cosine_similarity(pred, gt, dim=-1)).mean()
            losses["L_mot"] = 0.5 * mse + 0.5 * cos
            total = total + self.lambda_mot * losses["L_mot"]

        # Temporal coherence: SmoothL1 (scalar regression, log-zscore normalized)
        if "z_tc" in fast_out and "gt_tc_embed" in targets:
            z_tc = fast_out["z_tc"]
            gt_tc = targets["gt_tc_embed"]
            if gt_tc.dim() != z_tc.dim():
                gt_tc = gt_tc.reshape_as(z_tc)
            losses["L_tc"] = F.smooth_l1_loss(z_tc, gt_tc)
            total = total + self.lambda_tc * losses["L_tc"]

        # Motion direction: CrossEntropy (8-class)
        if "z_dir" in fast_out and "gt_direction_class" in targets:
            logits = fast_out["z_dir"]  # (B, 8)
            labels = targets["gt_direction_class"].long()
            losses["L_dir"] = F.cross_entropy(logits, labels)
            total = total + self.lambda_dir * losses["L_dir"]

        return total, losses


class GuidanceLoss(nn.Module):
    """Guidance consistency: head predictions should match stimulus embeddings."""
    def __init__(self, lambda_gk=0.5, lambda_gt=0.5, lambda_gm=0.5):
        super().__init__()
        self.lambda_gk = lambda_gk
        self.lambda_gt = lambda_gt
        self.lambda_gm = lambda_gm

    def forward(self, slow_out, fast_out, video_embed=None, text_embed=None):
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
        if "z_mot" in fast_out and video_embed is not None:
            # z_mot is (B, 1920) flow tokens — compare using MSE with pooled video embed
            # Skip if shapes don't match (guidance loss is secondary)
            pass

        return total, losses


class AuxAlignmentLoss(nn.Module):
    """InfoNCE: pull eeg_cls toward fmri_cls (detached) for curriculum training."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_cls, fmri_cls):
        """
        Args:
            eeg_cls: (B, D) EEG CLS token
            fmri_cls: (B, D) fMRI CLS token (will be detached)
        Returns:
            loss: scalar InfoNCE loss
        """
        fmri_cls = fmri_cls.detach()
        eeg_norm = F.normalize(eeg_cls, dim=-1)
        fmri_norm = F.normalize(fmri_cls, dim=-1)
        logits = eeg_norm @ fmri_norm.T / self.temperature
        labels = torch.arange(eeg_cls.shape[0], device=eeg_cls.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
