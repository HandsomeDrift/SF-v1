"""CineBrain-SF v1 loss functions: alignment, slow, fast, guidance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """5-way cross-modal alignment loss (extends existing CLIP loss)."""
    def __init__(self, lambda_fv=1.0, lambda_ft=1.0, lambda_ev=1.0, lambda_et=1.0, lambda_fe=0.5):
        super().__init__()
        self.lambdas = {"fv": lambda_fv, "ft": lambda_ft, "ev": lambda_ev, "et": lambda_et, "fe": lambda_fe}
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)  # ln(1/0.07)

    def contrastive(self, feat_a, feat_b):
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_ab = scale * feat_a @ feat_b.T
        logits_ba = logits_ab.T
        labels = torch.arange(feat_a.shape[0], device=feat_a.device)
        return (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels)) / 2

    def forward(self, slow_out, fast_out, video_embed=None, text_embed=None):
        """
        Args:
            slow_out: dict with "fmri_cls" (B, 1152)
            fast_out: dict with "eeg_cls" (B, 1152)
            video_embed: (B, 1152) from SigLIP, optional (only in training)
            text_embed: (B, 1152) from SigLIP, optional
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total = torch.tensor(0.0, device=slow_out["fmri_cls"].device)
        fmri_cls = slow_out["fmri_cls"]
        eeg_cls = fast_out["eeg_cls"]

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
        """
        Args:
            slow_out: dict with z_key, z_txt, z_str
            targets: dict with gt_keyframe_embed, gt_text_embed, gt_structure_embed
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(slow_out.values())).device)

        if "z_key" in slow_out and "gt_keyframe_embed" in targets:
            losses["L_key"] = F.mse_loss(slow_out["z_key"], targets["gt_keyframe_embed"])
            total = total + self.lambda_key * losses["L_key"]
        if "z_txt" in slow_out and "gt_text_embed" in targets:
            losses["L_txt"] = 1.0 - F.cosine_similarity(
                slow_out["z_txt"], targets["gt_text_embed"], dim=-1
            ).mean()
            total = total + self.lambda_txt * losses["L_txt"]
        if "z_str" in slow_out and "gt_structure_embed" in targets:
            losses["L_str"] = F.mse_loss(slow_out["z_str"], targets["gt_structure_embed"])
            total = total + self.lambda_str * losses["L_str"]

        return total, losses


class FastBranchLoss(nn.Module):
    """Supervision for fast branch heads."""
    def __init__(self, lambda_dyn=1.0, lambda_mot=1.0, lambda_tc=0.5):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_mot = lambda_mot
        self.lambda_tc = lambda_tc

    def forward(self, fast_out, targets):
        """
        Args:
            fast_out: dict with z_dyn, z_mot, z_tc
            targets: dict with gt_dynamics_embed, gt_motion_embed, gt_tc_embed
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(fast_out.values())).device)

        if "z_dyn" in fast_out and "gt_dynamics_embed" in targets:
            losses["L_dyn"] = F.mse_loss(fast_out["z_dyn"], targets["gt_dynamics_embed"])
            total = total + self.lambda_dyn * losses["L_dyn"]
        if "z_mot" in fast_out and "gt_motion_embed" in targets:
            losses["L_mot"] = F.mse_loss(fast_out["z_mot"], targets["gt_motion_embed"])
            total = total + self.lambda_mot * losses["L_mot"]
        if "z_tc" in fast_out and "gt_tc_embed" in targets:
            losses["L_tc"] = F.mse_loss(fast_out["z_tc"], targets["gt_tc_embed"])
            total = total + self.lambda_tc * losses["L_tc"]

        return total, losses


class GuidanceLoss(nn.Module):
    """Guidance consistency: generated content should match guidance signals."""
    def __init__(self, lambda_gk=0.5, lambda_gt=0.5, lambda_gm=0.5):
        super().__init__()
        self.lambda_gk = lambda_gk
        self.lambda_gt = lambda_gt
        self.lambda_gm = lambda_gm

    def forward(self, slow_out, fast_out, video_embed=None, text_embed=None):
        """
        Consistency between head predictions and stimulus embeddings.
        Uses cosine similarity loss.
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(slow_out.values())).device)

        if "z_key" in slow_out and video_embed is not None:
            losses["L_gk"] = 1.0 - F.cosine_similarity(
                slow_out["z_key"], video_embed, dim=-1
            ).mean()
            total = total + self.lambda_gk * losses["L_gk"]
        if "z_txt" in slow_out and text_embed is not None:
            losses["L_gt"] = 1.0 - F.cosine_similarity(
                slow_out["z_txt"], text_embed, dim=-1
            ).mean()
            total = total + self.lambda_gt * losses["L_gt"]
        if "z_mot" in fast_out and video_embed is not None:
            losses["L_gm"] = 1.0 - F.cosine_similarity(
                fast_out["z_mot"], video_embed, dim=-1
            ).mean()
            total = total + self.lambda_gm * losses["L_gm"]

        return total, losses
