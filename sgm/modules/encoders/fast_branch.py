"""Fast Motion-Dynamics Branch: EEG → z_dyn, z_mot, z_tc, z_dir"""
import torch
import torch.nn as nn
from sgm.modules.encoders.temporal_conv import TemporalAttentionPool


class DynamicsHead(nn.Module):
    """3-class motion intensity classification: slow/mid/fast."""
    def __init__(self, in_dim=2048, num_classes=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, num_classes),
        )

    def forward(self, x):
        """x: (B, S, D) → z_dyn: (B, num_classes) logits via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class MotionHead(nn.Module):
    """Predict PCA-reduced motion flow embedding (128-dim)."""
    def __init__(self, in_dim=2048, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_mot: (B, out_dim) via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class TemporalCoherenceHead(nn.Module):
    """Predict temporal coherence (OFS log-zscore, scalar regression)."""
    def __init__(self, in_dim=2048, out_dim=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_tc: (B,) scalar via mean pooling + regression"""
        return self.proj(x.mean(dim=1)).squeeze(-1)


class MotionDirectionHead(nn.Module):
    """8-class motion direction classification."""
    def __init__(self, in_dim=2048, num_classes=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, num_classes),
        )

    def forward(self, x):
        """x: (B, S, D) → z_dir: (B, num_classes) logits via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class FastBranch(nn.Module):
    """
    Fast Motion-Dynamics Branch.
    Wraps the existing EEG encoder and adds prediction heads.
    """
    def __init__(
        self,
        eeg_encoder,
        embed_dim=2048,
        head_dim=1152,  # unused, kept for API compat
        use_dynamics_head=True,
        use_motion_head=True,
        use_temporal_coherence_head=True,
        use_direction_head=True,
        motion_pca_dim=128,
        num_dyn_classes=3,
        num_dir_classes=8,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.use_dynamics_head = use_dynamics_head
        self.use_motion_head = use_motion_head
        self.use_temporal_coherence_head = use_temporal_coherence_head
        self.use_direction_head = use_direction_head

        if use_dynamics_head:
            self.dynamics_head = DynamicsHead(embed_dim, num_classes=num_dyn_classes)
        if use_motion_head:
            self.motion_head = MotionHead(embed_dim, out_dim=motion_pca_dim)
        if use_temporal_coherence_head:
            self.tc_head = TemporalCoherenceHead(embed_dim, out_dim=1)
        if use_direction_head:
            self.direction_head = MotionDirectionHead(embed_dim, num_classes=num_dir_classes)

        # Shared temporal attention pooling (replaces mean pooling in heads)
        self.temporal_pool = TemporalAttentionPool(dim=embed_dim)

    def forward(self, eeg):
        """
        Args:
            eeg: (B, 5, 64, 800)
        Returns:
            dict with keys:
                "eeg_cls": (B, 1152) for CLIP alignment
                "eeg_spatial": (B, 226, 2048) raw spatial features
                "fast_feat": (B, 226, 2048) features for fusion
                "z_dyn": (B, 3) dynamics class logits
                "z_mot": (B, 128) PCA motion embedding
                "z_tc": (B,) temporal coherence scalar
                "z_dir": (B, 8) direction class logits
        """
        eeg_cls, eeg_spatial = self.eeg_encoder(eeg)

        # Temporal-aware pooled features for heads
        pooled_feat = self.temporal_pool(eeg_spatial)  # (B, D)
        pooled_3d = pooled_feat.unsqueeze(1)  # (B, 1, D)

        out = {
            "eeg_cls": eeg_cls,
            "eeg_spatial": eeg_spatial,
            "fast_feat": eeg_spatial,
        }

        if self.use_dynamics_head:
            out["z_dyn"] = self.dynamics_head(pooled_3d)
        if self.use_motion_head:
            out["z_mot"] = self.motion_head(pooled_3d)
        if self.use_temporal_coherence_head:
            out["z_tc"] = self.tc_head(pooled_3d)
        if self.use_direction_head:
            out["z_dir"] = self.direction_head(pooled_3d)

        return out
