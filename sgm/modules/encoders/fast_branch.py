"""Fast Motion-Dynamics Branch: EEG → z_dyn, z_mot, z_tc"""
import torch
import torch.nn as nn


class DynamicsHead(nn.Module):
    """Predict dynamic pattern embedding z_dyn from fast features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_dyn: (B, out_dim) via mean pooling"""
        return self.proj(x.mean(dim=1))


class MotionHead(nn.Module):
    """Predict motion latent z_mot from fast features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_mot: (B, out_dim) via mean pooling"""
        return self.proj(x.mean(dim=1))


class TemporalCoherenceHead(nn.Module):
    """Predict temporal coherence token z_tc from fast features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_tc: (B, out_dim) via mean pooling"""
        return self.proj(x.mean(dim=1))


class FastBranch(nn.Module):
    """
    Fast Motion-Dynamics Branch.
    Wraps the existing EEG encoder and adds three prediction heads.
    All heads can be individually disabled via config.
    """
    def __init__(
        self,
        eeg_encoder,
        embed_dim=2048,
        head_dim=1152,
        use_dynamics_head=True,
        use_motion_head=True,
        use_temporal_coherence_head=True,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder

        self.use_dynamics_head = use_dynamics_head
        self.use_motion_head = use_motion_head
        self.use_temporal_coherence_head = use_temporal_coherence_head

        if use_dynamics_head:
            self.dynamics_head = DynamicsHead(embed_dim, head_dim)
        if use_motion_head:
            self.motion_head = MotionHead(embed_dim, head_dim)
        if use_temporal_coherence_head:
            self.tc_head = TemporalCoherenceHead(embed_dim, head_dim)

    def forward(self, eeg):
        """
        Args:
            eeg: (B, 5, 64, 800)
        Returns:
            dict with keys:
                "eeg_cls": (B, 1152) for CLIP alignment
                "eeg_spatial": (B, 226, 2048) raw spatial features
                "fast_feat": (B, 226, 2048) features for fusion
                "z_dyn": (B, 1152) dynamics embedding (if enabled)
                "z_mot": (B, 1152) motion latent (if enabled)
                "z_tc": (B, 1152) temporal coherence token (if enabled)
        """
        eeg_cls, eeg_spatial = self.eeg_encoder(eeg)

        out = {
            "eeg_cls": eeg_cls,
            "eeg_spatial": eeg_spatial,
            "fast_feat": eeg_spatial,
        }

        if self.use_dynamics_head:
            out["z_dyn"] = self.dynamics_head(eeg_spatial)
        if self.use_motion_head:
            out["z_mot"] = self.motion_head(eeg_spatial)
        if self.use_temporal_coherence_head:
            out["z_tc"] = self.tc_head(eeg_spatial)

        return out
