"""Fast Motion-Dynamics Branch: EEG → z_dyn, z_mot, z_tc"""
import torch
import torch.nn as nn


class DynamicsHead(nn.Module):
    """Predict dynamics score z_dyn (scalar) from fast features.
    Supervised by OFS (optical flow score) — a single scalar per clip.
    """
    def __init__(self, in_dim=2048, out_dim=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_dyn: (B,) scalar via mean pooling + regression"""
        return self.proj(x.mean(dim=1)).squeeze(-1)


class MotionHead(nn.Module):
    """Predict motion flow token z_mot from fast features.
    Supervised by RAFT patch-pooled flow magnitude — (B, num_patches).
    Default num_patches = (520//16) * (960//16) = 32*60 = 1920.
    """
    def __init__(self, in_dim=2048, out_dim=1920):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_mot: (B, num_patches) via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class TemporalCoherenceHead(nn.Module):
    """Predict temporal coherence score z_tc (scalar) from fast features.
    Supervised by OFS score — a single scalar per clip.
    """
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
        head_dim=1152,  # unused, kept for API compat
        use_dynamics_head=True,
        use_motion_head=True,
        use_temporal_coherence_head=True,
        motion_token_dim=1920,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder

        self.use_dynamics_head = use_dynamics_head
        self.use_motion_head = use_motion_head
        self.use_temporal_coherence_head = use_temporal_coherence_head

        if use_dynamics_head:
            self.dynamics_head = DynamicsHead(embed_dim, out_dim=1)  # scalar OFS
        if use_motion_head:
            self.motion_head = MotionHead(embed_dim, out_dim=motion_token_dim)  # flow tokens
        if use_temporal_coherence_head:
            self.tc_head = TemporalCoherenceHead(embed_dim, out_dim=1)  # scalar OFS

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
