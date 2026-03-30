"""Slow Semantic-Structure Branch: fMRI → z_key, z_txt, z_str"""
import torch
import torch.nn as nn
from sgm.modules.encoders.common import CrossAttention, FeedForward


class AudiovisualContextAdapter(nn.Module):
    """Cross-attention adapter fusing visual and auditory fMRI representations."""
    def __init__(self, embed_dim=2048, num_heads=16, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, embed_dim * 4, dropout)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feat, auditory_feat):
        """
        Args:
            visual_feat: (B, S, D) visual ROI features
            auditory_feat: (B, S_a, D) auditory ROI features
        Returns:
            (B, S, D) context-enriched visual features
        """
        h = self.cross_attn(self.norm_q(visual_feat), self.norm_kv(auditory_feat))
        visual_feat = visual_feat + self.dropout(h)
        h = self.ff(self.norm_ff(visual_feat))
        visual_feat = visual_feat + self.dropout(h)
        return visual_feat


class KeyframeHead(nn.Module):
    """Predict keyframe latent z_key from slow features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_key: (B, out_dim) via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class SceneTextHead(nn.Module):
    """Predict scene-text semantic embedding z_txt from slow features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_txt: (B, out_dim) via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class StructureHead(nn.Module):
    """Predict structure latent z_str from slow features (spatial-preserving)."""
    def __init__(self, in_dim=2048, out_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_str: (B, S, out_dim) spatial-preserving"""
        return self.proj(x)


class SlowBranch(nn.Module):
    """
    Slow Semantic-Structure Branch.
    Wraps the existing fMRI encoder and adds:
    - Optional auditory ROI encoding + audiovisual context adapter
    - Three prediction heads: keyframe, scene-text, structure

    All heads can be individually disabled via config.
    """
    def __init__(
        self,
        fmri_encoder,
        auditory_encoder=None,
        embed_dim=2048,
        head_dim=1152,
        use_auditory=False,
        use_keyframe_head=True,
        use_scene_text_head=True,
        use_structure_head=True,
    ):
        super().__init__()
        self.fmri_encoder = fmri_encoder
        self.use_auditory = use_auditory and auditory_encoder is not None
        if self.use_auditory:
            self.auditory_encoder = auditory_encoder
            self.av_adapter = AudiovisualContextAdapter(embed_dim)

        self.use_keyframe_head = use_keyframe_head
        self.use_scene_text_head = use_scene_text_head
        self.use_structure_head = use_structure_head

        if use_keyframe_head:
            self.keyframe_head = KeyframeHead(embed_dim, head_dim)
        if use_scene_text_head:
            self.scene_text_head = SceneTextHead(embed_dim, head_dim)
        if use_structure_head:
            self.structure_head = StructureHead(embed_dim, embed_dim)

    def forward(self, fmri, auditory_fmri=None):
        """
        Args:
            fmri: (B, 5, seq_len) visual ROI fMRI
            auditory_fmri: (B, 5, aud_seq_len) auditory ROI fMRI, optional
        Returns:
            dict with keys:
                "fmri_cls": (B, 1152) for CLIP alignment
                "fmri_spatial": (B, 226, 2048) raw spatial features
                "slow_feat": (B, 226, 2048) context-enriched features (for fusion)
                "z_key": (B, 1152) keyframe latent (if enabled)
                "z_txt": (B, 1152) scene-text embedding (if enabled)
                "z_str": (B, 226, 2048) structure latent (if enabled)
        """
        fmri_cls, fmri_spatial = self.fmri_encoder(fmri)

        slow_feat = fmri_spatial
        out = {
            "fmri_cls": fmri_cls,
            "fmri_spatial": fmri_spatial,
        }

        if self.use_auditory and auditory_fmri is not None:
            _, aud_spatial = self.auditory_encoder(auditory_fmri)
            slow_feat = self.av_adapter(fmri_spatial, aud_spatial)

        out["slow_feat"] = slow_feat

        if self.use_keyframe_head:
            out["z_key"] = self.keyframe_head(slow_feat)
        if self.use_scene_text_head:
            out["z_txt"] = self.scene_text_head(slow_feat)
        if self.use_structure_head:
            out["z_str"] = self.structure_head(slow_feat)

        return out
