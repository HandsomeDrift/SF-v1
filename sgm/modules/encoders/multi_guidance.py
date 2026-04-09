"""Multi-Guidance Decoder Adapter v3.1: per-channel cross-attention + alpha weighting.

Each guidance signal has its own cross-attention with z_b (spatial selectivity),
then alpha weights control each channel's contribution via additive residual.
This avoids the dead-alpha problem while maintaining spatial selectivity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidanceCrossAttention(nn.Module):
    """Lightweight single-channel cross-attention: z_b queries one guidance token."""
    def __init__(self, brain_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = brain_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(brain_dim, brain_dim)
        self.k_proj = nn.Linear(brain_dim, brain_dim)
        self.v_proj = nn.Linear(brain_dim, brain_dim)
        self.out_proj = nn.Linear(brain_dim, brain_dim)

    def forward(self, query, kv):
        """query: (B, S, D), kv: (B, 1, D) → (B, S, D)"""
        B, Sq, D = query.shape
        Sk = kv.shape[1]
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(query).reshape(B, Sq, H, Dh).transpose(1, 2)
        k = self.k_proj(kv).reshape(B, Sk, H, Dh).transpose(1, 2)
        v = self.v_proj(kv).reshape(B, Sk, H, Dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Sq, D)
        return self.out_proj(out)


class MultiGuidanceAdapter(nn.Module):
    """
    v3.1: Per-channel cross-attention + alpha-weighted additive residual.
    Each guidance channel gets its own cross-attention (spatial selectivity),
    then alpha controls the contribution (no dead-alpha problem).
    """
    def __init__(
        self,
        brain_dim=4096,
        head_dim=1152,
        num_spatial=226,
        use_keyframe_guidance=True,
        use_text_guidance=True,
        use_motion_guidance=True,
        use_brain_latent_guidance=True,
        mot_input_dim=2048,
        use_temporal_guidance=False,
        guidance_num_heads=16,
    ):
        super().__init__()
        self.use_keyframe_guidance = use_keyframe_guidance
        self.use_text_guidance = use_text_guidance
        self.use_motion_guidance = use_motion_guidance
        self.use_brain_latent_guidance = use_brain_latent_guidance
        self.use_temporal_guidance = use_temporal_guidance
        self.brain_dim = brain_dim

        # Per-channel projections + cross-attention
        if use_keyframe_guidance:
            self.key_proj = nn.Linear(head_dim, brain_dim)
            self.key_attn = GuidanceCrossAttention(brain_dim, guidance_num_heads)

        if use_text_guidance:
            self.txt_proj = nn.Linear(head_dim, brain_dim)
            self.txt_attn = GuidanceCrossAttention(brain_dim, guidance_num_heads)

        if use_motion_guidance:
            self.mot_proj = nn.Linear(mot_input_dim, brain_dim)
            self.mot_attn = GuidanceCrossAttention(brain_dim, guidance_num_heads)

        if use_temporal_guidance:
            self.temporal_proj = nn.Linear(head_dim, brain_dim)
            self.temporal_gate = nn.Linear(head_dim, 1)

        self.norm = nn.LayerNorm(brain_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, brain_dim),
        )

    def forward(self, z_b, alphas, slow_out, fast_out):
        """
        Each guidance channel: proj → cross-attn(z_b, guidance) → alpha * result
        All channels added to z_b as residuals.
        """
        context = z_b.clone()
        q = self.norm(z_b)

        if self.use_keyframe_guidance and "z_key" in slow_out:
            g_key = self.key_proj(slow_out["z_key"]).unsqueeze(1)  # (B, 1, D)
            context = context + alphas["alpha_key"].unsqueeze(-1) * self.key_attn(q, g_key)

        if self.use_text_guidance and "z_txt" in slow_out:
            g_txt = self.txt_proj(slow_out["z_txt"]).unsqueeze(1)
            context = context + alphas["alpha_txt"].unsqueeze(-1) * self.txt_attn(q, g_txt)

        if self.use_motion_guidance:
            eeg_feat = fast_out.get("eeg_pooled_proj", None)
            if eeg_feat is not None:
                g_mot = self.mot_proj(eeg_feat).unsqueeze(1)
                if self.use_temporal_guidance and "global_dyn_token" in fast_out:
                    g_temporal = self.temporal_proj(fast_out["global_dyn_token"])
                    gate = torch.sigmoid(self.temporal_gate(fast_out["global_dyn_token"]))
                    g_mot = g_mot + gate.unsqueeze(-1) * g_temporal.unsqueeze(1)
                context = context + alphas["alpha_mot"].unsqueeze(-1) * self.mot_attn(q, g_mot)

        if self.use_brain_latent_guidance:
            context = context + alphas["alpha_brain"].unsqueeze(-1) * z_b

        return self.out_proj(context)
