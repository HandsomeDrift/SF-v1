"""Multi-Guidance Decoder Adapter v3: cross-attention guidance injection.

P2-2 redesign: Replace global vector broadcast with cross-attention.
Spatial tokens (z_b) query guidance embeddings, allowing different positions
to selectively attend to different guidance signals.

v2→v3 changes:
  - Guidance signals stacked as key/value tokens for cross-attention
  - Spatial positions can selectively attend to relevant guidance
  - Alpha weights modulate guidance token contributions before attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiGuidanceAdapter(nn.Module):
    """
    v3: Cross-attention guidance injection.
    Guidance signals (keyframe, text, motion, temporal) are projected to tokens,
    then z_b attends to them via cross-attention for spatial selectivity.
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

        # Project each guidance signal to brain_dim tokens
        if use_keyframe_guidance:
            self.key_proj = nn.Linear(head_dim, brain_dim)
        if use_text_guidance:
            self.txt_proj = nn.Linear(head_dim, brain_dim)
        if use_motion_guidance:
            self.mot_proj = nn.Linear(mot_input_dim, brain_dim)
        if use_temporal_guidance:
            self.temporal_proj = nn.Linear(head_dim, brain_dim)
            self.temporal_gate = nn.Linear(head_dim, 1)

        # Cross-attention: z_b (spatial) queries guidance tokens
        self.cross_attn_q = nn.Linear(brain_dim, brain_dim)
        self.cross_attn_k = nn.Linear(brain_dim, brain_dim)
        self.cross_attn_v = nn.Linear(brain_dim, brain_dim)
        self.cross_attn_out = nn.Linear(brain_dim, brain_dim)
        self.num_heads = guidance_num_heads
        self.head_dim = brain_dim // guidance_num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(brain_dim)
        self.norm_kv = nn.LayerNorm(brain_dim)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, brain_dim),
        )

    def _cross_attend(self, query, key_value):
        """Multi-head cross-attention: query (B, Sq, D) attends to key_value (B, Sk, D)."""
        B, Sq, _ = query.shape
        Sk = key_value.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.cross_attn_q(query).reshape(B, Sq, H, Dh).transpose(1, 2)
        k = self.cross_attn_k(key_value).reshape(B, Sk, H, Dh).transpose(1, 2)
        v = self.cross_attn_v(key_value).reshape(B, Sk, H, Dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Sq, self.brain_dim)
        return self.cross_attn_out(out)

    def forward(self, z_b, alphas, slow_out, fast_out):
        """
        Args:
            z_b: (B, S, brain_dim) fused brain latent
            alphas: dict of (B, 1) weights
            slow_out: dict from SlowBranch with z_key, z_txt
            fast_out: dict from FastBranch with eeg_pooled_proj, global_dyn_token
        Returns:
            context: (B, S, brain_dim) final conditioning for DiT
        """
        B = z_b.shape[0]
        guidance_tokens = []  # unweighted tokens for cross-attention
        alpha_weights = []     # per-token alpha weights (applied AFTER attention)

        # Collect guidance tokens WITHOUT alpha weighting
        # Alpha is applied after cross-attention to avoid dead gradient problem
        if self.use_keyframe_guidance and "z_key" in slow_out:
            g_key = self.key_proj(slow_out["z_key"]).unsqueeze(1)  # (B, 1, D)
            guidance_tokens.append(g_key)
            alpha_weights.append(alphas["alpha_key"].unsqueeze(-1))  # (B, 1, 1)

        if self.use_text_guidance and "z_txt" in slow_out:
            g_txt = self.txt_proj(slow_out["z_txt"]).unsqueeze(1)
            guidance_tokens.append(g_txt)
            alpha_weights.append(alphas["alpha_txt"].unsqueeze(-1))

        if self.use_motion_guidance:
            eeg_feat = fast_out.get("eeg_pooled_proj", None)
            if eeg_feat is not None:
                g_mot = self.mot_proj(eeg_feat).unsqueeze(1)
                if self.use_temporal_guidance and "global_dyn_token" in fast_out:
                    g_temporal = self.temporal_proj(fast_out["global_dyn_token"])
                    gate = torch.sigmoid(self.temporal_gate(fast_out["global_dyn_token"]))
                    g_mot = g_mot + gate.unsqueeze(-1) * g_temporal.unsqueeze(1)
                guidance_tokens.append(g_mot)
                alpha_weights.append(alphas["alpha_mot"].unsqueeze(-1))

        if self.use_brain_latent_guidance:
            g_brain = z_b.mean(dim=1, keepdim=True)
            guidance_tokens.append(g_brain)
            alpha_weights.append(alphas["alpha_brain"].unsqueeze(-1))

        if len(guidance_tokens) > 0:
            # Stack unweighted guidance tokens: (B, N_guide, brain_dim)
            guide_kv = torch.cat(guidance_tokens, dim=1)
            # Cross-attention: all tokens participate equally
            q = self.norm_q(z_b)
            kv = self.norm_kv(guide_kv)
            attn_out = self._cross_attend(q, kv)  # (B, S, brain_dim)

            # Apply alpha weights AFTER attention via weighted residual
            # Each guidance channel's contribution is scaled by its alpha
            alpha_scale = torch.cat(alpha_weights, dim=1).mean(dim=1, keepdim=True)  # (B, 1, 1)
            context = z_b + alpha_scale * attn_out
        else:
            context = z_b

        return self.out_proj(context)
