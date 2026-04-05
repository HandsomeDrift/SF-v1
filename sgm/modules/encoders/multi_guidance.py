"""Multi-Guidance Decoder Adapter v2: combine guidance channels into DiT context.

v2 changes: motion guidance now uses distilled EEG pooled features (2048-dim)
instead of classification head outputs (140-dim cat of dyn/mot/tc/dir).

P1 extension: optional gated residual adapter for temporal dynamics guidance.
"""
import torch
import torch.nn as nn


class MultiGuidanceAdapter(nn.Module):
    """
    Computes guidance signals and combines them with fused brain latent
    to produce the final context tensor for the DiT decoder.

    v2: motion guidance uses eeg_pooled_proj (2048-dim distilled features)
    instead of concatenated classification outputs (140-dim).

    P1: optional temporal guidance via gated residual from global_dyn_token.
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
        mot_input_dim=2048,  # v2: distilled EEG feature dim (was 140)
        # P1: temporal dynamics guidance
        use_temporal_guidance=False,
    ):
        super().__init__()
        self.use_keyframe_guidance = use_keyframe_guidance
        self.use_text_guidance = use_text_guidance
        self.use_motion_guidance = use_motion_guidance
        self.use_brain_latent_guidance = use_brain_latent_guidance
        self.use_temporal_guidance = use_temporal_guidance
        self.mot_input_dim = mot_input_dim

        if use_keyframe_guidance:
            self.key_proj = nn.Linear(head_dim, brain_dim)
        if use_text_guidance:
            self.txt_proj = nn.Linear(head_dim, brain_dim)
        if use_motion_guidance:
            self.mot_proj = nn.Linear(mot_input_dim, brain_dim)

        # P1: temporal dynamics gated residual adapter
        if use_temporal_guidance:
            self.temporal_proj = nn.Linear(head_dim, brain_dim)
            self.temporal_gate = nn.Linear(head_dim, 1)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, brain_dim),
        )

    def forward(self, z_b, alphas, slow_out, fast_out):
        """
        Args:
            z_b: (B, S, brain_dim) fused brain latent
            alphas: dict of (B, 1) weights
            slow_out: dict from SlowBranch with z_key, z_txt
            fast_out: dict from FastBranch with eeg_pooled_proj, and optionally global_dyn_token
        Returns:
            context: (B, S, brain_dim) final conditioning for DiT
        """
        context = z_b.clone()

        if self.use_keyframe_guidance and "z_key" in slow_out:
            g_key = self.key_proj(slow_out["z_key"]).unsqueeze(1)
            context = context + alphas["alpha_key"].unsqueeze(-1) * g_key

        if self.use_text_guidance and "z_txt" in slow_out:
            g_txt = self.txt_proj(slow_out["z_txt"]).unsqueeze(1)
            context = context + alphas["alpha_txt"].unsqueeze(-1) * g_txt

        if self.use_motion_guidance:
            # v2: use distilled EEG pooled features as motion guidance
            eeg_feat = fast_out.get("eeg_pooled_proj", None)
            if eeg_feat is not None:
                g_mot = self.mot_proj(eeg_feat).unsqueeze(1)  # (B, 1, brain_dim)

                # P1: enhance motion guidance with temporal dynamics (gated residual)
                if self.use_temporal_guidance and "global_dyn_token" in fast_out:
                    g_temporal = self.temporal_proj(fast_out["global_dyn_token"])  # (B, brain_dim)
                    gate = torch.sigmoid(self.temporal_gate(fast_out["global_dyn_token"]))  # (B, 1)
                    g_mot = g_mot + gate.unsqueeze(-1) * g_temporal.unsqueeze(1)

                context = context + alphas["alpha_mot"].unsqueeze(-1) * g_mot

        if self.use_brain_latent_guidance:
            context = context + alphas["alpha_brain"].unsqueeze(-1) * z_b

        return self.out_proj(context)
