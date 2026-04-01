"""
SFBrainEmbedder: Drop-in replacement for BrainmbedderCLIP.
When SF branches are disabled, degrades to CineSync-equivalent behavior.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sgm.modules.encoders.modules import AbstractEmbModel
from sgm.modules.encoders.fmri_encoder_custom import CustomfMRITransformer
from sgm.modules.encoders.eeg_encoder_custom import CustomEEGTransformer
from sgm.modules.encoders.slow_branch import SlowBranch
from sgm.modules.encoders.fast_branch import FastBranch
from sgm.modules.encoders.gated_fusion import CrossModalGatedFusion
from sgm.modules.encoders.multi_guidance import MultiGuidanceAdapter


class SFBrainEmbedder(AbstractEmbModel):
    """
    Slow-Fast Brain Embedder for CineBrain-SF v1.

    Modes:
    - use_slow_branch=False, use_fast_branch=False → CineSync baseline (Linear fusion)
    - use_slow_branch=True, use_fast_branch=True → Full SF v1
    """
    def __init__(
        self,
        # fMRI encoder params
        in_channels=5,
        seq_len=8405,
        embed_dim=2048,
        num_spatial=226,
        fmri_num_layers=24,
        eeg_num_layers=12,
        clip_dim=1152,
        # SF config
        use_slow_branch=True,
        use_fast_branch=True,
        use_gated_fusion=True,
        use_multi_guidance=True,
        # Slow branch config
        use_auditory=False,
        use_keyframe_head=True,
        use_scene_text_head=True,
        use_structure_head=True,
        # Fast branch config
        use_dynamics_head=True,
        use_motion_head=True,
        use_temporal_coherence_head=True,
        use_direction_head=True,
        motion_pca_dim=128,
        num_dyn_classes=3,
        num_dir_classes=8,
        # Fusion config
        fusion_hidden_dim=2048,
        fusion_num_layers=4,
        fixed_weights=False,
        # Guidance config
        use_keyframe_guidance=True,
        use_text_guidance=True,
        use_motion_guidance=True,
        use_brain_latent_guidance=True,
        # Training mode
        mode="infer",
        # Checkpoint
        ckpt="",
    ):
        super().__init__()
        self.mode = mode
        self.use_slow_branch = use_slow_branch
        self.use_fast_branch = use_fast_branch
        self.use_gated_fusion = use_gated_fusion
        self.use_multi_guidance = use_multi_guidance

        # Core encoders (same as BrainmbedderCLIP)
        self.fmri_encoder = CustomfMRITransformer(
            clip_dim=clip_dim, in_channels=in_channels,
            seq_len=seq_len, num_layers=fmri_num_layers, num_spatial=num_spatial
        )
        self.eeg_encoder = CustomEEGTransformer(
            clip_dim=clip_dim, num_layers=eeg_num_layers, num_spatial=num_spatial
        )

        # Auditory fMRI encoder (lighter: half the layers of visual encoder)
        if use_auditory:
            self.auditory_encoder = CustomfMRITransformer(
                clip_dim=clip_dim, in_channels=in_channels,
                seq_len=10541, num_layers=fmri_num_layers // 2, num_spatial=num_spatial
            )

        # CLIP alignment (same as BrainmbedderCLIP)
        self.v_clip_linear = nn.Conv1d(33, 1, 1)
        self.fmri_v_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fmri_t_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eeg_v_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eeg_t_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eeg_fmri_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Baseline fallback: simple linear fusion (CineSync-style)
        self.fmri_eeg_linear = nn.Linear(4096, 4096)

        # SF branches (optional)
        if use_slow_branch:
            self.slow_branch = SlowBranch(
                fmri_encoder=self.fmri_encoder,
                auditory_encoder=self.auditory_encoder if use_auditory else None,
                embed_dim=embed_dim,
                head_dim=clip_dim,
                use_auditory=use_auditory,
                use_keyframe_head=use_keyframe_head,
                use_scene_text_head=use_scene_text_head,
                use_structure_head=use_structure_head,
            )

        if use_fast_branch:
            self.fast_branch = FastBranch(
                eeg_encoder=self.eeg_encoder,
                embed_dim=embed_dim,
                head_dim=clip_dim,
                use_dynamics_head=use_dynamics_head,
                use_motion_head=use_motion_head,
                use_temporal_coherence_head=use_temporal_coherence_head,
                use_direction_head=use_direction_head,
                motion_pca_dim=motion_pca_dim,
                num_dyn_classes=num_dyn_classes,
                num_dir_classes=num_dir_classes,
            )

        if use_gated_fusion and use_slow_branch and use_fast_branch:
            self.gated_fusion = CrossModalGatedFusion(
                slow_dim=embed_dim,
                fast_dim=embed_dim,
                hidden_dim=fusion_hidden_dim,
                output_dim=4096,
                num_heads=16,
                num_layers=fusion_num_layers,
                num_spatial=num_spatial,
                fixed_weights=fixed_weights,
            )

        if use_multi_guidance and use_slow_branch and use_fast_branch:
            self.guidance_adapter = MultiGuidanceAdapter(
                brain_dim=4096,
                head_dim=clip_dim,  # for keyframe/text heads (1152)
                num_spatial=num_spatial,
                use_keyframe_guidance=use_keyframe_guidance,
                use_text_guidance=use_text_guidance,
                use_motion_guidance=use_motion_guidance,
                use_brain_latent_guidance=use_brain_latent_guidance,
                mot_input_dim=140,  # 3(dyn) + 128(mot) + 1(tc) + 8(dir)
            )

    def forward(self, batch, siglip_model):
        self.dtype = self.v_clip_linear.weight.dtype
        clip_loss = torch.tensor(0.0, device=self.v_clip_linear.weight.device, dtype=self.dtype)

        # CLIP loss (training only)
        if self.mode == "train":
            clip_loss = self._compute_clip_loss(batch, siglip_model)

        fmri = batch["fmri"].to(self.dtype)
        eeg = batch["eeg"].to(self.dtype)

        # --- SF v1 path ---
        if self.use_slow_branch and self.use_fast_branch:
            auditory_fmri = batch.get("fmri_auditory")
            if auditory_fmri is not None:
                auditory_fmri = auditory_fmri.to(self.dtype)
            slow_out = self.slow_branch(fmri, auditory_fmri=auditory_fmri)
            fast_out = self.fast_branch(eeg)

            if self.use_gated_fusion:
                z_b, alphas = self.gated_fusion(slow_out["slow_feat"], fast_out["fast_feat"])
            else:
                # Fallback: simple concat + linear (CineSync-style)
                z_b = self.fmri_eeg_linear(
                    torch.cat([slow_out["fmri_spatial"], fast_out["eeg_spatial"]], dim=-1)
                )
                alphas = {k: torch.tensor([[0.25]], device=z_b.device, dtype=z_b.dtype).expand(z_b.shape[0], -1)
                          for k in ["alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"]}

            if self.use_multi_guidance:
                context = self.guidance_adapter(z_b, alphas, slow_out, fast_out)
            else:
                context = z_b

            # Store intermediate outputs for loss computation
            self._last_slow_out = slow_out
            self._last_fast_out = fast_out
            self._last_alphas = alphas

            return context, clip_loss

        # --- Baseline path (no SF branches) ---
        else:
            fmri_cls, fmri_spatial = self.fmri_encoder(fmri)
            eeg_cls, eeg_spatial = self.eeg_encoder(eeg)
            context = self.fmri_eeg_linear(
                torch.cat([fmri_spatial, eeg_spatial], dim=-1)
            )
            self._last_slow_out = {"fmri_cls": fmri_cls, "fmri_spatial": fmri_spatial}
            self._last_fast_out = {"eeg_cls": eeg_cls, "eeg_spatial": eeg_spatial}
            self._last_alphas = {}
            return context, clip_loss

    def _compute_clip_loss(self, batch, siglip_model):
        """Compute 5-way CLIP contrastive loss (same logic as BrainmbedderCLIP)."""
        with torch.no_grad():
            self.fmri_v_clip_logit_scale.clamp_(0, math.log(100))
            self.fmri_t_clip_logit_scale.clamp_(0, math.log(100))
            self.eeg_v_clip_logit_scale.clamp_(0, math.log(100))
            self.eeg_t_clip_logit_scale.clamp_(0, math.log(100))
            self.eeg_fmri_clip_logit_scale.clamp_(0, math.log(100))

            B = batch["text"].shape[0]
            inputs = {
                "input_ids": batch["text"].cuda().reshape(B, 64),
                "pixel_values": batch["video"].cuda().reshape(B * 33, 3, 384, 384),
            }
            with torch.amp.autocast("cuda", enabled=False):
                outputs = siglip_model(**inputs)
            img_embeds = outputs["image_embeds"].to(self.dtype).reshape(B, 33, 1152)
            text_embeds = outputs["text_embeds"].to(self.dtype)

        img_embeds = self.v_clip_linear(img_embeds).squeeze(1)  # (B, 1152)

        fmri_cls, _ = self.fmri_encoder(batch["fmri"].to(self.dtype))
        eeg_cls, _ = self.eeg_encoder(batch["eeg"].to(self.dtype))

        loss = (
            self._clip_loss(fmri_cls, img_embeds, self.fmri_v_clip_logit_scale.exp())
            + self._clip_loss(fmri_cls, text_embeds, self.fmri_t_clip_logit_scale.exp())
            + self._clip_loss(eeg_cls, img_embeds, self.eeg_v_clip_logit_scale.exp())
            + self._clip_loss(eeg_cls, text_embeds, self.eeg_t_clip_logit_scale.exp())
            + self._clip_loss(eeg_cls, fmri_cls, self.eeg_fmri_clip_logit_scale.exp())
        )
        return loss

    @staticmethod
    def _clip_loss(feat_a, feat_b, logit_scale):
        logits_ab = logit_scale * feat_a @ feat_b.T
        labels = torch.arange(feat_a.shape[0], device=feat_a.device)
        return (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ab.T, labels)) / 2
