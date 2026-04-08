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
        # P1: Temporal dynamics
        use_temporal_dynamics=False,
        num_temporal_queries=9,
        temporal_d_model=512,
        use_temporal_guidance=False,
        use_causal_mask=False,
        sparse_attn_drop=0.0,
        flow_codebook_k=0,
        # Training mode
        mode="infer",
        # Checkpoint
        ckpt="",
        # Branch freezing for curriculum training
        freeze_slow_branch=False,
        freeze_fast_branch=False,
    ):
        super().__init__()
        self.mode = mode
        self.use_slow_branch = use_slow_branch
        self.use_fast_branch = use_fast_branch
        self.freeze_slow_branch = freeze_slow_branch
        self.freeze_fast_branch = freeze_fast_branch
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
                use_temporal_dynamics=use_temporal_dynamics,
                num_temporal_queries=num_temporal_queries,
                temporal_d_model=temporal_d_model,
                use_causal_mask=use_causal_mask,
                sparse_attn_drop=sparse_attn_drop,
                flow_codebook_k=flow_codebook_k,
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
                mot_input_dim=2048,  # v2: distilled EEG pooled feature dim
                use_temporal_guidance=use_temporal_guidance,
            )

        # P1-4: MoCo queue for contrastive alignment at bs=1
        self.queue_size = 512
        self.register_buffer("queue_fmri", torch.randn(self.queue_size, clip_dim))
        self.register_buffer("queue_eeg", torch.randn(self.queue_size, clip_dim))
        self.register_buffer("queue_video", torch.randn(self.queue_size, clip_dim))
        self.register_buffer("queue_text", torch.randn(self.queue_size, clip_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # Normalize initial random queues
        self.queue_fmri = F.normalize(self.queue_fmri, dim=1)
        self.queue_eeg = F.normalize(self.queue_eeg, dim=1)
        self.queue_video = F.normalize(self.queue_video, dim=1)
        self.queue_text = F.normalize(self.queue_text, dim=1)

        # Apply branch freezing for curriculum training (C-02 fix)
        if self.freeze_slow_branch and hasattr(self, 'slow_branch'):
            for p in self.slow_branch.parameters():
                p.requires_grad_(False)
            print("[SFBrainEmbedder] Slow branch frozen (%d params)" %
                  sum(p.numel() for p in self.slow_branch.parameters()))
        if self.freeze_fast_branch and hasattr(self, 'fast_branch'):
            for p in self.fast_branch.parameters():
                p.requires_grad_(False)
            print("[SFBrainEmbedder] Fast branch frozen (%d params)" %
                  sum(p.numel() for p in self.fast_branch.parameters()))

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

            # P1-4: Update MoCo queues (training only, no grad)
            if self.mode == "train":
                self._update_queues(slow_out, fast_out, batch)

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

    @torch.no_grad()
    def _update_queues(self, slow_out, fast_out, batch):
        """P1-4: Update MoCo queues with current batch embeddings."""
        B = slow_out["fmri_cls"].shape[0]
        ptr = int(self.queue_ptr)

        self.queue_fmri[ptr:ptr+B] = F.normalize(slow_out["fmri_cls"].detach().float(), dim=1)
        self.queue_eeg[ptr:ptr+B] = F.normalize(fast_out["eeg_cls"].detach().float(), dim=1)

        # Video/text embeddings come from targets (may not be available at embedder level)
        # They'll be updated via update_target_queues() called from loss computation
        self.queue_ptr[0] = (ptr + B) % self.queue_size

    @torch.no_grad()
    def update_target_queues(self, video_embed, text_embed):
        """P1-4: Update video/text queues (called from loss after targets available)."""
        if video_embed is None and text_embed is None:
            return
        B = video_embed.shape[0] if video_embed is not None else text_embed.shape[0]
        # Use same ptr as brain queues (they advance together)
        ptr = (int(self.queue_ptr) - B) % self.queue_size
        if video_embed is not None:
            self.queue_video[ptr:ptr+B] = F.normalize(video_embed.detach().float(), dim=1)
        if text_embed is not None:
            self.queue_text[ptr:ptr+B] = F.normalize(text_embed.detach().float(), dim=1)

    def get_queues(self):
        """P1-4: Return current queue contents for alignment loss."""
        return {
            "fmri": self.queue_fmri.detach(),
            "eeg": self.queue_eeg.detach(),
            "video": self.queue_video.detach(),
            "text": self.queue_text.detach(),
        }

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
