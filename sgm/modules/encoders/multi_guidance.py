"""Multi-Guidance Decoder Adapter: combine 4 guidance channels into DiT context."""
import torch
import torch.nn as nn


class MultiGuidanceAdapter(nn.Module):
    """
    Computes 4 guidance signals and combines them with fused brain latent
    to produce the final context tensor for the DiT decoder.

    g_key = alpha_key * z_key_proj
    g_txt = alpha_txt * z_txt_proj
    g_mot = alpha_mot * z_mot_proj (cat of z_dyn, z_mot, z_tc)
    g_brain = alpha_brain * z_b

    Final context = LayerNorm(Linear(z_b + g_key + g_txt + g_mot + g_brain))
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
        mot_input_dim=1922,
    ):
        super().__init__()
        self.use_keyframe_guidance = use_keyframe_guidance
        self.use_text_guidance = use_text_guidance
        self.use_motion_guidance = use_motion_guidance
        self.use_brain_latent_guidance = use_brain_latent_guidance
        self.mot_input_dim = mot_input_dim

        if use_keyframe_guidance:
            self.key_proj = nn.Linear(head_dim, brain_dim)
        if use_text_guidance:
            self.txt_proj = nn.Linear(head_dim, brain_dim)
        if use_motion_guidance:
            self.mot_proj = nn.Linear(mot_input_dim, brain_dim)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, brain_dim),
        )

    def forward(self, z_b, alphas, slow_out, fast_out):
        """
        Args:
            z_b: (B, S, brain_dim) fused brain latent
            alphas: dict of (B, 1) weights
            slow_out: dict from SlowBranch with z_key, z_txt, z_str
            fast_out: dict from FastBranch with z_dyn, z_mot, z_tc
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
            B = z_b.shape[0]
            device, dtype = z_b.device, z_b.dtype
            # z_dyn: (B,) or (B,1) scalar, z_mot: (B, mot_dim), z_tc: (B,) or (B,1) scalar
            z_dyn = fast_out.get("z_dyn", torch.zeros(B, device=device, dtype=dtype))
            mot_vec_dim = self.mot_input_dim - 2  # exclude z_dyn(1) and z_tc(1)
            z_mot = fast_out.get("z_mot", torch.zeros(B, mot_vec_dim, device=device, dtype=dtype))
            z_tc = fast_out.get("z_tc", torch.zeros(B, device=device, dtype=dtype))
            # Ensure all are at least 1D
            if z_dyn.dim() == 0: z_dyn = z_dyn.unsqueeze(0)
            if z_dyn.dim() == 1: z_dyn = z_dyn.unsqueeze(-1)  # (B,1)
            if z_tc.dim() == 0: z_tc = z_tc.unsqueeze(0)
            if z_tc.dim() == 1: z_tc = z_tc.unsqueeze(-1)  # (B,1)
            mot_cat = torch.cat([z_dyn, z_mot, z_tc], dim=-1)  # (B, mot_dim+2)
            g_mot = self.mot_proj(mot_cat).unsqueeze(1)
            context = context + alphas["alpha_mot"].unsqueeze(-1) * g_mot

        if self.use_brain_latent_guidance:
            context = context + alphas["alpha_brain"].unsqueeze(-1) * z_b

        return self.out_proj(context)
