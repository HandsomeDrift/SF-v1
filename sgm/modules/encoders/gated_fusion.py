"""Cross-Modal Gated Fusion: slow_feat + fast_feat → alpha weights + z_b"""
import torch
import torch.nn as nn
from sgm.modules.encoders.common import TransformerEncoderLayer


class CrossModalGatedFusion(nn.Module):
    """
    Learns gating weights (alpha_key, alpha_txt, alpha_mot, alpha_brain)
    and produces a fused brain latent z_b from slow and fast branch features.
    Supports fixed_weights mode for ablation.
    """
    def __init__(
        self,
        slow_dim=2048,
        fast_dim=2048,
        hidden_dim=2048,
        output_dim=4096,
        num_heads=16,
        num_layers=4,
        num_spatial=226,
        num_alphas=4,
        fixed_weights=False,
        dropout=0.1,
    ):
        super().__init__()
        self.fixed_weights = fixed_weights
        self.num_alphas = num_alphas

        # Project slow+fast concat to hidden dim
        self.input_proj = nn.Linear(slow_dim + fast_dim, hidden_dim)

        # Modality embeddings
        self.modality_embed = nn.Parameter(torch.randn(2, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_spatial * 2, hidden_dim))

        # Fusion transformer
        self.fusion_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection for z_b
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Gating network: produces alpha weights from pooled fusion features
        if not fixed_weights:
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_alphas),
                nn.Sigmoid(),
            )
        else:
            self.register_buffer(
                "fixed_alpha", torch.ones(num_alphas) / num_alphas
            )

    def forward(self, slow_feat, fast_feat):
        """
        Args:
            slow_feat: (B, S, D_slow) from SlowBranch
            fast_feat: (B, S, D_fast) from FastBranch
        Returns:
            z_b: (B, S, output_dim) fused brain latent
            alphas: dict {"alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"} each (B, 1)
        """
        B, S, _ = slow_feat.shape

        combined = torch.cat([slow_feat, fast_feat], dim=-1)  # (B, S, D_s+D_f)
        h = self.input_proj(combined)  # (B, S, hidden)

        # Two-stream with modality embeddings
        h_slow = h + self.modality_embed[0]
        h_fast = h + self.modality_embed[1]
        h = torch.cat([h_slow, h_fast], dim=1)  # (B, 2S, hidden)
        h = h + self.pos_embed[:, :h.shape[1], :]

        for layer in self.fusion_layers:
            h = layer(h)
        h = self.norm(h)

        # z_b from first S tokens (slow-aligned) projected to output_dim
        z_b = self.output_proj(h[:, :S, :])  # (B, S, output_dim)

        # Gating
        if self.fixed_weights:
            alpha_vec = self.fixed_alpha.unsqueeze(0).expand(B, -1)  # (B, 4)
        else:
            pooled = h.mean(dim=1)  # (B, hidden)
            alpha_vec = self.gate_net(pooled)  # (B, 4)

        alphas = {
            "alpha_key": alpha_vec[:, 0:1],    # (B, 1)
            "alpha_txt": alpha_vec[:, 1:2],
            "alpha_mot": alpha_vec[:, 2:3],
            "alpha_brain": alpha_vec[:, 3:4],
        }

        return z_b, alphas
