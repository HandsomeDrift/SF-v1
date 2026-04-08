"""Cross-Modal Gated Fusion v2: slow_feat + fast_feat → alpha weights + z_b

P2-1 redesign: Replace concat+shared_projection+take_slow with cross-attention mixing.
Slow tokens query fast tokens via cross-attention, allowing fast branch information
to directly influence the output instead of being squeezed through a bottleneck.
"""
import torch
import torch.nn as nn
from sgm.modules.encoders.common import CrossAttention, TransformerEncoderLayer


class CrossModalGatedFusion(nn.Module):
    """
    v2: Cross-attention fusion.
    - Separate projections for slow/fast (no information bottleneck)
    - Slow tokens attend to fast tokens via cross-attention
    - Self-attention refines the fused representation
    - Gating network determines guidance channel weights
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
        self.hidden_dim = hidden_dim

        # Separate projections (no bottleneck from concat)
        self.slow_proj = nn.Linear(slow_dim, hidden_dim)
        self.fast_proj = nn.Linear(fast_dim, hidden_dim)

        # Cross-attention: slow queries attend to fast keys/values
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": CrossAttention(hidden_dim, num_heads, dropout),
                "cross_norm_q": nn.LayerNorm(hidden_dim),
                "cross_norm_kv": nn.LayerNorm(hidden_dim),
                "self_attn": TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout),
            })
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection for z_b
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Gating network: pooled slow+fast → alpha weights
        if not fixed_weights:
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_alphas),
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

        # Separate projections
        h_slow = self.slow_proj(slow_feat)  # (B, S, hidden)
        h_fast = self.fast_proj(fast_feat)  # (B, S, hidden)

        # Cross-attention layers: slow queries fast, then self-attention refines
        h = h_slow
        for layer in self.cross_attn_layers:
            # Cross-attention: slow attends to fast
            q = layer["cross_norm_q"](h)
            kv = layer["cross_norm_kv"](h_fast)
            h = h + layer["cross_attn"](q, kv)
            # Self-attention + FFN
            h = layer["self_attn"](h)

        h = self.norm(h)

        # z_b: all S tokens contribute (not just slow-aligned)
        z_b = self.output_proj(h)  # (B, S, output_dim)

        # Gating: pool both modalities for gate decisions
        if self.fixed_weights:
            alpha_vec = self.fixed_alpha.unsqueeze(0).expand(B, -1)
        else:
            pooled = torch.cat([h_slow.mean(dim=1), h_fast.mean(dim=1)], dim=-1)  # (B, 2*hidden)
            alpha_vec = self.gate_net(pooled)  # (B, 4)

        alphas = {
            "alpha_key": alpha_vec[:, 0:1],
            "alpha_txt": alpha_vec[:, 1:2],
            "alpha_mot": alpha_vec[:, 2:3],
            "alpha_brain": alpha_vec[:, 3:4],
        }

        return z_b, alphas
