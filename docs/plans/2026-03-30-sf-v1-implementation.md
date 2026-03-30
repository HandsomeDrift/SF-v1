# CineBrain-SF v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 CineBrain/CineSync 代码上实现 Slow-Fast 双分支架构，包括 Slow Branch、Fast Branch、Cross-Modal Gated Fusion、Multi-Guidance Decoder Adapter 和对应的损失函数，支持三阶段训练。

**Architecture:** 基于现有 `BrainmbedderCLIP` 扩展为 `SFBrainEmbedder`，新增 Slow/Fast 分支和 Gated Fusion，替换原有 Linear 融合。Decoder 侧新增 Multi-Guidance Adapter，将 4 条 guidance 通道投影为 DiT cross-attention 条件。所有新模块可通过配置开关关闭，关闭时退化为原 CineSync baseline。

**Tech Stack:** PyTorch, SAT (SwissArmyTransformer), DeepSpeed, OmegaConf, CogVideoX-5B

**Codebase Root:** `/home/drift/ts3/SF-v1/CineBrain/`

---

## 现有关键接口参考

```
fMRI encoder: CustomfMRITransformer(in_channels=5, seq_len=8405, embed_dim=2048, num_spatial=226)
  → forward(x: (B,5,8405)) → (cls: (B,1152), spatial: (B,226,2048))

EEG encoder: CustomEEGTransformer(d_model=2048, num_spatial=226, num_layers=12)
  → forward(x: (B,5,64,800)) → (cls: (B,1152), spatial: (B,226,2048))

BrainmbedderCLIP.forward(batch, siglip_model)
  → cat(fmri_spatial, eeg_spatial) → Linear(4096→4096) → (B,226,4096), clip_loss

GeneralConditioner 按 output dim 路由: 3D → "crossattn" → DiT context
DiT text_proj: Linear(4096→3072) 在 ImagePatchEmbeddingMixin 中
```

---

## Task 2: Build Slow Branch

### 2.1 创建公共 Transformer 组件

**Files:**
- Create: `sgm/modules/encoders/common.py`

提取 fmri/eeg/fusion 三文件中重复的 `MultiHeadAttention`, `FeedForward`, `CustomTransformerEncoderLayer` 到公共模块。

```python
# sgm/modules/encoders/common.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, attn_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: query attends to key/value from another modality."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, context):
        B, Sq, _ = query.shape
        Sc = context.shape[1]
        q = self.q_proj(query).reshape(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Sq, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(self.norm1(src), src_mask)
        src = src + self.dropout1(src2)
        src2 = self.feed_forward(self.norm2(src))
        src = src + self.dropout2(src2)
        return src
```

### 2.2 创建 Slow Branch

**Files:**
- Create: `sgm/modules/encoders/slow_branch.py`

```python
# sgm/modules/encoders/slow_branch.py
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
        self.fmri_encoder = fmri_encoder  # existing CustomfMRITransformer (shared, not owned)
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
```

---

## Task 3: Build Fast Branch

**Files:**
- Create: `sgm/modules/encoders/fast_branch.py`

```python
# sgm/modules/encoders/fast_branch.py
"""Fast Motion-Dynamics Branch: EEG → z_dyn, z_mot, z_tc"""
import torch
import torch.nn as nn


class DynamicsHead(nn.Module):
    """Predict dynamic pattern embedding z_dyn from fast features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_dyn: (B, out_dim) via mean pooling"""
        return self.proj(x.mean(dim=1))


class MotionHead(nn.Module):
    """Predict motion latent z_mot from fast features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_mot: (B, out_dim) via mean pooling"""
        return self.proj(x.mean(dim=1))


class TemporalCoherenceHead(nn.Module):
    """Predict temporal coherence token z_tc from fast features."""
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_tc: (B, out_dim) via mean pooling"""
        return self.proj(x.mean(dim=1))


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
        head_dim=1152,
        use_dynamics_head=True,
        use_motion_head=True,
        use_temporal_coherence_head=True,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder  # existing CustomEEGTransformer (shared, not owned)

        self.use_dynamics_head = use_dynamics_head
        self.use_motion_head = use_motion_head
        self.use_temporal_coherence_head = use_temporal_coherence_head

        if use_dynamics_head:
            self.dynamics_head = DynamicsHead(embed_dim, head_dim)
        if use_motion_head:
            self.motion_head = MotionHead(embed_dim, head_dim)
        if use_temporal_coherence_head:
            self.tc_head = TemporalCoherenceHead(embed_dim, head_dim)

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
```

---

## Task 4: Build Cross-Modal Gated Fusion

**Files:**
- Create: `sgm/modules/encoders/gated_fusion.py`

```python
# sgm/modules/encoders/gated_fusion.py
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
            # Fixed equal weights
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
```

---

## Task 5: Multi-Guidance Decoder Adapter

**Files:**
- Create: `sgm/modules/encoders/multi_guidance.py`

Multi-Guidance Adapter 将 4 条 guidance 通道组合并投影为 DiT 能接收的 `context` tensor。它不修改 DiT 内部，而是在 conditioner 输出和 DiT 输入之间做适配。

```python
# sgm/modules/encoders/multi_guidance.py
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

    Final context = project(z_b + g_key + g_txt + g_mot + g_brain)
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
    ):
        super().__init__()
        self.use_keyframe_guidance = use_keyframe_guidance
        self.use_text_guidance = use_text_guidance
        self.use_motion_guidance = use_motion_guidance
        self.use_brain_latent_guidance = use_brain_latent_guidance

        # Project each guidance vector (B, head_dim) → (B, 1, brain_dim) for broadcasting
        if use_keyframe_guidance:
            self.key_proj = nn.Linear(head_dim, brain_dim)
        if use_text_guidance:
            self.txt_proj = nn.Linear(head_dim, brain_dim)
        if use_motion_guidance:
            self.mot_proj = nn.Linear(head_dim * 3, brain_dim)  # cat(z_dyn, z_mot, z_tc)

        # Final output projection (keep at brain_dim=4096 for DiT text_proj compatibility)
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
            g_key = self.key_proj(slow_out["z_key"]).unsqueeze(1)  # (B, 1, brain_dim)
            context = context + alphas["alpha_key"].unsqueeze(-1) * g_key

        if self.use_text_guidance and "z_txt" in slow_out:
            g_txt = self.txt_proj(slow_out["z_txt"]).unsqueeze(1)
            context = context + alphas["alpha_txt"].unsqueeze(-1) * g_txt

        if self.use_motion_guidance:
            mot_parts = []
            for k in ["z_dyn", "z_mot", "z_tc"]:
                if k in fast_out:
                    mot_parts.append(fast_out[k])
                else:
                    mot_parts.append(torch.zeros_like(next(iter(fast_out.values()))))
            mot_cat = torch.cat(mot_parts, dim=-1)  # (B, head_dim*3)
            g_mot = self.mot_proj(mot_cat).unsqueeze(1)
            context = context + alphas["alpha_mot"].unsqueeze(-1) * g_mot

        if self.use_brain_latent_guidance:
            context = context + alphas["alpha_brain"].unsqueeze(-1) * z_b

        return self.out_proj(context)
```

---

## Task 6: SF Losses

**Files:**
- Create: `sgm/modules/diffusionmodules/sf_losses.py`

```python
# sgm/modules/diffusionmodules/sf_losses.py
"""CineBrain-SF v1 loss functions: alignment, slow, fast, guidance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """5-way cross-modal alignment loss (extends existing CLIP loss)."""
    def __init__(self, lambda_fv=1.0, lambda_ft=1.0, lambda_ev=1.0, lambda_et=1.0, lambda_fe=0.5):
        super().__init__()
        self.lambdas = {"fv": lambda_fv, "ft": lambda_ft, "ev": lambda_ev, "et": lambda_et, "fe": lambda_fe}
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)  # ln(1/0.07)

    def contrastive(self, feat_a, feat_b):
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_ab = scale * feat_a @ feat_b.T
        logits_ba = logits_ab.T
        labels = torch.arange(feat_a.shape[0], device=feat_a.device)
        return (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels)) / 2

    def forward(self, slow_out, fast_out, video_embed=None, text_embed=None):
        """
        Args:
            slow_out: dict with "fmri_cls" (B, 1152)
            fast_out: dict with "eeg_cls" (B, 1152)
            video_embed: (B, 1152) from SigLIP, optional (only in training)
            text_embed: (B, 1152) from SigLIP, optional
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total = torch.tensor(0.0, device=slow_out["fmri_cls"].device)
        fmri_cls = slow_out["fmri_cls"]
        eeg_cls = fast_out["eeg_cls"]

        if video_embed is not None:
            losses["L_fv"] = self.contrastive(fmri_cls, video_embed)
            losses["L_ev"] = self.contrastive(eeg_cls, video_embed)
            total = total + self.lambdas["fv"] * losses["L_fv"] + self.lambdas["ev"] * losses["L_ev"]
        if text_embed is not None:
            losses["L_ft"] = self.contrastive(fmri_cls, text_embed)
            losses["L_et"] = self.contrastive(eeg_cls, text_embed)
            total = total + self.lambdas["ft"] * losses["L_ft"] + self.lambdas["et"] * losses["L_et"]

        losses["L_fe"] = self.contrastive(fmri_cls, eeg_cls)
        total = total + self.lambdas["fe"] * losses["L_fe"]

        return total, losses


class SlowBranchLoss(nn.Module):
    """Supervision for slow branch heads."""
    def __init__(self, lambda_key=1.0, lambda_txt=1.0, lambda_str=1.0):
        super().__init__()
        self.lambda_key = lambda_key
        self.lambda_txt = lambda_txt
        self.lambda_str = lambda_str

    def forward(self, slow_out, targets):
        """
        Args:
            slow_out: dict with z_key, z_txt, z_str
            targets: dict with gt_keyframe_embed, gt_text_embed, gt_structure_embed
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(slow_out.values())).device)

        if "z_key" in slow_out and "gt_keyframe_embed" in targets:
            losses["L_key"] = F.mse_loss(slow_out["z_key"], targets["gt_keyframe_embed"])
            total = total + self.lambda_key * losses["L_key"]
        if "z_txt" in slow_out and "gt_text_embed" in targets:
            losses["L_txt"] = 1.0 - F.cosine_similarity(
                slow_out["z_txt"], targets["gt_text_embed"], dim=-1
            ).mean()
            total = total + self.lambda_txt * losses["L_txt"]
        if "z_str" in slow_out and "gt_structure_embed" in targets:
            losses["L_str"] = F.mse_loss(slow_out["z_str"], targets["gt_structure_embed"])
            total = total + self.lambda_str * losses["L_str"]

        return total, losses


class FastBranchLoss(nn.Module):
    """Supervision for fast branch heads."""
    def __init__(self, lambda_dyn=1.0, lambda_mot=1.0, lambda_tc=0.5):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_mot = lambda_mot
        self.lambda_tc = lambda_tc

    def forward(self, fast_out, targets):
        """
        Args:
            fast_out: dict with z_dyn, z_mot, z_tc
            targets: dict with gt_dynamics_embed, gt_motion_embed, gt_tc_embed
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(fast_out.values())).device)

        if "z_dyn" in fast_out and "gt_dynamics_embed" in targets:
            losses["L_dyn"] = F.mse_loss(fast_out["z_dyn"], targets["gt_dynamics_embed"])
            total = total + self.lambda_dyn * losses["L_dyn"]
        if "z_mot" in fast_out and "gt_motion_embed" in targets:
            losses["L_mot"] = F.mse_loss(fast_out["z_mot"], targets["gt_motion_embed"])
            total = total + self.lambda_mot * losses["L_mot"]
        if "z_tc" in fast_out and "gt_tc_embed" in targets:
            losses["L_tc"] = F.mse_loss(fast_out["z_tc"], targets["gt_tc_embed"])
            total = total + self.lambda_tc * losses["L_tc"]

        return total, losses


class GuidanceLoss(nn.Module):
    """Guidance consistency: generated content should match guidance signals."""
    def __init__(self, lambda_gk=0.5, lambda_gt=0.5, lambda_gm=0.5):
        super().__init__()
        self.lambda_gk = lambda_gk
        self.lambda_gt = lambda_gt
        self.lambda_gm = lambda_gm

    def forward(self, slow_out, fast_out, video_embed=None, text_embed=None):
        """
        Consistency between head predictions and stimulus embeddings.
        Uses cosine similarity loss.
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(slow_out.values())).device)

        if "z_key" in slow_out and video_embed is not None:
            losses["L_gk"] = 1.0 - F.cosine_similarity(
                slow_out["z_key"], video_embed, dim=-1
            ).mean()
            total = total + self.lambda_gk * losses["L_gk"]
        if "z_txt" in slow_out and text_embed is not None:
            losses["L_gt"] = 1.0 - F.cosine_similarity(
                slow_out["z_txt"], text_embed, dim=-1
            ).mean()
            total = total + self.lambda_gt * losses["L_gt"]
        if "z_mot" in fast_out and video_embed is not None:
            losses["L_gm"] = 1.0 - F.cosine_similarity(
                fast_out["z_mot"], video_embed, dim=-1
            ).mean()
            total = total + self.lambda_gm * losses["L_gm"]

        return total, losses
```

---

## Task 7: Integrate — SFBrainEmbedder

**Files:**
- Create: `sgm/modules/encoders/sf_embedder.py`
- Modify: `sgm/modules/encoders/modules.py` — add import, keep `BrainmbedderCLIP` untouched

这是核心集成模块，替代 `BrainmbedderCLIP` 成为新的 conditioner embedder。

```python
# sgm/modules/encoders/sf_embedder.py
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

        # CLIP alignment (same as BrainmbedderCLIP)
        self.v_clip_linear = nn.Conv1d(33, 1, 1)
        self.fmri_v_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fmri_t_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eeg_v_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eeg_t_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eeg_fmri_clip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Baseline fallback: simple linear fusion
        self.fmri_eeg_linear = nn.Linear(4096, 4096)

        # SF branches (optional)
        if use_slow_branch:
            self.slow_branch = SlowBranch(
                fmri_encoder=self.fmri_encoder,
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
                head_dim=clip_dim,
                num_spatial=num_spatial,
                use_keyframe_guidance=use_keyframe_guidance,
                use_text_guidance=use_text_guidance,
                use_motion_guidance=use_motion_guidance,
                use_brain_latent_guidance=use_brain_latent_guidance,
            )

    def forward(self, batch, siglip_model):
        self.dtype = self.v_clip_linear.weight.dtype
        clip_loss = 0.0

        # --- CLIP loss (training only) ---
        if self.mode == "train":
            clip_loss = self._compute_clip_loss(batch, siglip_model)

        fmri = batch["fmri"].to(self.dtype)
        eeg = batch["eeg"].to(self.dtype)

        # --- SF v1 path ---
        if self.use_slow_branch and self.use_fast_branch:
            slow_out = self.slow_branch(fmri)
            fast_out = self.fast_branch(eeg)

            if self.use_gated_fusion:
                z_b, alphas = self.gated_fusion(slow_out["slow_feat"], fast_out["fast_feat"])
            else:
                # Fallback: simple concat + linear (CineSync-style)
                z_b = self.fmri_eeg_linear(
                    torch.cat([slow_out["fmri_spatial"], fast_out["eeg_spatial"]], dim=-1)
                )
                alphas = {k: torch.tensor([[0.25]], device=z_b.device).expand(z_b.shape[0], -1)
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
```

**修改 `modules.py`:** 仅在文件末尾添加 import，不改动 `BrainmbedderCLIP`：

在 `modules.py` 末尾添加:
```python
# SF v1 embedder — imported here so GeneralConditioner can instantiate it via config
from sgm.modules.encoders.sf_embedder import SFBrainEmbedder  # noqa: F401
```

---

## Task 8: Configuration Files

**Files:**
- Create: `configs/sf_v1/cinebrain_sf_v1_model.yaml` — 模型配置
- Create: `configs/sf_v1/sf_v1_train_stage1.yaml` — Stage I 训练
- Create: `configs/sf_v1/sf_v1_train_stage2.yaml` — Stage II 训练
- Create: `configs/sf_v1/sf_v1_train_stage3.yaml` — Stage III 联合训练

配置的核心改动：将 conditioner 的 embedder target 从 `sgm.modules.encoders.modules.BrainmbedderCLIP` 改为 `sgm.modules.encoders.sf_embedder.SFBrainEmbedder`，并添加 SF 专用参数。

**Stage 切换通过 loss_fn_config 中的 `training_stage` 参数控制**：
- `branch_pretrain`: 仅计算 L_align + L_slow + L_fast
- `fusion`: 上述 + L_guide
- `joint`: L_total (含 L_diff)

---

## Task 9: Modify Loss Integration

**Files:**
- Modify: `sgm/modules/diffusionmodules/loss.py` — 扩展 `VideoDiffusionLossBrain`

在 `VideoDiffusionLossBrain.__call__` 中，获取 conditioner 的 `_last_slow_out` / `_last_fast_out` / `_last_alphas`，调用 SF losses，按 training_stage 选择性累加。

---

## Task 10: Modify Training Infrastructure

**Files:**
- Modify: `diffusion_video_brain.py` — `disable_untrainable_params` 添加新模块前缀
- Modify: `data_video.py` — `BrainDataset` 添加 supervision target 构建（keyframe/text/structure embed from SigLIP）

---

## 实施顺序

```
Task 2.1  → 公共组件 (common.py)
Task 2.2  → Slow Branch (slow_branch.py)
Task 3    → Fast Branch (fast_branch.py)
Task 4    → Gated Fusion (gated_fusion.py)
Task 5    → Multi-Guidance Adapter (multi_guidance.py)
Task 6    → SF Losses (sf_losses.py)
Task 7    → SFBrainEmbedder 集成 (sf_embedder.py + modules.py)
Task 8    → 配置文件
Task 9    → Loss 集成到训练循环
Task 10   → 训练基础设施修改
```

每个 Task 完成后 commit，保持 baseline 始终可运行。
