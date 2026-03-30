# CineBrain 仓库审计报告 — CineBrain-SF v1

> 审计日期: 2026-03-30
> 目标: 基于现有 CineBrain/CineSync 代码实现 SF v1 Slow-Fast 双分支架构

---

## A. 仓库结构摘要

```
CineBrain/
├── train_video_fmri.py          # 训练入口（SAT DeepSpeed 训练循环）
├── diffusion_video_brain.py     # 扩散引擎（SATVideoDiffusionEngineBrain）
├── data_video.py                # 数据加载（BrainDataset）
├── sample_brain_va.py           # 推理采样
├── get_metric.py                # 评估驱动脚本
├── arguments.py                 # 参数解析 + 配置注入
├── local_config.py              # 本地路径配置加载器
├── dit_video_concat.py          # DiT 主模型（3D patch, I2V 支持）
├── dit_video_concat_fmri.py     # DiT fMRI 变体（2D patch, 简化版） ← 当前使用
├── finetune_multi_gpus.sh       # 4-GPU 训练脚本
├── inference.sh                 # 单 GPU 推理脚本
├── configs/
│   ├── cogvideox_5b_lora_brain_va.yaml   # 模型/LoRA/条件器配置
│   ├── sft_5b_brain_va_clip.yaml         # 训练超参（per-subject）
│   └── infer_brain_va_5b_sub*.yaml       # 推理配置（per-subject）
├── models/
│   └── eval_metrics.py          # 评估指标实现（SSIM/PSNR/FVD/CLIP/VideoMAE等）
├── sgm/modules/
│   ├── encoders/
│   │   ├── fmri_encoder_custom.py  # fMRI Transformer 编码器（24层）
│   │   ├── eeg_encoder_custom.py   # EEG Transformer 编码器（12层）
│   │   ├── fusion.py               # CrossModalFusionTransformer（未使用）
│   │   └── modules.py              # GeneralConditioner + BrainmbedderCLIP
│   ├── diffusionmodules/
│   │   ├── loss.py                 # VideoDiffusionLoss / VideoDiffusionLossBrain
│   │   ├── denoiser.py             # DiscreteDenoiser（VP参数化）
│   │   ├── sampling.py             # VPSDEDPMPP2MSampler（DPM-Solver++ 2M SDE）
│   │   └── lora.py                 # HuggingFace 风格 LoRA（备用）
│   └── ...
├── sat/                         # SAT 框架（SwissArmyTransformer）
│   ├── model/finetune/lora2.py  # SAT LoRA Mixin（训练使用此版本，rank=128）
│   ├── training/deepspeed_training.py  # DeepSpeed 训练循环
│   └── ...
├── vae_modules/                 # VAE 编解码器（ContextParallel 3D）
└── tools/                       # 辅助脚本（权重转换、数据处理）
```

---

## B. 数据流摘要

### 训练数据流

```
[BrainDataset]
  fMRI (.npy) x5 → cat → (B, 5, 8405)     ← visual ROI 体素
  EEG  (.npy) x5 → cat → (B, 5, 64, 800)  ← 5段 x 64通道 x 800时间点
  Video (.mp4) → 33帧 → (B, 33, C, H, W)
  Text → SigLIP tokenize → (B, 1, 64)

            ↓

[SATVideoDiffusionEngineBrain.shared_step]
  video → VAE encode → latent z (B, T, 16, H/8, W/8)
  可选: noised_image_input (第一帧加噪 concat)

            ↓

[VideoDiffusionLoss.__call__]
  1. conditioner(batch) → GeneralConditioner → BrainmbedderCLIP:
     fMRI → CustomfMRITransformer(24层) → fmri_cls (B,1152) + fmri_embed (B,226,2048)
     EEG  → CustomEEGTransformer(12层)  → eeg_cls  (B,1152) + eeg_embed  (B,226,2048)
     cat(fmri_embed, eeg_embed) → Linear(4096→4096) → context (B, 226, 4096)
     [训练时] 5路 CLIP 对比损失 (fmri↔video, fmri↔text, eeg↔video, eeg↔text, eeg↔fmri)

  2. sigma_sampler → alpha_cumprod_sqrt (VP 参数化)
  3. noised = z * alpha + noise * sqrt(1-alpha²)
  4. model_output = denoiser(DiT, noised, alpha, cond)
  5. loss = v-prediction 加权 MSE + contrastive_loss

            ↓

[DiT (dit_video_concat_fmri.DiffusionTransformer)]
  context (B,226,4096) → text_proj(4096→3072) → 与 visual tokens 拼接
  42层 Transformer, hidden=3072, heads=48
  AdaLN 双路调制 (text/image 各自 shift/scale/gate)
  3D RoPE (T/H/W 分频)
  SAT LoraMixin rank=128
```

### 推理数据流

```
fMRI + EEG → BrainmbedderCLIP → cond/uc (CFG: force_uc_zero_embeddings=["fmri"])
           → VPSDEDPMPP2MSampler (50步, DynamicCFG scale=6)
           → VAE decode → video mp4
```

### 关键 Tensor 形状

| 位置 | 形状 |
|------|------|
| fMRI 原始 | `(B, 5, 8405)` — 5个 fMRI volume, 8405 体素 |
| EEG 原始 | `(B, 5, 64, 800)` — 5段, 64通道, 800时间点 |
| fMRI encoder 输出 | `(B, 226, 2048)` spatial + `(B, 1152)` CLS |
| EEG encoder 输出 | `(B, 226, 2048)` spatial + `(B, 1152)` CLS |
| 融合后 context | `(B, 226, 4096)` |
| DiT 内部 hidden | `(B, 226+T*H*W, 3072)` text+visual tokens |
| Video latent | `(B, T, 16, H/8, W/8)` 如 `(B, 9, 16, 60, 90)` |
| 输出视频 | `(B, 33, 3, 480, 720)` |

---

## C. 现有模块映射 → SF v1 需求

| SF v1 需求 | 现有模块 | 状态 |
|---|---|---|
| fMRI 视觉 ROI 编码 | `CustomfMRITransformer` | ✅ 可复用，24层 Transformer，输出 (226, 2048) |
| fMRI 听觉 ROI 编码 | 无 | ❌ 需新建（或从 fMRI encoder 分离 auditory ROI） |
| EEG 编码 | `CustomEEGTransformer` | ✅ 可复用，12层 Transformer，输出 (226, 2048) |
| 多模态融合 | `BrainmbedderCLIP` 中 `fmri_eeg_linear` | ⚠️ 仅 Linear，需替换为 CrossModalGatedFusion |
| Fusion Transformer | `CrossModalFusionTransformer` (fusion.py) | ⚠️ 已有实现但未集成到 pipeline |
| Decoder (CogVideoX DiT) | `dit_video_concat_fmri.DiffusionTransformer` | ✅ 可复用，需添加 multi-guidance adapter |
| LoRA 微调 | `sat.model.finetune.lora2.LoraMixin` | ✅ rank=128 |
| 对比学习 (CLIP) | `BrainmbedderCLIP` 5路 CLIP loss | ✅ 可复用，需扩展为 SF alignment loss |
| VAE 编解码 | `VideoAutoencoderInferenceWrapper` | ✅ 冻结，直接复用 |
| 扩散损失 | `VideoDiffusionLoss` / `VideoDiffusionLossBrain` | ✅ 可复用，需添加 SF 专用 loss |
| 数据加载 | `BrainDataset` | ✅ 需扩展支持 auditory ROI 分离 |
| 评估指标 | `eval_metrics.py` + `get_metric.py` | ✅ 可复用，需添加中间指标 |
| 训练循环 | SAT `training_main` + DeepSpeed | ✅ 直接复用 |
| 配置系统 | YAML + OmegaConf + local_config | ✅ 可扩展 |

---

## D. 可复用模块识别

### 直接复用（不修改）
1. **VAE** — `vae_modules/` + `VideoAutoencoderInferenceWrapper`，冻结
2. **SAT 框架** — `sat/` 全部，训练循环、DeepSpeed、检查点管理
3. **评估指标** — `models/eval_metrics.py` + `get_metric.py`，10类指标
4. **LoRA** — `sat.model.finetune.lora2.LoraMixin`
5. **采样器** — `VPSDEDPMPP2MSampler`
6. **去噪器** — `DiscreteDenoiser`
7. **位置编码** — `Rotary3DPositionEmbeddingMixin`
8. **工具脚本** — `tools/`

### 需修改扩展
1. **`BrainmbedderCLIP`** (modules.py) — 改为 SF 双分支调度器
2. **`GeneralConditioner`** (modules.py) — 需支持多 guidance 通道输出
3. **`CustomfMRITransformer`** — 需支持 visual/auditory ROI 分离输入
4. **`BrainDataset`** (data_video.py) — 需加载 auditory ROI
5. **`VideoDiffusionLossBrain`** (loss.py) — 需添加 SF 专用损失项
6. **DiT** (dit_video_concat_fmri.py) — 需添加 multi-guidance conditioning 接口
7. **训练脚本** (train_video_fmri.py) — 需支持三阶段训练切换
8. **配置文件** — 需新增 SF v1 配置

---

## E. 新模块挂接方案

### E1. Slow Branch 挂接

```
sgm/modules/encoders/slow_branch.py  [新建]
├── VisualEncoderWrapper   — 包装 CustomfMRITransformer，取 visual ROI 子集
├── AuditoryEncoderWrapper — 新建 auditory ROI 编码器（可复用 fMRI encoder 架构）
├── AudiovisualContextAdapter — cross-attention 融合 visual + auditory
├── KeyframeHead           — Linear/MLP → z_key
├── SceneTextHead          — Linear/MLP → z_txt
└── StructureHead          — Linear/MLP → z_str
```

**挂接点**: `BrainmbedderCLIP.forward()` 中 fMRI 编码后、融合前。

### E2. Fast Branch 挂接

```
sgm/modules/encoders/fast_branch.py  [新建]
├── SpatialEncoderWrapper  — 包装 CustomEEGTransformer 空间路径
├── TemporalEncoderWrapper — 包装 CustomEEGTransformer 时间路径
├── DynamicsHead           — Linear/MLP → z_dyn
├── MotionHead             — Linear/MLP → z_mot (latent mode 优先)
└── TemporalCoherenceHead  — Linear/MLP → z_tc
```

**挂接点**: `BrainmbedderCLIP.forward()` 中 EEG 编码后、融合前。

### E3. Cross-Modal Gated Fusion 挂接

```
sgm/modules/encoders/gated_fusion.py  [新建]
└── CrossModalGatedFusion
    — 输入: slow representations + fast representations
    — 输出: alpha_key, alpha_txt, alpha_mot, alpha_brain + z_b (fused latent)
    — gating: sigmoid MLP 或 dual-transformer + gate
```

**挂接点**: 替换 `BrainmbedderCLIP` 中的 `fmri_eeg_linear`，或新建 `SFBrainEmbedder` 类。

### E4. Multi-Guidance Decoder Adapter 挂接

```
sgm/modules/encoders/multi_guidance.py  [新建]
└── MultiGuidanceDecoderAdapter
    — 计算 4 条 guidance: g_key, g_txt, g_mot, g_brain
    — 输出: 拼接/投影为 DiT 的 context tensor
```

**挂接点**: `GeneralConditioner` 输出 → DiT `context` 参数之间。

### E5. 损失函数挂接

```
sgm/modules/diffusionmodules/sf_losses.py  [新建]
├── AlignmentLoss   — 5路 cross-modal alignment (扩展现有 CLIP loss)
├── SlowBranchLoss  — keyframe + scene-text + structure MSE/cosine
├── FastBranchLoss  — dynamics + motion + temporal coherence
└── GuidanceLoss    — guidance 一致性 (g_key↔keyframe, g_txt↔text, g_mot↔motion)
```

**挂接点**: `VideoDiffusionLossBrain.__call__` 中，在 diffusion loss 之后累加。

### E6. 配置系统挂接

```
configs/sf_v1/
├── cinebrain_sf_v1_model.yaml     # 模型配置 (slow/fast/fusion/guidance 开关)
├── sf_v1_train_stage1.yaml        # Stage I: 分支预训练
├── sf_v1_train_stage2.yaml        # Stage II: 融合训练
├── sf_v1_train_stage3.yaml        # Stage III: 联合解码
└── sf_v1_ablation_*.yaml          # 消融实验配置
```

**挂接点**: `arguments.py` 的 `--base` 参数，无需修改参数解析逻辑。

---

## F. 配置系统检查

| 检查项 | 结果 |
|--------|------|
| 配置管理方式 | OmegaConf 合并多个 YAML，`--base` 接收多配置 |
| 新模型注册 | 通过 YAML `target` 字段指定类路径，`instantiate_from_config` 动态加载 |
| Loss 权重新增 | YAML `loss_fn_config.params` 下添加新参数即可 |
| Guidance 开关 | 需在 model config 中新增 `slow_branch/fast_branch/fusion/guidance` 开关 |
| Auditory ROI 开关 | 需在 conditioner config 中新增 `auditory_enabled` 参数 |
| 本地路径 | `local_config.yaml` 占位符替换机制已就绪 |
| 三阶段切换 | 可通过不同 YAML 配置切换，无需改 trainer 代码 |

---

## G. 最小侵入实现计划

### 原则
1. **不新建仓库** — 在现有目录结构中扩展
2. **不破坏 baseline** — 所有新模块可通过配置开关关闭，关闭时退化为原 CineSync
3. **新增文件优先** — 尽量在新文件中实现，减少对现有文件的修改

### 需修改的现有文件（最小集）

| 文件 | 修改内容 |
|------|----------|
| `sgm/modules/encoders/modules.py` | 新建 `SFBrainEmbedder` 类（或扩展 `BrainmbedderCLIP`），注册到 conditioner |
| `sgm/modules/diffusionmodules/loss.py` | 扩展 `VideoDiffusionLossBrain`，添加 SF loss 调用 |
| `data_video.py` | `BrainDataset` 添加 auditory ROI 加载（可选，受 config 控制） |
| `diffusion_video_brain.py` | `disable_untrainable_params` 添加新模块的可训练参数前缀 |
| `train_video_fmri.py` | 三阶段训练逻辑（可能通过 config 控制，不改代码） |

### 需新建的文件

| 文件 | 内容 |
|------|------|
| `sgm/modules/encoders/slow_branch.py` | SlowBranch + 所有子模块 |
| `sgm/modules/encoders/fast_branch.py` | FastBranch + 所有子模块 |
| `sgm/modules/encoders/gated_fusion.py` | CrossModalGatedFusion |
| `sgm/modules/encoders/multi_guidance.py` | MultiGuidanceDecoderAdapter |
| `sgm/modules/diffusionmodules/sf_losses.py` | 4类 SF 专用损失 |
| `configs/sf_v1/*.yaml` | 模型/训练/消融配置 |

---

## H. 风险点与建议

### 高风险
1. **显存压力**: CogVideoX-5B (42层, 3072 hidden) + LoRA rank=128 已经很吃显存。新增 Slow/Fast branch + Gated Fusion 会进一步增加。建议：Stage I 冻结 DiT，仅训练分支；评估是否需要降低 LoRA rank。
2. **fMRI 输入维度**: 当前 fMRI 维度 `seq_len=8405` 但配置中 `seq_len=18946`（两个 config 不一致），需要确认实际 fMRI 数据的维度。Auditory ROI 的维度和分离方式待定。
3. **CLIP 对比损失硬编码关闭**: `BrainmbedderCLIP.forward` 中 `mode="infer"` 硬编码，导致训练时不计算对比损失。SF v1 需要修复此问题。

### 中风险
4. **代码重复**: `MultiHeadAttention/FeedForward/TransformerEncoderLayer` 在三个文件中重复定义。建议提取到公共模块，但不作为 v1 必须任务。
5. **两个 DiT 版本**: `dit_video_concat.py` 和 `dit_video_concat_fmri.py` 功能重叠。当前训练使用 fMRI 版本。Multi-guidance adapter 需决定挂到哪个版本。
6. **SigLIP 无条件加载**: `GeneralConditioner.__init__` 中 SigLIP 模型始终加载，即使推理时不需要。推理阶段会浪费显存。

### 低风险
7. **Python `is not` bug**: `modules.py` 第 365 行使用 `is not` 做字符串比较，应改为 `!=`。
8. **配置文件命名**: 现有配置按 subject 分离，SF v1 需要新的命名约定。
9. **Motion head**: Flow token 实现为 optional，优先使用 latent mode。

---

## I. 审计结论

CineBrain 代码库结构清晰，模块化程度较好。基于 SAT 框架的插件式设计（Mixin + instantiate_from_config）使得新模块挂接相对容易。核心挂接点在 `BrainmbedderCLIP`（条件编码）和 `VideoDiffusionLossBrain`（损失函数）两处。

**建议实施顺序**: Task 1 (Baseline freeze) → Task 2-3 (Slow/Fast branch) → Task 6 (Losses) → Task 4 (Fusion) → Task 5 (Decoder upgrade) → Task 7 (Stage-wise training) → Task 8 (Eval) → Task 9 (Docs)

将 Task 6 提前到 Task 4 之前，因为分支的训练验证需要对应的损失函数。
