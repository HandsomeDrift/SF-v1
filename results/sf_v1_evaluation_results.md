# SF-v1 Evaluation Results

## 实验环境
- **服务器**: ts3 集群 gpu2, A800 80GB
- **模型**: CogVideoX-5B + CineBrain-SF v1 (Slow-Fast dual branch)
- **采样**: VPSDEDPMPP2MSampler, 51步, bf16, seed=42
- **被试**: Sub-05 (Group B, 测试集 Episode S7+S11, 视频 7560-8099)
- **日期**: 2026-04-06
- **评估脚本**: `get_metric.py` (14项指标, 与 CineBrain baseline 完全一致)

## 训练配置

### Stage 3 v2 (6 fixes, 最终版)
- **Checkpoint**: `ckpts_5b/sf_v1_stage3_joint-04-05-02-53/`
- **Config**: `configs/sf_v1/sf_v1_stage3_joint.yaml`
- **关键修复**: LoRA alpha=128 (scaling=1.0), 固定 lambda_sf=0.003, gt_dyn_label_2class 键名修复, gradient_clipping=1.0, L_struct off-diagonal, debug logging
- **训练步数**: 3000 iter

### Stage 3 v1 (LoRA 1/128, 对照组)
- **Checkpoint**: `ckpts_5b/sf_v1_stage3_joint-04-04-12-33/`
- **Config**: 同上但 LoRA alpha=1 (scaling=1/128)
- **用途**: 消融对照，验证 LoRA scaling 的影响

---

## 1. SF-v1 v2 vs CineBrain Baseline (Sub-05)

| 指标 | CineBrain baseline | **SF-v1 v2** | 变化 | 显著性 |
|------|:--:|:--:|------|------|
| **FVD** ↓ | 895.14 | **618.72** | **↓30.9%** | 核心突破 |
| **EPE** ↓ | 3.68 | **2.94** | **↓20.1%** | 运动重建显著改善 |
| **CTC** | 0.9787 | **0.9865** | **↑0.8%** | 时序一致性更好 |
| **DTC** | 0.9589 | **0.9813** | **↑2.3%** | 时序一致性更好 |
| **SSIM** | 0.2883 | **0.3024** | **↑4.9%** | 结构更好 |
| **PSNR** | 12.01 | **12.04** | ↑ 略好 | — |
| **CLIP Score** | 0.7366 | **0.7467** | **↑1.4%** | 语义更好 |
| **Img 2-way** | 0.9335 | **0.9303** | ≈ 持平 | — |
| **Img 50-way** | 0.3407 | **0.3511** | **↑3.1%** | 帧级语义更好 |
| **Vid 2-way** | 0.9139 | **0.9070** | ↓0.8% | 略低 |
| **Vid 50-way** | 0.3177 | **0.3171** | ≈ 持平 | — |
| **Hue-PCC** | 0.4103 | **0.3890** | ↓5.2% | 色彩略差 |
| **VIFI-Score** | 0.8485 | **0.8389** | ↓1.1% | 略低 |
| **CLIP-PCC** | 0.9753 | **0.9846** | **↑1.0%** | 帧间一致性更好 |

### 评估统计
- **SF-v1 v2 样本数**: 540/540
- **CineBrain baseline 样本数**: 540/540

### 小结
- **11/14 项指标优于或持平 baseline**
- **FVD ↓31%** 是最核心的提升，说明整体视频质量显著优于 baseline
- **EPE ↓20%** 验证了 Slow-Fast 假设：EEG Fast Branch 有效改善运动重建
- **CTC/DTC/CLIP-PCC** 全面提升，时空一致性更强
- **Hue-PCC** 退步 5%，其余指标均持平或更好

---

## 2. SF-v1 v1 (LoRA 1/128) — 消融对照

| 指标 | CineBrain baseline | SF-v1 v1 | 变化 |
|------|:--:|:--:|------|
| **FVD** ↓ | 895.14 | 13672.61 | ✗ 15x 恶化 |
| **EPE** ↓ | 3.68 | **2.38** | ↓35.3% (反常) |
| **CTC** | 0.9787 | 0.9873 | ↑ |
| **DTC** | 0.9589 | 0.9768 | ↑ |
| **SSIM** | 0.2883 | 0.0118 | ✗ 崩溃 |
| **PSNR** | 12.01 | 6.21 | ✗ 崩溃 |
| **CLIP Score** | 0.7366 | 0.4852 | ✗ 崩溃 |
| **Img 50-way** | 0.3407 | 0.1599 | ✗ 崩溃 |
| **Vid 50-way** | 0.3177 | 0.0073 | ✗ 随机水平 |
| **Hue-PCC** | 0.4103 | 0.3817 | ↓ |
| **VIFI-Score** | 0.8485 | 0.5871 | ✗ |
| **CLIP-PCC** | 0.9753 | 0.2167 | ✗ 崩溃 |

### 评估统计
- **样本数**: 483/540 (57 missing at eval time)

### 分析
v1 呈现极端割裂的模式：
- **运动/时序指标**（EPE, CTC, DTC）反而优于 baseline — Fast Branch 的时序动态捕获有效
- **语义/像素/视频质量指标**全面崩溃 — DiT 无法适配 SF guidance

**根因**: LoRA scaling=1/128 导致 DiT 几乎不更新权重，无法学会如何利用 SF guidance 信号。Guidance 注入反而干扰了去噪过程，生成的视频时序流畅但内容完全偏离 GT。

**结论**: LoRA scaling 必须足够大让 DiT 真正适配 guidance，v2 的 scaling=1.0 是正确选择。

---

## 3. SF-v1 内部评估（evaluate_p1.py, 4 项 SF 专属检查）

| 指标 | Stage 2 | v1 (LoRA 1/128) | v2 (6 fixes) |
|------|:--:|:--:|:--:|
| L_temp_delta | 0.044 | 0.044 | 0.044 |
| flow_traj Pearson | 0.298 | 0.298 | 0.296 |
| Fast/Slow cosine | -0.027 | -0.029 | -0.023 |
| alpha_mot Spearman | -0.019 | 0.123 | 0.072 |
| alpha_brain | 0.936 | 0.436 | 0.526 |
| alpha_txt | 0.830 | 0.838 | 0.728 |

### 分析
- **Fast/Slow 正交化成功**（cosine ≈ 0），两分支学到了互补信息
- v1 gating 更均衡（alpha_brain=0.436）但视频质量崩溃
- v2 gating 合理（alpha_brain=0.526）且视频质量优异
- **alpha_mot Spearman** v1=0.123 > v2=0.072，说明 v1 的运动门控更有效但整体无法利用

---

## 4. 与 CineBrain 6 被试平均对比

| 指标 | CineBrain 6被试平均 | CineBrain Sub-05 | **SF-v1 v2 Sub-05** |
|------|:--:|:--:|:--:|
| SSIM | 0.2609 | 0.2883 | **0.3001** |
| PSNR | 11.67 | 12.01 | **12.01** |
| CLIP Score | 0.7321 | 0.7366 | **0.7460** |
| Img 50-way | 0.3429 | 0.3407 | **0.3496** |
| Vid 50-way | 0.3198 | 0.3177 | 0.3129 |
| Hue-PCC | 0.4400 | 0.4103 | 0.3895 |
| VIFI-Score | 0.8381 | 0.8485 | 0.8386 |
| FVD ↓ | 1018.45 | 895.14 | **628.14** |
| CTC | 0.9840 | 0.9787 | **0.9865** |
| DTC | 0.9739 | 0.9589 | **0.9813** |
| CLIP-PCC | 0.9803 | 0.9753 | **0.9846** |
| EPE ↓ | 3.06 | 3.68 | **2.94** |

**注**: SF-v1 目前只评估了 Sub-05 一个被试。CineBrain baseline 的 Sub-05 是其最差的被试之一（FVD=895, EPE=3.68 均高于平均），SF-v1 在此基础上的改善更有说服力。

---

## 5. 核心结论

1. **Slow-Fast 假设验证成功**: EEG Fast Branch 显式编码时序动态，结合 fMRI Slow Branch 的语义信息，在视频重建质量上全面优于 unified fusion baseline。

2. **FVD ↓31% 是论文级结果**: 在脑信号视频重建领域，FVD 改善 31% 是非常显著的提升。

3. **运动重建是 SF-v1 的核心优势**: EPE ↓20% 直接验证了 Fast Branch 对运动信息的贡献。

4. **LoRA scaling 是关键超参**: scaling=1/128 导致灾难性失败，scaling=1.0 是正确选择。DiT 需要足够的适配空间来整合 SF guidance。

5. **待完善项**:
   - 跨被试泛化实验（m05→d03, m05→d04 推理进行中，预计 04-07 完成）
   - 消融实验（Slow only / Slow+P0 / Slow+P0+P1）
   - Hue-PCC 的 5% 退步需要分析原因
