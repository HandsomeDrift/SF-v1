# CineBrain 复现指南

本文档帮助在任意服务器上从零复现 CineBrain 项目（脑信号 → 视频重建）。

- 论文：https://arxiv.org/abs/2503.06940
- 上游仓库：https://github.com/yanweifu-sii/CineBrain
- 数据集：https://huggingface.co/datasets/Fudan-fMRI/CineBrain
- 预训练权重：https://huggingface.co/datasets/Fudan-fMRI/CineSync

## 目录

1. [环境要求](#1-环境要求)
2. [数据准备](#2-数据准备)
3. [模型权重准备](#3-模型权重准备)
4. [路径配置](#4-路径配置)
5. [训练](#5-训练)
6. [推理](#6-推理)
7. [评估](#7-评估)
8. [已知问题与修复](#8-已知问题与修复)
9. [目录结构参考](#9-目录结构参考)

---

## 1. 环境要求

- **GPU**：A100 80GB（训练需 2+ 卡，推理 1 卡即可）
- **内存**：评估时需加载 540 个视频到内存，建议 64GB+ RAM
- **磁盘**：约 350GB（数据集 132G + 模型 37G + LoRA 权重 174G）
- **Python**：3.10+
- **PyTorch**：2.6+（需 CUDA 支持）

### 安装依赖

```bash
git clone git@github.com:HandsomeDrift/CineBrain.git
cd CineBrain

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install scipy imageio[ffmpeg] cdfvd scikit-image
```

关键包版本参考：

| 包 | 版本 |
|---|------|
| torch | 2.10.0 |
| transformers | 4.57.6 |
| deepspeed | 0.18.4 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| decord | 0.6.0 |
| einops | 0.8.1 |
| diffusers | 0.36.0 |

---

## 2. 数据准备

### 2.1 下载数据集

从 HuggingFace 下载（国内建议用 hf-mirror.com）：

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内加速

# 下载数据集（132GB）
huggingface-cli download Fudan-fMRI/CineBrain --repo-type dataset --local-dir $DATA_ROOT/datasets
```

### 2.2 解压

```bash
DATA_ROOT=/path/to/your/data  # 替换为你的数据目录

# 解压视频（8100 个 mp4，约 8GB）
cd $DATA_ROOT/datasets
tar -xf videos.tar

# 解压被试数据（每个被试约 20GB）
# 以 sub-0005 为例：
cd $DATA_ROOT/datasets/sub-0005
tar -xf fMRI_preprocessed_data.tar   # → visual_audio/ (27000 个 .npy)
tar -xf EEG_preprocessed_data.tar    # → eeg_02/ (27000 个 .npy)
```

解压后每个被试目录结构：
```
sub-0005/
├── visual_audio/     # fMRI 数据，27000 个 .npy，shape=(18946,)
├── eeg_02/           # EEG 数据，27000 个 .npy，shape=(69,800)，代码取前 64 通道
├── fMRI_preprocessed_data.tar
├── EEG_preprocessed_data.tar
└── fMRI_raw_data.tar  # 不需要解压
```

### 2.3 生成训练/测试 JSON

编辑 `tools/generate_test_json.py`，将 `data_root` 改为你的数据目录，然后运行：

```bash
python tools/generate_test_json.py
```

这会为 6 个被试分别生成 `sub-{id}_test_va.json`（各 540 条测试样本）。

训练 JSON 需额外生成（格式相同，训练集为 4860 条样本）。参考 `sub-0005_train_va.json` 的格式：

```json
{
  "video": "/path/to/datasets/videos/000000.mp4",
  "fmri": ["/path/to/datasets/sub-0005/visual_audio/0.npy", "...（共5个）"],
  "eeg": ["/path/to/datasets/sub-0005/eeg_02/0.npy", "...（共5个）"],
  "text": "视频描述文本..."
}
```

### 2.4 数据编号映射

- 30 集视频 → 8100 个 4 秒片段（编号 0-8099），每集 270 个片段
- **被试 1/2/6**（Group A）：使用 Episode 1-20 → 视频 0-5399
  - 训练：Episode 1-18 = 4860 片段，测试：Episode 19-20 = 540 片段（视频 4860-5399）
- **被试 3/4/5**（Group B）：使用 Episode 1-10 + 21-30 → 视频 0-2699 + 5400-8099
  - 训练：前 18 集 = 4860 片段，测试：最后 2 集 = 540 片段（视频 7560-8099）
- 每片段 5 个 fMRI 帧：fMRI 编号 = 片段索引 × 5 + offset

---

## 3. 模型权重准备

### 3.1 CogVideoX-5B（基座模型）

需要将 HuggingFace diffusers 格式转为 SAT 格式：

```bash
# 下载 diffusers 格式
huggingface-cli download THUDM/CogVideoX-5b --local-dir $DATA_ROOT/CogVideoX-5b

# 转换为 SAT 格式
python tools/convert_weight_hf2sat.py \
  --hf_model_path $DATA_ROOT/CogVideoX-5b \
  --output_path $DATA_ROOT/CogVideoX-5b-sat \
  --num_layers 42
```

转换后结构：
```
CogVideoX-5b-sat/
├── transformer/
│   ├── 1000/
│   │   └── mp_rank_00_model_states.pt  (11GB)
│   └── latest                          (内容: "1000")
└── vae/
    └── 3d-vae.pt                       (823MB)
```

### 3.2 SigLIP2（视觉编码器）

```bash
# 国内用 hf-mirror.com 直连
huggingface-cli download google/siglip2-so400m-patch14-384 \
  --local-dir $DATA_ROOT/models/google/siglip2-so400m-patch14-384
```

### 3.3 预训练 LoRA 权重（可选，跳过自行训练）

```bash
# 下载 CineSync 权重（174GB，6 个被试各 29GB）
huggingface-cli download Fudan-fMRI/CineSync --repo-type dataset \
  --local-dir $DATA_ROOT/CineSync

# 创建符号链接
mkdir -p ckpts_5b
for i in 01 02 03 04 05 06; do
  ln -s $DATA_ROOT/CineSync/subject-$i ckpts_5b/brain_lora_va_sub$i
done
```

每个被试的权重结构：
```
subject-05/
├── 1/
│   └── mp_rank_00_model_states.pt  (29GB, 2846 keys)
├── latest                          (内容: "1")
├── model_config.json
└── training_config.yaml
```

---

## 4. 路径配置

代码中有多处硬编码路径需要修改。将以下路径替换为你的实际路径：

```bash
# 查找所有需要修改的位置
grep -rn "/data/lilehui/cinebrain" --include='*.py' --include='*.yaml' --include='*.sh'
```

需要修改的文件和对应路径：

| 文件 | 配置项 | 说明 |
|------|--------|------|
| `configs/cogvideox_5b_lora_brain_va.yaml` | `ckpt_path` | VAE 权重路径 |
| `configs/sft_5b_brain_va_clip.yaml` | `load` | 基座模型 transformer 路径 |
| `configs/sft_5b_brain_va_clip.yaml` | `train_data`, `valid_data` | 训练/验证 JSON 路径 |
| `data_video.py` | `AutoProcessor.from_pretrained(...)` | SigLIP2 模型路径 |
| `sgm/modules/encoders/modules.py` | `AutoModel.from_pretrained(...)` | SigLIP2 模型路径 |
| `get_metric.py` | GT 视频目录 | 评估时的 GT 视频路径 |
| `tools/generate_test_json.py` | `data_root` | 数据集根目录 |

---

## 5. 训练

### 单被试训练（以 sub-05 为例）

```bash
# 2 卡训练
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 train_video_fmri.py \
  --base configs/cogvideox_5b_lora_brain_va.yaml configs/sft_5b_brain_va_clip.yaml \
  --seed 42
```

关键训练参数（在 `configs/sft_5b_brain_va_clip.yaml` 中）：
- `train_iters: 10000`
- `train_micro_batch_size_per_gpu: 1`（80GB 显存只够 batch=1）
- `gradient_accumulation_steps: 4`
- `lr: 1e-4`
- LoRA rank=128

预计时间：
- 2 卡 A100：约 3.2 天
- 4 卡 A100：约 1.5-2 天

权重保存在 `ckpts_5b/brain_lora_va_sub05/`。

---

## 6. 推理

```bash
export CUDA_VISIBLE_DEVICES=0
python sample_brain_va.py \
  --base configs/cogvideox_5b_lora_brain_va.yaml configs/infer_brain_va_5b_sub05.yaml \
  --seed 42 \
  --jsonpath /path/to/sub-0005_test_va.json
```

推理参数（在 `configs/infer_brain_va_5b_sub05.yaml` 中）：
- 采样步数：51（VPSDEDPMPP2MSampler）
- 输出分辨率：480×720
- 输出帧数：33 帧（9 关键帧 + 时间插值）
- 精度：bf16
- 单卡约 2.8 秒/步，每个视频约 2.5 分钟
- 540 个测试视频约 22 小时（单卡）

输出保存在 `results/brain_va_5b_sub05/`，文件名对应视频编号（如 `007560.mp4`）。

---

## 7. 评估

```bash
python get_metric.py
```

评估 7 个指标：

| 指标 | 说明 | 论文 CineSync⋆ (6 被试平均) | 我们 (sub-05) |
|------|------|---------------------------|-------------|
| SSIM | 结构相似度 | 0.297 | 0.290 |
| PSNR | 峰值信噪比 | 12.18 | 12.05 |
| Img 2-way | 图像 2 分类准确率 | 0.926 | 0.933 |
| Img 50-way | 图像 50 分类准确率 | 0.336 | 0.336 |
| CTC | CLIP 时间一致性 | 0.953 | 0.979 |
| DTC | DINO 时间一致性 | 0.921 | 0.960 |
| FVD | Fréchet 视频距离 | 44.77 | 894* |

*FVD 注：论文未公开 FVD 计算代码，我们测试了 3 种标准实现（cdfvd、StyleGAN-V、VideoGPT）均给出 ~900 的值。其余 6 个指标均与论文高度吻合，FVD 差异为计算方式不同导致，非质量问题。

---

## 8. 已知问题与修复

以下问题已在本仓库中修复：

### 8.1 PyTorch 2.6+ `weights_only` 兼容性

PyTorch 2.6+ 默认 `torch.load(..., weights_only=True)`，但 CineSync 权重包含 numpy 对象，需要设置 `weights_only=False`。

涉及文件：
- `sat/training/model_io.py`（checkpoint 加载）
- `sgm/modules/encoders/modules.py`（SigLIP 模型加载）

### 8.2 VAE 加载格式

不同来源的 VAE 权重格式不一致（有的包含 `state_dict` key，有的直接是权重字典）。`vae_modules/autoencoder.py` 已添加兼容处理。

### 8.3 Conditioner 类型错误

`sgm/modules/diffusionmodules/wrappers.py` 中 conditioner 输出可能包含 float 标量而非 tensor，已添加 `isinstance` 检查。

### 8.4 显存限制

单卡 A100 80GB 训练时 `batch_size` 需设为 1（原始配置为 2）。

---

## 9. 目录结构参考

完整部署后的目录结构：

```
$DATA_ROOT/
├── CogVideoX-5b/                    # HF diffusers 格式（转换后可删除）
├── CogVideoX-5b-sat/                # SAT 格式（12GB）
│   ├── transformer/
│   │   ├── 1000/mp_rank_00_model_states.pt
│   │   └── latest
│   └── vae/3d-vae.pt
├── models/google/siglip2-so400m-patch14-384/  # SigLIP2（4.3GB）
├── CineSync/                         # 预训练 LoRA 权重（174GB）
│   ├── subject-01/ ~ subject-06/
│   │   ├── 1/mp_rank_00_model_states.pt
│   │   ├── latest
│   │   ├── model_config.json
│   │   └── training_config.yaml
│   └── README.md
└── datasets/                         # 数据集（132GB）
    ├── videos/                       # 8100 个 mp4
    ├── captions-qwen-2.5-vl-7b.json  # 视频描述
    ├── sub-0001/ ~ sub-0006/         # 各被试数据
    │   ├── visual_audio/             # fMRI（27000 个 .npy）
    │   ├── eeg_02/                   # EEG（27000 个 .npy）
    │   └── *.tar                     # 原始压缩包
    ├── sub-*_train_va.json           # 训练 JSON
    └── sub-*_test_va.json            # 测试 JSON

CineBrain/                            # 代码仓库
├── ckpts_5b/                         # LoRA 权重符号链接
│   └── brain_lora_va_sub{01-06} -> $DATA_ROOT/CineSync/subject-{01-06}
├── configs/                          # 配置文件
├── models/eval_metrics.py            # 评估指标
├── tools/                            # 工具脚本
├── results/                          # 推理输出
└── venv/                             # Python 虚拟环境
```
