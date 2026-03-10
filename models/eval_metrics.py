import numpy as np
from typing import List, Optional, Tuple
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import torch
from einops import rearrange
import torch.nn.functional as F
from transformers import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logging.set_verbosity_error()


# ==================== Batched feature extraction ====================

@torch.no_grad()
def _preprocess_images_gpu(images_np, target_size, mean, std):
    """Fast GPU-accelerated image preprocessing: resize + normalize."""
    from torchvision.transforms.functional import normalize as tv_normalize, resize as tv_resize
    t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    t = tv_resize(t, list(target_size), antialias=True)
    t = tv_normalize(t, mean.tolist(), std.tolist())
    return t


@torch.no_grad()
def _batch_vit_logits(images_np, processor, model, device, batch_size=256):
    """Batch ViT inference with GPU-accelerated preprocessing."""
    target_size = (processor.size.get('height', 224), processor.size.get('width', 224))
    mean = torch.tensor(processor.image_mean)
    std = torch.tensor(processor.image_std)

    all_logits = []
    for i in range(0, len(images_np), batch_size):
        batch = images_np[i:i+batch_size]
        inputs = _preprocess_images_gpu(batch, target_size, mean, std)
        logits = model(inputs.to(device, torch.float16)).logits.detach().cpu()
        all_logits.append(logits)
    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def _batch_clip_features(images_np, processor, model, device, batch_size=256):
    """Batch CLIP inference with GPU-accelerated preprocessing.
    processor can be AutoProcessor (has .image_processor) or raw image processor.
    """
    if hasattr(processor, 'image_processor'):
        img_proc = processor.image_processor
    else:
        img_proc = processor
    target_size = (img_proc.size.get('shortest_edge', 224),) * 2
    mean = torch.tensor(img_proc.image_mean)
    std = torch.tensor(img_proc.image_std)

    all_feats = []
    for i in range(0, len(images_np), batch_size):
        batch = images_np[i:i+batch_size]
        inputs = _preprocess_images_gpu(batch, target_size, mean, std)
        feats = model(inputs.to(device, torch.float16)).image_embeds.float().cpu()
        all_feats.append(feats)
    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def _batch_videomae_logits(videos_np, processor, model, device, batch_size=16):
    """Batch VideoMAE inference with GPU-accelerated preprocessing."""
    from torchvision.transforms.functional import normalize, resize
    target_size = (processor.size.get('shortest_edge', 224),) * 2
    mean = torch.tensor(processor.image_mean, dtype=torch.float32)
    std = torch.tensor(processor.image_std, dtype=torch.float32)

    all_logits = []
    for i in range(0, len(videos_np), batch_size):
        if i % (batch_size * 10) == 0:
            print(f'      videomae progress: {i}/{len(videos_np)}')
        batch = videos_np[i:i+batch_size]
        t_batch = torch.from_numpy(batch).permute(0, 1, 4, 2, 3).float() / 255.0
        B, T, C, H, W = t_batch.shape
        t_flat = t_batch.reshape(B * T, C, H, W)
        t_resized = resize(t_flat, list(target_size), antialias=True)
        t_resized = normalize(t_resized, mean.tolist(), std.tolist())
        _, C2, h, w = t_resized.shape
        t_resized = t_resized.reshape(B, T, C2, h, w)
        logits = model(t_resized.to(device, torch.float16)).logits.detach().cpu()
        all_logits.append(logits)
    return torch.cat(all_logits, dim=0)


# ==================== Model loaders (load once, reuse) ====================

def load_vit_model(cache_dir='.cache', device='cuda'):
    """Load ViT model and processor once."""
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', cache_dir=cache_dir)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                      cache_dir=cache_dir).to(device, torch.float16)
    model.eval()
    return processor, model


def load_clip_model(cache_dir='.cache', device='cuda'):
    """Load CLIP model and processor once."""
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32",
                                                          cache_dir=cache_dir).to(device, torch.float16)
    model.eval()
    return processor, model


# ==================== N-way accuracy ====================

def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    if isinstance(class_id, int):
        class_id = [class_id]
    pick_range = [i for i in np.arange(len(pred)) if i not in class_id]
    corrects = 0
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        for gt_id in class_id:
            pred_picked = torch.cat([pred[gt_id].unsqueeze(0), pred[idxs_picked]])
            pred_picked = pred_picked.argsort(descending=False)[-top_k:]
            if 0 in pred_picked:
                corrects += 1
                break
    return corrects / num_trials, np.sqrt(corrects / num_trials * (1 - corrects / num_trials) / num_trials)


# ==================== Image classification (batched, model reuse) ====================

@torch.no_grad()
def img_classify_metric(
        pred_videos: np.array,
        gt_videos: np.array,
        n_way: list = [50],
        num_trials: int = 100,
        top_k: int = 1,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        return_std: bool = False,
        batch_size: int = 256,
        preloaded: Optional[Tuple] = None
):
    for nway in n_way:
        assert nway > top_k

    if preloaded is not None:
        processor, model = preloaded
    else:
        processor, model = load_vit_model(cache_dir, device)

    pred_logits = _batch_vit_logits(pred_videos, processor, model, device, batch_size)
    gt_logits = _batch_vit_logits(gt_videos, processor, model, device, batch_size)

    acc_list = [[] for _ in range(len(n_way))]
    std_list = [[] for _ in range(len(n_way))]
    for idx in range(len(pred_videos)):
        gt_class_id = gt_logits[idx].argsort(descending=False)[-3:]
        pred_out = pred_logits[idx].softmax(-1)
        for i, nway in enumerate(n_way):
            acc, std = n_way_top_k_acc(pred_out, gt_class_id, nway, num_trials, top_k)
            acc_list[i].append(acc)
            std_list[i].append(std)
    if return_std:
        return acc_list, std_list
    return acc_list


# ==================== Video classification (batched) ====================

@torch.no_grad()
def video_classify_metric(
        pred_videos: np.array,
        gt_videos: np.array,
        n_way: list = [50],
        num_trials: int = 100,
        top_k: int = 1,
        num_frames: int = 6,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        return_std: bool = False,
        batch_size: int = 16
):
    for nway in n_way:
        assert nway > top_k
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics',
                                                       cache_dir=cache_dir)
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics',
                                                           num_frames=num_frames,
                                                           cache_dir=cache_dir).to(device, torch.float16)
    model.eval()

    print(f'    [video_classify] batch computing logits for {len(pred_videos)} videos...')
    pred_logits = _batch_videomae_logits(pred_videos, processor, model, device, batch_size)
    gt_logits = _batch_videomae_logits(gt_videos, processor, model, device, batch_size)

    acc_list = [[] for _ in range(len(n_way))]
    std_list = [[] for _ in range(len(n_way))]
    for idx in range(len(pred_videos)):
        gt_class_id = gt_logits[idx].argsort(descending=False)[-3:]
        pred_out = pred_logits[idx].softmax(-1)
        for i, nway in enumerate(n_way):
            acc, std = n_way_top_k_acc(pred_out, gt_class_id, nway, num_trials, top_k)
            acc_list[i].append(acc)
            std_list[i].append(std)
    if return_std:
        return acc_list, std_list
    return acc_list


# ==================== CLIP score (batched, model reuse) ====================

@torch.no_grad()
def clip_score_only(
        pred_videos: np.array,
        gt_videos: np.array,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256,
        preloaded: Optional[Tuple] = None
):
    """Batched CLIP score computation. Accepts preloaded=(processor, model)."""
    if preloaded is not None:
        clip_processor, clip_model = preloaded
    else:
        clip_processor, clip_model = load_clip_model(cache_dir, device)

    pred_feats = _batch_clip_features(pred_videos, clip_processor, clip_model, device, batch_size)
    gt_feats = _batch_clip_features(gt_videos, clip_processor, clip_model, device, batch_size)
    pred_feats = F.normalize(pred_feats, dim=-1)
    gt_feats = F.normalize(gt_feats, dim=-1)
    scores = F.cosine_similarity(pred_feats, gt_feats, dim=-1).numpy()
    return np.mean(scores), np.std(scores)


# ==================== Pixel-level metrics ====================

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    if len(img.shape) == 3:
        img = rearrange(img, 'c h w -> h w c')
    elif len(img.shape) == 4:
        img = rearrange(img, 'f c h w -> f h w c')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img

def ssim_score_only(pred_videos: np.array, gt_videos: np.array, **kwargs):
    pred_videos = channel_last(pred_videos)
    gt_videos = channel_last(gt_videos)
    scores = []
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(ssim_metric(pred, gt))
    return np.mean(scores), np.std(scores)

def psnr_score_only(pred_videos: np.array, gt_videos: np.array, **kwargs):
    pred_videos = channel_last(pred_videos)
    gt_videos = channel_last(gt_videos)
    scores = []
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(psnr(gt.astype(np.float64), pred.astype(np.float64), data_range=255))
    return np.mean(scores), np.std(scores)

def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)

def remove_overlap(pred_videos, gt_videos, scene_seg_list, get_scene_seg=False):
    pred_list = []
    gt_list = []
    seg_dict = {}
    for pred, gt, seg in zip(pred_videos, gt_videos, scene_seg_list):
        if '-' not in seg:
            if get_scene_seg:
                if seg not in seg_dict.keys():
                    seg_dict[seg] = seg
                    pred_list.append(pred)
                    gt_list.append(gt)
            else:
                pred_list.append(pred)
                gt_list.append(gt)
    return np.stack(pred_list), np.stack(gt_list)


# ==================== Temporal consistency (batched, model reuse) ====================

@torch.no_grad()
def clip_temporal_consistency(
        pred_videos: np.array,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256,
        preloaded: Optional[Tuple] = None
):
    """CTC: CLIP Temporal Consistency. Accepts preloaded=(processor, model)."""
    if preloaded is not None:
        clip_processor, clip_model = preloaded
    else:
        clip_processor, clip_model = load_clip_model(cache_dir, device)

    n, t = pred_videos.shape[:2]
    all_frames = pred_videos.reshape(n * t, *pred_videos.shape[2:])
    print(f'    [CTC] batch extracting CLIP features for {len(all_frames)} frames...')
    all_feats = _batch_clip_features(all_frames, clip_processor, clip_model, device, batch_size)
    all_feats = F.normalize(all_feats, dim=-1).view(n, t, -1)

    scores = []
    for i in range(n):
        sims = F.cosine_similarity(all_feats[i, :-1], all_feats[i, 1:], dim=-1)
        scores.append(sims.mean().item())
    return np.mean(scores), np.std(scores)


@torch.no_grad()
def dino_temporal_consistency(
        pred_videos: np.array,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256
):
    """DTC: DINO Temporal Consistency."""
    from transformers import ViTModel, ViTImageProcessor as DinoProcessor
    dino_processor = DinoProcessor.from_pretrained("facebook/dino-vitb16", cache_dir=cache_dir)
    dino_model = ViTModel.from_pretrained("facebook/dino-vitb16", cache_dir=cache_dir).to(device, torch.float16)
    dino_model.eval()

    n, t = pred_videos.shape[:2]
    all_frames = pred_videos.reshape(n * t, *pred_videos.shape[2:])
    print(f'    [DTC] batch extracting DINO features for {len(all_frames)} frames...')

    target_size = (dino_processor.size.get('height', 224), dino_processor.size.get('width', 224))
    mean = torch.tensor(dino_processor.image_mean)
    std = torch.tensor(dino_processor.image_std)

    all_feats = []
    for i in range(0, len(all_frames), batch_size):
        batch = all_frames[i:i+batch_size]
        inputs = _preprocess_images_gpu(batch, target_size, mean, std)
        feats = dino_model(inputs.to(device, torch.float16)).last_hidden_state[:, 0].float().cpu()
        all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0)
    all_feats = F.normalize(all_feats, dim=-1).view(n, t, -1)

    scores = []
    for i in range(n):
        sims = F.cosine_similarity(all_feats[i, :-1], all_feats[i, 1:], dim=-1)
        scores.append(sims.mean().item())
    return np.mean(scores), np.std(scores)


# ==================== FVD ====================

def compute_fvd(
        pred_videos: np.array,
        gt_videos: np.array,
        device: Optional[str] = 'cuda',
):
    """FVD: Fréchet Video Distance using I3D features (FP32 for numerical stability)."""
    from cdfvd.fvd import cdfvd

    fvd_calculator = cdfvd(model='i3d', n_real='full', n_fake='full',
                           device=str(device), half_precision=False)
    result = fvd_calculator.compute_fvd(
        real_videos=gt_videos.astype(np.uint8),
        fake_videos=pred_videos.astype(np.uint8)
    )
    return result
