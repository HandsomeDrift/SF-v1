from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from sat import mpu


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]

        # [2, 13, 16, 60, 90],[2] dict_keys(['crossattn', 'concat'])  dict_keys(['idx'])
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss

class VideoDiffusionLossBrain(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]

        # [2, 13, 16, 60, 90],[2] dict_keys(['crossattn', 'concat'])  dict_keys(['idx'])
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        loss = self.get_loss(model_output, input, w)
        contrastive_loss = cond["contrastive_loss"] / cond["contrastive_loss"].item() * loss.item()
        return loss + contrastive_loss

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLossSF(VideoDiffusionLoss):
    """
    SF v1 loss: extends VideoDiffusionLoss with Slow-Fast branch losses.

    Training stages:
    - branch_pretrain: L_align + L_slow + L_fast (no diffusion loss)
    - fusion: L_align + L_slow + L_fast + L_guide (no diffusion loss)
    - joint: L_diff + L_align + L_slow + L_fast + L_guide
    """
    def __init__(self, training_stage="joint", sf_loss_config=None, lambda_sf=0.003,
                 router_loss_type="focal", router_focal_gamma=2.0, router_focal_alpha=0.25,
                 router_lambda_schedule="cosine_warmup", router_lambda_start=0.5,
                 router_lambda_end=0.05, router_warmup_iters=500, **kwargs):
        super().__init__(**kwargs)
        self.training_stage = training_stage
        self.lambda_sf = lambda_sf

        # Router loss configuration
        self.router_loss_type = router_loss_type  # "bce", "focal", "soft_focal"
        self.router_focal_gamma = router_focal_gamma
        self.router_focal_alpha = router_focal_alpha
        self.router_lambda_schedule = router_lambda_schedule  # "fixed", "cosine_warmup", "linear_decay"
        self.router_lambda_start = router_lambda_start
        self.router_lambda_end = router_lambda_end
        self.router_warmup_iters = router_warmup_iters
        self.global_step = 0

        from sgm.modules.diffusionmodules.sf_losses import (
            AlignmentLoss, SlowBranchLoss, FastBranchDistillLoss, GuidanceLoss
        )

        cfg = sf_loss_config or {}
        self.align_loss = AlignmentLoss(
            lambda_fv=cfg.get("lambda_fv", 1.0),
            lambda_ft=cfg.get("lambda_ft", 1.0),
            lambda_ev=cfg.get("lambda_ev", 1.0),
            lambda_et=cfg.get("lambda_et", 1.0),
            lambda_fe=cfg.get("lambda_fe", 0.5),
        )
        self.slow_loss = SlowBranchLoss(
            lambda_key=cfg.get("lambda_key", 1.0),
            lambda_txt=cfg.get("lambda_txt", 1.0),
            lambda_str=cfg.get("lambda_str", 1.0),
        )
        self.fast_loss = FastBranchDistillLoss(
            lambda_cls=cfg.get("lambda_distill_cls", 1.0),
            lambda_spatial=cfg.get("lambda_distill_spatial", 1.0),
            # P1 loss weights
            lambda_temporal_delta=cfg.get("lambda_temporal_delta", 1.0),
            lambda_temporal_abs=cfg.get("lambda_temporal_abs", 0.2),
            lambda_flow_traj=cfg.get("lambda_flow_traj", 0.3),
            lambda_dyn=cfg.get("lambda_dyn", 0.1),
            lambda_struct=cfg.get("lambda_struct", 0.0),
        )
        self.guide_loss = GuidanceLoss(
            lambda_gk=cfg.get("lambda_gk", 0.5),
            lambda_gt=cfg.get("lambda_gt", 0.5),
            lambda_gm=cfg.get("lambda_gm", 0.5),
        )

        from sgm.modules.diffusionmodules.sf_losses import AuxAlignmentLoss
        self.aux_align = AuxAlignmentLoss()
        self.lambda_aux = cfg.get("lambda_aux", 0.0)  # 0 by default, enable via config

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """
        Focal Loss: 降低易分类样本的权重，聚焦难样本

        Args:
            pred: (B,) 预测的 alpha_mot (0-1)
            target: (B,) ground truth (0 or 1)
            alpha: 类别平衡权重
            gamma: 聚焦参数，越大越关注难样本

        Returns:
            scalar loss
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        # pt = 预测正确的概率
        pt = torch.where(target == 1, pred, 1 - pred)
        # focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** gamma
        # 类别平衡
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        loss = alpha_t * focal_weight * bce
        return loss.mean()

    def soft_focal_loss(self, pred, target, gamma=2.0, margin=0.2):
        """
        Soft Focal Loss: 只对偏离 margin 的样本施加损失

        对于已经分对且置信度足够的样本（如 target=1, pred>0.8），不施加梯度。
        这避免了模型为了把 alpha 推到极值而破坏画质。

        Args:
            pred: (B,) 预测的 alpha_mot (0-1)
            target: (B,) ground truth (0 or 1)
            gamma: 聚焦参数
            margin: 置信度阈值，超过此阈值的样本不计入损失

        Returns:
            scalar loss
        """
        # 计算预测正确的概率
        pt = torch.where(target == 1, pred, 1 - pred)
        # 只对 pt < (1 - margin) 的样本计算损失
        mask = (pt < (1 - margin)).float()
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        focal_weight = (1 - pt) ** gamma
        loss = mask * focal_weight * bce
        # 归一化：除以有效样本数
        return loss.sum() / (mask.sum() + 1e-8)

    def get_router_lambda(self):
        """
        根据训练步数计算当前的 router loss 权重

        Returns:
            当前步的 lambda_router
        """
        if self.router_lambda_schedule == "fixed":
            return self.router_lambda_start

        elif self.router_lambda_schedule == "cosine_warmup":
            # 前 warmup_iters 步：从 start 保持高权重强制激活 Gate
            # 后续步：cosine 衰减到 end 恢复画质
            if self.global_step < self.router_warmup_iters:
                return self.router_lambda_start
            else:
                progress = (self.global_step - self.router_warmup_iters) / max(1, 2000 - self.router_warmup_iters)
                progress = min(progress, 1.0)
                # cosine 衰减
                lambda_t = self.router_lambda_end + 0.5 * (self.router_lambda_start - self.router_lambda_end) * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                return lambda_t.item() if isinstance(lambda_t, torch.Tensor) else lambda_t

        elif self.router_lambda_schedule == "linear_decay":
            # 线性衰减
            progress = self.global_step / max(1, 2000)
            progress = min(progress, 1.0)
            return self.router_lambda_start + (self.router_lambda_end - self.router_lambda_start) * progress

        else:
            return self.router_lambda_start

    def __call__(self, network, denoiser, conditioner, input, batch):
        # Run conditioner (SFBrainEmbedder) to get cond and populate _last_slow_out etc.
        cond = conditioner(batch)

        # Get SF intermediate outputs from the embedder
        embedder = conditioner.embedders[0]
        slow_out = getattr(embedder, '_last_slow_out', {})
        fast_out = getattr(embedder, '_last_fast_out', {})

        # Move sf_targets to device and match dtype (bf16-safe)
        targets = batch.get("sf_targets", {})
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    targets[k] = v.to(device=input.device, dtype=input.dtype)
                else:
                    targets[k] = v.to(device=input.device)

        # Initialize loss accumulator matching input dtype (bf16)
        sf_total = input.new_tensor(0.0)

        # Alignment loss (P1-4: pass MoCo queues for bs=1 contrastive)
        if "fmri_cls" in slow_out and "eeg_cls" in fast_out:
            video_embed = targets.get("gt_keyframe_embed", None)
            text_embed = targets.get("gt_text_embed", None)
            queues = None
            brain_embedder = conditioner.embedders[0] if conditioner is not None else None
            if brain_embedder is not None and hasattr(brain_embedder, 'get_queues'):
                queues = brain_embedder.get_queues()
                brain_embedder.update_target_queues(video_embed, text_embed)
            l_align, _ = self.align_loss(slow_out, fast_out, video_embed, text_embed, queues=queues)
            sf_total = sf_total + l_align

        # Slow branch loss
        if "z_key" in slow_out and targets:
            l_slow, _ = self.slow_loss(slow_out, targets)
            sf_total = sf_total + l_slow

        # Fast branch loss (P1: now passes targets for temporal dynamics losses)
        if "eeg_cls_proj" in fast_out and "fmri_cls" in slow_out:
            l_fast, fast_losses = self.fast_loss(fast_out, slow_out, targets)
            sf_total = sf_total + l_fast

        # Auxiliary EEG-fMRI alignment (curriculum Stage 2+)
        if self.lambda_aux > 0 and "fmri_cls" in slow_out and "eeg_cls" in fast_out:
            l_aux = self.aux_align(fast_out["eeg_cls"], slow_out["fmri_cls"])
            sf_total = sf_total + self.lambda_aux * l_aux

        # Guidance loss (stage 2+): pass video/text embeds from targets
        if self.training_stage in ("fusion", "joint"):
            video_embed = targets.get("gt_keyframe_embed", None)
            text_embed = targets.get("gt_text_embed", None)
            if "z_key" in slow_out and (video_embed is not None or text_embed is not None):
                l_guide, _ = self.guide_loss(slow_out, fast_out, targets, video_embed, text_embed)
                sf_total = sf_total + l_guide

        # Diffusion loss (stage 2 fusion + stage 3 joint)
        if self.training_stage in ("fusion", "joint"):
            additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}
            alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
            alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
            idx = idx.to(input.device)
            noise = torch.randn_like(input)

            mp_size = mpu.get_model_parallel_world_size()
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

            additional_model_inputs["idx"] = idx

            if self.offset_noise_level > 0.0:
                noise = (
                    noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
                )

            noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
                (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
            )

            if "concat_images" in batch.keys():
                cond["concat"] = batch["concat_images"]

            model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
            w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)

            if self.min_snr_value is not None:
                w = torch.clamp(w, max=self.min_snr_value)

            diff_loss = self.get_loss(model_output, input, w)

            # Gating supervision: alpha_mot vs dyn_labels (Router Loss)
            # 使用 Focal Loss + 权重退火策略解决 Router-DiT 冲突
            alphas = getattr(embedder, '_last_alphas', {})
            if self.training_stage == "joint" and "alpha_mot" in alphas and "gt_dyn_label_2class" in targets:
                pred = alphas["alpha_mot"].float().reshape(-1)
                target = targets["gt_dyn_label_2class"].float().reshape(-1)

                # 根据配置选择 loss 类型
                if self.router_loss_type == "focal":
                    l_router = self.focal_loss(pred, target,
                                              alpha=self.router_focal_alpha,
                                              gamma=self.router_focal_gamma)
                elif self.router_loss_type == "soft_focal":
                    l_router = self.soft_focal_loss(pred, target,
                                                    gamma=self.router_focal_gamma,
                                                    margin=0.2)
                else:  # "bce"
                    l_router = F.binary_cross_entropy(pred, target)

                # 动态权重调度
                lambda_router = self.get_router_lambda()

                # 直接加到 sf_total（不再做复杂的重新缩放）
                sf_total = sf_total + l_router * lambda_router

                # 记录用于日志
                l_router_val = l_router.detach().float().item()
                lambda_router_val = lambda_router

                # 更新全局步数
                self.global_step += 1

            # Fixed SF loss weight (replaces old dynamic scaling that made lambdas ineffective)
            raw_sf = sf_total.detach().float().item()
            sf_total = sf_total * self.lambda_sf

            # Store loss breakdown for logging (include raw sf_total and ratio for diagnostics)
            self._last_loss_breakdown = {
                "debug/raw_sf_total": raw_sf,
                "debug/diff_loss": diff_loss.detach().float().mean().item(),
                "debug/sf_diff_ratio": (raw_sf * self.lambda_sf) / (diff_loss.detach().float().mean().item() + 1e-8),
            }
            if 'l_align' in locals():
                self._last_loss_breakdown["sf/L_align"] = l_align.detach().float().item()
            if 'l_slow' in locals():
                self._last_loss_breakdown["sf/L_slow"] = l_slow.detach().float().item()
            if 'l_fast' in locals():
                self._last_loss_breakdown["sf/L_fast"] = l_fast.detach().float().item()
            if "fast_losses" in locals():
                for fk, fv in fast_losses.items():
                    self._last_loss_breakdown["sf/" + fk] = fv.detach().float().item()
            if 'l_aux' in locals():
                self._last_loss_breakdown["sf/L_aux"] = l_aux.detach().float().item()
            if 'l_guide' in locals():
                self._last_loss_breakdown["sf/L_guide"] = l_guide.detach().float().item()
            if 'l_router_val' in locals():
                self._last_loss_breakdown["sf/L_router"] = l_router_val
                self._last_loss_breakdown["debug/lambda_router"] = lambda_router_val
            self._last_loss_breakdown["sf/total"] = sf_total.detach().float().item()

            return diff_loss + sf_total.to(diff_loss.dtype)

        # Store loss breakdown for logging
        self._last_loss_breakdown = {}
        if 'l_align' in locals():
            self._last_loss_breakdown["sf/L_align"] = l_align.detach().float().item()
        if 'l_slow' in locals():
            self._last_loss_breakdown["sf/L_slow"] = l_slow.detach().float().item()
        if 'l_fast' in locals():
            self._last_loss_breakdown["sf/L_fast"] = l_fast.detach().float().item()
            if "fast_losses" in locals():
                for fk, fv in fast_losses.items():
                    self._last_loss_breakdown["sf/" + fk] = fv.detach().float().item()
        if 'l_aux' in locals():
            self._last_loss_breakdown["sf/L_aux"] = l_aux.detach().float().item()
        if 'l_guide' in locals():
            self._last_loss_breakdown["sf/L_guide"] = l_guide.detach().float().item()
        self._last_loss_breakdown["sf/total"] = sf_total.detach().float().item()

        # For branch_pretrain stage: return SF loss only (no diffusion)
        return sf_total.to(input.dtype)
