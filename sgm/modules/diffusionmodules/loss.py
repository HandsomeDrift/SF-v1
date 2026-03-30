from typing import List, Optional, Union

import torch
import torch.nn as nn
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
    def __init__(self, training_stage="joint", sf_loss_config=None, **kwargs):
        super().__init__(**kwargs)
        self.training_stage = training_stage

        from sgm.modules.diffusionmodules.sf_losses import (
            AlignmentLoss, SlowBranchLoss, FastBranchLoss, GuidanceLoss
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
        self.fast_loss = FastBranchLoss(
            lambda_dyn=cfg.get("lambda_dyn", 1.0),
            lambda_mot=cfg.get("lambda_mot", 1.0),
            lambda_tc=cfg.get("lambda_tc", 0.5),
        )
        self.guide_loss = GuidanceLoss(
            lambda_gk=cfg.get("lambda_gk", 0.5),
            lambda_gt=cfg.get("lambda_gt", 0.5),
            lambda_gm=cfg.get("lambda_gm", 0.5),
        )

    def __call__(self, network, denoiser, conditioner, input, batch):
        # Run conditioner (SFBrainEmbedder) to get cond and populate _last_slow_out etc.
        cond = conditioner(batch)

        # Get SF intermediate outputs from the embedder
        embedder = conditioner.embedders[0]
        slow_out = getattr(embedder, '_last_slow_out', {})
        fast_out = getattr(embedder, '_last_fast_out', {})

        # Compute SF auxiliary losses (always)
        sf_total = torch.tensor(0.0, device=input.device)
        sf_log = {}

        # Alignment loss (fmri_cls / eeg_cls exist in both SF and baseline modes)
        if "fmri_cls" in slow_out and "eeg_cls" in fast_out:
            l_align, d_align = self.align_loss(slow_out, fast_out)
            sf_total = sf_total + l_align
            sf_log.update(d_align)

        # Slow branch loss
        if "z_key" in slow_out:
            targets = batch.get("sf_targets", {})
            l_slow, d_slow = self.slow_loss(slow_out, targets)
            sf_total = sf_total + l_slow
            sf_log.update(d_slow)

        # Fast branch loss
        if "z_dyn" in fast_out:
            targets = batch.get("sf_targets", {})
            l_fast, d_fast = self.fast_loss(fast_out, targets)
            sf_total = sf_total + l_fast
            sf_log.update(d_fast)

        # Guidance loss (stage 2+)
        if self.training_stage in ("fusion", "joint"):
            if "z_key" in slow_out or "z_mot" in fast_out:
                l_guide, d_guide = self.guide_loss(slow_out, fast_out)
                sf_total = sf_total + l_guide
                sf_log.update(d_guide)

        # Diffusion loss (stage 3 only)
        if self.training_stage == "joint":
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
                w = min(w, self.min_snr_value)

            diff_loss = self.get_loss(model_output, input, w)

            # Normalize SF loss to same scale as diffusion loss
            if sf_total.item() > 0:
                sf_total = sf_total / sf_total.item() * diff_loss.item()
            return diff_loss + sf_total

        # For branch_pretrain / fusion stages: return SF loss only (no diffusion)
        return sf_total
