import os, sys, time
import numpy as np
from local_config import get_paths
from models.eval_metrics import (
    load_vit_model, load_clip_model,
    clip_score_only, ssim_score_only, psnr_score_only,
    img_classify_metric, video_classify_metric,
    clip_temporal_consistency, dino_temporal_consistency,
    compute_fvd
)
import imageio.v3 as iio
import torch

def main(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    gt_list = []
    pred_list = []
    print('loading test results ...')
    for i in range(7560, 8100):
        if i % 100 == 0:
            print(f'  loading {i-7560}/540 ...')
        pred = iio.imread(os.path.join(data_path, f'{str(i).zfill(6)}.mp4'), index=None)
        gt = iio.imread(
            os.path.join(get_paths()["video_dir"], f'{str(i).zfill(6)}.mp4'),
            index=None
        )
        gt_list.append(gt)
        pred_list.append(pred[:33])
    print(f'test results loaded in {time.time()-t0:.1f}s')

    gt_list = np.stack(gt_list)
    pred_list = np.stack(pred_list)
    print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

    n_way = [2, 50]
    num_trials = 100
    top_k = 1

    # ==================== Load models once ====================
    t1 = time.time()
    print('\nLoading shared models ...')
    vit_processor, vit_model = load_vit_model(device=device)
    clip_processor, clip_model = load_clip_model(device=device)
    print(f'  Models loaded in {time.time()-t1:.1f}s')

    # ==================== 1. Video classification ====================
    t1 = time.time()
    print('\n[1/7] Video classification (VideoMAE) ...')
    vid_acc, vid_std = video_classify_metric(
        pred_list, gt_list,
        n_way=n_way, top_k=top_k, num_trials=num_trials,
        num_frames=gt_list.shape[1], return_std=True, device=device
    )
    for i, nway in enumerate(n_way):
        print(f'  Video {nway}-way: {np.mean(vid_acc[i]):.4f} +- {np.mean(vid_std[i]):.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 2. FVD (FP32) ====================
    t1 = time.time()
    print('\n[2/7] FVD (I3D, FP32) ...')
    fvd_score = compute_fvd(pred_list, gt_list, device=device)
    print(f'  FVD: {fvd_score:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 3. CTC (reuse CLIP) ====================
    t1 = time.time()
    print('\n[3/7] CTC (CLIP Temporal Consistency) ...')
    ctc_mean, ctc_std = clip_temporal_consistency(
        pred_list, device=device, preloaded=(clip_processor, clip_model)
    )
    print(f'  CTC: {ctc_mean:.4f} +- {ctc_std:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 4. DTC ====================
    t1 = time.time()
    print('\n[4/7] DTC (DINO Temporal Consistency) ...')
    dtc_mean, dtc_std = dino_temporal_consistency(pred_list, device=device)
    print(f'  DTC: {dtc_mean:.4f} +- {dtc_std:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 5-7. Per-frame metrics (reuse models) ====================
    t_frame_start = time.time()
    print('\n[5-7/7] Per-frame metrics (SSIM, PSNR, CLIP, Img Classification) ...')
    n_frames = pred_list.shape[1]
    ssim_aver, ssim_std_l = [], []
    psnr_aver, psnr_std_l = [], []
    clip_aver, clip_std_l = [], []
    acc_aver = [[] for _ in range(len(n_way))]
    acc_std  = [[] for _ in range(len(n_way))]

    for fi in range(n_frames):
        t1 = time.time()
        pred_frame = pred_list[:, fi]
        gt_frame = gt_list[:, fi]

        # SSIM + PSNR (CPU)
        s_mean, s_std = ssim_score_only(pred_frame, gt_frame)
        ssim_aver.append(s_mean); ssim_std_l.append(s_std)
        p_mean, p_std = psnr_score_only(pred_frame, gt_frame)
        psnr_aver.append(p_mean); psnr_std_l.append(p_std)

        # CLIP Score (reuse model)
        c_mean, c_std = clip_score_only(pred_frame, gt_frame, device=device,
                                        preloaded=(clip_processor, clip_model))
        clip_aver.append(c_mean); clip_std_l.append(c_std)

        # Img classification (reuse model)
        a_list, s_list = img_classify_metric(
            pred_frame, gt_frame,
            n_way=n_way, top_k=top_k, num_trials=num_trials,
            return_std=True, device=device,
            preloaded=(vit_processor, vit_model)
        )
        for idx, nway in enumerate(n_way):
            acc_aver[idx].append(np.mean(a_list[idx]))
            acc_std[idx].append(np.mean(s_list[idx]))

        elapsed = time.time() - t1
        print(f'  frame {fi}/{n_frames}: ssim={s_mean:.4f} psnr={p_mean:.2f} clip={c_mean:.4f} '
              f'2way={np.mean(a_list[0]):.4f} 50way={np.mean(a_list[1]):.4f} ({elapsed:.1f}s)')

    print(f'  Per-frame total: {time.time()-t_frame_start:.1f}s')

    # ==================== Summary ====================
    print('\n' + '='*60)
    print('                  EVALUATION RESULTS')
    print('='*60)
    print(f'SSIM:           {np.mean(ssim_aver):.4f} +- {np.mean(ssim_std_l):.4f}')
    print(f'PSNR:           {np.mean(psnr_aver):.4f} +- {np.mean(psnr_std_l):.4f}')
    print(f'CLIP Score:     {np.mean(clip_aver):.4f} +- {np.mean(clip_std_l):.4f}')
    for i, nway in enumerate(n_way):
        print(f'Img {nway:2d}-way:     {np.mean(acc_aver[i]):.4f} +- {np.mean(acc_std[i]):.4f}')
    for i, nway in enumerate(n_way):
        print(f'Video {nway:2d}-way:   {np.mean(vid_acc[i]):.4f} +- {np.mean(vid_std[i]):.4f}')
    print(f'FVD:            {fvd_score:.4f}')
    print(f'CTC:            {ctc_mean:.4f} +- {ctc_std:.4f}')
    print(f'DTC:            {dtc_mean:.4f} +- {dtc_std:.4f}')
    print('='*60)
    print(f'Total time: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main(data_path='results/brain_va_5b_sub05')
