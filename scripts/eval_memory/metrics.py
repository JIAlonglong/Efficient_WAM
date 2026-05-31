"""Evaluation metrics for memory experiments."""

import torch
import torch.nn.functional as F
from typing import Dict


def _to_btcwh(x: torch.Tensor) -> torch.Tensor:
    """Convert 5D tensor to [B, T, C, H, W]. Handles both [B,C,T,H,W] and [B,T,C,H,W]."""
    if x.dim() != 5:
        return x
    # Heuristic: if dim 2 has value 3 and dim 1 doesn't, it's [B, C, T, H, W]
    if x.shape[2] == 3 and x.shape[1] != 3:
        return x.permute(0, 2, 1, 3, 4)
    return x


def compute_action_mse(pred_actions: torch.Tensor, gt_actions: torch.Tensor) -> Dict[str, float]:
    """
    Compute Action MSE between predicted and ground-truth actions.

    Args:
        pred_actions: [B, chunk_size, action_dim]
        gt_actions: [B, chunk_size, action_dim]

    Returns:
        Dict with 'action_mse', 'action_mse_per_step', 'action_mse_per_dim'
    """
    total_mse = F.mse_loss(pred_actions, gt_actions, reduction='mean').item()
    per_step = F.mse_loss(pred_actions, gt_actions, reduction='none')
    per_step_mean = per_step.mean(dim=(0, 2)).tolist()
    per_dim_mean = per_step.mean(dim=(0, 1)).tolist()

    return {
        'action_mse': total_mse,
        'action_mse_per_step': per_step_mean,
        'action_mse_per_dim': per_dim_mean,
    }


class LPIPSMetric:
    """Video LPIPS using a pretrained backbone."""

    def __init__(self, device: str = 'cuda', net: str = 'alex'):
        self.device = device
        self.lpips_fn = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net).to(device).eval()
        except ImportError:
            print("WARNING: lpips package not installed. Video LPIPS will be skipped.")

    @torch.no_grad()
    def compute(self, pred_frames: torch.Tensor, gt_frames: torch.Tensor) -> Dict[str, float]:
        if self.lpips_fn is None:
            return {'video_lpips': float('nan'), 'video_lpips_per_frame': []}

        pred_frames = _to_btcwh(pred_frames)
        gt_frames = _to_btcwh(gt_frames)

        B, T, C, H, W = pred_frames.shape
        pred_norm = pred_frames * 2.0 - 1.0
        gt_norm = gt_frames * 2.0 - 1.0

        per_frame_lpips = []
        for t in range(T):
            val = self.lpips_fn(pred_norm[:, t], gt_norm[:, t]).mean().item()
            per_frame_lpips.append(val)

        return {
            'video_lpips': sum(per_frame_lpips) / len(per_frame_lpips),
            'video_lpips_per_frame': per_frame_lpips,
        }


def compute_video_mse(pred_frames: torch.Tensor, gt_frames: torch.Tensor) -> Dict[str, float]:
    """
    Compute simple Video MSE.

    Args:
        pred_frames: [B, C, T, H, W] from model
        gt_frames: [B, T, C, H, W] from dataset

    Returns:
        Dict with 'video_mse' and 'video_mse_per_frame'
    """
    pred_frames = _to_btcwh(pred_frames)
    gt_frames = _to_btcwh(gt_frames)

    B, T, C, H, W = pred_frames.shape
    total_mse = F.mse_loss(pred_frames, gt_frames, reduction='mean').item()
    per_frame = F.mse_loss(pred_frames, gt_frames, reduction='none')
    per_frame_mean = per_frame.mean(dim=(0, 2, 3, 4)).tolist()

    return {
        'video_mse': total_mse,
        'video_mse_per_frame': per_frame_mean,
    }
