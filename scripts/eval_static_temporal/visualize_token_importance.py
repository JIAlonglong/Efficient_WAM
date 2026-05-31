#!/usr/bin/env python3
"""Visualize which video tokens the understanding tokens attend to in Motus.

Computes per-token importance scores as a proxy for attention weights
(flash_attention does not expose attention weights), then renders
heatmaps overlaid on the original video frames.

Usage
-----
    python visualize_token_importance.py \
        --config configs/robotwin.yaml \
        --checkpoint /path/to/checkpoint_step_100000 \
        --output_dir ./vis_output \
        --num_samples 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib headless backend -- must be set before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Token importance computation
# ---------------------------------------------------------------------------

def compute_token_spatial_importance(
    video_tokens: torch.Tensor,
    und_tokens: torch.Tensor,
    grid_sizes: Tuple[int, int, int],
) -> torch.Tensor:
    """Compute importance of each video token based on understanding tokens.

    Uses L2-norm of each video token as a proxy for "information density".
    Higher-norm tokens carry more signal and are therefore more important.

    Parameters
    ----------
    video_tokens : [B, N, 3072]
        Video tokens from the patch embedding (before or after joint attention).
    und_tokens : [B, L, 512]
        Understanding tokens (used only for the projection-based variant).
    grid_sizes : (T, H, W)
        Latent grid dimensions *after* patch_embedding, e.g. (3, 6, 5).

    Returns
    -------
    importance : [B, T, H, W]
        Spatial importance map per frame, values in roughly [0, inf]
        (not normalised here -- callers decide how to display).
    """
    T, H, W = grid_sizes
    B, N, D = video_tokens.shape
    assert N == T * H * W, f"N={N} != T*H*W={T * H * W}"

    # Simple importance: L2 norm of each video token
    token_norms = video_tokens.float().norm(dim=-1)  # [B, N]

    # Reshape to spatial grid
    importance = token_norms.view(B, T, H, W)  # [B, T, H, W]

    # Normalise per sample to [0, 1]
    imp_min = importance.amin(dim=(1, 2, 3), keepdim=True)
    imp_max = importance.amax(dim=(1, 2, 3), keepdim=True)
    importance = (importance - imp_min) / (imp_max - imp_min + 1e-8)

    return importance


def compute_token_importance_with_projection(
    video_tokens: torch.Tensor,
    und_tokens: torch.Tensor,
    grid_sizes: Tuple[int, int, int],
) -> torch.Tensor:
    """Compute importance using cosine similarity to the global understanding feature.

    Projects video tokens down to the understanding dimension (512) by
    truncation, then computes cosine similarity with the mean understanding
    token.  This gives a *semantic* importance signal: tokens whose
    representation is more aligned with the language-conditioned features
    score higher.

    Parameters
    ----------
    video_tokens : [B, N, 3072]
    und_tokens   : [B, L, 512]
    grid_sizes   : (T, H, W) -- after patch_embedding

    Returns
    -------
    importance : [B, T, H, W]  values in roughly [-1, 1], then scaled to [0, 1]
    """
    T, H, W = grid_sizes
    B, N, D = video_tokens.shape

    # Global understanding feature (mean pool over sequence)
    und_global = und_tokens.float().mean(dim=1)  # [B, 512]

    # Truncate video tokens to 512 dims (simple projection proxy)
    video_proj = video_tokens[:, :, :512].float()  # [B, N, 512]

    # Cosine similarity per token
    similarity = F.cosine_similarity(
        video_proj, und_global.unsqueeze(1), dim=-1
    )  # [B, N]

    # Reshape and scale to [0, 1]
    importance = similarity.view(B, T, H, W)
    imp_min = importance.amin(dim=(1, 2, 3), keepdim=True)
    imp_max = importance.amax(dim=(1, 2, 3), keepdim=True)
    importance = (importance - imp_min) / (imp_max - imp_min + 1e-8)

    return importance


# ---------------------------------------------------------------------------
# 2. Heatmap overlay
# ---------------------------------------------------------------------------

def create_heatmap_overlay(
    frame_image: np.ndarray,
    importance_map: np.ndarray,
    alpha: float = 0.65,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay an importance heatmap on the original frame.

    Parameters
    ----------
    frame_image : [H_img, W_img, 3]  uint8, RGB
    importance_map : [H, W]  float, values in [0, 1]
    alpha : overlay transparency (higher = more visible heatmap)
    colormap : matplotlib colormap name

    Returns
    -------
    overlay : [H_img, W_img, 3]  uint8
    """
    import cv2

    H_img, W_img = frame_image.shape[:2]

    # Resize importance map to match frame size
    importance_resized = cv2.resize(
        importance_map, (W_img, H_img), interpolation=cv2.INTER_LINEAR
    )

    # Normalise to [0, 1]
    imp_min = importance_resized.min()
    imp_max = importance_resized.max()
    importance_resized = (importance_resized - imp_min) / (imp_max - imp_min + 1e-8)

    # Apply colormap -> [H, W, 4] (RGBA), take RGB
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(importance_resized)[:, :, :3]  # [H, W, 3], float [0, 1]

    # Lighten the frame first to improve contrast with overlay
    frame_float = frame_image.astype(np.float32) / 255.0
    frame_light = np.clip(frame_float * 0.7 + 0.15, 0, 1)  # lighten dark frames

    # Blend
    overlay = (1.0 - alpha) * frame_light + alpha * heatmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return overlay


# ---------------------------------------------------------------------------
# 3. Temporal importance grid (3x3)
# ---------------------------------------------------------------------------

def create_temporal_importance_grid(
    frames: List[np.ndarray],
    importance_maps: List[np.ndarray],
    save_path: str | Path,
    title: str = "Token Importance: Understanding -> Video",
    max_frames: int = 3,
) -> None:
    """Create a 3-row grid: original frames, heatmaps, overlays.

    Parameters
    ----------
    frames : list of numpy arrays [H, W, 3], uint8
    importance_maps : list of numpy arrays [H, W], float in [0, 1]
    save_path : where to save the figure
    title : figure title
    max_frames : how many temporal steps to show (up to len(frames))
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(frames), max_frames)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 15))
    if n == 1:
        axes = axes.reshape(3, 1)

    for t in range(n):
        # Row 0: original frame
        axes[0, t].imshow(frames[t])
        axes[0, t].set_title(f"Frame {t}", fontsize=13)
        axes[0, t].axis("off")

        # Row 1: raw heatmap
        im = axes[1, t].imshow(importance_maps[t], cmap="YlOrRd", vmin=0, vmax=1)
        axes[1, t].set_title("Importance Map", fontsize=13)
        axes[1, t].axis("off")

        # Row 2: overlay
        overlay = create_heatmap_overlay(frames[t], importance_maps[t])
        axes[2, t].imshow(overlay)
        axes[2, t].set_title("Overlay", fontsize=13)
        axes[2, t].axis("off")

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.33, 0.015, 0.33])
    fig.colorbar(im, cax=cbar_ax, label="Semantic Importance")

    plt.suptitle(title, fontsize=15, y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved temporal importance grid to {save_path}")


# ---------------------------------------------------------------------------
# 4. Comparison grid
# ---------------------------------------------------------------------------

def create_comparison_grid(
    frames: List[np.ndarray],
    semantic_imp: List[np.ndarray],
    temporal_novelty: List[np.ndarray],
    save_path: str | Path,
    max_frames: int = 3,
) -> None:
    """Side-by-side comparison of different importance signals.

    Columns = temporal steps, Rows = {Semantic, Temporal Novelty, Random}.

    Parameters
    ----------
    frames : list of [H, W, 3] uint8
    semantic_imp : list of [H, W] float
    temporal_novelty : list of [H, W] float
    save_path : output path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(frames), max_frames)
    row_labels = ["Semantic", "Temporal Novelty", "Random"]
    colormaps = ["YlOrRd", "YlGnBu", "Greens"]
    importance_groups = [semantic_imp, temporal_novelty, None]

    fig, axes = plt.subplots(3, n, figsize=(5 * n, 15))
    if n == 1:
        axes = axes.reshape(3, 1)

    for row, (imp_group, label, cmap_name) in enumerate(
        zip(importance_groups, row_labels, colormaps)
    ):
        for t in range(n):
            if imp_group is not None:
                imp_map = imp_group[t]
            else:
                # Random baseline
                rng = np.random.RandomState(42 + t)
                imp_map = rng.rand(*frames[t].shape[:2]).astype(np.float32)

            overlay = create_heatmap_overlay(frames[t], imp_map, colormap=cmap_name)
            axes[row, t].imshow(overlay)

            if t == 0:
                axes[row, t].set_ylabel(
                    label, fontsize=13, rotation=0, labelpad=80, va="center"
                )
            axes[row, t].set_title(f"Frame {t}" if row == 0 else "", fontsize=13)
            axes[row, t].axis("off")

    plt.suptitle("Importance Signal Comparison", fontsize=15, y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison grid to {save_path}")


# ---------------------------------------------------------------------------
# 5. Temporal novelty (cross-frame difference)
# ---------------------------------------------------------------------------

def compute_temporal_novelty(
    video_tokens: torch.Tensor,
    grid_sizes: Tuple[int, int, int],
) -> torch.Tensor:
    """Compute temporal novelty as L2 change between adjacent frames.

    Frames that change a lot between steps score higher.

    Parameters
    ----------
    video_tokens : [B, N, D]
    grid_sizes   : (T, H, W)

    Returns
    -------
    novelty : [B, T, H, W]  values in [0, 1]
    """
    T, H, W = grid_sizes
    B, N, D = video_tokens.shape

    tokens = video_tokens.float().view(B, T, H, W, D)

    # Compute frame-to-frame L2 difference (temporal gradient)
    novelty = torch.zeros(B, T, H, W, device=video_tokens.device)
    for t in range(1, T):
        diff = (tokens[:, t] - tokens[:, t - 1]).norm(dim=-1)  # [B, H, W]
        novelty[:, t] = diff

    # Normalise to [0, 1]
    n_min = novelty.amin(dim=(1, 2, 3), keepdim=True)
    n_max = novelty.amax(dim=(1, 2, 3), keepdim=True)
    novelty = (novelty - n_min) / (n_max - n_min + 1e-8)

    return novelty


# ---------------------------------------------------------------------------
# 6. Main runner
# ---------------------------------------------------------------------------

def _frames_to_numpy(video_frames: torch.Tensor) -> List[np.ndarray]:
    """Convert a [B, C, H, W] tensor (0-1 float) to list of [H, W, 3] uint8."""
    frames = []
    for b in range(video_frames.shape[0]):
        f = video_frames[b].permute(1, 2, 0).cpu().float().numpy()
        f = np.clip(f, 0, 1)
        f = (f * 255).astype(np.uint8)
        frames.append(f)
    return frames


def run_visualization(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 4,
    num_inference_steps: int = 10,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
) -> None:
    """End-to-end visualisation pipeline.

    1. Load model and a batch from RoboTwin2.
    2. Run forward pass, extract video_tokens and und_tokens.
    3. Compute importance maps.
    4. Create and save visualisations.
    5. Save raw importance data to .npy files.
    """
    from omegaconf import OmegaConf
    from models.motus import Motus, MotusConfig
    from data.dataset import create_dataset, collate_fn
    from torch.utils.data import DataLoader

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    config = OmegaConf.load(config_path)
    config.common.action_chunk_size = (
        config.common.num_video_frames * config.common.video_action_freq_ratio
    )
    logger.info(f"Loaded config from {config_path}")

    # ── Build model ────────────────────────────────────────────────────────
    model_config = MotusConfig(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        action_expert_dim=config.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=config.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=config.model.action_expert.norm_eps,
        und_expert_hidden_size=config.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=config.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=config.model.und_expert.norm_eps,
        vlm_adapter_input_dim=config.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=config.model.und_expert.vlm.projector_type,
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=batch_size,
        training_mode=getattr(config, "training_mode", "finetune"),
        load_pretrained_backbones=False,  # we load from checkpoint
    )

    logger.info("Creating Motus model...")
    model = Motus(model_config)

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        ckpt_file = ckpt_path / "mp_rank_00_model_states.pt"
        if not ckpt_file.exists():
            # Try accelerator-style checkpoint
            alt = ckpt_path / "pytorch_model_0.bin"
            if alt.exists():
                ckpt_file = alt
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {ckpt_path}. "
                    "Expected mp_rank_00_model_states.pt or pytorch_model_0.bin."
                )
    else:
        ckpt_file = ckpt_path

    logger.info(f"Loading checkpoint from {ckpt_file}")
    state_dict = torch.load(ckpt_file, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device).eval()

    # ── Compute grid sizes (after VAE 16x spatial + patch_embedding 2x) ───
    # VAE spatial downsample: 16x, temporal: 4x
    # patch_embedding Conv3d(kernel=(1,2,2), stride=(1,2,2)): 2x spatial
    # Total: 32x spatial, 4x temporal
    grid_T = 1 + config.common.num_video_frames // 4
    grid_H = config.common.video_height // 32   # 384//32 = 12
    grid_W = config.common.video_width // 32    # 320//32 = 10
    grid_sizes = (grid_T, grid_H, grid_W)
    N = grid_T * grid_H * grid_W
    logger.info(
        f"Grid sizes: T={grid_T}, H={grid_H}, W={grid_W}, N={N}"
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    logger.info("Creating dataset...")
    dataset = create_dataset(config, val=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ── Get a batch ────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch = None
    for b in loader:
        if b is not None:
            batch = b
            break
    if batch is None:
        raise RuntimeError("Could not fetch a valid batch from the dataset.")

    # ── Move to device ─────────────────────────────────────────────────────
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.to(device, dtype=model.dtype)
        elif isinstance(v, dict):
            batch_dev[k] = {
                kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            batch_dev[k] = v

    # ── Extract tokens (using the logic from extract_motus_tokens) ─────────
    logger.info("Extracting video and understanding tokens...")
    with torch.no_grad():
        video_frames_raw = batch_dev["video_frames"]
        first_frame_raw = batch_dev["first_frame"]

        # Encode video through VAE
        first_frame_norm = (first_frame_raw * 2.0 - 1.0).unsqueeze(2)
        video_normalized = (video_frames_raw * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)
        clean_latent = model.video_model.encode_video(full_video.to(model.dtype))

        # Video tokens from patch_embedding (clean latent, no noise)
        video_tokens = model.video_module.prepare_input(clean_latent.to(model.dtype))
        logger.info(f"Video tokens shape: {video_tokens.shape}")

        # Understanding tokens from VLM
        vlm_inputs = batch_dev.get("vlm_inputs", None)
        if vlm_inputs is not None:
            und_tokens = model.und_module.extract_und_features(vlm_inputs)
        else:
            logger.warning("No VLM inputs in batch, creating dummy und_tokens")
            und_tokens = torch.zeros(
                batch_dev["video_frames"].shape[0],
                32,
                512,
                device=device,
                dtype=model.dtype,
            )
        logger.info(f"Understanding tokens shape: {und_tokens.shape}")

    # ── Compute importance maps ────────────────────────────────────────────
    logger.info("Computing importance maps...")

    # Method 1: L2-norm based importance
    importance_norm = compute_token_spatial_importance(
        video_tokens, und_tokens, grid_sizes
    )  # [B, T, H, W]

    # Method 2: Cosine-similarity based importance
    importance_cosine = compute_token_importance_with_projection(
        video_tokens, und_tokens, grid_sizes
    )  # [B, T, H, W]

    # Temporal novelty
    temporal_novelty = compute_temporal_novelty(video_tokens, grid_sizes)

    logger.info(
        f"Importance maps computed: norm={importance_norm.shape}, "
        f"cosine={importance_cosine.shape}, novelty={temporal_novelty.shape}"
    )

    # ── Prepare frames for visualisation ───────────────────────────────────
    # The latent has grid_T frames: frame 0 = condition, frames 1..grid_T-1 = predicted.
    # For pixel-space display we pick the first frame + evenly-spaced target frames
    # from the original video to roughly correspond to the latent slots.
    B_imp = importance_norm.shape[0]
    num_target_frames = video_frames_raw.shape[1]  # e.g. 8

    first_frames_np = _frames_to_numpy(first_frame_raw[:B_imp])  # list of [H, W, 3]

    # Pick (grid_T - 1) evenly spaced target frames to align with latent frames 1..grid_T-1
    target_indices = np.linspace(0, num_target_frames - 1, grid_T - 1, dtype=int)
    video_frames_for_vis = video_frames_raw[:B_imp][:, target_indices]  # [B, grid_T-1, C, H, W]
    video_frames_np = _frames_to_numpy(video_frames_for_vis.reshape(-1, *video_frames_for_vis.shape[2:]))
    # Reshape back: list of B lists, each with (grid_T-1) frames
    video_frames_per_sample = [
        video_frames_np[s * (grid_T - 1) : (s + 1) * (grid_T - 1)]
        for s in range(B_imp)
    ]

    # ── Save raw importance data ───────────────────────────────────────────
    np.save(
        output_path / "importance_norm.npy",
        importance_norm.cpu().float().numpy(),
    )
    np.save(
        output_path / "importance_cosine.npy",
        importance_cosine.cpu().float().numpy(),
    )
    np.save(
        output_path / "temporal_novelty.npy",
        temporal_novelty.cpu().float().numpy(),
    )
    np.save(
        output_path / "video_tokens.npy",
        video_tokens.cpu().float().numpy(),
    )
    np.save(
        output_path / "und_tokens.npy",
        und_tokens.cpu().float().numpy(),
    )
    logger.info(f"Raw importance data saved to {output_path}")

    # ── Generate visualisations per sample ─────────────────────────────────
    for sample_idx in range(min(B_imp, num_samples)):
        logger.info(f"Generating visualisation for sample {sample_idx}...")

        # Build frame list: first frame (condition) + target frames
        frame_list = [first_frames_np[sample_idx]]
        frame_list.extend(video_frames_per_sample[sample_idx])

        # Per-frame importance maps (from latent grid)
        imp_maps_cosine = []
        imp_maps_norm = []
        imp_maps_novelty = []
        for t_idx in range(grid_T):
            imp_maps_cosine.append(importance_cosine[sample_idx, t_idx].cpu().float().numpy())
            imp_maps_norm.append(importance_norm[sample_idx, t_idx].cpu().float().numpy())
            imp_maps_novelty.append(temporal_novelty[sample_idx, t_idx].cpu().float().numpy())

        # Use up to 3 frames for the grid
        display_n = min(len(frame_list), 3)

        # Grid 1: Cosine-similarity importance
        create_temporal_importance_grid(
            frame_list[:display_n],
            imp_maps_cosine[:display_n],
            output_path / f"sample{sample_idx}_cosine_importance.png",
            title=f"Sample {sample_idx} -- Cosine Similarity Importance",
        )

        # Grid 2: L2-norm importance
        create_temporal_importance_grid(
            frame_list[:display_n],
            imp_maps_norm[:display_n],
            output_path / f"sample{sample_idx}_norm_importance.png",
            title=f"Sample {sample_idx} -- L2 Norm Importance",
        )

        # Grid 3: Temporal novelty
        create_temporal_importance_grid(
            frame_list[:display_n],
            imp_maps_novelty[:display_n],
            output_path / f"sample{sample_idx}_temporal_novelty.png",
            title=f"Sample {sample_idx} -- Temporal Novelty",
        )

        # Grid 4: Comparison (semantic vs novelty vs random)
        create_comparison_grid(
            frame_list[:display_n],
            imp_maps_cosine[:display_n],
            imp_maps_novelty[:display_n],
            output_path / f"sample{sample_idx}_comparison.png",
        )

    # ── Aggregate statistics ───────────────────────────────────────────────
    logger.info("=== Token Importance Statistics ===")
    for name, imp in [
        ("Cosine", importance_cosine),
        ("L2 Norm", importance_norm),
        ("Temporal Novelty", temporal_novelty),
    ]:
        imp_np = imp.cpu().float().numpy()
        logger.info(
            f"  {name}: mean={imp_np.mean():.4f}, std={imp_np.std():.4f}, "
            f"min={imp_np.min():.4f}, max={imp_np.max():.4f}"
        )

    # Per-frame breakdown
    logger.info("Per-frame importance (cosine similarity):")
    for t in range(grid_T):
        frame_imp = importance_cosine[:, t].cpu().float().numpy()
        logger.info(
            f"  Frame {t}: mean={frame_imp.mean():.4f}, "
            f"std={frame_imp.std():.4f}, "
            f"min={frame_imp.min():.4f}, max={frame_imp.max():.4f}"
        )

    logger.info(f"Visualisation complete. All outputs saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise token importance in Motus: "
        "which video tokens the understanding tokens attend to."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/robotwin.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to model checkpoint (directory containing "
            "mp_rank_00_model_states.pt, or the .pt file itself)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vis_token_importance",
        help="Directory to save visualisations and .npy data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to visualise",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for loading data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_visualization(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
