#!/usr/bin/env python3
"""
Static Temporal Ablation Experiments for Motus.

Implements two ablation studies that measure the importance of temporal
video context for action prediction:

  1. Single-Frame Baseline (SFB): Replace *all* video_frames with the
     first_frame. This tests how well the model predicts actions when it
     has only the initial observation and no temporal context.

  2. Temporal Gaps (TG): Leave out one frame at a time (replace it with
     the first_frame). This produces a per-frame importance profile
     showing which timestep carries the most predictive information.

Usage:
    python evaluate_ablation.py \
        --config configs/robotwin.yaml \
        --checkpoint checkpoints/.../mp_rank_00_model_states.pt \
        --output_dir results/ablation \
        --num_eval_batches 5
"""

import os
import sys
import json
import copy
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.set_loglevel("WARNING")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — add project root so we can import Motus modules
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from omegaconf import OmegaConf

from data.dataset import create_dataset, collate_fn
from train.sample import inference_sample, compute_action_metrics

logger = logging.getLogger(__name__)


# =========================================================================
# Helper: build a modified copy of a batch with certain video frames replaced
# =========================================================================

def _replace_frames(
    batch: Dict[str, torch.Tensor],
    frame_indices: List[int],
    replace_value: str = "first",
) -> Dict[str, torch.Tensor]:
    """
    Return a shallow copy of *batch* whose ``video_frames`` have the
    specified indices replaced.

    Replacement strategies
    ----------------------
    - ``"first"`` : replace with the conditioning first_frame (broadcast
      from ``[B, C, H, W]`` to the target slot).
    - ``"zero"``  : zero out the frame tensor.
    - ``"noise"`` : replace with i.i.d. Gaussian noise matching the
      tensor's dtype/device.

    The original batch is **not** mutated.
    """
    modified = dict(batch)  # shallow copy — enough because we only touch video_frames
    vf = batch["video_frames"].clone()  # [B, T, C, H, W]
    ff = batch["first_frame"]           # [B, C, H, W]
    B, T, C, H, W = vf.shape

    for t in frame_indices:
        if t < 0 or t >= T:
            logger.warning("Frame index %d out of range [0, %d), skipping", t, T)
            continue

        if replace_value == "first":
            vf[:, t] = ff  # broadcast [B, C, H, W] -> [B, C, H, W]
        elif replace_value == "zero":
            vf[:, t] = 0.0
        elif replace_value == "noise":
            vf[:, t] = torch.randn_like(vf[:, t])
        else:
            raise ValueError(f"Unknown replace_value: {replace_value!r}")

    modified["video_frames"] = vf
    return modified


# =========================================================================
# Function 1: evaluate_with_frame_drop
# =========================================================================

def evaluate_with_frame_drop(
    model,
    dataloader,
    accelerator,
    config,
    drop_mode: str = "none",
    drop_frame_idx: Optional[int] = None,
    replace_value: str = "first",
    num_eval_batches: int = 2,
) -> Dict[str, float]:
    """
    Run evaluation on *num_eval_batches* batches while optionally
    modifying the video frames before inference.

    Parameters
    ----------
    model : torch.nn.Module
        The Motus model (may be wrapped by Accelerate / DDP).
    dataloader : DataLoader
        Validation dataloader.
    accelerator : accelerate.Accelerator
        HuggingFace Accelerate handle (used only for device queries).
    config : OmegaConf
        Full experiment config.
    drop_mode : str
        ``"none"``         — normal evaluation (no frame modification).
        ``"single_frame"`` — replace *all* video frames with the first
                             frame (SFB experiment).
        ``"leave_out_N"``  — replace frame *drop_frame_idx* only (TG
                             experiment, single frame).
    drop_frame_idx : int or None
        Which frame to drop when ``drop_mode == "leave_out_N"``.
    replace_value : str
        How to replace frames: ``"first"``, ``"zero"``, or ``"noise"``.
    num_eval_batches : int
        Number of batches to evaluate over.

    Returns
    -------
    dict
        Keys: ``action_mse_loss``, ``action_l2_error``,
        ``action_mse_std``, ``action_l2_std``, ``video_mse``.
    """
    from collections import defaultdict

    was_training = model.training
    model.eval()

    metrics: Dict[str, list] = defaultdict(list)

    for step, batch in enumerate(dataloader):
        if step >= num_eval_batches:
            break
        if batch is None:
            continue

        # -----------------------------------------------------------
        # 1. Apply frame drop / replacement on the batch copy
        # -----------------------------------------------------------
        if drop_mode == "none":
            eval_batch = batch
        elif drop_mode == "single_frame":
            # Replace ALL video frames (indices 0 .. T-1) with first_frame
            num_frames = batch["video_frames"].shape[1]
            eval_batch = _replace_frames(
                batch, list(range(num_frames)), replace_value=replace_value
            )
        elif drop_mode == "leave_out_N":
            if drop_frame_idx is None:
                raise ValueError(
                    "drop_frame_idx must be set when drop_mode='leave_out_N'"
                )
            eval_batch = _replace_frames(
                batch, [drop_frame_idx], replace_value=replace_value
            )
        else:
            raise ValueError(f"Unknown drop_mode: {drop_mode!r}")

        # -----------------------------------------------------------
        # 2. Run inference (uses first_frame, state, language, vlm only)
        # -----------------------------------------------------------
        predicted_frames, predicted_actions = inference_sample(
            model, eval_batch, config
        )

        # predicted_frames: [B, T_pred, C, H, W] in [0, 1]
        predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        # Ground-truth video frames (NOT the modified ones) for video MSE
        gt_frames = batch["video_frames"].to(predicted_frames.device)

        # Video reconstruction metric
        video_mse = torch.nn.functional.mse_loss(
            predicted_frames, gt_frames, reduction="mean"
        ).item()
        metrics["video_mse"].append(video_mse)

        # Action prediction metrics
        if "action_sequence" in batch and predicted_actions is not None:
            gt_actions = batch["action_sequence"][
                :, : predicted_actions.shape[1]
            ].to(predicted_actions.device)
            action_metrics = compute_action_metrics(predicted_actions, gt_actions)
            for key, value in action_metrics.items():
                metrics[f"action_{key}"].append(value)

    # -----------------------------------------------------------
    # 3. Aggregate
    # -----------------------------------------------------------
    result: Dict[str, float] = {}
    for key, values in metrics.items():
        if values:
            result[key] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))

    if was_training:
        model.train()

    return result


# =========================================================================
# Convenience wrappers called by run_experiments.py
# =========================================================================

def run_sfb_experiment(
    model,
    dataloader,
    config,
    num_eval_batches: int = 2,
    num_inference_steps: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run the Single-Frame Bias (SFB) experiment.

    Returns a dict with at least ``sfb_score``, ``baseline_mse``, ``sfb_mse``.
    """
    config.model.inference.num_inference_timesteps = num_inference_steps

    baseline = evaluate_with_frame_drop(
        model, dataloader, None, config,
        drop_mode="none",
        num_eval_batches=num_eval_batches,
    )
    sfb = evaluate_with_frame_drop(
        model, dataloader, None, config,
        drop_mode="single_frame",
        replace_value="first",
        num_eval_batches=num_eval_batches,
    )

    baseline_mse = baseline.get("action_mse_loss", float("nan"))
    sfb_mse = sfb.get("action_mse_loss", float("nan"))
    sfb_score = baseline_mse / sfb_mse if (sfb_mse and sfb_mse > 0) else float("inf")

    return {
        "sfb_score": sfb_score,
        "baseline_mse": baseline_mse,
        "sfb_mse": sfb_mse,
        "baseline": baseline,
        "sfb": sfb,
    }


def run_tg_experiment(
    model,
    dataloader,
    config,
    num_eval_batches: int = 2,
    num_inference_steps: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run the Temporal-Gain (TG) experiment – leave out each frame one at a time.

    Returns a dict with ``tg_profile`` (list of per-frame MSE values) and
    the full per-frame detail.
    """
    config.model.inference.num_inference_timesteps = num_inference_steps

    num_video_frames = (
        getattr(config.common, "num_video_frames", None)
        or getattr(getattr(config.model, "wan", None), "num_video_frames", 6)
    )

    baseline = evaluate_with_frame_drop(
        model, dataloader, None, config,
        drop_mode="none",
        num_eval_batches=num_eval_batches,
    )

    tg_profile: List[float] = []
    per_frame: Dict[str, Dict] = {}
    for t in range(num_video_frames):
        tg_t = evaluate_with_frame_drop(
            model, dataloader, None, config,
            drop_mode="leave_out_N",
            drop_frame_idx=t,
            replace_value="first",
            num_eval_batches=num_eval_batches,
        )
        tg_profile.append(tg_t.get("action_mse_loss", float("nan")))
        per_frame[f"frame_{t}"] = tg_t

    return {
        "tg_profile": tg_profile,
        "num_video_frames": num_video_frames,
        "baseline": baseline,
        "per_frame": per_frame,
    }


# =========================================================================
# Function 2: run_sfb_and_tg_experiments
# =========================================================================

def run_sfb_and_tg_experiments(
    model,
    dataloader,
    accelerator,
    config,
    num_eval_batches: int = 2,
    replace_value: str = "first",
    output_dir: str = "results/ablation",
) -> Dict[str, float]:
    """
    Execute the full ablation suite:

    1. **Baseline** — normal evaluation, no frame modification.
    2. **SFB** — replace *all* video_frames with the first_frame.
    3. **TG** — leave out each video frame one at a time (frame 0 .. T-1).

    All results are persisted to ``<output_dir>/ablation_results.json``.

    Returns
    -------
    dict
        Flat dictionary with keys like ``baseline_action_mse_loss``,
        ``sfb_action_mse_loss``, ``tg_no_0_action_mse_loss``, etc.
    """
    # Determine num_video_frames from config
    num_video_frames = (
        getattr(config.common, "num_video_frames", None)
        or getattr(getattr(config.model, "wan", None), "num_video_frames", 6)
    )

    os.makedirs(output_dir, exist_ok=True)

    all_results: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Running BASELINE evaluation ...")
    baseline = evaluate_with_frame_drop(
        model, dataloader, accelerator, config,
        drop_mode="none",
        num_eval_batches=num_eval_batches,
    )
    for k, v in baseline.items():
        all_results[f"baseline_{k}"] = v
    logger.info("  Baseline action_mse_loss = %.6f  (std %.6f)",
                baseline.get("action_mse_loss", float("nan")),
                baseline.get("action_mse_std", 0.0))
    logger.info("  Baseline action_l2_error = %.6f",
                baseline.get("action_l2_error", float("nan")))

    # ------------------------------------------------------------------
    # 2. SFB — replace ALL video frames with first_frame
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Running SFB (single-frame baseline) evaluation ...")
    sfb = evaluate_with_frame_drop(
        model, dataloader, accelerator, config,
        drop_mode="single_frame",
        replace_value=replace_value,
        num_eval_batches=num_eval_batches,
    )
    for k, v in sfb.items():
        all_results[f"sfb_{k}"] = v
    logger.info("  SFB action_mse_loss = %.6f  (std %.6f)",
                sfb.get("action_mse_loss", float("nan")),
                sfb.get("action_mse_std", 0.0))
    logger.info("  SFB action_l2_error = %.6f",
                sfb.get("action_l2_error", float("nan")))

    # ------------------------------------------------------------------
    # 3. TG — leave out each frame individually
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Running TG (temporal-gap) evaluation for %d frames ...",
                num_video_frames)
    for t in range(num_video_frames):
        tg_t = evaluate_with_frame_drop(
            model, dataloader, accelerator, config,
            drop_mode="leave_out_N",
            drop_frame_idx=t,
            replace_value=replace_value,
            num_eval_batches=num_eval_batches,
        )
        for k, v in tg_t.items():
            all_results[f"tg_no_{t}_{k}"] = v
        logger.info("  TG t=%d  action_mse_loss = %.6f  action_l2_error = %.6f",
                    t,
                    tg_t.get("action_mse_loss", float("nan")),
                    tg_t.get("action_l2_error", float("nan")))

    # ------------------------------------------------------------------
    # 4. Save JSON
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "ablation_results.json")
    # Convert numpy / torch scalars to plain Python floats for JSON
    serializable = {k: float(v) for k, v in all_results.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return all_results


# =========================================================================
# Function 3: plot_results
# =========================================================================

def plot_results(
    results: Dict[str, float],
    output_dir: str = "results/ablation",
    num_video_frames: Optional[int] = None,
) -> None:
    """
    Generate two publication-quality figures:

    1. **SFB bar chart** — baseline vs. single-frame action MSE and L2.
    2. **TG profile** — per-frame action MSE and L2 when each frame is
       individually removed.

    Figures are saved to ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Infer number of video frames from the result keys if not given
    if num_video_frames is None:
        tg_keys = [k for k in results if k.startswith("tg_no_") and k.endswith("_action_mse_loss")]
        num_video_frames = len(tg_keys)

    # ------------------------------------------------------------------
    # Figure 1: SFB bar chart
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))

    # MSE
    ax = axes1[0]
    labels = ["Baseline", "SFB"]
    mse_vals = [
        results.get("baseline_action_mse_loss", 0.0),
        results.get("sfb_action_mse_loss", 0.0),
    ]
    mse_stds = [
        results.get("baseline_action_mse_std", 0.0),
        results.get("sfb_action_mse_std", 0.0),
    ]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(labels, mse_vals, yerr=mse_stds, capsize=5, color=colors, edgecolor="black")
    ax.set_ylabel("Action MSE Loss")
    ax.set_title("SFB: Action MSE")
    # Annotate values on bars
    for bar, val in zip(bars, mse_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mse_stds[0] * 0.3 + 1e-5,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    # L2
    ax = axes1[1]
    l2_vals = [
        results.get("baseline_action_l2_error", 0.0),
        results.get("sfb_action_l2_error", 0.0),
    ]
    l2_stds = [
        results.get("baseline_action_l2_std", 0.0),
        results.get("sfb_action_l2_std", 0.0),
    ]
    bars = ax.bar(labels, l2_vals, yerr=l2_stds, capsize=5, color=colors, edgecolor="black")
    ax.set_ylabel("Action L2 Error")
    ax.set_title("SFB: Action L2 Error")
    for bar, val in zip(bars, l2_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + l2_stds[0] * 0.3 + 1e-5,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    fig1.suptitle("Single-Frame Baseline (SFB) Ablation", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    sfb_path = os.path.join(output_dir, "sfb_comparison.png")
    fig1.savefig(sfb_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info("SFB figure saved to %s", sfb_path)

    # ------------------------------------------------------------------
    # Figure 2: TG profile bar chart
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(max(10, num_video_frames * 1.5), 5))

    frame_labels = [f"t={t}" for t in range(num_video_frames)]
    tg_mse = []
    tg_l2 = []
    for t in range(num_video_frames):
        tg_mse.append(results.get(f"tg_no_{t}_action_mse_loss", 0.0))
        tg_l2.append(results.get(f"tg_no_{t}_action_l2_error", 0.0))

    # Reference lines from baseline and SFB
    baseline_mse = results.get("baseline_action_mse_loss", None)
    sfb_mse = results.get("sfb_action_mse_loss", None)
    baseline_l2 = results.get("baseline_action_l2_error", None)
    sfb_l2 = results.get("sfb_action_l2_error", None)

    # MSE panel
    ax = axes2[0]
    tg_mse_stds = [
        results.get(f"tg_no_{t}_action_mse_std", 0.0)
        for t in range(num_video_frames)
    ]
    bars = ax.bar(frame_labels, tg_mse, yerr=tg_mse_stds, capsize=4,
                  color="#55A868", edgecolor="black")
    if baseline_mse is not None:
        ax.axhline(baseline_mse, color="#4C72B0", linestyle="--", linewidth=1.5,
                    label=f"Baseline ({baseline_mse:.4f})")
    if sfb_mse is not None:
        ax.axhline(sfb_mse, color="#DD8452", linestyle=":", linewidth=1.5,
                    label=f"SFB ({sfb_mse:.4f})")
    ax.set_ylabel("Action MSE Loss")
    ax.set_title("TG: Action MSE")
    ax.set_xlabel("Dropped frame")
    ax.legend(fontsize=9)

    # L2 panel
    ax = axes2[1]
    tg_l2_stds = [
        results.get(f"tg_no_{t}_action_l2_std", 0.0)
        for t in range(num_video_frames)
    ]
    bars = ax.bar(frame_labels, tg_l2, yerr=tg_l2_stds, capsize=4,
                  color="#55A868", edgecolor="black")
    if baseline_l2 is not None:
        ax.axhline(baseline_l2, color="#4C72B0", linestyle="--", linewidth=1.5,
                    label=f"Baseline ({baseline_l2:.4f})")
    if sfb_l2 is not None:
        ax.axhline(sfb_l2, color="#DD8452", linestyle=":", linewidth=1.5,
                    label=f"SFB ({sfb_l2:.4f})")
    ax.set_ylabel("Action L2 Error")
    ax.set_title("TG: Action L2 Error")
    ax.set_xlabel("Dropped frame")
    ax.legend(fontsize=9)

    fig2.suptitle("Temporal Gaps (TG) Ablation — Per-Frame Importance",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    tg_path = os.path.join(output_dir, "tg_profile.png")
    fig2.savefig(tg_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("TG figure saved to %s", tg_path)


# =========================================================================
# Standalone entry-point
# =========================================================================

def load_config(config_path: str) -> OmegaConf:
    """Load and validate a YAML config (mirrors train.py logic)."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = OmegaConf.load(config_path)
    config.common.action_chunk_size = (
        config.common.num_video_frames * config.common.video_action_freq_ratio
    )
    return config


def build_model(config: OmegaConf, checkpoint_path: str, device: str = "cuda"):
    """
    Instantiate the Motus model and load a checkpoint.

    This mirrors the logic in ``train.py::create_model_and_optimizer``
    but without an optimizer — we only need the model for inference.
    """
    from models.motus import Motus, MotusConfig

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
        batch_size=1,  # evaluation batch size; will be overridden
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        training_mode=getattr(config, "training_mode", "finetune"),
        load_pretrained_backbones=False,  # we load from ckpt, not pretrained
    )

    model = Motus(model_config)

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        # Try standard DeepSpeed layout
        for candidate in [
            ckpt_path / "mp_rank_00_model_states.pt",
            ckpt_path / "pytorch_model" / "mp_rank_00_model_states.pt",
        ]:
            if candidate.exists():
                ckpt_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_path}. "
                "Expected mp_rank_00_model_states.pt"
            )

    logger.info("Loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("module", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Checkpoint loaded: %d missing, %d unexpected keys", len(missing), len(unexpected)
    )
    if missing:
        logger.warning("Missing keys: %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:10])

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Motus Static-Temporal Ablation (SFB & TG experiments)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (e.g. configs/robotwin.yaml)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (directory or .pt file)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/ablation",
        help="Directory for results JSON and figures",
    )
    parser.add_argument(
        "--num_eval_batches", type=int, default=2,
        help="Number of batches to evaluate per experiment",
    )
    parser.add_argument(
        "--replace_value", type=str, default="first",
        choices=["first", "zero", "noise"],
        help="How to replace dropped frames",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip the baseline experiment (reuse from JSON if present)",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    # ---- Logging ----
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Config ----
    config = load_config(args.config)
    num_video_frames = getattr(config.common, "num_video_frames", 6)

    # ---- Model ----
    logger.info("Building model ...")
    model = build_model(config, args.checkpoint, device=args.device)

    # ---- Validation dataset ----
    logger.info("Creating validation dataset ...")
    val_dataset = create_dataset(config, val=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Accelerator (CPU-only — we handle device placement ourselves)
    try:
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=(args.device == "cpu"))
    except Exception:
        accelerator = None

    # ---- Run experiments ----
    results = run_sfb_and_tg_experiments(
        model=model,
        dataloader=val_dataloader,
        accelerator=accelerator,
        config=config,
        num_eval_batches=args.num_eval_batches,
        replace_value=args.replace_value,
        output_dir=args.output_dir,
    )

    # ---- Plot ----
    plot_results(
        results,
        output_dir=args.output_dir,
        num_video_frames=num_video_frames,
    )

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("-" * 60)
    logger.info(
        "  Baseline  MSE: %.6f   L2: %.6f",
        results.get("baseline_action_mse_loss", float("nan")),
        results.get("baseline_action_l2_error", float("nan")),
    )
    logger.info(
        "  SFB       MSE: %.6f   L2: %.6f",
        results.get("sfb_action_mse_loss", float("nan")),
        results.get("sfb_action_l2_error", float("nan")),
    )
    for t in range(num_video_frames):
        logger.info(
            "  TG t=%-3d  MSE: %.6f   L2: %.6f",
            t,
            results.get(f"tg_no_{t}_action_mse_loss", float("nan")),
            results.get(f"tg_no_{t}_action_l2_error", float("nan")),
        )
    logger.info("=" * 60)
    logger.info(
        "All results saved to %s",
        os.path.join(args.output_dir, "ablation_results.json"),
    )


if __name__ == "__main__":
    main()
