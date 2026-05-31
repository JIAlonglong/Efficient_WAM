#!/usr/bin/env python3
"""
Main experiment runner for Section 4.2 (Phase 1 basic experiments)
of the Motion-Aware Temporal Memory research plan.

Experiments:
  1.1: History KV Concatenation (Training-Free)
       - Prepend first-frame K/V from all 30 MoT layers to current frame's
         K/V during denoising.  Evaluates whether static visual context helps
         action and video prediction.
  1.2: First-Frame Memory
       - Extract VLM features (und_tokens) once from the first frame and
         reuse across all denoising steps instead of re-extracting each step.
         Evaluates whether VLM features change meaningfully across noise levels.

Usage:
    python run_memory_experiments.py \
        --config ../../configs/robotwin.yaml \
        --checkpoint /path/to/Motus_robotwin2/mp_rank_00_model_states.pt \
        --output_dir ./results \
        --num_eval_batches 5 \
        --num_inference_steps 50 \
        --experiments baseline history_kv first_frame_memory
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

_inference_root = _project_root / "inference" / "robotwin"
sys.path.insert(0, str(_inference_root))

# ---------------------------------------------------------------------------
# Monkey-patch lerobot to allow loading v2.1 datasets with v3.0 code
# (identical to the patches in eval_static_temporal/run_experiments.py)
# ---------------------------------------------------------------------------
import lerobot.datasets.utils as _lerobot_utils
import lerobot.datasets.lerobot_dataset as _lerobot_ds
import pandas as _pd

# 1. Bypass version compatibility check
_original_check = _lerobot_utils.check_version_compatibility


def _patched_check(repo_id, version_to_check, current_version, enforce_breaking_major=True):
    logger.info(
        f"Bypassing version check for dataset {repo_id} "
        f"(v{version_to_check} vs codebase v{current_version})"
    )
    return


_lerobot_utils.check_version_compatibility = _patched_check
try:
    import lerobot.datasets.backward_compatibility as _bc
    if hasattr(_bc, "check_version_compatibility"):
        _bc.check_version_compatibility = _patched_check
except ImportError:
    pass


# 2. Patch get_safe_version to skip HF Hub lookup for local datasets
def _patched_get_safe_version(repo_id, revision=None):
    """Skip Hub lookup; return current codebase version for local datasets."""
    return _lerobot_ds.CODEBASE_VERSION


_lerobot_utils.get_safe_version = _patched_get_safe_version
_lerobot_ds.get_safe_version = _patched_get_safe_version


# 3. Patch load_tasks to support v2.1 tasks.jsonl format
_original_load_tasks = _lerobot_utils.load_tasks


def _patched_load_tasks(root, episodes=None):
    """Load tasks from either tasks.parquet (v3.0) or tasks.jsonl (v2.1)."""
    parquet_path = root / "meta" / "tasks.parquet"
    jsonl_path = root / "meta" / "tasks.jsonl"
    if parquet_path.exists():
        return _original_load_tasks(root, episodes)
    elif jsonl_path.exists():
        import json as _json
        tasks = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(_json.loads(line))
        return _pd.DataFrame(tasks)
    else:
        logger.warning(f"No tasks file found in {root / 'meta'}, returning empty DataFrame")
        return _pd.DataFrame()


_lerobot_utils.load_tasks = _patched_load_tasks


# 4. Patch load_episodes to support v2.1 episodes.jsonl format
_original_load_episodes = _lerobot_utils.load_episodes


def _patched_load_episodes(root):
    """Load episodes from either episodes/ parquet dir (v3.0) or episodes.jsonl (v2.1)."""
    episodes_dir = root / "meta" / "episodes"
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_dir.exists() and list(episodes_dir.glob("*/*.parquet")):
        return _original_load_episodes(root)
    elif episodes_jsonl.exists():
        import json as _json
        records = []
        with open(episodes_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = _json.loads(line)
                    flat = {
                        "episode_index": rec.get("episode_index", 0),
                        "length": rec.get("length", 0),
                    }
                    tasks = rec.get("tasks", [])
                    flat["tasks"] = tasks[0] if tasks else ""
                    records.append(flat)
        from datasets import Dataset as _HFDataset
        return _HFDataset.from_list(records)
    else:
        logger.warning(f"No episodes file found in {root / 'meta'}")
        from datasets import Dataset as _HFDataset
        return _HFDataset.from_list([])


_lerobot_utils.load_episodes = _patched_load_episodes


# 5. Patch get_video_file_path to handle episodes without per-video chunk columns
_original_get_video_file_path = _lerobot_ds.LeRobotDatasetMetadata.get_video_file_path


def _patched_get_video_file_path(self, ep_index, vid_key):
    """Handle episodes that lack per-video chunk/file index columns,
    and use the actual on-disk layout: videos/chunk-{idx}/{key}/episode_{idx}.mp4"""
    info = getattr(self, "info", None)
    chunks_size = info.get("chunks_size", 1000) if isinstance(info, dict) else 1000
    chunk_idx = ep_index // chunks_size
    file_idx = ep_index
    return Path(f"videos/chunk-{chunk_idx:03d}/{vid_key}/episode_{file_idx:06d}.mp4")


_lerobot_ds.LeRobotDatasetMetadata.get_video_file_path = _patched_get_video_file_path


# 6. Fix the video_path template in info to match actual disk layout
_original_load_info = _lerobot_utils.load_info


def _patched_load_info(root):
    info = _original_load_info(root)
    if "video_path" in info:
        info["video_path"] = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"
    return info


_lerobot_utils.load_info = _patched_load_info

# --- End monkey-patch ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section 4.2: Memory Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Motus checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--num_eval_batches", type=int, default=5, help="Number of batches to evaluate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["baseline", "history_kv", "first_frame_memory"],
        help="Experiments to run",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model & data loading
# ---------------------------------------------------------------------------

def load_model_and_data(config_path: str, checkpoint_path: str, device: str):
    """Load Motus model and validation dataset with lerobot v2.1 compatibility patches."""
    import yaml
    from omegaconf import OmegaConf

    # Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)

    # Override inference steps
    config.model.inference.num_inference_timesteps = 50

    # Load model
    from models.motus import Motus, MotusConfig

    logger.info("Loading Motus model...")

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
        action_expert_norm_eps=float(config.model.action_expert.norm_eps),
        und_expert_hidden_size=config.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=config.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=float(config.model.und_expert.norm_eps),
        vlm_adapter_input_dim=config.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=config.model.und_expert.vlm.projector_type,
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=config.training.batch_size,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        training_mode="finetune",
        load_pretrained_backbones=False,
    )

    model = Motus(model_config)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        if os.path.isdir(checkpoint_path):
            ckpt_file = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
        else:
            ckpt_file = checkpoint_path

        state_dict = torch.load(ckpt_file, map_location="cpu")
        if "module" in state_dict:
            state_dict = state_dict["module"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    # Load dataset
    from data.dataset import create_dataset, collate_fn
    from torch.utils.data import DataLoader

    logger.info("Loading validation dataset...")
    val_dataset = create_dataset(config, val=True)
    dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return model, dataloader, config


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _import_metrics():
    """Lazy-import metrics to allow graceful handling if lpips is missing."""
    from metrics import compute_action_mse, LPIPSMetric, compute_video_mse
    return compute_action_mse, LPIPSMetric, compute_video_mse


def aggregate_metrics(
    action_metrics_list: list,
    video_metrics_list: list,
) -> dict:
    """Average metrics across batches."""
    compute_action_mse, _, compute_video_mse = _import_metrics()

    avg_action_mse = float(np.mean([m["action_mse"] for m in action_metrics_list]))
    avg_video_lpips = float(np.nanmean([m.get("video_lpips", np.nan) for m in video_metrics_list]))
    avg_video_mse = float(np.mean([m["video_mse"] for m in video_metrics_list]))

    # Per-step aggregation
    all_per_step = [m["action_mse_per_step"] for m in action_metrics_list]
    max_len = max(len(s) for s in all_per_step)
    padded = [s + [np.nan] * (max_len - len(s)) for s in all_per_step]
    avg_per_step = np.nanmean(padded, axis=0).tolist()

    return {
        "action_mse": avg_action_mse,
        "action_mse_per_step": avg_per_step,
        "video_lpips": avg_video_lpips,
        "video_mse": avg_video_mse,
        "num_batches": len(action_metrics_list),
    }


# ---------------------------------------------------------------------------
# Baseline evaluation (no memory)
# ---------------------------------------------------------------------------

def evaluate_baseline(
    model,
    dataloader,
    config,
    num_eval_batches: int,
    num_inference_steps: int,
    device: str,
) -> dict:
    """Run standard inference (no history KV, no cached und_tokens) and compute metrics.

    This is the reference baseline: every denoising step re-extracts VLM features
    and no first-frame context is injected into the attention layers.
    """
    from metrics import compute_action_mse, LPIPSMetric, compute_video_mse

    lpips_metric = LPIPSMetric(device=device)
    action_metrics_agg = []
    video_metrics_agg = []

    logger.info(f"  Evaluating baseline on {num_eval_batches} batches ...")

    for i, batch in enumerate(dataloader):
        if i >= num_eval_batches:
            break

        first_frame = batch["first_frame"].to(device)       # [B, C, H, W]
        video_frames = batch["video_frames"].to(device)     # [B, T, C, H, W]
        state = batch["initial_state"].to(device)             # [B, state_dim]
        gt_actions = batch["action_sequence"].to(device)     # [B, chunk_size, action_dim]
        language_embeddings = batch["language_embedding"]    # Tensor [B, seq_len, dim]
        vlm_inputs = batch["vlm_inputs"]                     # Dict

        with torch.no_grad():
            pred_frames, pred_actions = model.inference_step(
                first_frame=first_frame,
                state=state,
                num_inference_steps=num_inference_steps,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
            )

        action_metrics = compute_action_mse(pred_actions, gt_actions)
        video_metrics = lpips_metric.compute(pred_frames, video_frames)
        video_mse_metrics = compute_video_mse(pred_frames, video_frames)
        video_metrics.update(video_mse_metrics)

        action_metrics_agg.append(action_metrics)
        video_metrics_agg.append(video_metrics)

        logger.info(
            f"    Batch {i + 1}/{num_eval_batches}: "
            f"action_mse={action_metrics['action_mse']:.6f}"
        )

    return aggregate_metrics(action_metrics_agg, video_metrics_agg)


# ---------------------------------------------------------------------------
# Experiment 1.1: History KV Concatenation
# ---------------------------------------------------------------------------

def evaluate_history_kv(
    model,
    dataloader,
    config,
    num_eval_batches: int,
    num_inference_steps: int,
    device: str,
) -> dict:
    """Prepend first-frame K/V from all MoT layers during denoising.

    For each batch:
      1. Encode the first frame to a clean latent via the VAE.
      2. Run the first-frame latent through all 30 MoT layers to extract
         per-layer K/V pairs.
      3. During denoising, prepend these K/V pairs to the current frame's
         K/V in every self-attention call (cross-attention style).
    """
    from metrics import compute_action_mse, LPIPSMetric, compute_video_mse
    from model_patches import apply_patches, extract_first_frame_kv, patched_inference_step

    apply_patches()

    lpips_metric = LPIPSMetric(device=device)
    action_metrics_agg = []
    video_metrics_agg = []

    num_layers = config.num_layers if hasattr(config, "num_layers") else 30
    logger.info(f"  Evaluating History KV on {num_eval_batches} batches (num_layers={num_layers}) ...")

    for i, batch in enumerate(dataloader):
        if i >= num_eval_batches:
            break

        first_frame = batch["first_frame"].to(device)       # [B, C, H, W]
        video_frames = batch["video_frames"].to(device)     # [B, T, C, H, W]
        state = batch["initial_state"].to(device)             # [B, state_dim]
        gt_actions = batch["action_sequence"].to(device)     # [B, chunk_size, action_dim]
        language_embeddings = batch["language_embedding"]    # Tensor [B, seq_len, dim]
        vlm_inputs = batch["vlm_inputs"]                     # Dict

        # Encode first frame to clean latent
        with torch.no_grad():
            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
            condition_frame_latent = model.video_model.encode_video(
                first_frame_norm.to(model.dtype)
            )  # [B, C', 1, H', W']

        # Extract per-layer history K/V from the first frame
        history_kv = extract_first_frame_kv(model, condition_frame_latent, num_layers)

        with torch.no_grad():
            pred_frames, pred_actions = patched_inference_step(
                model,
                first_frame,
                state,
                num_inference_steps,
                language_embeddings,
                vlm_inputs,
                history_kv=history_kv,
            )

        action_metrics = compute_action_mse(pred_actions, gt_actions)
        video_metrics = lpips_metric.compute(pred_frames, video_frames)
        video_mse_metrics = compute_video_mse(pred_frames, video_frames)
        video_metrics.update(video_mse_metrics)

        action_metrics_agg.append(action_metrics)
        video_metrics_agg.append(video_metrics)

        logger.info(
            f"    Batch {i + 1}/{num_eval_batches}: "
            f"action_mse={action_metrics['action_mse']:.6f}"
        )

    return aggregate_metrics(action_metrics_agg, video_metrics_agg)


# ---------------------------------------------------------------------------
# Experiment 1.2: First-Frame Memory (cached VLM features)
# ---------------------------------------------------------------------------

def evaluate_first_frame_memory(
    model,
    dataloader,
    config,
    num_eval_batches: int,
    num_inference_steps: int,
    device: str,
) -> dict:
    """Extract VLM features once from the first frame and reuse across denoising.

    The baseline re-extracts VLM features at every denoising step.  This
    experiment caches the features from the first (cleanest) step and reuses
    them unchanged, testing whether the VLM features change meaningfully
    across noise levels.
    """
    from metrics import compute_action_mse, LPIPSMetric, compute_video_mse
    from model_patches import apply_patches, patched_inference_step

    apply_patches()

    lpips_metric = LPIPSMetric(device=device)
    action_metrics_agg = []
    video_metrics_agg = []

    logger.info(f"  Evaluating First Frame Memory on {num_eval_batches} batches ...")

    for i, batch in enumerate(dataloader):
        if i >= num_eval_batches:
            break

        first_frame = batch["first_frame"].to(device)       # [B, C, H, W]
        video_frames = batch["video_frames"].to(device)     # [B, T, C, H, W]
        state = batch["initial_state"].to(device)             # [B, state_dim]
        gt_actions = batch["action_sequence"].to(device)     # [B, chunk_size, action_dim]
        language_embeddings = batch["language_embedding"]    # Tensor [B, seq_len, dim]
        vlm_inputs = batch["vlm_inputs"]                     # Dict

        # Extract VLM features once from the first frame (clean input)
        with torch.no_grad():
            first_frame_und_tokens = model.und_module.extract_und_features(vlm_inputs)

        # Run inference with cached und_tokens (no per-step re-extraction)
        with torch.no_grad():
            pred_frames, pred_actions = patched_inference_step(
                model,
                first_frame,
                state,
                num_inference_steps,
                language_embeddings,
                vlm_inputs,
                first_frame_und_tokens=first_frame_und_tokens,
            )

        action_metrics = compute_action_mse(pred_actions, gt_actions)
        video_metrics = lpips_metric.compute(pred_frames, video_frames)
        video_mse_metrics = compute_video_mse(pred_frames, video_frames)
        video_metrics.update(video_mse_metrics)

        action_metrics_agg.append(action_metrics)
        video_metrics_agg.append(video_metrics)

        logger.info(
            f"    Batch {i + 1}/{num_eval_batches}: "
            f"action_mse={action_metrics['action_mse']:.6f}"
        )

    return aggregate_metrics(action_metrics_agg, video_metrics_agg)


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

# Pretty names for each experiment key
_METHOD_NAMES = {
    "baseline": "Baseline (no history)   ",
    "history_kv": "History KV (first frame) ",
    "first_frame_memory": "First Frame Memory      ",
}


def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table of all experiment results."""
    header = (
        "\n"
        "=" * 68 + "\n"
        "  Section 4.2: Memory Experiments Results\n"
        "=" * 68 + "\n"
        f"{'Method':<28s}| {'Action MSE':<12s}| {'Video LPIPS':<12s}| {'Video MSE':<12s}\n"
        "-" * 68
    )
    print(header)

    for key in ["baseline", "history_kv", "first_frame_memory"]:
        if key not in results:
            continue
        r = results[key]
        name = _METHOD_NAMES.get(key, key)
        action_mse = r.get("action_mse", float("nan"))
        video_lpips = r.get("video_lpips", float("nan"))
        video_mse = r.get("video_mse", float("nan"))
        lpips_str = f"{video_lpips:.4f}" if not np.isnan(video_lpips) else "N/A"
        print(
            f"{name}| {action_mse:<12.6f}| {lpips_str:<12s}| {video_mse:<12.6f}"
        )

    print("=" * 68)

    # Print deltas relative to baseline
    if "baseline" in results:
        base_action = results["baseline"].get("action_mse", None)
        base_lpips = results["baseline"].get("video_lpips", None)
        base_vmse = results["baseline"].get("video_mse", None)

        print("\n  Delta vs baseline:")
        for key in ["history_kv", "first_frame_memory"]:
            if key not in results:
                continue
            r = results[key]
            name = _METHOD_NAMES.get(key, key).strip()
            parts = []
            if base_action is not None:
                d = r.get("action_mse", 0) - base_action
                parts.append(f"action_mse {d:+.6f}")
            if base_lpips is not None and not np.isnan(base_lpips):
                d = r.get("video_lpips", 0) - base_lpips
                parts.append(f"video_lpips {d:+.4f}")
            if base_vmse is not None:
                d = r.get("video_mse", 0) - base_vmse
                parts.append(f"video_mse {d:+.6f}")
            print(f"    {name}: {'  '.join(parts)}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Section 4.2: Memory Experiments")
    logger.info("=" * 60)
    logger.info(f"Config:       {args.config}")
    logger.info(f"Checkpoint:   {args.checkpoint}")
    logger.info(f"Output dir:   {args.output_dir}")
    logger.info(f"Batches:      {args.num_eval_batches}")
    logger.info(f"Steps:        {args.num_inference_steps}")
    logger.info(f"Experiments:  {args.experiments}")
    logger.info(f"Device:       {args.device}")
    logger.info(f"Seed:         {args.seed}")

    model, dataloader, config = load_model_and_data(
        args.config, args.checkpoint, args.device
    )

    results = {}

    # ---- Baseline (no memory) ----
    if "baseline" in args.experiments:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Running Baseline (no memory)...")
        logger.info("=" * 60)
        results["baseline"] = evaluate_baseline(
            model=model,
            dataloader=dataloader,
            config=config,
            num_eval_batches=args.num_eval_batches,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        )
        logger.info(f"  Baseline action_mse = {results['baseline']['action_mse']:.6f}")

    # ---- Experiment 1.1: History KV ----
    if "history_kv" in args.experiments:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Running History KV experiment (Exp 1.1)...")
        logger.info("=" * 60)
        results["history_kv"] = evaluate_history_kv(
            model=model,
            dataloader=dataloader,
            config=config,
            num_eval_batches=args.num_eval_batches,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        )
        logger.info(f"  History KV action_mse = {results['history_kv']['action_mse']:.6f}")

    # ---- Experiment 1.2: First-Frame Memory ----
    if "first_frame_memory" in args.experiments:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Running First Frame Memory experiment (Exp 1.2)...")
        logger.info("=" * 60)
        results["first_frame_memory"] = evaluate_first_frame_memory(
            model=model,
            dataloader=dataloader,
            config=config,
            num_eval_batches=args.num_eval_batches,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        )
        logger.info(
            f"  First Frame Memory action_mse = "
            f"{results['first_frame_memory']['action_mse']:.6f}"
        )

    # ---- Save results ----
    output_file = os.path.join(args.output_dir, "memory_experiments.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_inference_steps": args.num_inference_steps,
        "num_eval_batches": args.num_eval_batches,
        "experiments_run": args.experiments,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Print comparison table ----
    print_comparison_table(results)

    logger.info(f"Results saved to {output_file}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
