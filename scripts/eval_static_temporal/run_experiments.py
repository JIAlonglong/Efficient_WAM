#!/usr/bin/env python3
"""
Main experiment runner for Static vs Temporal Gap Analysis.

Runs Exp1-4:
  Exp1: Single Frame Bias Score (SFB)
  Exp2: Temporal Gain Profile (TG)
  Exp3: Semantic vs Temporal Redundancy (Kendall tau)
  Exp4: Token Importance Heatmap

Usage:
    python run_experiments.py \
        --config ../../configs/robotwin.yaml \
        --checkpoint /path/to/Motus_robotwin2/mp_rank_00_model_states.pt \
        --output_dir ./results \
        --num_eval_batches 5 \
        --num_inference_steps 50
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# --- Monkey-patch lerobot to allow loading v2.1 datasets with v3.0 code ---
import lerobot.datasets.utils as _lerobot_utils
import lerobot.datasets.lerobot_dataset as _lerobot_ds
import pandas as _pd

# 1. Bypass version compatibility check
_original_check = _lerobot_utils.check_version_compatibility
def _patched_check(repo_id, version_to_check, current_version, enforce_breaking_major=True):
    logger.info(f"Bypassing version check for dataset {repo_id} (v{version_to_check} vs codebase v{current_version})")
    return
_lerobot_utils.check_version_compatibility = _patched_check
try:
    import lerobot.datasets.backward_compatibility as _bc
    if hasattr(_bc, 'check_version_compatibility'):
        _bc.check_version_compatibility = _patched_check
except ImportError:
    pass

# 2. Patch get_safe_version to skip HF Hub lookup for local datasets
def _patched_get_safe_version(repo_id, revision=None):
    """Skip Hub lookup; return current codebase version for local datasets."""
    return _lerobot_ds.CODEBASE_VERSION
_lerobot_utils.get_safe_version = _patched_get_safe_version
_lerobot_ds.get_safe_version = _patched_get_safe_version  # Also patch in lerobot_dataset module

# 3. Patch load_tasks to support v2.1 tasks.jsonl format
_original_load_tasks = _lerobot_utils.load_tasks
def _patched_load_tasks(root, episodes=None):
    """Load tasks from either tasks.parquet (v3.0) or tasks.jsonl (v2.1)."""
    parquet_path = root / "meta" / "tasks.parquet"
    jsonl_path = root / "meta" / "tasks.jsonl"
    if parquet_path.exists():
        return _original_load_tasks(root, episodes)
    elif jsonl_path.exists():
        tasks = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    import json as _json
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
        import pyarrow as pa
        records = []
        with open(episodes_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = _json.loads(line)
                    # Flatten: keep episode_index, length, and first task
                    flat = {
                        "episode_index": rec.get("episode_index", 0),
                        "length": rec.get("length", 0),
                    }
                    # Add tasks as a string (first task)
                    tasks = rec.get("tasks", [])
                    flat["tasks"] = tasks[0] if tasks else ""
                    records.append(flat)
        table = pa.table(records)
        return _lerobot_utils.load_nested_dataset.__wrapped__(root / "meta") if hasattr(_lerobot_utils.load_nested_dataset, '__wrapped__') else _pd.DataFrame(records)
    else:
        logger.warning(f"No episodes file found in {root / 'meta'}")
        return _pd.DataFrame()

# Simpler approach: just use pandas to read jsonl
def _patched_load_episodes_simple(root):
    """Load episodes from episodes.jsonl (v2.1 format)."""
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
        # Create a HuggingFace Dataset from the records
        from datasets import Dataset as _HFDataset
        return _HFDataset.from_list(records)
    else:
        logger.warning(f"No episodes file found in {root / 'meta'}")
        from datasets import Dataset as _HFDataset
        return _HFDataset.from_list([])

_lerobot_utils.load_episodes = _patched_load_episodes_simple

# 5. Patch get_video_file_path to handle episodes without per-video chunk columns
#    and to use the correct on-disk path layout (chunk before video_key).
import lerobot.datasets.lerobot_dataset as _lerobot_ds_mod
_original_get_video_file_path = _lerobot_ds_mod.LeRobotDatasetMetadata.get_video_file_path
def _patched_get_video_file_path(self, ep_index, vid_key):
    """Handle episodes that lack per-video chunk/file index columns,
    and use the actual on-disk layout: videos/chunk-{idx}/{key}/episode_{idx}.mp4"""
    # Compute chunk/file from episode_index and chunks_size
    info = getattr(self, 'info', None)
    chunks_size = info.get('chunks_size', 1000) if isinstance(info, dict) else 1000
    chunk_idx = ep_index // chunks_size
    file_idx = ep_index
    # The actual on-disk layout puts chunk first, then video_key:
    # videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4
    return Path(f"videos/chunk-{chunk_idx:03d}/{vid_key}/episode_{file_idx:06d}.mp4")
_lerobot_ds_mod.LeRobotDatasetMetadata.get_video_file_path = _patched_get_video_file_path

# Also fix the video_path template in info to match actual disk layout
_original_load_info = _lerobot_utils.load_info
def _patched_load_info(root):
    info = _original_load_info(root)
    if 'video_path' in info:
        # Fix: the default template puts video_key before chunk, but actual disk layout is reversed
        info['video_path'] = 'videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4'
    return info
_lerobot_utils.load_info = _patched_load_info

# --- End monkey-patch ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Static vs Temporal Gap Analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Motus checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_eval_batches', type=int, default=5, help='Number of batches to evaluate')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Denoising steps')
    parser.add_argument('--experiments', type=str, nargs='+', default=['exp1', 'exp2', 'exp3', 'exp4'],
                        help='Experiments to run')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def load_model_and_data(config_path, checkpoint_path, device):
    """Load Motus model and RoboTwin2 dataset."""
    import yaml
    from omegaconf import OmegaConf

    # Load config
    with open(config_path, 'r') as f:
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
        training_mode='finetune',
        load_pretrained_backbones=False,  # Skip loading pretrained backbones; load from checkpoint instead
    )

    model = Motus(model_config)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        if os.path.isdir(checkpoint_path):
            ckpt_file = os.path.join(checkpoint_path, 'mp_rank_00_model_states.pt')
        else:
            ckpt_file = checkpoint_path

        state_dict = torch.load(ckpt_file, map_location='cpu')
        if 'module' in state_dict:
            state_dict = state_dict['module']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    # Load dataset
    from data.dataset import create_dataset, collate_fn
    from torch.utils.data import DataLoader
    logger.info("Loading RoboTwin2 dataset...")
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and data only if needed (exp1/exp2 use it directly;
    # exp3/exp4 load their own models via standalone scripts).
    needs_model = bool(set(args.experiments) & {'exp1', 'exp2'})
    model, dataloader, config = None, None, None
    if needs_model:
        model, dataloader, config = load_model_and_data(
            args.config, args.checkpoint, args.device
        )

    results = {}

    # ==================== Exp1: SFB Score ====================
    if 'exp1' in args.experiments:
        logger.info("=" * 60)
        logger.info("Running Exp1: Single Frame Bias Score")
        logger.info("=" * 60)

        from evaluate_ablation import run_sfb_experiment
        sfb_results = run_sfb_experiment(
            model=model,
            dataloader=dataloader,
            config=config,
            num_eval_batches=args.num_eval_batches,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        )
        results['exp1_sfb'] = sfb_results

        # Save
        with open(os.path.join(args.output_dir, 'exp1_sfb.json'), 'w') as f:
            json.dump(sfb_results, f, indent=2, default=str)
        logger.info(f"Exp1 SFB Score: {sfb_results.get('sfb_score', 'N/A')}")

    # ==================== Exp2: TG Profile ====================
    if 'exp2' in args.experiments:
        logger.info("=" * 60)
        logger.info("Running Exp2: Temporal Gain Profile")
        logger.info("=" * 60)

        from evaluate_ablation import run_tg_experiment
        tg_results = run_tg_experiment(
            model=model,
            dataloader=dataloader,
            config=config,
            num_eval_batches=args.num_eval_batches,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        )
        results['exp2_tg'] = tg_results

        with open(os.path.join(args.output_dir, 'exp2_tg.json'), 'w') as f:
            json.dump(tg_results, f, indent=2, default=str)
        logger.info(f"Exp2 TG Profile: {tg_results.get('tg_profile', 'N/A')}")

    # ==================== Exp3: Semantic vs Temporal ====================
    if 'exp3' in args.experiments:
        logger.info("=" * 60)
        logger.info("Running Exp3: Semantic vs Temporal Redundancy")
        logger.info("=" * 60)

        from analyze_semantic_temporal import run_analysis
        from argparse import Namespace as _NS
        st_args = _NS(
            config=args.config,
            checkpoint=args.checkpoint,
            output_dir=os.path.join(args.output_dir, 'exp3'),
            num_batches=args.num_eval_batches,
        )
        run_analysis(st_args)

        # Read back the saved results
        st_result_file = os.path.join(args.output_dir, 'exp3', 'semantic_temporal_analysis.json')
        with open(st_result_file) as f:
            st_results = json.load(f)
        results['exp3_semantic_temporal'] = st_results

        with open(os.path.join(args.output_dir, 'exp3_kendall_tau.json'), 'w') as f:
            json.dump(st_results, f, indent=2, default=str)
        logger.info(f"Exp3 Kendall tau: {st_results.get('kendall_tau', {}).get('mean_tau', 'N/A')}")

    # ==================== Exp4: Heatmap ====================
    if 'exp4' in args.experiments:
        logger.info("=" * 60)
        logger.info("Running Exp4: Token Importance Heatmap")
        logger.info("=" * 60)

        from visualize_token_importance import run_visualization
        run_visualization(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=os.path.join(args.output_dir, 'exp4'),
            num_samples=min(args.num_eval_batches, 2),
            batch_size=min(args.num_eval_batches, 2),
            device=args.device,
            seed=args.seed,
        )
        results['exp4_heatmap'] = {'output_dir': os.path.join(args.output_dir, 'exp4')}
        logger.info(f"Exp4 visualizations saved to {args.output_dir}/exp4/")

    # ==================== Summary ====================
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'num_inference_steps': args.num_inference_steps,
        'num_eval_batches': args.num_eval_batches,
        'results': results,
    }

    # Print key metrics
    if 'exp1_sfb' in results:
        sfb = results['exp1_sfb'].get('sfb_score', 'N/A')
        logger.info(f"  SFB Score: {sfb}")
        if isinstance(sfb, (int, float)):
            if sfb > 0.9:
                logger.info(f"  → Static information is SUFFICIENT (SFB > 0.9)")
            elif sfb > 0.8:
                logger.info(f"  → Gray zone (0.8 < SFB < 0.9), need further analysis")
            else:
                logger.info(f"  → Temporal information is IMPORTANT (SFB < 0.8)")

    if 'exp2_tg' in results:
        tg = results['exp2_tg'].get('tg_profile', [])
        logger.info(f"  TG Profile: {tg}")

    if 'exp3_semantic_temporal' in results:
        tau = results['exp3_semantic_temporal'].get('kendall_tau', {}).get('mean_tau', 'N/A')
        logger.info(f"  Kendall tau: {tau}")
        if isinstance(tau, (int, float)):
            if tau > 0.5:
                logger.info(f"  → Semantic importance correlates with temporal novelty")
            elif tau > 0.2:
                logger.info(f"  → Weak correlation, mixed signals")
            else:
                logger.info(f"  → No correlation, semantic guidance cannot capture temporal dynamics")

    # Save full summary
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {args.output_dir}/")
    logger.info("Done!")


if __name__ == '__main__':
    main()
