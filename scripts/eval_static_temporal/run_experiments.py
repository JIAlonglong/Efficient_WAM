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
        batch_size=config.training.batch_size,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        training_mode='finetune',
        load_pretrained_backbones=True,
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
    from data.dataset import create_dataloader
    logger.info("Loading RoboTwin2 dataset...")
    dataloader = create_dataloader(
        config=config,
        split='val',
        batch_size=1,
        num_workers=4,
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

    # Load model and data
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
        st_results = run_analysis(
            model=model,
            dataloader=dataloader,
            config=config,
            num_batches=args.num_eval_batches,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'exp3'),
        )
        results['exp3_semantic_temporal'] = st_results

        with open(os.path.join(args.output_dir, 'exp3_kendall_tau.json'), 'w') as f:
            json.dump(st_results, f, indent=2, default=str)
        logger.info(f"Exp3 Kendall tau: {st_results.get('mean_tau', 'N/A')}")

    # ==================== Exp4: Heatmap ====================
    if 'exp4' in args.experiments:
        logger.info("=" * 60)
        logger.info("Running Exp4: Token Importance Heatmap")
        logger.info("=" * 60)

        from visualize_token_importance import run_visualization
        vis_results = run_visualization(
            model=model,
            dataloader=dataloader,
            config=config,
            num_batches=min(args.num_eval_batches, 2),  # Vis only needs a few
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'exp4'),
        )
        results['exp4_heatmap'] = vis_results
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
        tau = results['exp3_semantic_temporal'].get('mean_tau', 'N/A')
        logger.info(f"  Kendall τ: {tau}")
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
