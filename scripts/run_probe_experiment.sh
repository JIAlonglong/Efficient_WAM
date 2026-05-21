#!/bin/bash
# ============================================================================
# Experiment 1: Probe Training Script
# ============================================================================
# Trains probes on frozen Motus tokens to predict action sequences.
# Evaluates semantic quality of Understanding Tokens.
#
# Usage:
#   bash scripts/run_probe_experiment.sh                    # Default config
#   bash scripts/run_probe_experiment.sh configs/probe.yaml # Custom config
#
# Prerequisites:
#   - Motus checkpoint downloaded and configured
#   - Dataset available at the path specified in config
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG_FILE="${1:-configs/probe.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-probe_results}"

echo "============================================"
echo "Motus Probe Experiment 1"
echo "============================================"
echo "Config:      $CONFIG_FILE"
echo "Output dir:  $OUTPUT_DIR"
echo "Project dir: $PROJECT_DIR"
echo "============================================"

# ── Step 1: Verify environment ────────────────────────────────────────
echo ""
echo "[Step 1/4] Checking environment..."

python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
import yaml
print(f'  PyYAML:  OK')
import wandb
print(f'  WandB:   {wandb.__version__}')
" || {
    echo "ERROR: Missing dependencies. Install with:"
    echo "  pip install torch pyyaml wandb matplotlib numpy tqdm"
    exit 1
}

# ── Step 2: Token extraction (if needed) ──────────────────────────────
echo ""
echo "[Step 2/4] Token extraction..."
echo "  If tokens are already cached, this step will be skipped."
echo "  For static debugging (no checkpoint), use the test mode below."
echo ""

# Uncomment and set checkpoint path for actual extraction:
# python3 -c "
# import sys; sys.path.insert(0, '.')
# from src.probe import MotusProbeDataset
# from models.motus import Motus, MotusConfig
# import torch
#
# # Load config and model
# import yaml
# with open('$CONFIG_FILE') as f:
#     config = yaml.safe_load(f)
#
# # Load frozen Motus model
# motus_config = MotusConfig()
# motus = Motus(motus_config)
# motus.load_checkpoint(config['model']['checkpoint_path'])
# motus.eval()
# motus.requires_grad_(False)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # ... create dataloader and extract tokens
# "

# ── Step 3: Probe training ────────────────────────────────────────────
echo ""
echo "[Step 3/4] Training probes..."
echo ""

# Uncomment for actual training:
# python3 -c "
# import sys; sys.path.insert(0, '.')
# import yaml
# import torch
# from torch.utils.data import DataLoader
# from src.probe import MotusProbeDataset, train_action_probe
#
# with open('$CONFIG_FILE') as f:
#     config = yaml.safe_load(f)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Load cached datasets
# train_dataset = MotusProbeDataset(cache_dir=config['cache']['cache_dir'] + '/train')
# test_dataset = MotusProbeDataset(cache_dir=config['cache']['cache_dir'] + '/test')
#
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=config['training']['batch_size'],
#     shuffle=True, num_workers=config['data']['num_workers'],
# )
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=config['training']['batch_size'],
#     shuffle=False, num_workers=config['data']['num_workers'],
# )
#
# # Run training
# results = train_action_probe(
#     config={**config['training'], 'probe_configs': config['probe_configs']},
#     train_loader=train_loader,
#     test_loader=test_loader,
#     device=device,
#     output_dir=config['output']['output_dir'],
#     wandb_mode=config['wandb']['mode'],
# )
#
# print('\\nResults:')
# for name, res in results.items():
#     print(f'  {name}: MSE={res[\"best_mse\"]:.6f} (epoch {res[\"best_epoch\"]})')
# "

echo "  [Skipped - set up model checkpoint to enable]"

# ── Step 4: Visualization ─────────────────────────────────────────────
echo ""
echo "[Step 4/4] Generating visualizations..."
echo ""

# Uncomment for actual visualization:
# python3 -c "
# import sys; sys.path.insert(0, '.')
# import json
# from src.visualize_probe import generate_all_plots
#
# with open('${OUTPUT_DIR}/probe_results.json') as f:
#     results = json.load(f)
#
# generate_all_plots(results, output_dir='${OUTPUT_DIR}/plots')
# print('Plots saved to ${OUTPUT_DIR}/plots/')
# "

echo "  [Skipped - no results to visualize]"

echo ""
echo "============================================"
echo "Done!"
echo "============================================"

# ── Static test mode ──────────────────────────────────────────────────
# Run this to verify imports and basic logic without a checkpoint:
#
#   python3 -c "
#   import sys; sys.path.insert(0, '.')
#   from src.probe import (
#       MotusLinearProbe, MotusTemporalProbe, MotusSpatiotemporalProbe,
#       create_probe, compute_regression_metrics,
#   )
#   import torch
#
#   # Test probe creation
#   for ptype in ['linear', 'temporal', 'spatiotemporal']:
#       probe = create_probe(ptype, feature_dim=3072, n_frames=1)
#       x = torch.randn(2, 64, 3072)
#       out = probe(x)
#       print(f'{ptype}: input {x.shape} -> output {out.shape}')
#
#   # Test metrics
#   preds = torch.randn(4, 16, 14)
#   targets = torch.randn(4, 16, 14)
#   metrics = compute_regression_metrics(preds, targets)
#   print(f'MSE: {metrics[\"mse\"]:.4f}, Correlations: {len(metrics[\"correlations\"])} dims')
#   print('All static tests passed!')
#   "
