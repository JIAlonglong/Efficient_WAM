"""Evaluation metrics for Motus probing experiments.

Computes:
  - MSE (Mean Squared Error) for overall regression quality
  - Per-dimension Pearson correlation for fine-grained analysis
  - Per-step (chunk position) MSE for temporal analysis

Reference: semantic-wm's _evaluate_probe() adapted from classification to regression.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_probe(
    probe: nn.Module,
    dataloader: DataLoader,
    feature_key: str,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a probe on a dataloader.

    Parameters
    ----------
    probe : nn.Module
        Trained probe model.
    dataloader : DataLoader
        DataLoader yielding batches with feature_key and "action".
    feature_key : str
        Key for the input feature in each batch (e.g. "video_before").
    device : torch.device
        Device for computation.

    Returns
    -------
    dict with:
        "mse": float - overall MSE
        "correlations": list[float] - per-dimension Pearson correlations
        "per_step_mse": list[float] - MSE per chunk position
    """
    probe.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[feature_key].to(device)
            pred = probe(x)
            all_preds.append(pred.cpu())
            all_targets.append(batch["action"])

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return compute_regression_metrics(preds, targets)


def evaluate_probe_mse_correlation(
    probe: nn.Module,
    dataloader: DataLoader,
    feature_key: str,
    device: torch.device,
) -> Dict[str, float]:
    """Alias for evaluate_probe with same interface."""
    return evaluate_probe(probe, dataloader, feature_key, device)


def compute_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute regression metrics between predictions and targets.

    Parameters
    ----------
    preds : (B, chunk_size, action_dim) tensor
    targets : (B, chunk_size, action_dim) tensor

    Returns
    -------
    dict with "mse", "correlations", "per_step_mse"
    """
    assert preds.shape == targets.shape, (
        f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
    )

    # Overall MSE
    mse = F.mse_loss(preds, targets).item()

    # Per-dimension Pearson correlation
    chunk_size, action_dim = preds.shape[1], preds.shape[2]
    correlations = compute_pearson_correlations(preds, targets)

    # Per-step MSE
    per_step_mse = compute_per_step_mse(preds, targets)

    return {
        "mse": mse,
        "correlations": correlations,
        "per_step_mse": per_step_mse,
    }


def compute_pearson_correlations(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> List[float]:
    """Compute per-dimension Pearson correlation coefficients.

    Parameters
    ----------
    preds : (B, chunk_size, action_dim)
    targets : (B, chunk_size, action_dim)

    Returns
    -------
    list of float : Pearson r for each action dimension.
    """
    action_dim = preds.shape[-1]
    correlations = []

    # Flatten batch and chunk dimensions for each action dim
    preds_flat = preds.reshape(-1, action_dim).numpy()
    targets_flat = targets.reshape(-1, action_dim).numpy()

    for d in range(action_dim):
        p = preds_flat[:, d]
        t = targets_flat[:, d]
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            correlations.append(0.0)
        else:
            corr = np.corrcoef(p, t)[0, 1]
            correlations.append(float(corr) if np.isfinite(corr) else 0.0)

    return correlations


def compute_per_step_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> List[float]:
    """Compute MSE for each position in the action chunk.

    Parameters
    ----------
    preds : (B, chunk_size, action_dim)
    targets : (B, chunk_size, action_dim)

    Returns
    -------
    list of float : MSE at each chunk position.
    """
    chunk_size = preds.shape[1]
    per_step = []
    for t in range(chunk_size):
        step_mse = F.mse_loss(preds[:, t], targets[:, t]).item()
        per_step.append(step_mse)
    return per_step


def compute_mae(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute Mean Absolute Error."""
    return F.l1_loss(preds, targets).item()


def compare_probe_results(
    results: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """Generate a summary comparison of multiple probe results.

    Parameters
    ----------
    results : dict
        Mapping from probe_name -> metrics dict (as returned by evaluate_probe).

    Returns
    -------
    dict with summary strings.
    """
    summary = {}

    # Sort by MSE
    sorted_probes = sorted(results.items(), key=lambda x: x[1].get("mse", float("inf")))

    lines = ["Probe Comparison (sorted by MSE):"]
    lines.append("-" * 60)
    for name, metrics in sorted_probes:
        mse = metrics.get("mse", float("nan"))
        corrs = metrics.get("correlations", [])
        mean_corr = np.mean(corrs) if corrs else 0.0
        lines.append(f"  {name:30s}  MSE={mse:.6f}  MeanCorr={mean_corr:.4f}")
    lines.append("-" * 60)

    summary["text"] = "\n".join(lines)
    summary["best_probe"] = sorted_probes[0][0] if sorted_probes else "N/A"
    summary["best_mse"] = sorted_probes[0][1].get("mse", float("inf")) if sorted_probes else float("inf")

    return summary
