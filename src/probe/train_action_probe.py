"""Training loop for Motus action prediction probes.

Trains probes on frozen Motus token features to predict action sequences.
Uses MSELoss + AdamW + CosineAnnealingLR. Logs to WandB.

Reference: semantic-wm's train_success_probe() adapted from classification
(BCEWithLogitsLoss) to regression (MSELoss).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate_probe import evaluate_probe
from .motus_probe import create_probe, get_probe_param_count

logger = logging.getLogger(__name__)


def train_single_probe(
    probe: nn.Module,
    probe_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_key: str,
    device: torch.device,
    output_dir: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    wandb_run=None,
) -> Dict[str, float]:
    """Train a single probe model.

    Parameters
    ----------
    probe : nn.Module
        Probe model to train.
    probe_name : str
        Name for logging and checkpointing.
    train_loader : DataLoader
        Training data (cached tokens).
    test_loader : DataLoader
        Test data (cached tokens).
    feature_key : str
        Key for input features in batch dict.
    device : torch.device
        Training device.
    output_dir : Path
        Directory for checkpoints.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for AdamW.
    wandb_run : optional
        WandB run object for logging.

    Returns
    -------
    dict with training results (best_mse, best_epoch, correlations).
    """
    probe = probe.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_mse = float("inf")
    best_epoch = 0

    logger.info(
        "Training probe '%s' (%s) with %d params, feature_key='%s'",
        probe_name,
        probe.__class__.__name__,
        get_probe_param_count(probe),
        feature_key,
    )

    for epoch in range(epochs):
        # ── Training ────────────────────────────────────────────────────
        probe.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[{probe_name}] Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[feature_key].to(device)
            target = batch["action"].to(device)

            pred = probe(x)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        train_mse = total_loss / max(n_batches, 1)

        # ── Evaluation ──────────────────────────────────────────────────
        test_metrics = evaluate_probe(probe, test_loader, feature_key, device)
        test_mse = test_metrics["mse"]

        log_msg = (
            f"[{probe_name}] Epoch {epoch+1}/{epochs}: "
            f"train_mse={train_mse:.6f}  test_mse={test_mse:.6f}"
        )
        logger.info(log_msg)

        # WandB logging
        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    f"probe/{probe_name}/train_mse": train_mse,
                    f"probe/{probe_name}/test_mse": test_mse,
                    f"probe/{probe_name}/lr": scheduler.get_last_lr()[0],
                    f"probe/{probe_name}/epoch": epoch + 1,
                }
            )

        # ── Checkpoint ──────────────────────────────────────────────────
        if test_mse < best_mse:
            best_mse = test_mse
            best_epoch = epoch + 1
            ckpt_path = output_dir / f"{probe_name}_best.pt"
            torch.save(
                {
                    "probe": probe.state_dict(),
                    "probe_name": probe_name,
                    "probe_class": probe.__class__.__name__,
                    "feature_key": feature_key,
                    "epoch": epoch + 1,
                    "test_mse": best_mse,
                    "test_correlations": test_metrics["correlations"],
                },
                ckpt_path,
            )

    # Final metrics
    final_metrics = evaluate_probe(probe, test_loader, feature_key, device)
    results = {
        "best_mse": best_mse,
        "best_epoch": best_epoch,
        "final_mse": final_metrics["mse"],
        "final_correlations": final_metrics["correlations"],
        "final_per_step_mse": final_metrics["per_step_mse"],
        "checkpoint": str(output_dir / f"{probe_name}_best.pt"),
    }

    return results


def train_action_probe(
    config: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path = "probe_results",
    wandb_mode: str = "disabled",
) -> Dict[str, Dict[str, float]]:
    """Train all configured probes and compare results.

    This is the main entry point for Experiment 1 probe training.

    Parameters
    ----------
    config : dict
        Probe experiment configuration. Expected keys:
          - probe_configs: list of dicts, each with:
              - name: str
              - type: "linear" | "temporal" | "spatiotemporal"
              - feature_key: str (e.g. "video_before", "und_after")
              - feature_dim: int
              - n_frames: int (optional, for temporal/spatiotemporal)
              - n_patches: int (optional, for spatiotemporal)
              - action_dim: int (default 14)
              - chunk_size: int (default 16)
          - epochs: int (default 50)
          - lr: float (default 1e-3)
          - weight_decay: float (default 1e-4)
    train_loader : DataLoader
        Training data loader (cached probe dataset).
    test_loader : DataLoader
        Test data loader (cached probe dataset).
    device : torch.device
        Training device.
    output_dir : str or Path
        Directory for results and checkpoints.
    wandb_mode : str
        WandB mode ("disabled", "online", "offline").

    Returns
    -------
    dict mapping probe_name -> results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB ───────────────────────────────────────────────────────────
    wandb_run = None
    if wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.get("wandb_project", "motus-probe"),
                name=config.get("run_name", "probe-experiment-1"),
                mode=wandb_mode,
                config=config,
                dir=str(output_dir),
            )
        except Exception as e:
            logger.warning("Failed to initialize WandB: %s", e)

    # ── Training parameters ─────────────────────────────────────────────
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)

    # ── Train each probe ────────────────────────────────────────────────
    probe_configs = config.get("probe_configs", [])
    all_results = {}

    for probe_cfg in probe_configs:
        name = probe_cfg["name"]
        probe_type = probe_cfg["type"]
        feature_key = probe_cfg["feature_key"]
        feature_dim = probe_cfg["feature_dim"]

        logger.info("=" * 60)
        logger.info("Starting probe: %s (type=%s, feature=%s)", name, probe_type, feature_key)
        logger.info("=" * 60)

        # Create probe
        probe = create_probe(
            probe_type=probe_type,
            feature_dim=feature_dim,
            n_frames=probe_cfg.get("n_frames", 1),
            action_dim=probe_cfg.get("action_dim", 14),
            chunk_size=probe_cfg.get("chunk_size", 16),
            n_patches=probe_cfg.get("n_patches", 64),
            n_heads=probe_cfg.get("n_heads", 8),
            n_layers=probe_cfg.get("n_layers", 1),
        )

        # Train
        results = train_single_probe(
            probe=probe,
            probe_name=name,
            train_loader=train_loader,
            test_loader=test_loader,
            feature_key=feature_key,
            device=device,
            output_dir=output_dir / "checkpoints",
            epochs=probe_cfg.get("epochs", epochs),
            lr=probe_cfg.get("lr", lr),
            weight_decay=probe_cfg.get("weight_decay", weight_decay),
            wandb_run=wandb_run,
        )

        all_results[name] = results
        logger.info(
            "[%s] Done. Best MSE=%.6f at epoch %d",
            name, results["best_mse"], results["best_epoch"],
        )

    # ── Save summary ────────────────────────────────────────────────────
    summary_path = output_dir / "probe_results.json"
    # Convert for JSON serialization
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {
            k: v if not isinstance(v, list) else v
            for k, v in res.items()
        }
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", summary_path)

    # ── Log summary to WandB ────────────────────────────────────────────
    if wandb_run is not None:
        import wandb

        for name, res in all_results.items():
            wandb.log({f"summary/{name}/best_mse": res["best_mse"]})

    # ── Print comparison ────────────────────────────────────────────────
    from .evaluate_probe import compare_probe_results

    comparison = compare_probe_results(all_results)
    logger.info("\n%s", comparison["text"])

    if wandb_run is not None:
        wandb_run.finish()

    return all_results
