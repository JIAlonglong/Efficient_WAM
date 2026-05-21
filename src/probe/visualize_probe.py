"""Visualization utilities for Motus probing experiments.

Generates:
  - Bar chart comparing MSE across probes
  - Per-dimension correlation heatmap
  - Per-step MSE curves
  - Layer-wise analysis plot

All plots saved as PNG files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import matplotlib to avoid issues in headless environments
_plt = None


def _get_plt():
    """Lazy-import matplotlib.pyplot."""
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def plot_probe_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str | Path,
    title: str = "Probe MSE Comparison",
    figsize: tuple = (12, 6),
) -> Path:
    """Bar chart comparing MSE across all probes.

    Parameters
    ----------
    results : dict
        Mapping from probe_name -> metrics dict (needs "mse" key).
    output_path : str or Path
        Path to save the figure.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Path to saved figure.
    """
    plt = _get_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    mses = [results[n].get("mse", float("nan")) for n in names]

    # Sort by MSE
    sorted_pairs = sorted(zip(names, mses), key=lambda x: x[1])
    names_sorted = [p[0] for p in sorted_pairs]
    mses_sorted = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names_sorted)))
    bars = ax.barh(range(len(names_sorted)), mses_sorted, color=colors)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel("MSE (lower is better)")
    ax.set_title(title)
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, mses_sorted):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved probe comparison plot to %s", output_path)
    return output_path


def plot_correlation_heatmap(
    results: Dict[str, Dict[str, float]],
    action_dim: int = 14,
    output_path: str | Path = "correlation_heatmap.png",
    title: str = "Per-Dimension Pearson Correlation",
    figsize: tuple = (14, 8),
    dim_labels: Optional[List[str]] = None,
) -> Path:
    """Heatmap of per-dimension Pearson correlations across probes.

    Parameters
    ----------
    results : dict
        Mapping from probe_name -> metrics dict (needs "correlations" key).
    action_dim : int
        Number of action dimensions.
    output_path : str or Path
        Path to save the figure.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    dim_labels : list of str, optional
        Labels for action dimensions. Default: ["dim_0", "dim_1", ...].

    Returns
    -------
    Path to saved figure.
    """
    plt = _get_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dim_labels is None:
        dim_labels = [f"dim_{i}" for i in range(action_dim)]

    names = list(results.keys())
    corr_matrix = np.zeros((len(names), action_dim))

    for i, name in enumerate(names):
        corrs = results[name].get("correlations", [0.0] * action_dim)
        for j, c in enumerate(corrs):
            if j < action_dim:
                corr_matrix[i, j] = c

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(action_dim))
    ax.set_xticklabels(dim_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(title)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")

    # Add text annotations
    for i in range(len(names)):
        for j in range(action_dim):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", output_path)
    return output_path


def plot_per_step_mse(
    results: Dict[str, Dict[str, float]],
    output_path: str | Path = "per_step_mse.png",
    title: str = "MSE per Action Chunk Position",
    figsize: tuple = (12, 6),
) -> Path:
    """Line plot of MSE at each action chunk position.

    Parameters
    ----------
    results : dict
        Mapping from probe_name -> metrics dict (needs "per_step_mse" key).
    output_path : str or Path
        Path to save the figure.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Path to saved figure.
    """
    plt = _get_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    for name, metrics in results.items():
        per_step = metrics.get("per_step_mse", [])
        if per_step:
            ax.plot(range(len(per_step)), per_step, marker="o", markersize=3, label=name)

    ax.set_xlabel("Action Chunk Position")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-step MSE plot to %s", output_path)
    return output_path


def plot_layer_wise_analysis(
    layer_results: Dict[int, Dict[str, float]],
    output_path: str | Path = "layer_wise_mse.png",
    title: str = "Probe MSE vs. Joint Attention Layer",
    figsize: tuple = (12, 6),
) -> Path:
    """Plot MSE as a function of the number of joint attention layers applied.

    Parameters
    ----------
    layer_results : dict
        Mapping from num_layers -> metrics dict (needs "mse" key).
    output_path : str or Path
        Path to save the figure.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Path to saved figure.
    """
    plt = _get_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = sorted(layer_results.keys())
    mses = [layer_results[l]["mse"] for l in layers]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers, mses, marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Number of Joint Attention Layers")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Annotate minimum
    min_idx = np.argmin(mses)
    ax.annotate(
        f"Best: {mses[min_idx]:.4f}",
        xy=(layers[min_idx], mses[min_idx]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved layer-wise analysis to %s", output_path)
    return output_path


def generate_all_plots(
    results: Dict[str, Dict[str, float]],
    output_dir: str | Path = "probe_plots",
    action_dim: int = 14,
    dim_labels: Optional[List[str]] = None,
) -> List[Path]:
    """Generate all standard plots for a probe experiment.

    Parameters
    ----------
    results : dict
        Mapping from probe_name -> metrics dict.
    output_dir : str or Path
        Directory for output plots.
    action_dim : int
        Number of action dimensions.
    dim_labels : list of str, optional
        Labels for action dimensions.

    Returns
    -------
    list of Path : paths to generated plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    generated.append(plot_probe_comparison(
        results, output_dir / "probe_comparison.png"
    ))

    generated.append(plot_correlation_heatmap(
        results, action_dim, output_dir / "correlation_heatmap.png",
        dim_labels=dim_labels,
    ))

    # Only plot per-step if any probe has per_step_mse
    has_per_step = any("per_step_mse" in m for m in results.values())
    if has_per_step:
        generated.append(plot_per_step_mse(
            results, output_dir / "per_step_mse.png"
        ))

    return generated
