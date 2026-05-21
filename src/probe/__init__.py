"""Motus Probe: Experiment 1 - Understanding Tokens semantic quality verification.

This package implements probing experiments to evaluate the semantic quality
of Understanding Tokens by training probes on frozen Motus encoder features
to predict action sequences.

Modules:
    extract_motus_tokens: Token extraction from frozen Motus model
    motus_probe: Probe architectures (Linear, Temporal, Spatiotemporal)
    motus_probe_dataset: Dataset with disk caching for extracted tokens
    train_action_probe: Training loop for action prediction probes
    evaluate_probe: Evaluation metrics (MSE, Pearson correlation)
    visualize_probe: Visualization utilities for probe results
"""

from .extract_motus_tokens import extract_motus_tokens, extract_motus_tokens_from_batch
from .motus_probe import (
    MotusLinearProbe,
    MotusTemporalProbe,
    MotusSpatiotemporalProbe,
    create_probe,
)
from .motus_probe_dataset import MotusProbeDataset
from .train_action_probe import train_action_probe, train_single_probe
from .evaluate_probe import evaluate_probe, evaluate_probe_mse_correlation
from .visualize_probe import (
    plot_probe_comparison,
    plot_correlation_heatmap,
    plot_layer_wise_analysis,
)

__all__ = [
    # Token extraction
    "extract_motus_tokens",
    "extract_motus_tokens_from_batch",
    # Probe architectures
    "MotusLinearProbe",
    "MotusTemporalProbe",
    "MotusSpatiotemporalProbe",
    "create_probe",
    # Dataset
    "MotusProbeDataset",
    # Training
    "train_action_probe",
    "train_single_probe",
    # Evaluation
    "evaluate_probe",
    "evaluate_probe_mse_correlation",
    # Visualization
    "plot_probe_comparison",
    "plot_correlation_heatmap",
    "plot_layer_wise_analysis",
]
