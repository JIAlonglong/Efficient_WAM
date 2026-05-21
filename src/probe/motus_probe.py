"""Probe architectures for Motus token analysis.

Three probe types adapted from semantic-wm to work with Motus (B, N, D) tokens:

  - MotusLinearProbe:       Mean pool -> Linear -> action prediction
  - MotusTemporalProbe:     CLS + 1-layer Transformer -> action prediction
  - MotusSpatiotemporalProbe: T x S tokens + 2-layer Transformer -> action prediction

All probes take (B, N, D) or (B, T, N, D) input and output
(B, chunk_size, action_dim) action predictions.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MotusLinearProbe(nn.Module):
    """Linear probe on mean-pooled Motus tokens.

    Simplest probe: mean-pool over the token sequence dimension, then
    a single linear layer to predict action sequences.

    Input:  (B, N, D) or (B, T, N, D) token features
    Output: (B, chunk_size, action_dim) predicted actions

    Parameters
    ----------
    feature_dim : int
        Token feature dimension (3072 for video, 512 for und).
    action_dim : int
        Action space dimension (default 14).
    chunk_size : int
        Number of action steps to predict (default 16).
    pool_mode : str
        Pooling strategy: "mean" or "cls" (default "mean").
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = 14,
        chunk_size: int = 16,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.head = nn.Linear(feature_dim, action_dim * chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D) or (B, T, N, D)

        Returns
        -------
        (B, chunk_size, action_dim)
        """
        if x.dim() == 4:
            # Multi-frame: (B, T, N, D) -> mean over spatial patches N -> (B, T, D)
            # Then mean over time T -> (B, D)
            x = x.mean(dim=2)  # (B, T, D)
        # Pool over sequence dimension
        x = x.mean(dim=1)  # (B, D)
        return self.head(x).view(-1, self.chunk_size, self.action_dim)


class MotusTemporalProbe(nn.Module):
    """Temporal probe with CLS token and Transformer encoder.

    Captures temporal dependencies across multiple frames using a
    CLS token + 1-layer Transformer, then predicts actions.

    Input:  (B, T, N, D) multi-frame token features
    Output: (B, chunk_size, action_dim) predicted actions

    Parameters
    ----------
    feature_dim : int
        Token feature dimension.
    n_frames : int
        Number of temporal frames.
    action_dim : int
        Action space dimension (default 14).
    chunk_size : int
        Number of action steps to predict (default 16).
    n_heads : int
        Number of attention heads (default 8).
    """

    def __init__(
        self,
        feature_dim: int,
        n_frames: int,
        action_dim: int = 14,
        chunk_size: int = 16,
        n_heads: int = 8,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_frames = n_frames

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(feature_dim, action_dim * chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D) single-frame or (B, T, N, D) multi-frame token features

        Returns
        -------
        (B, chunk_size, action_dim)
        """
        if x.dim() == 3:
            # Single frame: (B, N, D) -> treat as 1 frame -> (B, 1, D)
            x = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        elif x.dim() == 4:
            # Multi-frame: (B, T, N, D) -> mean pool over spatial patches -> (B, T, D)
            x = x.mean(dim=2)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, D)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.transformer(x)
        return self.head(x[:, 0]).view(-1, self.chunk_size, self.action_dim)


class MotusSpatiotemporalProbe(nn.Module):
    """Spatiotemporal probe: each spatial patch is an independent token.

    Flattens T x N tokens into a single sequence, adds temporal + spatial
    position embeddings, then uses a 2-layer Transformer with a CLS token
    to predict actions.

    Input:  (B, T, N, D) multi-frame token features
    Output: (B, chunk_size, action_dim) predicted actions

    Parameters
    ----------
    feature_dim : int
        Token feature dimension.
    n_frames : int
        Number of temporal frames.
    n_patches : int
        Number of spatial patches per frame (after any pooling).
    action_dim : int
        Action space dimension (default 14).
    chunk_size : int
        Number of action steps to predict (default 16).
    n_heads : int
        Number of attention heads (default 8).
    n_layers : int
        Number of Transformer layers (default 2).
    """

    def __init__(
        self,
        feature_dim: int,
        n_frames: int,
        n_patches: int = 64,
        action_dim: int = 14,
        chunk_size: int = 16,
        n_heads: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_frames = n_frames

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.temporal_embed = nn.Parameter(torch.randn(1, n_frames, 1, feature_dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, n_patches, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(feature_dim, action_dim * chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D) single-frame or (B, T, N, D) multi-frame token features

        Returns
        -------
        (B, chunk_size, action_dim)
        """
        if x.dim() == 3:
            # Single frame: (B, N, D) -> (B, 1, N, D)
            x = x.unsqueeze(1)

        B, T, N, D = x.shape

        # Adaptive pooling if N doesn't match expected n_patches
        if N != self.n_patches:
            # Reshape for adaptive pooling: (B*T, D, N) -> pool -> (B*T, D, n_patches)
            x_flat = x.reshape(B * T, N, D).transpose(1, 2)  # (B*T, D, N)
            x_flat = F.adaptive_avg_pool1d(x_flat, self.n_patches)  # (B*T, D, n_patches)
            x = x_flat.transpose(1, 2).reshape(B, T, self.n_patches, D)

        # Add position embeddings (broadcast over batch)
        x = x + self.temporal_embed[:, :T] + self.spatial_embed

        # Flatten temporal and spatial: (B, T*N, D)
        x = x.reshape(B, T * self.n_patches, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T*N+1, D)

        x = self.transformer(x)
        return self.head(x[:, 0]).view(-1, self.chunk_size, self.action_dim)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_probe(
    probe_type: str,
    feature_dim: int,
    n_frames: int = 1,
    action_dim: int = 14,
    chunk_size: int = 16,
    n_patches: int = 64,
    n_heads: int = 8,
    n_layers: int = 1,
    pool_mode: str = "mean",
) -> nn.Module:
    """Create a probe model by type name.

    Parameters
    ----------
    probe_type : str
        One of "linear", "temporal", "spatiotemporal".
    feature_dim : int
        Input feature dimension (3072 for video, 512 for und).
    n_frames : int
        Number of temporal frames (used by temporal and spatiotemporal).
    action_dim : int
        Action space dimension.
    chunk_size : int
        Action chunk size.
    n_patches : int
        Number of spatial patches (used by spatiotemporal).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer layers (used by spatiotemporal).
    pool_mode : str
        Pooling mode for linear probe.

    Returns
    -------
    nn.Module : probe model.

    Raises
    ------
    ValueError
        If probe_type is not recognized.
    """
    if probe_type == "linear":
        return MotusLinearProbe(
            feature_dim=feature_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            pool_mode=pool_mode,
        )
    elif probe_type == "temporal":
        return MotusTemporalProbe(
            feature_dim=feature_dim,
            n_frames=n_frames,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_heads=n_heads,
        )
    elif probe_type == "spatiotemporal":
        return MotusSpatiotemporalProbe(
            feature_dim=feature_dim,
            n_frames=n_frames,
            n_patches=n_patches,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_heads=n_heads,
            n_layers=n_layers,
        )
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")


def get_probe_param_count(probe: nn.Module) -> int:
    """Count trainable parameters in a probe."""
    return sum(p.numel() for p in probe.parameters() if p.requires_grad)
