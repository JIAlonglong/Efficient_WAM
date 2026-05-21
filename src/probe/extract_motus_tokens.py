"""Token extraction from frozen Motus model.

Extracts 4 types of tokens from the Motus model:
  - video_before: Video tokens before Joint Attention [B, N, 3072]
  - video_after:  Video tokens after Joint Attention  [B, N, 3072]
  - und_before:   Understanding tokens before Joint Attention [B, L, 512]
  - und_after:    Understanding tokens after Joint Attention  [B, L, 512]

All extraction is done with `@torch.no_grad()` on a fully frozen model.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default token dimensions from Motus architecture
VIDEO_TOKEN_DIM = 3072  # WAN hidden dim
UND_TOKEN_DIM = 512     # Understanding Expert hidden dim
ACTION_TOKEN_DIM = 1024 # Action Expert hidden dim


@torch.no_grad()
def extract_motus_tokens(
    motus_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    extract_before: bool = True,
    extract_after: bool = True,
    num_layers: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Extract frozen tokens from Motus model.

    Runs a forward pass through the Motus model to extract video and
    understanding tokens before and/or after the joint attention layers.

    Parameters
    ----------
    motus_model : nn.Module
        The Motus model (should be in eval mode with frozen parameters).
    batch : dict
        Batch from the Motus dataloader, expected keys:
          - "video_frames": [B, num_frames, C, H, W] target video frames
          - "first_frame": [B, C, H, W] first frame for conditioning
          - "vlm_inputs": VLM input dict or list of dicts
          - "language_embeddings": pre-encoded T5 embeddings
          - "actions": [B, chunk_size, action_dim] action sequences
          - "state": [B, state_dim] robot state
    device : torch.device
        Device to run extraction on.
    extract_before : bool
        Whether to extract tokens before joint attention.
    extract_after : bool
        Whether to extract tokens after joint attention.
    num_layers : int, optional
        Number of joint attention layers to process. If None, uses all
        layers from the model config.

    Returns
    -------
    dict with keys (depending on extract_before/extract_after):
        "video_before": [B, N, 3072] video tokens before joint attention
        "video_after":  [B, N, 3072] video tokens after joint attention
        "und_before":   [B, L, 512]  understanding tokens before joint attention
        "und_after":    [B, L, 512]  understanding tokens after joint attention
    """
    motus_model.eval()

    config = motus_model.config
    if num_layers is None:
        num_layers = config.num_layers

    # ── Prepare video tokens ────────────────────────────────────────────
    video_frames = batch["video_frames"].to(device, dtype=motus_model.dtype)
    first_frame = batch["first_frame"].to(device, dtype=motus_model.dtype)

    # Normalize to [-1, 1] and encode via VAE
    first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
    video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

    clean_latent = motus_model.video_model.encode_video(full_video.to(motus_model.dtype))

    # Use clean latent directly (no noise for probing)
    video_tokens_before = motus_model.video_module.prepare_input(clean_latent.to(motus_model.dtype))

    # ── Prepare understanding tokens ────────────────────────────────────
    vlm_inputs = batch["vlm_inputs"]
    if isinstance(vlm_inputs, dict):
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in vlm_inputs.items()}
    elif isinstance(vlm_inputs, list):
        vlm_inputs = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in item.items()}
            for item in vlm_inputs
        ]

    und_tokens_before = motus_model.und_module.extract_und_features(vlm_inputs)

    result = {}
    if extract_before:
        result["video_before"] = video_tokens_before.cpu()
        result["und_before"] = und_tokens_before.cpu()

    # ── Prepare action tokens (needed for joint attention) ──────────────
    actions = batch["actions"].to(device, dtype=motus_model.dtype)
    B = actions.shape[0]

    # Encode actions without noise (clean actions for probing)
    if motus_model.action_expert.config.num_registers > 0 and motus_model.action_expert.registers is not None:
        registers = motus_model.action_expert.registers.expand(B, -1, -1)
    else:
        registers = None

    if config.training_mode == 'pretrain':
        action_tokens = motus_model.action_expert.input_encoder(None, actions, registers)
    else:
        state = batch["state"].to(device, dtype=motus_model.dtype)
        state_tokens = state.unsqueeze(1)
        action_tokens = motus_model.action_expert.input_encoder(state_tokens, actions, registers)

    # ── Run joint attention layers ──────────────────────────────────────
    if extract_after:
        # Clone tokens for processing through joint attention
        video_tokens = video_tokens_before.clone()
        und_tokens = und_tokens_before.clone()

        # Get time embeddings (use t=0 for clean tokens)
        t_zero = torch.zeros(B, device=device, dtype=motus_model.dtype)
        video_head_time_emb, video_adaln_params = motus_model.video_module.get_time_embedding(
            t_zero, video_tokens.shape[1]
        )
        action_head_time_emb, action_adaln_params = motus_model.action_module.get_time_embedding(
            t_zero, action_tokens.shape[1]
        )

        # Pre-process T5 context for cross-attention
        language_embeddings = batch.get("language_embeddings")
        if language_embeddings is not None:
            if isinstance(language_embeddings, list):
                language_embeddings = [emb.to(device, dtype=motus_model.dtype) for emb in language_embeddings]
            processed_t5_context = motus_model.video_module.preprocess_t5_embeddings(language_embeddings)
        else:
            # Create dummy T5 context if not provided
            seq_len = video_tokens.shape[1]
            processed_t5_context = torch.zeros(
                B, 512, VIDEO_TOKEN_DIM, device=device, dtype=motus_model.dtype
            )

        # Process through joint attention layers
        with torch.autocast(device_type="cuda", dtype=motus_model.video_model.precision):
            for layer_idx in range(num_layers):
                video_adaln_modulation = motus_model.video_module.compute_adaln_modulation(
                    video_adaln_params, layer_idx
                )
                action_adaln_modulation = motus_model.action_module.compute_adaln_modulation(
                    action_adaln_params, layer_idx
                )

                # Trimodal joint attention
                video_tokens, action_tokens, und_tokens = motus_model.video_module.process_joint_attention(
                    video_tokens, action_tokens,
                    video_adaln_modulation, action_adaln_modulation,
                    layer_idx,
                    motus_model.action_expert.blocks[layer_idx],
                    und_tokens,
                    motus_model.und_expert.blocks[layer_idx],
                )

                # WAN cross-attention with T5
                video_tokens = motus_model.video_module.process_cross_attention(
                    video_tokens, video_adaln_params, layer_idx, processed_t5_context
                )

                # FFNs
                video_tokens = motus_model.video_module.process_ffn(
                    video_tokens, video_adaln_modulation, layer_idx
                )
                action_tokens = motus_model.action_module.process_ffn(
                    action_tokens, action_adaln_modulation, layer_idx
                )
                und_tokens = motus_model.und_module.process_ffn(und_tokens, layer_idx)

        result["video_after"] = video_tokens.cpu()
        result["und_after"] = und_tokens.cpu()

    return result


@torch.no_grad()
def extract_motus_tokens_from_batch(
    motus_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    token_types: Optional[list] = None,
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper to extract specific token types.

    Parameters
    ----------
    motus_model : nn.Module
        Frozen Motus model.
    batch : dict
        Batch from dataloader.
    device : torch.device
        Device for computation.
    token_types : list of str, optional
        Which tokens to extract. Options: "video_before", "video_after",
        "und_before", "und_after". If None, extracts all four.

    Returns
    -------
    dict with requested token tensors.
    """
    if token_types is None:
        token_types = ["video_before", "video_after", "und_before", "und_after"]

    extract_before = any(t.endswith("_before") for t in token_types)
    extract_after = any(t.endswith("_after") for t in token_types)

    all_tokens = extract_motus_tokens(
        motus_model, batch, device,
        extract_before=extract_before,
        extract_after=extract_after,
    )

    return {k: v for k, v in all_tokens.items() if k in token_types}


def get_token_dim(token_type: str) -> int:
    """Get the feature dimension for a token type.

    Parameters
    ----------
    token_type : str
        One of "video_before", "video_after", "und_before", "und_after",
        or just "video" / "und".

    Returns
    -------
    int : feature dimension (3072 for video, 512 for und).
    """
    if "video" in token_type:
        return VIDEO_TOKEN_DIM
    elif "und" in token_type:
        return UND_TOKEN_DIM
    else:
        raise ValueError(f"Unknown token type: {token_type}")
