"""
Runtime monkey-patches for the Motus model to support history KV concatenation
and first-frame memory for probing experiments.

Usage:
    from model_patches import apply_patches, extract_first_frame_kv, patched_inference_step

    apply_patches()  # idempotent, safe to call multiple times

    # Extract first-frame K/V cache
    history_kv = extract_first_frame_kv(model, first_frame_latent, num_layers=30)

    # Run patched inference
    frames, actions = patched_inference_step(
        model, first_frame, state, num_inference_steps,
        language_embeddings, vlm_inputs,
        history_kv=history_kv,
    )

All functions run under torch.no_grad() -- inference only.
No existing model files are modified permanently; patches are applied at runtime.
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------------
# Import helpers from the project
# ---------------------------------------------------------------------------
# The wan package lives under bak/ which may not be on sys.path.  We follow
# the same convention used by inference/robotwin/Motus/models/motus.py.
from pathlib import Path

_BAK_ROOT = str(
    (Path(__file__).resolve().parents[2] / "inference" / "robotwin" / "Motus" / "bak").resolve()
)
if _BAK_ROOT not in sys.path:
    sys.path.insert(0, _BAK_ROOT)

from wan.modules.model import WanSelfAttention, rope_apply  # noqa: E402
from wan.modules.attention import flash_attention  # noqa: E402

# Where VideoModule lives -- we import it lazily so the caller can import this
# module before the model is fully constructed.

# ---------------------------------------------------------------------------
# Guard flag: make patches idempotent
# ---------------------------------------------------------------------------
_PATCHES_APPLIED = False


# ============================================================================
# 1. Patched WanSelfAttention.forward
# ============================================================================

def patched_wan_self_attn_forward(
    self: WanSelfAttention,
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    action_q: torch.Tensor | None = None,
    action_k: torch.Tensor | None = None,
    action_v: torch.Tensor | None = None,
    und_q: torch.Tensor | None = None,
    und_k: torch.Tensor | None = None,
    und_v: torch.Tensor | None = None,
    history_kv: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple:
    """Patched forward that optionally prepends history K/V.

    When *history_kv* is provided it is a tuple ``(hist_k, hist_v)`` where
    each tensor has shape ``[B, L_hist, N, D]`` (batch, history_len,
    num_heads, head_dim).  History tokens are **prepended** before the
    action/understanding tokens in the concatenated K/V sequence.

    After flash-attention, the first ``L_hist`` output positions (which
    correspond to the history context) are discarded -- only video, action,
    and understanding outputs are returned.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # -- Q/K/V projection for the video expert --
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    # ------------------------------------------------------------------
    # Trimodal MoT branch (video + action + understanding)
    # ------------------------------------------------------------------
    if action_q is not None or und_q is not None:
        L_x = q.size(1)

        # Apply RoPE to video tokens only
        # Truncate grid_sizes to match actual batch size B
        gs = grid_sizes[:b]  # [B, 3]
        q_video_rope = rope_apply(q, gs, freqs)
        k_video_rope = rope_apply(k, gs, freqs)

        # Build concatenation parts: [history?, video, action?, und?]
        q_parts: list[torch.Tensor] = [q_video_rope]
        k_parts: list[torch.Tensor] = [k_video_rope]
        v_parts: list[torch.Tensor] = [v]

        # --- History KV (prepended BEFORE action/und but AFTER video) ---
        L_hist = 0
        if history_kv is not None:
            hist_k, hist_v = history_kv
            L_hist = hist_k.size(1)
            # History K/V already have RoPE applied at extraction time --
            # they are NOT re-encoded here.
            # Prepend: order is [video_rope, hist, action, und]
            k_parts.insert(1, hist_k)
            v_parts.insert(1, hist_v)
            # q_parts does NOT get history queries -- history is K/V only
            # (cross-attention style from the model's perspective)

        # --- Action tokens ---
        if action_q is not None:
            q_parts.append(action_q)
            k_parts.append(action_k)
            v_parts.append(action_v)
            L_action = action_q.size(1)
        else:
            L_action = 0

        # --- Understanding tokens ---
        if und_q is not None:
            q_parts.append(und_q)
            k_parts.append(und_k)
            v_parts.append(und_v)
            L_und = und_q.size(1)
        else:
            L_und = 0

        # Concatenate all parts along sequence dimension
        q_cat = torch.cat(q_parts, dim=1)
        k_cat = torch.cat(k_parts, dim=1)
        v_cat = torch.cat(v_parts, dim=1)

        # k_lens must include history tokens so flash_attn_varlen_func doesn't
        # truncate K/V.  Output length still equals Q length (no history in Q).
        if L_hist > 0:
            seq_lens = seq_lens + L_hist

        attn_out = flash_attention(
            q=q_cat,
            k=k_cat,
            v=v_cat,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # --------------------------------------------------------------
        # Split outputs back to respective modalities
        # --------------------------------------------------------------
        # flash_attention output length = Q length = L_x + L_action + L_und
        # (history is K/V-only, so it does NOT appear in the output)

        x_out = attn_out[:, :L_x, :, :]
        outputs = [x_out]

        start_idx = L_x
        if action_q is not None:
            action_out = attn_out[:, start_idx : start_idx + L_action, :, :]
            outputs.append(action_out)
            start_idx += L_action
        else:
            outputs.append(None)

        if und_q is not None:
            und_out = attn_out[:, start_idx : start_idx + L_und, :, :]
            outputs.append(und_out)
        else:
            outputs.append(None)

        # Project WAN branch through output projection
        x_out = x_out.flatten(2)
        x_out = self.o(x_out)
        outputs[0] = x_out

        return tuple(outputs)

    # ------------------------------------------------------------------
    # Standard branch (no MoT) -- unchanged from original
    # ------------------------------------------------------------------
    gs = grid_sizes[:b]  # truncate to actual batch size
    x = flash_attention(
        q=rope_apply(q, gs, freqs),
        k=rope_apply(k, gs, freqs),
        v=v,
        k_lens=seq_lens,
        window_size=self.window_size,
    )
    x = x.flatten(2)
    x = self.o(x)
    return x


# ============================================================================
# 2. Patched VideoModule.process_joint_attention
# ============================================================================

def patched_process_joint_attention(
    self,
    video_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    video_adaln_modulation: tuple,
    action_adaln_modulation: tuple,
    layer_idx: int,
    action_block: nn.Module,
    und_tokens: torch.Tensor,
    und_block: nn.Module,
    history_kv: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patched joint attention that forwards *history_kv* to WAN self-attention.

    The only difference from the original is that *history_kv* (a per-layer
    tuple of ``[B, L_hist, N, D]`` tensors) is forwarded to the WAN
    self-attention call when provided.
    """
    wan_layer = self.video_model.wan_model.blocks[layer_idx]

    # AdaLN params (already computed by caller)
    v_mod = video_adaln_modulation
    a_mod = action_adaln_modulation

    # Pre-attn normalization with AdaLN
    norm_video = (
        wan_layer.norm1(video_tokens).float()
        * (1 + v_mod[1].squeeze(2))
        + v_mod[0].squeeze(2)
    )
    norm_action = (
        action_block.norm1(action_tokens)
        * (1 + a_mod[1].squeeze(2))
        + a_mod[0].squeeze(2)
    )

    # Dimensions
    B, L_v, C = norm_video.shape
    L_a = norm_action.shape[1]
    n = self.video_model.wan_model.num_heads
    d = C // n

    # Action Q/K/V projection into WAN head space
    a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
    a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
    a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
    a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
    a_v = a_v_h.view(B, L_a, n, d)

    # Understanding Expert Q/K/V
    norm_und = und_block.norm1(und_tokens)
    L_u = norm_und.shape[1]
    u_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)
    u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
    u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
    u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
    u_v = u_v_h.view(B, L_u, n, d)

    # Seq lens and RoPE freqs
    seq_lens = torch.full(
        (B,), L_v + L_a + L_u, dtype=torch.long, device=self.device
    )
    freqs = self.video_model.wan_model.freqs
    if freqs.device != self.device:
        freqs = freqs.to(self.device)

    # ---- Call patched self_attn (now accepts history_kv) ----
    y, action_out_h, und_out_h = wan_layer.self_attn(
        norm_video,
        seq_lens,
        self.grid_sizes,
        freqs,
        action_q=a_q,
        action_k=a_k,
        action_v=a_v,
        und_q=u_q,
        und_k=u_k,
        und_v=u_v,
        history_kv=history_kv,
    )

    # Project Understanding Expert output
    und_out = und_block.wan_und_o(und_out_h.flatten(2))

    # Project back and residual connections
    action_out = action_block.wan_action_o(action_out_h.flatten(2))
    video_tokens = video_tokens + y * v_mod[2].squeeze(2)
    action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
    und_tokens = und_tokens + und_out

    return video_tokens, action_tokens, und_tokens


# ============================================================================
# 3. apply_patches()
# ============================================================================

def apply_patches() -> None:
    """Monkey-patch WanSelfAttention.forward and VideoModule.process_joint_attention.

    Idempotent: safe to call multiple times; only the first call has effect.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    # Ensure inference/robotwin/Motus is on sys.path so models.motus is importable
    # without triggering Motus/__init__.py (which imports deploy_policy etc.)
    _motus_pkg = str(
        (Path(__file__).resolve().parents[2] / "inference" / "robotwin" / "Motus").resolve()
    )
    if _motus_pkg not in sys.path:
        sys.path.insert(0, _motus_pkg)

    # Patch WanSelfAttention.forward
    WanSelfAttention.forward = patched_wan_self_attn_forward  # type: ignore[assignment]

    # Patch VideoModule.process_joint_attention — import via models.motus (not Motus.models.motus)
    # to avoid triggering Motus/__init__.py
    from models.motus import VideoModule
    VideoModule.process_joint_attention = patched_process_joint_attention  # type: ignore[assignment]

    _PATCHES_APPLIED = True


# ============================================================================
# 4. extract_first_frame_kv
# ============================================================================

@torch.no_grad()
def extract_first_frame_kv(
    model,
    first_frame_latent: torch.Tensor,
    num_layers: int = 30,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer K/V from the first frame's clean latent.

    All 30 layers receive the **same** input video tokens (the clean
    first-frame latent projected through the patch embedding).  Each layer's
    K/V projection weights differ, so the resulting (k, v) pairs are
    layer-specific even though the input is shared.  This is an acceptable
    approximation for baseline probing experiments.

    Args:
        model: A fully-initialised ``Motus`` instance.
        first_frame_latent: Clean latent tensor ``[B, C', 1, H', W']``
            (i.e. the VAE-encoded first frame).
        num_layers: Number of WAN transformer layers (default 30).

    Returns:
        A list of length *num_layers*, each element is a tuple
        ``(k, v)`` with shapes ``[B, L, num_heads, head_dim]``.
        ``k`` has RoPE already applied.
    """
    device = model.device
    dtype = model.video_model.precision  # typically bfloat16

    # Prepare video tokens from clean latent via patch embedding
    video_tokens = model.video_module.prepare_input(first_frame_latent.to(dtype))
    # video_tokens: [B, L, 3072] (hidden_dim = 3072)

    B = video_tokens.shape[0]
    n = model.video_model.wan_model.num_heads       # 24
    wan_dim = getattr(model.video_model.wan_model.config, 'dim', 3072)
    d = wan_dim // n                                 # 128 (= 3072 / 24)
    freqs = model.video_model.wan_model.freqs
    if freqs.device != device:
        freqs = freqs.to(device)

    # Compute grid_sizes from first_frame_latent, NOT from model.grid_sizes.
    # model.grid_sizes is for the full video latent (e.g. [3,12,10] for 3 frames),
    # but first_frame_latent may have a different temporal dimension (e.g. 1 frame).
    # WAN patch embedding: temporal patch_size=2, spatial patch_size=2
    _, C_lat, T_lat, H_lat, W_lat = first_frame_latent.shape
    T_patches = (T_lat + 1) // 2  # ceil(T/2)
    H_patches = H_lat // 2
    W_patches = W_lat // 2
    grid_sizes = torch.tensor([[T_patches, H_patches, W_patches]], device=device, dtype=torch.long)

    history_kv: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_idx in range(num_layers):
        self_attn = model.video_model.wan_model.blocks[layer_idx].self_attn

        # Compute per-layer K/V from the same input tokens
        # (Q is not needed for the history cache)
        k = self_attn.norm_k(self_attn.k(video_tokens)).view(B, -1, n, d)
        v = self_attn.v(video_tokens).view(B, -1, n, d)

        # Apply RoPE to K (history K/V carry spatial RoPE)
        k_rope = rope_apply(k, grid_sizes, freqs)

        history_kv.append((k_rope, v))

    return history_kv


# ============================================================================
# 5. patched_inference_step
# ============================================================================

@torch.no_grad()
def patched_inference_step(
    model,
    first_frame: torch.Tensor,
    state: torch.Tensor,
    num_inference_steps: int,
    language_embeddings: list,
    vlm_inputs: list,
    history_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    first_frame_und_tokens: Optional[torch.Tensor] = None,
) -> tuple:
    """Run Motus inference with optional history KV and cached understanding tokens.

    This is a standalone reimplementation of ``Motus.inference_step`` that
    threads *history_kv* (a list of per-layer ``(k, v)`` tuples) through
    every call to ``process_joint_attention``, and optionally reuses
    pre-extracted understanding tokens instead of re-extracting from the VLM
    at every denoising step.

    Args:
        model: A fully-initialised ``Motus`` instance.
        first_frame: ``[B, C, H, W]`` in ``[0, 1]``.
        state: ``[B, state_dim]`` robot state.
        num_inference_steps: Number of Euler denoising steps.
        language_embeddings: Pre-encoded T5 token embeddings (list or tensor).
        vlm_inputs: VLM inputs (images / text) for understanding features.
        history_kv: Optional list of length ``num_layers``, each a
            ``(k, v)`` tuple with shapes ``[B, L_hist, N, D]``.
            Passed to each layer's joint attention.
        first_frame_und_tokens: Optional pre-extracted understanding tokens
            ``[B, num_queries * num_layers, und_dim]``.  When provided the
            VLM extraction inside the denoising loop is skipped.

    Returns:
        ``(predicted_frames, predicted_actions)`` tuple.
    """
    B = first_frame.shape[0]
    device = model.device
    dtype = model.dtype

    # Move inputs
    language_embeddings = [e.to(device).to(dtype) for e in language_embeddings]
    state = state.to(device).to(dtype)
    first_frame = first_frame.to(device).to(dtype)

    # 1. Encode condition frame and initialise latents
    first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B,C,1,H,W]
    with torch.no_grad():
        condition_frame_latent = model.video_model.encode_video(first_frame_norm.to(dtype))

    B_lat, C_lat, f_lat, H_lat, W_lat = condition_frame_latent.shape
    num_total_latent_frames = 1 + model.config.num_video_frames // 4
    video_latent = torch.randn(
        (B_lat, C_lat, num_total_latent_frames, H_lat, W_lat),
        device=device,
        dtype=dtype,
    )
    video_latent[:, :, 0:1] = condition_frame_latent

    action_shape = (B, model.config.action_chunk_size, model.config.action_dim)
    action_latent = torch.randn(action_shape, device=device, dtype=dtype)

    # 2. Understanding features and T5 context
    if first_frame_und_tokens is not None:
        und_tokens = first_frame_und_tokens.to(device).to(dtype)
    else:
        und_tokens = model.und_module.extract_und_features(vlm_inputs)

    processed_t5_context = model.video_module.preprocess_t5_embeddings(language_embeddings)

    # 3. Denoising loop
    timesteps = torch.linspace(
        1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype
    )

    for i in range(num_inference_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t
        video_t_scaled = (t * 1000).expand(B).to(dtype)
        action_t_scaled = (t * 1000).expand(B).to(dtype)

        # Prepare tokens
        video_tokens = model.video_module.prepare_input(video_latent.to(dtype))
        state_tokens = state.unsqueeze(1).to(dtype)
        registers = model.action_expert.registers.expand(B, -1, -1)
        action_tokens = model.action_expert.input_encoder(
            state_tokens, action_latent, registers
        )

        # Understanding tokens: use cached or re-extract
        if first_frame_und_tokens is None:
            und_tokens = model.und_module.extract_und_features(vlm_inputs)

        # Trimodal MoT forward
        with torch.autocast(device_type="cuda", dtype=model.video_model.precision):
            # Time embeddings
            video_head_time_emb, video_adaln_params = (
                model.video_module.get_time_embedding(
                    video_t_scaled, video_tokens.shape[1]
                )
            )
            action_head_time_emb, action_adaln_params = (
                model.action_module.get_time_embedding(
                    action_t_scaled, action_tokens.shape[1]
                )
            )

            for layer_idx in range(model.config.num_layers):
                video_adaln_modulation = model.video_module.compute_adaln_modulation(
                    video_adaln_params, layer_idx
                )
                action_adaln_modulation = model.action_module.compute_adaln_modulation(
                    action_adaln_params, layer_idx
                )

                # Get per-layer history KV if available
                layer_history_kv = None
                if history_kv is not None:
                    layer_history_kv = history_kv[layer_idx]

                video_tokens, action_tokens, und_tokens = (
                    model.video_module.process_joint_attention(
                        video_tokens,
                        action_tokens,
                        video_adaln_modulation,
                        action_adaln_modulation,
                        layer_idx,
                        model.action_expert.blocks[layer_idx],
                        und_tokens,
                        model.und_expert.blocks[layer_idx],
                        history_kv=layer_history_kv,
                    )
                )

                # WAN cross-attention
                video_tokens = model.video_module.process_cross_attention(
                    video_tokens, video_adaln_params, layer_idx, processed_t5_context
                )

                # FFNs
                video_tokens = model.video_module.process_ffn(
                    video_tokens, video_adaln_modulation, layer_idx
                )
                action_tokens = model.action_module.process_ffn(
                    action_tokens, action_adaln_modulation, layer_idx
                )
                und_tokens = model.und_module.process_ffn(und_tokens, layer_idx)

            # Heads (velocities)
            video_velocity = model.video_module.apply_output_head(
                video_tokens, video_head_time_emb
            )
            action_pred_full = model.action_expert.decoder(
                action_tokens, action_head_time_emb
            )
            action_velocity = action_pred_full[
                :, 1 : -model.action_expert.config.num_registers, :
            ]

            # Euler integration
            video_latent = video_latent + video_velocity * dt
            action_latent = action_latent + action_velocity * dt

            # Teacher forcing: keep first frame clean
            video_latent[:, :, 0:1] = condition_frame_latent

    # 4. Decode outputs
    with torch.no_grad():
        decoded_frames = model.video_model.decode_video(video_latent)
        predicted_frames = decoded_frames[:, :, 1:]  # skip condition frame
        predicted_frames = (predicted_frames + 1.0) / 2.0
        predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

    predicted_actions = action_latent.float()
    return predicted_frames, predicted_actions


# ============================================================================
# 6. patched_inference_step_with_memory
# ============================================================================

@torch.no_grad()
def patched_inference_step_with_memory(
    model,
    first_frame: torch.Tensor,
    state: torch.Tensor,
    num_inference_steps: int,
    language_embeddings: list,
    vlm_inputs: list,
    history_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    first_frame_und_tokens: Optional[torch.Tensor] = None,
    memory_vlm_features: Optional[torch.Tensor] = None,
    memory_motion_features: Optional[torch.Tensor] = None,
    memory_injector: Optional[nn.Module] = None,
    memory_retriever: Optional[nn.Module] = None,
    top_k: int = 5,
    injection_mode: str = "concat",
) -> tuple:
    """Run Motus inference with temporal memory bank injection.

    This extends ``patched_inference_step`` with memory bank support.
    Memory is injected into Understanding Expert tokens at each denoising step.

    Memory injection modes:
      - "concat": concatenate memory tokens with und_tokens before MoT layers
      - "cross_attn": use cross-attention (memory_injector) to enhance und_tokens
      - "mean_add": average memory features and add to und_tokens

    Args:
        model: A fully-initialised ``Motus`` instance.
        first_frame: ``[B, C, H, W]`` in ``[0, 1]``.
        state: ``[B, state_dim]`` robot state.
        num_inference_steps: Number of Euler denoising steps.
        language_embeddings: Pre-encoded T5 token embeddings.
        vlm_inputs: VLM inputs for understanding features.
        history_kv: Optional per-layer K/V cache.
        first_frame_und_tokens: Optional pre-extracted understanding tokens.
        memory_vlm_features: ``[B, K, seq_len, und_dim]`` retrieved memory VLM features.
            If None, no memory injection is performed.
        memory_motion_features: ``[B, K, motion_dim]`` retrieved memory motion features (unused in injection, kept for analysis).
        memory_injector: MemoryInjector module (for cross_attn mode).
        memory_retriever: MemoryRetriever module (for dynamic retrieval).
        top_k: Number of memory entries to retrieve (if using retriever).
        injection_mode: How to inject memory into Understanding Expert.

    Returns:
        ``(predicted_frames, predicted_actions)`` tuple.
    """
    B = first_frame.shape[0]
    device = model.device
    dtype = model.dtype

    # Move inputs
    language_embeddings = [e.to(device).to(dtype) for e in language_embeddings]
    state = state.to(device).to(dtype)
    first_frame = first_frame.to(device).to(dtype)

    # 1. Encode condition frame and initialise latents
    first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B,C,1,H,W]
    with torch.no_grad():
        condition_frame_latent = model.video_model.encode_video(first_frame_norm.to(dtype))

    B_lat, C_lat, f_lat, H_lat, W_lat = condition_frame_latent.shape
    num_total_latent_frames = 1 + model.config.num_video_frames // 4
    video_latent = torch.randn(
        (B_lat, C_lat, num_total_latent_frames, H_lat, W_lat),
        device=device,
        dtype=dtype,
    )
    video_latent[:, :, 0:1] = condition_frame_latent

    action_shape = (B, model.config.action_chunk_size, model.config.action_dim)
    action_latent = torch.randn(action_shape, device=device, dtype=dtype)

    # 2. Understanding features and T5 context
    if first_frame_und_tokens is not None:
        und_tokens = first_frame_und_tokens.to(device).to(dtype)
    else:
        und_tokens = model.und_module.extract_und_features(vlm_inputs)

    # Store original und_tokens sequence length for concat mode trimming
    und_seq_len = und_tokens.shape[1]

    processed_t5_context = model.video_module.preprocess_t5_embeddings(language_embeddings)

    # 3. Memory preparation
    use_memory = memory_vlm_features is not None and memory_vlm_features.shape[1] > 0
    if use_memory:
        memory_vlm_features = memory_vlm_features.to(device).to(dtype)

    # 4. Denoising loop
    timesteps = torch.linspace(
        1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype
    )

    for i in range(num_inference_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t
        video_t_scaled = (t * 1000).expand(B).to(dtype)
        action_t_scaled = (t * 1000).expand(B).to(dtype)

        # Prepare tokens
        video_tokens = model.video_module.prepare_input(video_latent.to(dtype))
        state_tokens = state.unsqueeze(1).to(dtype)
        registers = model.action_expert.registers.expand(B, -1, -1)
        action_tokens = model.action_expert.input_encoder(
            state_tokens, action_latent, registers
        )

        # Understanding tokens: use cached or re-extract
        if first_frame_und_tokens is None:
            und_tokens = model.und_module.extract_und_features(vlm_inputs)

        # ---- Memory Injection into Understanding tokens ----
        if use_memory:
            if injection_mode == "concat":
                # Concatenate memory tokens: [B, K*seq_len + und_seq_len, und_dim]
                K = memory_vlm_features.shape[1]
                memory_flat = memory_vlm_features.reshape(B, -1, und_tokens.shape[-1])
                und_tokens = torch.cat([memory_flat, und_tokens], dim=1)

            elif injection_mode == "cross_attn" and memory_injector is not None:
                # Cross-attention injection
                und_tokens = memory_injector(und_tokens, memory_vlm_features)

            elif injection_mode == "mean_add":
                # Average memory and add to und_tokens
                memory_mean = memory_vlm_features.mean(dim=1)  # [B, seq_len, und_dim]
                # Match sequence length (truncate or pad)
                if memory_mean.shape[1] >= und_tokens.shape[1]:
                    memory_mean = memory_mean[:, :und_tokens.shape[1]]
                else:
                    # Pad memory_mean to match und_tokens length
                    pad_len = und_tokens.shape[1] - memory_mean.shape[1]
                    memory_mean = F.pad(memory_mean, (0, 0, 0, pad_len))
                und_tokens = und_tokens + memory_mean

        # Trimodal MoT forward
        with torch.autocast(device_type="cuda", dtype=model.video_model.precision):
            # Time embeddings
            video_head_time_emb, video_adaln_params = (
                model.video_module.get_time_embedding(
                    video_t_scaled, video_tokens.shape[1]
                )
            )
            action_head_time_emb, action_adaln_params = (
                model.action_module.get_time_embedding(
                    action_t_scaled, action_tokens.shape[1]
                )
            )

            for layer_idx in range(model.config.num_layers):
                video_adaln_modulation = model.video_module.compute_adaln_modulation(
                    video_adaln_params, layer_idx
                )
                action_adaln_modulation = model.action_module.compute_adaln_modulation(
                    action_adaln_params, layer_idx
                )

                # Get per-layer history KV if available
                layer_history_kv = None
                if history_kv is not None:
                    layer_history_kv = history_kv[layer_idx]

                video_tokens, action_tokens, und_tokens = (
                    model.video_module.process_joint_attention(
                        video_tokens,
                        action_tokens,
                        video_adaln_modulation,
                        action_adaln_modulation,
                        layer_idx,
                        model.action_expert.blocks[layer_idx],
                        und_tokens,
                        model.und_expert.blocks[layer_idx],
                        history_kv=layer_history_kv,
                    )
                )

                # WAN cross-attention
                video_tokens = model.video_module.process_cross_attention(
                    video_tokens, video_adaln_params, layer_idx, processed_t5_context
                )

                # FFNs
                video_tokens = model.video_module.process_ffn(
                    video_tokens, video_adaln_modulation, layer_idx
                )
                action_tokens = model.action_module.process_ffn(
                    action_tokens, action_adaln_modulation, layer_idx
                )
                und_tokens = model.und_module.process_ffn(und_tokens, layer_idx)

            # Heads (velocities)
            video_velocity = model.video_module.apply_output_head(
                video_tokens, video_head_time_emb
            )
            action_pred_full = model.action_expert.decoder(
                action_tokens, action_head_time_emb
            )
            action_velocity = action_pred_full[
                :, 1 : -model.action_expert.config.num_registers, :
            ]

            # Euler integration
            video_latent = video_latent + video_velocity * dt
            action_latent = action_latent + action_velocity * dt

            # Teacher forcing: keep first frame clean
            video_latent[:, :, 0:1] = condition_frame_latent

            # If concat mode, trim und_tokens back to original size for next step
            if use_memory and injection_mode == "concat":
                und_tokens = und_tokens[:, -und_seq_len:]

    # 5. Decode outputs
    with torch.no_grad():
        decoded_frames = model.video_model.decode_video(video_latent)
        predicted_frames = decoded_frames[:, :, 1:]  # skip condition frame
        predicted_frames = (predicted_frames + 1.0) / 2.0
        predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

    predicted_actions = action_latent.float()
    return predicted_frames, predicted_actions
