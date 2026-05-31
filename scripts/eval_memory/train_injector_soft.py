#!/usr/bin/env python3
"""
Train ActionMemoryInjector + RecencyWeightedRetriever jointly (Experiment 3.2b)

This extends train_injector.py by replacing the hard top-k retrieval with
RecencyWeightedRetriever (soft weighting with learnable decay).

Trainable components:
  - ActionMemoryInjector: cross-attention on action tokens
  - RecencyWeightedRetriever.log_decay: learnable time decay parameter

Usage:
    # Single GPU (smoke test)
    python train_injector_soft.py \
        --config configs/robotwin.yaml \
        --checkpoint pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt \
        --output_dir ./results \
        --num_epochs 1 --lr 1e-4 \
        --num_train_batches 3 --num_eval_batches 2 \
        --num_inference_steps 2 --bank_size 5

    # Multi-GPU (full training)
    accelerate launch --num_processes 4 \
        train_injector_soft.py \
        --config configs/robotwin.yaml \
        --checkpoint pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt \
        --output_dir ./results \
        --num_epochs 10 --lr 1e-4 \
        --num_train_batches 200 --num_eval_batches 50 \
        --num_inference_steps 10 --bank_size 5
"""

import os
import sys
import gc
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

_inference_root = _project_root / "inference" / "robotwin"
sys.path.insert(0, str(_inference_root))

# ---------------------------------------------------------------------------
# Monkey-patch lerobot (same as train_injector.py)
# ---------------------------------------------------------------------------
import lerobot.datasets.utils as _lerobot_utils
import lerobot.datasets.lerobot_dataset as _lerobot_ds
import pandas as _pd

def _patched_check(repo_id, version_to_check, current_version, enforce_breaking_major=True):
    return

_lerobot_utils.check_version_compatibility = _patched_check
try:
    import lerobot.datasets.backward_compatibility as _bc
    if hasattr(_bc, "check_version_compatibility"):
        _bc.check_version_compatibility = _patched_check
except ImportError:
    pass

def _patched_get_safe_version(repo_id, revision=None):
    return _lerobot_ds.CODEBASE_VERSION

_lerobot_utils.get_safe_version = _patched_get_safe_version
_lerobot_ds.get_safe_version = _patched_get_safe_version

_original_load_tasks = _lerobot_utils.load_tasks
def _patched_load_tasks(root, episodes=None):
    parquet_path = root / "meta" / "tasks.parquet"
    jsonl_path = root / "meta" / "tasks.jsonl"
    if parquet_path.exists():
        return _original_load_tasks(root, episodes)
    elif jsonl_path.exists():
        import json as _json
        tasks = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(_json.loads(line))
        return _pd.DataFrame(tasks)
    else:
        return _pd.DataFrame()

_lerobot_utils.load_tasks = _patched_load_tasks

_original_load_episodes = _lerobot_utils.load_episodes
def _patched_load_episodes(root):
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
        from datasets import Dataset as _HFDataset
        return _HFDataset.from_list(records)
    else:
        from datasets import Dataset as _HFDataset
        return _HFDataset.from_list([])

_lerobot_utils.load_episodes = _patched_load_episodes

_original_get_video_file_path = _lerobot_ds.LeRobotDatasetMetadata.get_video_file_path
def _patched_get_video_file_path(self, ep_index, vid_key):
    info = getattr(self, "info", None)
    chunks_size = info.get("chunks_size", 1000) if isinstance(info, dict) else 1000
    chunk_idx = ep_index // chunks_size
    file_idx = ep_index
    return Path(f"videos/chunk-{chunk_idx:03d}/{vid_key}/episode_{file_idx:06d}.mp4")

_lerobot_ds.LeRobotDatasetMetadata.get_video_file_path = _patched_get_video_file_path

_original_load_info = _lerobot_utils.load_info
def _patched_load_info(root):
    info = _original_load_info(root)
    if "video_path" in info:
        info["video_path"] = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"
    return info

_lerobot_utils.load_info = _patched_load_info

# --- End monkey-patch ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Injector + RecencyWeightedRetriever jointly")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_batches", type=int, default=50)
    parser.add_argument("--num_eval_batches", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=3,
                        help="Denoising steps during training (fewer = faster)")
    parser.add_argument("--bank_size", type=int, default=5)
    parser.add_argument("--init_decay", type=float, default=1.0,
                        help="Initial decay rate for RecencyWeightedRetriever")
    parser.add_argument("--eval_every_steps", type=int, default=100,
                        help="Evaluate every N training steps")
    parser.add_argument("--max_checkpoints", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model & data loading (same as train_injector.py)
# ---------------------------------------------------------------------------

def load_model_and_data(config_path: str, checkpoint_path: str, device: str, val: bool = False):
    import yaml
    from omegaconf import OmegaConf

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    config.model.inference.num_inference_timesteps = 50

    # Override dataset_dir if configured path doesn't exist
    if not os.path.exists(config.dataset.dataset_dir):
        alt_path = "/kpfs-intern/jialongliu/projects/Motus/data/robotwin2/robotwin_dataset_v2"
        if os.path.exists(alt_path):
            logger.warning(f"Dataset dir not found: {config.dataset.dataset_dir}, using {alt_path}")
            config.dataset.dataset_dir = alt_path

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
        training_mode="finetune",
        load_pretrained_backbones=False,
    )

    model = Motus(model_config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        if os.path.isdir(checkpoint_path):
            ckpt_file = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
        else:
            ckpt_file = checkpoint_path

        state_dict = torch.load(ckpt_file, map_location="cpu")
        if "module" in state_dict:
            state_dict = state_dict["module"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    # Freeze entire model
    for param in model.parameters():
        param.requires_grad = False

    from data.dataset import create_dataset, collate_fn
    from torch.utils.data import DataLoader

    split = "validation" if val else "training"
    logger.info(f"Loading {split} dataset...")
    dataset = create_dataset(config, val=val)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=not val,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return model, dataloader, config, dataset


# ---------------------------------------------------------------------------
# Motion Feature Extractor (same as train_injector.py)
# ---------------------------------------------------------------------------

class SimpleMotionExtractor:
    def __init__(self, feature_dim: int = 256, device: str = "cuda"):
        self.feature_dim = feature_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feature_dim),
        ).to(device)
        self.encoder.eval()

    @torch.no_grad()
    def extract(self, current_frame: torch.Tensor, previous_frame: torch.Tensor) -> torch.Tensor:
        diff = current_frame - previous_frame
        diff_mag = torch.sqrt(diff.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        return self.encoder(diff_mag)


# ---------------------------------------------------------------------------
# FIFO Memory Bank (same as train_injector.py)
# ---------------------------------------------------------------------------

class FIFOMemoryBank:
    def __init__(self, max_size: int = 5, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.clear()

    def clear(self):
        self.vlm_features: List[torch.Tensor] = []
        self.motion_features: List[torch.Tensor] = []
        self.timestamps: List[int] = []

    @property
    def size(self) -> int:
        return len(self.vlm_features)

    def add(self, vlm_features: torch.Tensor, motion_features: torch.Tensor, timestamp: int):
        if self.size >= self.max_size:
            self.vlm_features.pop(0)
            self.motion_features.pop(0)
            self.timestamps.pop(0)
        self.vlm_features.append(vlm_features.detach().clone())
        self.motion_features.append(motion_features.detach().clone())
        self.timestamps.append(timestamp)

    def get_all_vlm(self) -> Optional[torch.Tensor]:
        if self.size == 0:
            return None
        B, seq_len, dim = self.vlm_features[0].shape
        features = torch.zeros(B, self.max_size, seq_len, dim, device=self.device, dtype=self.vlm_features[0].dtype)
        for i, feat in enumerate(self.vlm_features):
            features[:, i] = feat
        return features

    def get_all_motion(self) -> Optional[torch.Tensor]:
        if self.size == 0:
            return None
        B, dim = self.motion_features[0].shape
        features = torch.zeros(B, self.max_size, dim, device=self.device, dtype=self.motion_features[0].dtype)
        for i, feat in enumerate(self.motion_features):
            features[:, i] = feat
        return features

    def get_timestamps(self) -> List[int]:
        return self.timestamps


# ---------------------------------------------------------------------------
# RecencyWeightedRetriever (from memory_bank.py, inlined for training)
# ---------------------------------------------------------------------------

class RecencyWeightedRetriever(nn.Module):
    """Recency-weighted soft retrieval: all entries participate in attention,
    weighted by motion relevance + learnable time decay."""

    def __init__(self, motion_dim: int = 256, learnable_decay: bool = True,
                 init_decay: float = 1.0):
        super().__init__()
        self.motion_dim = motion_dim
        if learnable_decay:
            self.log_decay = nn.Parameter(torch.tensor(float(init_decay)).log())
        else:
            self.register_buffer('log_decay', torch.tensor(float(init_decay)).log())

    @property
    def decay(self) -> torch.Tensor:
        return torch.exp(self.log_decay)

    def forward(
        self,
        query_motion: torch.Tensor,
        memory_bank: FIFOMemoryBank,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Soft retrieval with recency weighting.

        Args:
            query_motion: [B, motion_dim]
            memory_bank: FIFOMemoryBank instance

        Returns:
            weighted_vlm: [B, max_size, seq_len, und_dim]
            weights: [B, max_size] — per-entry weights (for analysis)
        """
        stored_vlm = memory_bank.get_all_vlm()
        stored_motion = memory_bank.get_all_motion()
        if stored_vlm is None:
            return None, None

        B, K = stored_motion.shape[:2]

        # 1. Motion relevance: cosine similarity
        motion_scores = F.cosine_similarity(
            query_motion.unsqueeze(1),  # [B, 1, motion_dim]
            stored_motion,               # [B, K, motion_dim]
            dim=-1
        )  # [B, K]

        # 2. Recency bias
        timestamps = torch.tensor(
            memory_bank.get_timestamps(), dtype=torch.float32, device=query_motion.device
        ).unsqueeze(0).expand(B, -1)  # [B, K]
        current_time = timestamps.max(dim=-1, keepdim=True).values  # [B, 1]
        age = current_time - timestamps  # [B, K]
        decay = self.decay
        recency_bias = -decay * age  # [B, K]

        # 3. Combined weight
        combined = motion_scores + recency_bias
        weights = F.softmax(combined, dim=-1)  # [B, K]

        # 4. Weighted features
        weighted_vlm = stored_vlm * weights.unsqueeze(-1).unsqueeze(-1)

        return weighted_vlm, weights


# ---------------------------------------------------------------------------
# ActionMemoryInjector (same as train_injector.py)
# ---------------------------------------------------------------------------

class ActionMemoryInjector(nn.Module):
    """Inject memory into action tokens via cross-attention."""
    def __init__(self, action_dim: int = 1024, memory_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = action_dim // num_heads

        self.q_proj = nn.Linear(action_dim, action_dim)
        self.k_proj = nn.Linear(memory_dim, action_dim)
        self.v_proj = nn.Linear(memory_dim, action_dim)
        self.out_proj = nn.Linear(action_dim, action_dim)
        self.norm_q = nn.LayerNorm(action_dim)
        self.norm_memory = nn.LayerNorm(memory_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, action_tokens: torch.Tensor, memory_features: torch.Tensor) -> torch.Tensor:
        B, S_a, D = action_tokens.shape
        K, S_m, D_m = memory_features.shape[1], memory_features.shape[2], memory_features.shape[3]
        input_dtype = action_tokens.dtype

        memory_flat = memory_features.reshape(B, K * S_m, D_m)

        q = self.norm_q(action_tokens.float())
        memory_normed = self.norm_memory(memory_flat.float())

        Q = self.q_proj(q).view(B, S_a, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = self.k_proj(memory_normed).view(B, K * S_m, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(memory_normed).view(B, K * S_m, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(Q, K_proj, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_a, D)
        output = self.out_proj(attn_output)

        return (action_tokens + torch.sigmoid(self.gate) * output).to(input_dtype)


# ---------------------------------------------------------------------------
# Helper: create VLM inputs for a single frame
# ---------------------------------------------------------------------------

def _create_vlm_inputs_for_frame(model, frame: torch.Tensor, batch: dict, device: str, vlm_processor=None) -> dict:
    from utils.vlm_utils import preprocess_vlm_messages

    original_vlm = batch["vlm_inputs"]
    if isinstance(original_vlm, list):
        text_instr = original_vlm[0].get("text", "")
    elif isinstance(original_vlm, dict):
        text_instr = original_vlm.get("text", "")
    else:
        text_instr = ""

    from data.utils.image_utils import tensor_to_pil
    if frame.dim() == 4:
        frame = frame[0]
    frame_pil = tensor_to_pil(frame.cpu().detach())

    vlm_tokens = preprocess_vlm_messages(text_instr, frame_pil, vlm_processor)
    return vlm_tokens


# ---------------------------------------------------------------------------
# Forward pass with soft memory injection
# ---------------------------------------------------------------------------

def forward_with_soft_memory(
    model,
    injector,
    retriever,
    first_frame, state, num_inference_steps,
    language_embeddings, vlm_inputs,
    memory_bank,
    first_frame_und_tokens,
    action_expert_blocks,
    und_expert_blocks,
    query_motion,
):
    """Forward pass with soft memory injection (RecencyWeightedRetriever)."""
    from model_patches import apply_patches
    apply_patches()

    B = first_frame.shape[0]
    device = model.device
    dtype = model.dtype

    language_embeddings = [e.to(device).to(dtype) for e in language_embeddings]
    state = state.to(device).to(dtype)
    first_frame = first_frame.to(device).to(dtype)

    # 1. Encode condition frame
    first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
    with torch.no_grad():
        condition_frame_latent = model.video_model.encode_video(first_frame_norm.to(dtype))

    B_lat, C_lat, f_lat, H_lat, W_lat = condition_frame_latent.shape
    num_total_latent_frames = 1 + model.config.num_video_frames // 4
    video_latent = torch.randn(
        (B_lat, C_lat, num_total_latent_frames, H_lat, W_lat),
        device=device, dtype=dtype,
    )
    video_latent[:, :, 0:1] = condition_frame_latent

    action_shape = (B, model.config.action_chunk_size, model.config.action_dim)
    action_latent = torch.randn(action_shape, device=device, dtype=dtype)

    # 2. Understanding features — UNCHANGED
    und_tokens = first_frame_und_tokens.to(device).to(dtype)

    processed_t5_context = model.video_module.preprocess_t5_embeddings(language_embeddings)

    # 3. Soft retrieval — weighted VLM features
    retrieved_vlm, weights = retriever(query_motion, memory_bank)
    mem_features = None
    if retrieved_vlm is not None:
        mem_features = retrieved_vlm.to(device).float()

    # 4. Denoising loop
    timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)

    for i in range(num_inference_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t
        video_t_scaled = (t * 1000).expand(B).to(dtype)
        action_t_scaled = (t * 1000).expand(B).to(dtype)

        video_tokens = model.video_module.prepare_input(video_latent.to(dtype))
        state_tokens = state.unsqueeze(1).to(dtype)
        registers = model.action_expert.registers.expand(B, -1, -1)
        action_tokens = model.action_expert.input_encoder(
            state_tokens, action_latent, registers
        )

        with torch.autocast(device_type="cuda", dtype=model.video_model.precision):
            video_head_time_emb, video_adaln_params = (
                model.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
            )
            action_head_time_emb, action_adaln_params = (
                model.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])
            )

            for layer_idx in range(model.config.num_layers):
                video_adaln_modulation = model.video_module.compute_adaln_modulation(
                    video_adaln_params, layer_idx
                )
                action_adaln_modulation = model.action_module.compute_adaln_modulation(
                    action_adaln_params, layer_idx
                )

                video_tokens, action_tokens, und_tokens = (
                    model.video_module.process_joint_attention(
                        video_tokens, action_tokens,
                        video_adaln_modulation, action_adaln_modulation,
                        layer_idx, action_expert_blocks[layer_idx],
                        und_tokens, und_expert_blocks[layer_idx],
                    )
                )

                video_tokens = model.video_module.process_cross_attention(
                    video_tokens, video_adaln_params, layer_idx, processed_t5_context
                )

                video_tokens = model.video_module.process_ffn(
                    video_tokens, video_adaln_modulation, layer_idx
                )
                action_tokens = model.action_module.process_ffn(
                    action_tokens, action_adaln_modulation, layer_idx
                )
                und_tokens = model.und_module.process_ffn(und_tokens, layer_idx)

            # Inject soft-weighted memory into action tokens
            if mem_features is not None:
                action_tokens = injector(action_tokens.float(), mem_features).to(dtype)

            video_velocity = model.video_module.apply_output_head(
                video_tokens, video_head_time_emb
            )
            action_pred_full = model.action_expert.decoder(
                action_tokens, action_head_time_emb
            )
            action_velocity = action_pred_full[
                :, 1 : -model.action_expert.config.num_registers, :
            ]

            video_latent = video_latent + video_velocity * dt
            action_latent = action_latent + action_velocity * dt
            video_latent[:, :, 0:1] = condition_frame_latent

    return action_latent.float()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model, injector, retriever, motion_extractor, dataloader,
    num_eval_batches, num_inference_steps, bank_size,
    device, vlm_processor, vlm_cache=None,
):
    """Evaluate with soft retrieval."""
    from model_patches import apply_patches
    apply_patches()

    injector.eval()
    retriever.eval()
    action_agg = []

    for i, batch in enumerate(dataloader):
        if i >= num_eval_batches:
            break

        first_frame = batch["first_frame"].to(device)
        video_frames = batch["video_frames"].to(device)
        state = batch["initial_state"].to(device)
        gt_actions = batch["action_sequence"].to(device)
        language_embeddings = batch["language_embedding"]
        vlm_inputs = batch["vlm_inputs"]

        B, T, C, H, W = video_frames.shape

        with torch.no_grad():
            if vlm_cache is not None and i in vlm_cache:
                cached = vlm_cache[i]
                first_frame_und_tokens = cached["first_frame_und"].to(device)
            else:
                first_frame_und_tokens = model.und_module.extract_und_features(vlm_inputs)

            # Build memory bank
            mem_bank = FIFOMemoryBank(max_size=bank_size, device=device)
            for t in range(T):
                if vlm_cache is not None and i in vlm_cache:
                    hist_und = vlm_cache[i]["hist_und"][t].to(device)
                    motion_feat = vlm_cache[i]["motion_feat"][t].to(device)
                else:
                    hist_frame = video_frames[:, t]
                    hist_vlm_inputs = _create_vlm_inputs_for_frame(
                        model, hist_frame, batch, device, vlm_processor=vlm_processor
                    )
                    hist_und = model.und_module.extract_und_features(hist_vlm_inputs)
                    if t == 0:
                        motion_feat = torch.zeros(B, motion_extractor.feature_dim, device=device)
                    else:
                        motion_feat = motion_extractor.extract(video_frames[:, t], video_frames[:, t-1])
                mem_bank.add(hist_und, motion_feat, timestamp=t)

            query_motion = torch.zeros(B, motion_extractor.feature_dim, device=device)

            # Forward with soft retrieval
            pred_actions = forward_with_soft_memory(
                model, injector, retriever, first_frame, state, num_inference_steps,
                language_embeddings, vlm_inputs, mem_bank, first_frame_und_tokens,
                model.action_expert.blocks, model.und_expert.blocks,
                query_motion,
            )

        action_mse = F.mse_loss(pred_actions.float(), gt_actions.float()).item()
        action_agg.append(action_mse)

    injector.train()
    retriever.train()
    return float(np.mean(action_agg))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Train Injector + RecencyWeightedRetriever (joint)")
    logger.info(f"  Process rank: {accelerator.process_index}, device: {device}")
    logger.info(f"  Init decay: {args.init_decay}")
    logger.info("=" * 60)

    distributed = accelerator.num_processes > 1

    # Load model + datasets
    model, train_dataloader, config, train_dataset = load_model_and_data(
        args.config, args.checkpoint, str(device), val=False
    )
    from data.dataset import create_dataset, collate_fn
    from torch.utils.data import DataLoader
    val_dataset = create_dataset(config, val=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True,
    )

    vlm_processor = getattr(train_dataset, 'vlm_processor', None)
    if vlm_processor is None:
        raise RuntimeError("Dataset does not have a vlm_processor attribute")

    # Create trainable modules
    injector = ActionMemoryInjector(action_dim=1024, memory_dim=512)
    retriever = RecencyWeightedRetriever(
        motion_dim=256, learnable_decay=True, init_decay=args.init_decay
    )
    injector.train()
    retriever.train()

    # Motion extractor (frozen)
    motion_extractor = SimpleMotionExtractor(device=str(device))

    # Optimizer — both injector and retriever are trainable
    trainable_params = list(injector.parameters()) + list(retriever.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Prepare with accelerator
    injector, retriever, optimizer, train_dataloader = accelerator.prepare(
        injector, retriever, optimizer, train_dataloader
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01,
    )

    # Freeze all model params
    for param in model.parameters():
        param.requires_grad = False

    n_injector = sum(p.numel() for p in injector.parameters() if p.requires_grad)
    n_retriever = sum(p.numel() for p in retriever.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: injector={n_injector:,}, retriever={n_retriever:,}, total={n_injector+n_retriever:,}")
    logger.info(f"Training: {args.num_epochs} epochs, {args.num_train_batches} batches/epoch")
    logger.info(f"Evaluation: every {args.eval_every_steps} steps, {args.num_eval_batches} batches")

    # ---------------------------------------------------------------------------
    # Pre-compute VLM features cache
    # ---------------------------------------------------------------------------
    logger.info("Pre-computing VLM features for all training batches...")
    vlm_cache = {}
    for cache_i, cache_batch in enumerate(train_dataloader):
        if cache_i >= args.num_train_batches:
            break
        with torch.no_grad():
            cache_vlm_inputs = cache_batch["vlm_inputs"]
            cache_first_und = model.und_module.extract_und_features(cache_vlm_inputs)
            cache_video_frames = cache_batch["video_frames"].to(device)
            B_c, T_c, C_c, H_c, W_c = cache_video_frames.shape
            cache_hist_und = []
            cache_motion_feat = []
            for t in range(T_c):
                hist_frame = cache_video_frames[:, t]
                hist_vlm_inputs = _create_vlm_inputs_for_frame(
                    model, hist_frame, cache_batch, str(device), vlm_processor=vlm_processor
                )
                hist_und = model.und_module.extract_und_features(hist_vlm_inputs)
                cache_hist_und.append(hist_und.detach().cpu())
                if t == 0:
                    motion_feat = torch.zeros(B_c, motion_extractor.feature_dim)
                else:
                    motion_feat = motion_extractor.extract(cache_video_frames[:, t], cache_video_frames[:, t-1]).cpu()
                cache_motion_feat.append(motion_feat)
        vlm_cache[cache_i] = {
            "first_frame_und": cache_first_und.detach().cpu(),
            "hist_und": cache_hist_und,
            "motion_feat": cache_motion_feat,
        }
        if (cache_i + 1) % 10 == 0:
            logger.info(f"  Cached {cache_i + 1}/{min(args.num_train_batches, len(train_dataloader))} batches")
    logger.info(f"VLM cache ready: {len(vlm_cache)} batches cached")

    # Pre-compute validation cache
    logger.info("Pre-computing VLM features for validation batches...")
    val_vlm_cache = {}
    for cache_i, cache_batch in enumerate(val_dataloader):
        if cache_i >= args.num_eval_batches:
            break
        with torch.no_grad():
            cache_vlm_inputs = cache_batch["vlm_inputs"]
            cache_first_und = model.und_module.extract_und_features(cache_vlm_inputs)
            cache_video_frames = cache_batch["video_frames"].to(device)
            B_c, T_c, C_c, H_c, W_c = cache_video_frames.shape
            cache_hist_und = []
            cache_motion_feat = []
            for t in range(T_c):
                hist_frame = cache_video_frames[:, t]
                hist_vlm_inputs = _create_vlm_inputs_for_frame(
                    model, hist_frame, cache_batch, str(device), vlm_processor=vlm_processor
                )
                hist_und = model.und_module.extract_und_features(hist_vlm_inputs)
                cache_hist_und.append(hist_und.detach().cpu())
                if t == 0:
                    motion_feat = torch.zeros(B_c, motion_extractor.feature_dim)
                else:
                    motion_feat = motion_extractor.extract(cache_video_frames[:, t], cache_video_frames[:, t-1]).cpu()
                cache_motion_feat.append(motion_feat)
        val_vlm_cache[cache_i] = {
            "first_frame_und": cache_first_und.detach().cpu(),
            "hist_und": cache_hist_und,
            "motion_feat": cache_motion_feat,
        }
    logger.info(f"Val VLM cache ready: {len(val_vlm_cache)} batches cached")

    # Training history
    history = {"train_loss": [], "val_mse": [], "lr": [], "step": [], "decay": []}
    best_val_mse = float("inf")
    best_step = -1
    global_step = 0

    recent_ckpts: deque = deque(maxlen=args.max_checkpoints)
    best_ckpt_path = os.path.join(args.output_dir, "injector_soft_best.pt")

    def _save_ckpt(step, val_mse, avg_loss, decay_val):
        raw_injector = accelerator.unwrap_model(injector)
        raw_retriever = accelerator.unwrap_model(retriever)
        ckpt = {
            "step": step,
            "injector_state_dict": raw_injector.state_dict(),
            "retriever_state_dict": raw_retriever.state_dict(),
            "decay": decay_val,
            "val_mse": val_mse,
            "train_loss": avg_loss,
            "args": vars(args),
        }
        ckpt_path = os.path.join(args.output_dir, f"injector_soft_step{step:06d}.pt")
        torch.save(ckpt, ckpt_path)
        recent_ckpts.append(ckpt_path)
        while len(recent_ckpts) > args.max_checkpoints:
            old_path = recent_ckpts.popleft()
            if os.path.exists(old_path) and old_path != best_ckpt_path:
                os.remove(old_path)
                logger.info(f"  Removed old checkpoint: {old_path}")
        if val_mse < best_val_mse:
            torch.save(ckpt, best_ckpt_path)
            return True
        return False

    for epoch in range(args.num_epochs):
        injector.train()
        retriever.train()
        epoch_losses = []

        for i, batch in enumerate(train_dataloader):
            if i >= args.num_train_batches:
                break

            first_frame = batch["first_frame"].to(device)
            video_frames = batch["video_frames"].to(device)
            state = batch["initial_state"].to(device)
            gt_actions = batch["action_sequence"].to(device)
            language_embeddings = batch["language_embedding"]
            vlm_inputs = batch["vlm_inputs"]

            B, T, C, H, W = video_frames.shape

            # Use cached VLM features
            cached = vlm_cache[i]
            first_frame_und_tokens = cached["first_frame_und"].to(device)

            # Build memory bank from cached features
            mem_bank = FIFOMemoryBank(max_size=args.bank_size, device=str(device))
            with torch.no_grad():
                for t in range(T):
                    hist_und = cached["hist_und"][t].to(device)
                    motion_feat = cached["motion_feat"][t].to(device)
                    mem_bank.add(hist_und, motion_feat, timestamp=t)

                query_motion = torch.zeros(B, motion_extractor.feature_dim, device=str(device))

            # Forward with soft memory injection
            pred_actions = forward_with_soft_memory(
                model, injector, retriever, first_frame, state, args.num_inference_steps,
                language_embeddings, vlm_inputs, mem_bank, first_frame_und_tokens,
                model.action_expert.blocks, model.und_expert.blocks,
                query_motion,
            )

            loss = F.mse_loss(pred_actions.float(), gt_actions.float())
            loss_val = loss.item()

            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss_val)
            global_step += 1

            del pred_actions, loss
            gc.collect()
            torch.cuda.empty_cache()

            if is_main and global_step % 10 == 0:
                decay_val = accelerator.unwrap_model(retriever).decay.item()
                logger.info(f"  Step {global_step}, epoch {epoch+1}, batch {i+1}/{args.num_train_batches}, "
                            f"loss={loss_val:.6f}, decay={decay_val:.4f}")

            # Step-based eval
            if is_main and global_step % args.eval_every_steps == 0:
                val_mse = evaluate(
                    model, injector, retriever, motion_extractor, val_dataloader,
                    args.num_eval_batches, args.num_inference_steps,
                    args.bank_size, str(device), vlm_processor,
                    vlm_cache=val_vlm_cache,
                )
                current_lr = scheduler.get_last_lr()[0]
                decay_val = accelerator.unwrap_model(retriever).decay.item()
                history["val_mse"].append(val_mse)
                history["step"].append(global_step)
                history["lr"].append(current_lr)
                history["decay"].append(decay_val)
                logger.info(f"  [Step {global_step}] Val MSE: {val_mse:.6f}, lr: {current_lr:.2e}, decay: {decay_val:.4f}")

                is_best = _save_ckpt(global_step, val_mse, float(np.mean(epoch_losses)), decay_val)
                if is_best:
                    best_val_mse = val_mse
                    best_step = global_step
                    logger.info(f"  ** New best: {val_mse:.6f} at step {global_step} **")

        scheduler.step()
        avg_train_loss = float(np.mean(epoch_losses))
        if is_main:
            history["train_loss"].append(avg_train_loss)
            decay_val = accelerator.unwrap_model(retriever).decay.item()
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}: train_loss={avg_train_loss:.6f}, decay={decay_val:.4f}")

    # Final eval
    if is_main and global_step % args.eval_every_steps != 0:
        val_mse = evaluate(
            model, injector, retriever, motion_extractor, val_dataloader,
            args.num_eval_batches, args.num_inference_steps,
            args.bank_size, str(device), vlm_processor,
            vlm_cache=val_vlm_cache,
        )
        decay_val = accelerator.unwrap_model(retriever).decay.item()
        _save_ckpt(global_step, val_mse, float(np.mean(epoch_losses)), decay_val)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_step = global_step

    # Save training history
    if is_main:
        history_path = os.path.join(args.output_dir, "train_history_soft.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info(f"Total steps: {global_step}")
        logger.info(f"Best Val MSE: {best_val_mse:.6f} (step {best_step})")
        logger.info(f"Final decay: {accelerator.unwrap_model(retriever).decay.item():.4f}")
        logger.info(f"Best checkpoint: {best_ckpt_path}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
