#!/usr/bin/env python3
"""
Phase 2 Core Experiments: Motion-Aware Temporal Memory Bank

Experiments:
  2.1: Temporal Memory Bank (no retrieval) - fixed-size memory, no selection
  2.2: Visual vs Motion Retrieval Comparison
  2.3: Memory Size Ablation

Usage:
    python run_phase2_experiments.py \
        --config ../../configs/robotwin_lerobot.yaml \
        --checkpoint /path/to/Motus_robotwin2/mp_rank_00_model_states.pt \
        --output_dir ./results \
        --num_eval_batches 5 \
        --num_inference_steps 10 \
        --experiments 2.1 2.2 2.3
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

_inference_root = _project_root / "inference" / "robotwin"
sys.path.insert(0, str(_inference_root))

# ---------------------------------------------------------------------------
# Monkey-patch lerobot (same as run_memory_experiments.py)
# ---------------------------------------------------------------------------
import lerobot.datasets.utils as _lerobot_utils
import lerobot.datasets.lerobot_dataset as _lerobot_ds
import pandas as _pd

_original_check = _lerobot_utils.check_version_compatibility

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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: Memory Bank Experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_eval_batches", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["2.1", "2.2", "2.3"],
        help="Experiments to run: 2.1, 2.2, 2.3",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model & data loading (reuse from run_memory_experiments.py)
# ---------------------------------------------------------------------------

def load_model_and_data(config_path: str, checkpoint_path: str, device: str):
    import yaml
    from omegaconf import OmegaConf

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    config.model.inference.num_inference_timesteps = 50

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

    from data.dataset import create_dataset, collate_fn
    from torch.utils.data import DataLoader

    logger.info("Loading validation dataset...")
    val_dataset = create_dataset(config, val=True)
    dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return model, dataloader, config, val_dataset


# ---------------------------------------------------------------------------
# Memory Bank utilities
# ---------------------------------------------------------------------------

class SimpleMotionExtractor:
    """Simple frame-difference based motion feature extractor (no external deps)."""

    def __init__(self, feature_dim: int = 256, device: str = "cuda"):
        self.feature_dim = feature_dim
        self.device = device
        # Use a small CNN for motion encoding
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
        """Extract motion features from frame pair.

        Args:
            current_frame: [B, C, H, W] in [0, 1]
            previous_frame: [B, C, H, W] in [0, 1]

        Returns:
            [B, feature_dim]
        """
        diff = current_frame - previous_frame
        diff_mag = torch.sqrt(diff.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        return self.encoder(diff_mag)


class SimpleMemoryBank:
    """Fixed-size memory bank for storing VLM and motion features."""

    def __init__(self, max_size: int = 20, device: str = "cuda"):
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


def retrieve_memory(
    query_motion: torch.Tensor,
    query_vlm: Optional[torch.Tensor],
    memory_bank: SimpleMemoryBank,
    retrieval_mode: str = "motion",
    top_k: int = 5,
    hybrid_alpha: float = 0.5,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Retrieve top-k memory entries based on retrieval mode.

    Returns:
        retrieved_vlm: [B, top_k, seq_len, und_dim] or None
        scores: [B, max_size] or None
    """
    if memory_bank.size == 0:
        return None, None

    stored_vlm = memory_bank.get_all_vlm()  # [B, N, seq_len, dim]
    stored_motion = memory_bank.get_all_motion()  # [B, N, motion_dim]
    B, N = stored_motion.shape[:2]

    if retrieval_mode == "visual" and query_vlm is not None:
        query_pooled = query_vlm.mean(dim=1)
        stored_pooled = stored_vlm.mean(dim=2)
        scores = F.cosine_similarity(query_pooled.unsqueeze(1), stored_pooled, dim=-1)

    elif retrieval_mode == "motion":
        scores = F.cosine_similarity(query_motion.unsqueeze(1), stored_motion, dim=-1)

    elif retrieval_mode == "hybrid":
        motion_scores = F.cosine_similarity(query_motion.unsqueeze(1), stored_motion, dim=-1)
        if query_vlm is not None:
            query_pooled = query_vlm.mean(dim=1)
            stored_pooled = stored_vlm.mean(dim=2)
            visual_scores = F.cosine_similarity(query_pooled.unsqueeze(1), stored_pooled, dim=-1)
        else:
            visual_scores = torch.zeros_like(motion_scores)
        scores = hybrid_alpha * motion_scores + (1 - hybrid_alpha) * visual_scores
    else:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")

    actual_k = min(top_k, memory_bank.size)
    _, top_k_indices = scores.topk(actual_k, dim=1)

    retrieved_vlm = torch.gather(
        stored_vlm, 1,
        top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, stored_vlm.shape[2], stored_vlm.shape[3]),
    )

    return retrieved_vlm, scores


# ---------------------------------------------------------------------------
# Cross-Attention Memory Injector
# ---------------------------------------------------------------------------

class CrossAttentionMemoryInjector(nn.Module):
    """Lightweight cross-attention for injecting memory into understanding tokens."""

    def __init__(self, und_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.und_dim = und_dim
        self.num_heads = num_heads
        self.head_dim = und_dim // num_heads

        self.q_proj = nn.Linear(und_dim, und_dim)
        self.k_proj = nn.Linear(und_dim, und_dim)
        self.v_proj = nn.Linear(und_dim, und_dim)
        self.out_proj = nn.Linear(und_dim, und_dim)
        self.norm_q = nn.LayerNorm(und_dim)
        self.norm_memory = nn.LayerNorm(und_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, und_tokens: torch.Tensor, memory_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            und_tokens: [B, S, D]
            memory_features: [B, K, S, D]
        Returns:
            [B, S, D]
        """
        B, S, D = und_tokens.shape
        K = memory_features.shape[1]
        input_dtype = und_tokens.dtype

        memory_flat = memory_features.reshape(B, K * S, D)

        # Cast to float32 for computation stability, then cast back
        q = self.norm_q(und_tokens.float())
        memory_normed = self.norm_memory(memory_flat.float())

        Q = self.q_proj(q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = self.k_proj(memory_normed).view(B, K * S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(memory_normed).view(B, K * S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(Q, K_proj, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_output)

        return (und_tokens + torch.sigmoid(self.gate) * output).to(input_dtype)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_action_mse(pred_actions: torch.Tensor, gt_actions: torch.Tensor) -> dict:
    """Compute action prediction MSE."""
    mse = F.mse_loss(pred_actions, gt_actions).item()
    per_step = []
    for t in range(pred_actions.shape[1]):
        per_step.append(F.mse_loss(pred_actions[:, t], gt_actions[:, t]).item())
    return {"action_mse": mse, "action_mse_per_step": per_step}


def compute_video_mse(pred_frames: torch.Tensor, gt_frames: torch.Tensor) -> dict:
    """Compute video prediction MSE.

    Handles both [B, C, T, H, W] and [B, T, C, H, W] formats by converting
    to [B, T, C, H, W] before comparison.
    """
    def _to_btcwh(x):
        if x.dim() != 5:
            return x
        if x.shape[2] == 3 and x.shape[1] != 3:
            return x.permute(0, 2, 1, 3, 4)
        return x

    pred_frames = _to_btcwh(pred_frames)
    gt_frames = _to_btcwh(gt_frames)

    # Ensure same number of frames (truncate to min)
    min_t = min(pred_frames.shape[1], gt_frames.shape[1])
    pred_frames = pred_frames[:, :min_t]
    gt_frames = gt_frames[:, :min_t]

    return {"video_mse": F.mse_loss(pred_frames, gt_frames).item()}


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment_2_1(
    model, dataloader, config, num_eval_batches, num_inference_steps, device,
    vlm_processor=None,
):
    """Experiment 2.1: Temporal Memory Bank (no retrieval).

    Tests: multi-frame VLM features (averaged) vs single-frame (first_frame_memory).
    Memory sizes: 1, 3, 5, 10 frames.
    """
    from model_patches import apply_patches, patched_inference_step, patched_inference_step_with_memory
    apply_patches()

    results = {}
    memory_sizes = [1, 3, 5, 10]

    for mem_size in memory_sizes:
        logger.info(f"  Experiment 2.1: memory_size={mem_size}")

        action_agg = []
        video_agg = []

        for i, batch in enumerate(dataloader):
            if i >= num_eval_batches:
                break

            first_frame = batch["first_frame"].to(device)
            video_frames = batch["video_frames"].to(device)
            state = batch["initial_state"].to(device)
            gt_actions = batch["action_sequence"].to(device)
            language_embeddings = batch["language_embedding"]
            vlm_inputs = batch["vlm_inputs"]

            with torch.no_grad():
                # Extract VLM features from first frame
                first_frame_und_tokens = model.und_module.extract_und_features(vlm_inputs)

                # Extract VLM features from video frames (use available frames as "historical")
                B, T, C, H, W = video_frames.shape
                available_frames = min(mem_size, T)

                # Use the first N video frames as historical context
                # (In real scenario, these would be from previous timesteps)
                memory_und_list = []
                for t in range(available_frames):
                    # Create VLM inputs for each historical frame
                    hist_frame = video_frames[:, t]  # [B, C, H, W]
                    hist_vlm_inputs = _create_vlm_inputs_for_frame(
                        model, hist_frame, batch, device, vlm_processor=vlm_processor
                    )
                    hist_und = model.und_module.extract_und_features(hist_vlm_inputs)
                    memory_und_list.append(hist_und)

                # Stack: [B, K, seq_len, und_dim]
                memory_vlm = torch.stack(memory_und_list, dim=1)

                # Run inference with memory (no retrieval, just averaged)
                # Average memory across frames
                memory_mean = memory_vlm.mean(dim=1, keepdim=True)  # [B, 1, seq_len, und_dim]

                pred_frames, pred_actions = patched_inference_step_with_memory(
                    model, first_frame, state, num_inference_steps,
                    language_embeddings, vlm_inputs,
                    first_frame_und_tokens=first_frame_und_tokens,
                    memory_vlm_features=memory_mean,
                    injection_mode="mean_add",
                )

            action_metrics = compute_action_mse(pred_actions, gt_actions)
            video_metrics = compute_video_mse(pred_frames, video_frames)

            action_agg.append(action_metrics)
            video_agg.append(video_metrics)

        avg_action_mse = np.mean([m["action_mse"] for m in action_agg])
        avg_video_mse = np.mean([m["video_mse"] for m in video_agg])

        results[f"memory_size_{mem_size}"] = {
            "action_mse": float(avg_action_mse),
            "video_mse": float(avg_video_mse),
            "memory_size": mem_size,
        }
        logger.info(f"    action_mse={avg_action_mse:.6f}, video_mse={avg_video_mse:.6f}")

    return results


def run_experiment_2_2(
    model, dataloader, config, num_eval_batches, num_inference_steps, device,
    vlm_processor=None,
):
    """Experiment 2.2: Visual vs Motion Retrieval Comparison.

    Compares three retrieval modes: visual, motion, hybrid.
    """
    from model_patches import apply_patches, patched_inference_step_with_memory
    apply_patches()

    results = {}
    retrieval_modes = ["visual", "motion", "hybrid"]
    top_k = 5

    motion_extractor = SimpleMotionExtractor(device=device)
    injector = CrossAttentionMemoryInjector(und_dim=512).to(device)

    for mode in retrieval_modes:
        logger.info(f"  Experiment 2.2: retrieval_mode={mode}")

        action_agg = []
        video_agg = []

        for i, batch in enumerate(dataloader):
            if i >= num_eval_batches:
                break

            first_frame = batch["first_frame"].to(device)
            video_frames = batch["video_frames"].to(device)
            state = batch["initial_state"].to(device)
            gt_actions = batch["action_sequence"].to(device)
            language_embeddings = batch["language_embedding"]
            vlm_inputs = batch["vlm_inputs"]

            with torch.no_grad():
                # Extract VLM features from first frame
                first_frame_und_tokens = model.und_module.extract_und_features(vlm_inputs)

                # Build memory bank from video frames
                B, T, C, H, W = video_frames.shape
                mem_bank = SimpleMemoryBank(max_size=20, device=device)

                for t in range(T):
                    hist_frame = video_frames[:, t]
                    hist_vlm_inputs = _create_vlm_inputs_for_frame(
                        model, hist_frame, batch, device, vlm_processor=vlm_processor
                    )
                    hist_und = model.und_module.extract_und_features(hist_vlm_inputs)

                    # Motion features
                    if t == 0:
                        # First frame: no motion
                        motion_feat = torch.zeros(B, motion_extractor.feature_dim, device=device)
                    else:
                        motion_feat = motion_extractor.extract(video_frames[:, t], video_frames[:, t-1])

                    mem_bank.add(hist_und, motion_feat, timestamp=t)

                # Query motion: use first frame's motion (zero)
                query_motion = torch.zeros(B, motion_extractor.feature_dim, device=device)
                query_vlm = first_frame_und_tokens

                # Retrieve
                retrieved_vlm, _ = retrieve_memory(
                    query_motion, query_vlm, mem_bank,
                    retrieval_mode=mode, top_k=top_k,
                )

                # Run inference with retrieved memory
                pred_frames, pred_actions = patched_inference_step_with_memory(
                    model, first_frame, state, num_inference_steps,
                    language_embeddings, vlm_inputs,
                    first_frame_und_tokens=first_frame_und_tokens,
                    memory_vlm_features=retrieved_vlm,
                    memory_injector=injector,
                    injection_mode="cross_attn",
                )

            action_metrics = compute_action_mse(pred_actions, gt_actions)
            video_metrics = compute_video_mse(pred_frames, video_frames)

            action_agg.append(action_metrics)
            video_agg.append(video_metrics)

        avg_action_mse = np.mean([m["action_mse"] for m in action_agg])
        avg_video_mse = np.mean([m["video_mse"] for m in video_agg])

        results[mode] = {
            "action_mse": float(avg_action_mse),
            "video_mse": float(avg_video_mse),
            "retrieval_mode": mode,
            "top_k": top_k,
        }
        logger.info(f"    action_mse={avg_action_mse:.6f}, video_mse={avg_video_mse:.6f}")

    return results


def run_experiment_2_3(
    model, dataloader, config, num_eval_batches, num_inference_steps, device,
    vlm_processor=None,
):
    """Experiment 2.3: Memory Size Ablation.

    Tests memory bank sizes: 1, 3, 5, 10, 20 with motion retrieval.
    """
    from model_patches import apply_patches, patched_inference_step_with_memory
    apply_patches()

    results = {}
    memory_sizes = [1, 3, 5, 10, 20]
    retrieval_mode = "motion"
    top_k = 5

    motion_extractor = SimpleMotionExtractor(device=device)
    injector = CrossAttentionMemoryInjector(und_dim=512).to(device)

    for mem_size in memory_sizes:
        logger.info(f"  Experiment 2.3: memory_bank_size={mem_size}")

        action_agg = []
        video_agg = []

        for i, batch in enumerate(dataloader):
            if i >= num_eval_batches:
                break

            first_frame = batch["first_frame"].to(device)
            video_frames = batch["video_frames"].to(device)
            state = batch["initial_state"].to(device)
            gt_actions = batch["action_sequence"].to(device)
            language_embeddings = batch["language_embedding"]
            vlm_inputs = batch["vlm_inputs"]

            with torch.no_grad():
                first_frame_und_tokens = model.und_module.extract_und_features(vlm_inputs)

                B, T, C, H, W = video_frames.shape
                available_frames = min(mem_size, T)
                mem_bank = SimpleMemoryBank(max_size=mem_size, device=device)

                for t in range(available_frames):
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
                query_vlm = first_frame_und_tokens

                retrieved_vlm, _ = retrieve_memory(
                    query_motion, query_vlm, mem_bank,
                    retrieval_mode=retrieval_mode, top_k=min(top_k, mem_size),
                )

                pred_frames, pred_actions = patched_inference_step_with_memory(
                    model, first_frame, state, num_inference_steps,
                    language_embeddings, vlm_inputs,
                    first_frame_und_tokens=first_frame_und_tokens,
                    memory_vlm_features=retrieved_vlm,
                    memory_injector=injector,
                    injection_mode="cross_attn",
                )

            action_metrics = compute_action_mse(pred_actions, gt_actions)
            video_metrics = compute_video_mse(pred_frames, video_frames)

            action_agg.append(action_metrics)
            video_agg.append(video_metrics)

        avg_action_mse = np.mean([m["action_mse"] for m in action_agg])
        avg_video_mse = np.mean([m["video_mse"] for m in video_agg])

        results[f"bank_size_{mem_size}"] = {
            "action_mse": float(avg_action_mse),
            "video_mse": float(avg_video_mse),
            "memory_bank_size": mem_size,
        }
        logger.info(f"    action_mse={avg_action_mse:.6f}, video_mse={avg_video_mse:.6f}")

    return results


# ---------------------------------------------------------------------------
# Helper: create VLM inputs for a single frame
# ---------------------------------------------------------------------------

def _create_vlm_inputs_for_frame(model, frame: torch.Tensor, batch: dict, device: str, vlm_processor=None) -> dict:
    """Create VLM inputs for a single frame (reusing text from batch).

    Args:
        frame: [B, C, H, W] in [0, 1]
        batch: original batch dict (for text/vlm_inputs template)
        device: device string
        vlm_processor: VLM processor instance (required)

    Returns:
        vlm_inputs dict suitable for model.und_module.extract_und_features
    """
    from utils.vlm_utils import preprocess_vlm_messages

    # Get text instruction from original vlm_inputs
    original_vlm = batch["vlm_inputs"]
    if isinstance(original_vlm, list):
        text_instr = original_vlm[0].get("text", "")
    elif isinstance(original_vlm, dict):
        text_instr = original_vlm.get("text", "")
    else:
        text_instr = ""

    # Convert frame to PIL for VLM processing
    from data.utils.image_utils import tensor_to_pil
    # Squeeze batch dimension if present: [B, C, H, W] -> [C, H, W]
    if frame.dim() == 4:
        frame = frame[0]
    # Move to CPU and detach for numpy conversion
    frame_pil = tensor_to_pil(frame.cpu().detach())

    vlm_tokens = preprocess_vlm_messages(text_instr, frame_pil, vlm_processor)
    return vlm_tokens


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results(all_results: dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("  Phase 2: Motion-Aware Temporal Memory Bank Experiments")
    print("=" * 70)

    for exp_name, results in all_results.items():
        print(f"\n--- {exp_name} ---")
        print(f"{'Method':<30s}| {'Action MSE':<12s}| {'Video MSE':<12s}")
        print("-" * 58)

        for key, r in results.items():
            action_mse = r.get("action_mse", float("nan"))
            video_mse = r.get("video_mse", float("nan"))
            print(f"{key:<30s}| {action_mse:<12.6f}| {video_mse:<12.6f}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 2: Motion-Aware Temporal Memory Bank Experiments")
    logger.info("=" * 60)
    logger.info(f"Config:       {args.config}")
    logger.info(f"Checkpoint:   {args.checkpoint}")
    logger.info(f"Output dir:   {args.output_dir}")
    logger.info(f"Batches:      {args.num_eval_batches}")
    logger.info(f"Steps:        {args.num_inference_steps}")
    logger.info(f"Experiments:  {args.experiments}")
    logger.info(f"Device:       {args.device}")

    model, dataloader, config, val_dataset = load_model_and_data(
        args.config, args.checkpoint, args.device
    )

    # Get VLM processor from dataset
    vlm_processor = getattr(val_dataset, 'vlm_processor', None)
    if vlm_processor is None:
        raise RuntimeError("Dataset does not have a vlm_processor attribute")

    all_results = {}

    if "2.1" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Running Experiment 2.1: Temporal Memory Bank (no retrieval)")
        logger.info("=" * 60)
        all_results["2.1_temporal_memory"] = run_experiment_2_1(
            model, dataloader, config,
            args.num_eval_batches, args.num_inference_steps, args.device,
            vlm_processor=vlm_processor,
        )

    if "2.2" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Running Experiment 2.2: Visual vs Motion Retrieval")
        logger.info("=" * 60)
        all_results["2.2_retrieval_comparison"] = run_experiment_2_2(
            model, dataloader, config,
            args.num_eval_batches, args.num_inference_steps, args.device,
            vlm_processor=vlm_processor,
        )

    if "2.3" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Running Experiment 2.3: Memory Size Ablation")
        logger.info("=" * 60)
        all_results["2.3_size_ablation"] = run_experiment_2_3(
            model, dataloader, config,
            args.num_eval_batches, args.num_inference_steps, args.device,
            vlm_processor=vlm_processor,
        )

    # Save results
    output_file = os.path.join(args.output_dir, "phase2_experiments.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_inference_steps": args.num_inference_steps,
        "num_eval_batches": args.num_eval_batches,
        "experiments_run": args.experiments,
        "results": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print_results(all_results)
    logger.info(f"\nResults saved to {output_file}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
