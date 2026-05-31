#!/usr/bin/env python3
"""
Experiment 3.2b: Recency-Weighted Soft Attention vs FIFO Top-K

Compares:
  - FIFO + top-k hard selection (baseline, MSE ~7.06)
  - Recency-Weighted soft retrieval with learnable decay

Usage:
    python run_recency_weighted_experiment.py \
        --config ../../configs/robotwin_lerobot.yaml \
        --checkpoint /path/to/Motus_robotwin2/mp_rank_00_model_states.pt \
        --output_dir ./results \
        --num_eval_batches 20 \
        --num_inference_steps 10
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
# Monkey-patch lerobot
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
    parser = argparse.ArgumentParser(description="Experiment 3.2b: Recency-Weighted Soft Attention")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_eval_batches", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--bank_size", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model & data loading (same as run_phase3_experiments.py)
# ---------------------------------------------------------------------------

def load_model_and_data(config_path: str, checkpoint_path: str, device: str):
    import yaml
    from omegaconf import OmegaConf

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    config.model.inference.num_inference_timesteps = 50

    # Override dataset_dir if the configured path doesn't exist
    from pathlib import Path as _Path
    configured_dir = _Path(config.dataset.dataset_dir)
    if not configured_dir.exists():
        # Try the v2 dataset path
        v2_path = _Path("/kpfs-intern/jialongliu/projects/Motus/data/robotwin2/robotwin_dataset_v2")
        if v2_path.exists():
            config.dataset.dataset_dir = str(v2_path)
            logger.info(f"  Override dataset_dir → {v2_path}")

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

    for param in model.parameters():
        param.requires_grad = False

    from data.dataset import create_dataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = create_dataset(config, val=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True,
    )

    return model, dataloader, config, dataset


# ---------------------------------------------------------------------------
# Motion Feature Extractor (same as run_phase3_experiments.py)
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
# Memory Banks
# ---------------------------------------------------------------------------

class FIFOMemoryBank:
    """FIFO memory bank with get_all interface."""
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


# ---------------------------------------------------------------------------
# Recency-Weighted Soft Retriever (learnable decay)
# ---------------------------------------------------------------------------

class RecencyWeightedRetriever(nn.Module):
    """Soft retrieval: all entries weighted by motion relevance + learnable recency decay."""

    def __init__(self, motion_dim: int = 256, init_decay: float = 1.0):
        super().__init__()
        self.motion_dim = motion_dim
        # log_decay ensures decay is always positive via exp()
        self.log_decay = nn.Parameter(torch.tensor(float(init_decay)).log())

    @property
    def decay(self) -> torch.Tensor:
        return torch.exp(self.log_decay)

    def forward(
        self,
        query_motion: torch.Tensor,
        memory_bank: FIFOMemoryBank,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            query_motion: [B, motion_dim]
            memory_bank: FIFOMemoryBank instance

        Returns:
            weighted_vlm: [B, max_size, S, D] — weighted VLM features
            weights: [B, max_size] — per-entry weights
        """
        if memory_bank.size == 0:
            return None, None

        stored_vlm = memory_bank.get_all_vlm()      # [B, K, S, D]
        stored_motion = memory_bank.get_all_motion()  # [B, K, motion_dim]
        B, K = stored_motion.shape[:2]

        # 1. Motion relevance
        motion_scores = F.cosine_similarity(
            query_motion.unsqueeze(1), stored_motion, dim=-1,
        )  # [B, K]

        # 2. Recency bias
        timestamps = torch.tensor(
            memory_bank.timestamps, dtype=torch.float32, device=query_motion.device,
        ).unsqueeze(0).expand(B, -1)  # [B, K]
        current_time = timestamps.max(dim=-1, keepdim=True).values
        age = current_time - timestamps  # [B, K]
        recency_bias = -self.decay * age

        # 3. Combined → softmax weights
        combined = motion_scores + recency_bias
        weights = F.softmax(combined, dim=-1)  # [B, K]

        # 4. Weighted features
        weighted_vlm = stored_vlm * weights.unsqueeze(-1).unsqueeze(-1)

        return weighted_vlm, weights


# ---------------------------------------------------------------------------
# Cross-Attention Memory Injector (same as run_phase3_experiments.py)
# ---------------------------------------------------------------------------

class CrossAttentionMemoryInjector(nn.Module):
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
        B, S, D = und_tokens.shape
        # Flatten memory: [B, K, S_mem, D] → [B, K*S_mem, D]
        # S_mem may differ from S (und_tokens seq_len), so flatten dynamically
        memory_flat = memory_features.reshape(B, -1, D)
        M = memory_flat.shape[1]  # total memory tokens = K * S_mem

        q = self.norm_q(und_tokens.float())
        memory_normed = self.norm_memory(memory_flat.float())

        Q = self.q_proj(q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = self.k_proj(memory_normed).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(memory_normed).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(Q, K_proj, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_output)

        return und_tokens + torch.sigmoid(self.gate) * output


# ---------------------------------------------------------------------------
# VLM inputs helper
# ---------------------------------------------------------------------------

def _create_vlm_inputs_for_frame(model, frame, batch, device, vlm_processor=None):
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
# Top-k retrieval (baseline)
# ---------------------------------------------------------------------------

def retrieve_memory_topk(
    query_motion, memory_bank, top_k=5,
):
    if memory_bank.size == 0:
        return None
    stored_vlm = memory_bank.get_all_vlm()
    stored_motion = memory_bank.get_all_motion()
    scores = F.cosine_similarity(query_motion.unsqueeze(1), stored_motion, dim=-1)
    actual_k = min(top_k, memory_bank.size)
    _, top_k_indices = scores.topk(actual_k, dim=1)
    retrieved_vlm = torch.gather(
        stored_vlm, 1,
        top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, stored_vlm.shape[2], stored_vlm.shape[3]),
    )
    return retrieved_vlm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_action_mse(pred_actions, gt_actions):
    return {"action_mse": F.mse_loss(pred_actions.float(), gt_actions.float()).item()}


def compute_video_mse(pred_frames, gt_frames):
    return {"video_mse": F.mse_loss(pred_frames.float(), gt_frames.float()).item()}


def evaluate_with_strategy(
    model, dataloader, num_eval_batches, num_inference_steps, device,
    memory_bank, retrieval_mode, top_k, injector, vlm_processor,
    motion_extractor, retriever=None,
):
    """Evaluate with either top-k hard or recency-weighted soft retrieval.

    If retriever is provided, uses soft retrieval; otherwise uses top-k hard.
    """
    from model_patches import apply_patches, patched_inference_step_with_memory
    apply_patches()

    action_agg = []
    all_weights = []  # collect weights for analysis

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

            # Build memory bank
            memory_bank.clear()
            for t in range(T):
                hist_frame = video_frames[:, t]
                hist_vlm_inputs = _create_vlm_inputs_for_frame(
                    model, hist_frame, batch, device, vlm_processor=vlm_processor
                )
                hist_und = model.und_module.extract_und_features(hist_vlm_inputs)
                if t == 0:
                    motion_feat = torch.zeros(B, motion_extractor.feature_dim, device=device)
                else:
                    motion_feat = motion_extractor.extract(video_frames[:, t], video_frames[:, t-1])
                memory_bank.add(hist_und, motion_feat, timestamp=t)

            query_motion = torch.zeros(B, motion_extractor.feature_dim, device=device)

            # Retrieve: soft or hard
            if retriever is not None:
                retrieved_vlm, weights = retriever(query_motion, memory_bank)
                if weights is not None:
                    all_weights.append(weights.detach().cpu())
            else:
                retrieved_vlm = retrieve_memory_topk(query_motion, memory_bank, top_k=top_k)

            pred_frames, pred_actions = patched_inference_step_with_memory(
                model, first_frame, state, num_inference_steps,
                language_embeddings, vlm_inputs,
                first_frame_und_tokens=first_frame_und_tokens,
                memory_vlm_features=retrieved_vlm,
                memory_injector=injector,
                injection_mode="cross_attn",
            )

        action_metrics = compute_action_mse(pred_actions, gt_actions)
        action_agg.append(action_metrics)

    avg_action_mse = np.mean([m["action_mse"] for m in action_agg])

    # Aggregate weights for analysis
    weight_stats = {}
    if all_weights:
        all_w = torch.cat(all_weights, dim=0)  # [N, K]
        weight_stats = {
            "mean_weights": all_w.mean(dim=0).tolist(),
            "std_weights": all_w.std(dim=0).tolist(),
            "entropy": -(all_w * (all_w + 1e-8).log()).sum(dim=-1).mean().item(),
        }

    return {
        "action_mse": float(avg_action_mse),
        "weight_stats": weight_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, dataloader, config, dataset = load_model_and_data(
        args.config, args.checkpoint, device,
    )
    vlm_processor = getattr(dataset, 'vlm_processor', None)

    motion_extractor = SimpleMotionExtractor(device=device)
    injector = CrossAttentionMemoryInjector(und_dim=512).to(device)

    results = {}

    # --- Baseline: FIFO + top-k hard ---
    logger.info("=" * 60)
    logger.info("Baseline: FIFO + top-k hard selection")
    logger.info("=" * 60)
    mem_bank = FIFOMemoryBank(max_size=args.bank_size, device=device)
    baseline_result = evaluate_with_strategy(
        model, dataloader, args.num_eval_batches, args.num_inference_steps, device,
        mem_bank, "motion", args.top_k, injector, vlm_processor, motion_extractor,
        retriever=None,
    )
    results["fifo_topk"] = baseline_result
    logger.info(f"  Action MSE: {baseline_result['action_mse']:.6f}")

    # --- Recency-Weighted Soft (different init_decay values) ---
    for init_decay in [0.1, 1.0, 10.0]:
        logger.info("=" * 60)
        logger.info(f"Recency-Weighted Soft: init_decay={init_decay}")
        logger.info("=" * 60)

        mem_bank = FIFOMemoryBank(max_size=args.bank_size, device=device)
        retriever = RecencyWeightedRetriever(
            motion_dim=motion_extractor.feature_dim, init_decay=init_decay,
        ).to(device)

        result = evaluate_with_strategy(
            model, dataloader, args.num_eval_batches, args.num_inference_steps, device,
            mem_bank, "motion", args.top_k, injector, vlm_processor, motion_extractor,
            retriever=retriever,
        )

        result["init_decay"] = init_decay
        result["learned_decay"] = retriever.decay.item()
        results[f"recency_soft_d{init_decay}"] = result

        logger.info(f"  Action MSE: {result['action_mse']:.6f}")
        logger.info(f"  Learned decay: {result['learned_decay']:.4f}")
        if result["weight_stats"]:
            logger.info(f"  Mean weights: {result['weight_stats']['mean_weights']}")
            logger.info(f"  Weight entropy: {result['weight_stats']['entropy']:.4f}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, res in results.items():
        decay_info = f", learned_decay={res.get('learned_decay', 'N/A')}" if 'learned_decay' in res else ""
        logger.info(f"  {name}: action_mse={res['action_mse']:.6f}{decay_info}")

    # Save results
    results_path = os.path.join(args.output_dir, "recency_weighted_results.json")
    # Convert tensors to serializable format
    serializable = {}
    for name, res in results.items():
        r = {"action_mse": res["action_mse"]}
        if "learned_decay" in res:
            r["learned_decay"] = res["learned_decay"]
        if "init_decay" in res:
            r["init_decay"] = res["init_decay"]
        if res.get("weight_stats"):
            r["weight_stats"] = res["weight_stats"]
        serializable[name] = r

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
