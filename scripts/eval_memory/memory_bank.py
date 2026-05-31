"""
Motion-Aware Temporal Memory Bank for Phase 2 experiments.

Core components:
  - OpticalFlowExtractor: computes motion features from frame differences
  - MemoryBank: stores and retrieves historical VLM features
  - MemoryRetriever: retrieves relevant memory entries (visual/motion/hybrid)
  - MemoryInjector: injects retrieved memory into Understanding Expert via cross-attention

All components are training-free (no gradient, no parameter updates).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Literal


# ============================================================================
# 1. Motion Feature Extractor (frame-difference based)
# ============================================================================

class MotionFeatureExtractor(nn.Module):
    """Extract motion features from consecutive frames using frame differences.

    Instead of optical flow (which requires cv2/kornia), we use:
      1. L2 frame difference: ||I_t - I_{t-1}||_2
      2. A small CNN to encode the difference into a compact feature vector

    This aligns with the diagnostic experiment's "temporal novelty" metric.
    """

    def __init__(self, input_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # Simple CNN encoder for frame difference
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
        )

        # Initialize with small weights (inference only, but good practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self, current_frame: torch.Tensor, previous_frame: torch.Tensor
    ) -> torch.Tensor:
        """Compute motion features from frame pair.

        Args:
            current_frame: [B, C, H, W] in [0, 1]
            previous_frame: [B, C, H, W] in [0, 1]

        Returns:
            motion_features: [B, feature_dim]
        """
        # Frame difference
        diff = current_frame - previous_frame  # [B, C, H, W]
        # L2 magnitude (channel-wise)
        diff_mag = torch.sqrt(diff.pow(2).mean(dim=1, keepdim=True) + 1e-8)  # [B, 1, H, W]
        # Repeat to 3 channels for CNN
        diff_input = diff_mag.repeat(1, 3, 1, 1)  # [B, 3, H, W]

        return self.encoder(diff_input)  # [B, feature_dim]


# ============================================================================
# 2. Memory Bank
# ============================================================================

class MemoryBank:
    """Fixed-size memory bank storing historical frame features.

    Each entry contains:
      - vlm_features: VLM output features (for injection into Understanding Expert)
      - motion_features: motion/flow features (for retrieval)
      - timestamp: frame index in the episode
    """

    def __init__(self, max_size: int = 20, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.clear()

    def clear(self):
        """Reset the memory bank."""
        self.vlm_features: List[torch.Tensor] = []  # list of [B, seq_len, und_dim]
        self.motion_features: List[torch.Tensor] = []  # list of [B, motion_dim]
        self.timestamps: List[int] = []

    @property
    def size(self) -> int:
        return len(self.vlm_features)

    def add(
        self,
        vlm_features: torch.Tensor,
        motion_features: torch.Tensor,
        timestamp: int,
    ):
        """Add a new entry to the memory bank.

        If the bank is full, the oldest entry is removed (FIFO).
        """
        if self.size >= self.max_size:
            # Remove oldest
            self.vlm_features.pop(0)
            self.motion_features.pop(0)
            self.timestamps.pop(0)

        self.vlm_features.append(vlm_features.detach().clone())
        self.motion_features.append(motion_features.detach().clone())
        self.timestamps.append(timestamp)

    def get_all_vlm_features(self) -> torch.Tensor:
        """Get all stored VLM features stacked.

        Returns:
            [B, max_size, seq_len, und_dim] (padded with zeros if not full)
        """
        if self.size == 0:
            return None

        # Get dimensions from first entry
        B, seq_len, und_dim = self.vlm_features[0].shape

        # Pad to max_size
        features = torch.zeros(
            B, self.max_size, seq_len, und_dim,
            device=self.device, dtype=self.vlm_features[0].dtype,
        )
        for i, feat in enumerate(self.vlm_features):
            features[:, i] = feat

        return features  # [B, max_size, seq_len, und_dim]

    def get_all_motion_features(self) -> torch.Tensor:
        """Get all stored motion features stacked.

        Returns:
            [B, max_size, motion_dim] (padded with zeros if not full)
        """
        if self.size == 0:
            return None

        B, motion_dim = self.motion_features[0].shape

        features = torch.zeros(
            B, self.max_size, motion_dim,
            device=self.device, dtype=self.motion_features[0].dtype,
        )
        for i, feat in enumerate(self.motion_features):
            features[:, i] = feat

        return features  # [B, max_size, motion_dim]


# ============================================================================
# 3. Memory Retriever
# ============================================================================

class MemoryRetriever(nn.Module):
    """Retrieve relevant memory entries using different similarity signals.

    Supports three retrieval modes:
      - "visual": cosine similarity on VLM features
      - "motion": cosine similarity on motion/flow features
      - "hybrid": weighted combination of visual and motion similarity
    """

    def __init__(
        self,
        retrieval_mode: Literal["visual", "motion", "hybrid"] = "motion",
        hybrid_alpha: float = 0.5,
    ):
        super().__init__()
        self.retrieval_mode = retrieval_mode
        self.hybrid_alpha = hybrid_alpha  # weight for motion in hybrid mode

    def forward(
        self,
        query_motion: torch.Tensor,
        query_vlm: Optional[torch.Tensor],
        memory_bank: MemoryBank,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k relevant memory entries.

        Args:
            query_motion: [B, motion_dim] current frame motion features
            query_vlm: [B, seq_len, und_dim] current frame VLM features (for visual retrieval)
            memory_bank: MemoryBank instance
            top_k: number of entries to retrieve

        Returns:
            retrieved_vlm: [B, top_k, seq_len, und_dim] retrieved VLM features
            scores: [B, max_size] similarity scores (for analysis)
        """
        if memory_bank.size == 0:
            return None, None

        # Get stored features
        stored_vlm = memory_bank.get_all_vlm_features()  # [B, N, seq_len, und_dim]
        stored_motion = memory_bank.get_all_motion_features()  # [B, N, motion_dim]
        B, N = stored_motion.shape[:2]

        # Compute similarity scores
        if self.retrieval_mode == "visual" and query_vlm is not None:
            # Visual similarity: average VLM features then cosine
            query_pooled = query_vlm.mean(dim=1)  # [B, und_dim]
            stored_pooled = stored_vlm.mean(dim=2)  # [B, N, und_dim]
            scores = F.cosine_similarity(
                query_pooled.unsqueeze(1), stored_pooled, dim=-1
            )  # [B, N]

        elif self.retrieval_mode == "motion":
            # Motion similarity: cosine on motion features
            scores = F.cosine_similarity(
                query_motion.unsqueeze(1), stored_motion, dim=-1
            )  # [B, N]

        elif self.retrieval_mode == "hybrid":
            # Hybrid: weighted combination
            # Motion similarity
            motion_scores = F.cosine_similarity(
                query_motion.unsqueeze(1), stored_motion, dim=-1
            )  # [B, N]

            # Visual similarity
            if query_vlm is not None:
                query_pooled = query_vlm.mean(dim=1)  # [B, und_dim]
                stored_pooled = stored_vlm.mean(dim=2)  # [B, N, und_dim]
                visual_scores = F.cosine_similarity(
                    query_pooled.unsqueeze(1), stored_pooled, dim=-1
                )  # [B, N]
            else:
                visual_scores = torch.zeros_like(motion_scores)

            scores = (
                self.hybrid_alpha * motion_scores
                + (1 - self.hybrid_alpha) * visual_scores
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

        # Clamp top_k to available entries
        actual_k = min(top_k, memory_bank.size)

        # Top-k retrieval
        _, top_k_indices = scores.topk(actual_k, dim=1)  # [B, actual_k]

        # Gather retrieved VLM features
        retrieved_vlm = torch.gather(
            stored_vlm, 1,
            top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, stored_vlm.shape[2], stored_vlm.shape[3]),
        )  # [B, actual_k, seq_len, und_dim]

        return retrieved_vlm, scores


# ============================================================================
# 3b. Recency-Weighted Soft Retriever
# ============================================================================

class RecencyWeightedRetriever(nn.Module):
    """Recency-weighted soft retrieval: all entries participate in attention,
    weighted by motion relevance + learnable time decay.

    Key difference from top-k hard selection:
      - top-k: selects k most relevant, discards the rest → information loss
      - soft: all entries weighted by relevance + recency → preserves all info,
              but recent/relevant entries contribute more

    The learnable decay parameter lets the model discover the optimal
    "how old is too old" timescale automatically.
    """

    def __init__(self, motion_dim: int = 256, learnable_decay: bool = True,
                 init_decay: float = 1.0):
        """
        Args:
            motion_dim: dimension of motion features
            learnable_decay: if True, decay is a learnable parameter
            init_decay: initial decay rate (higher = more recency-biased)
                - 0.1: strongly recency-biased (only recent frames matter)
                - 1.0: balanced (default)
                - 10.0: weakly recency-biased (all frames roughly equal)
        """
        super().__init__()
        self.motion_dim = motion_dim

        if learnable_decay:
            # log_decay ensures decay is always positive via exp()
            self.log_decay = nn.Parameter(torch.tensor(float(init_decay)).log())
        else:
            self.register_buffer('log_decay', torch.tensor(float(init_decay)).log())

    @property
    def decay(self) -> torch.Tensor:
        """Current decay rate (always positive)."""
        return torch.exp(self.log_decay)

    def forward(
        self,
        query_motion: torch.Tensor,
        memory_bank: 'MemoryBank',
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Soft retrieval with recency weighting.

        Args:
            query_motion: [B, motion_dim] current frame motion features
            memory_bank: MemoryBank instance

        Returns:
            weighted_vlm: [B, max_size, seq_len, und_dim] — weighted VLM features
                          (all entries, weighted by relevance + recency)
            weights: [B, max_size] — per-entry weights (for analysis/visualization)
        """
        if memory_bank.size == 0:
            return None, None

        stored_vlm = memory_bank.get_all_vlm_features()      # [B, K, S, D]
        stored_motion = memory_bank.get_all_motion_features()  # [B, K, motion_dim]
        B, K = stored_motion.shape[:2]

        # 1. Motion relevance: cosine similarity between query and stored motions
        motion_scores = F.cosine_similarity(
            query_motion.unsqueeze(1),  # [B, 1, motion_dim]
            stored_motion,               # [B, K, motion_dim]
            dim=-1,
        )  # [B, K]

        # 2. Recency bias: recent frames get higher weight
        #    timestamps are stored as a list; convert to tensor
        timestamps = torch.tensor(
            memory_bank.timestamps, dtype=torch.float32, device=query_motion.device,
        )  # [K]
        timestamps = timestamps.unsqueeze(0).expand(B, -1)  # [B, K]

        current_time = timestamps.max(dim=-1, keepdim=True).values  # [B, 1]
        age = current_time - timestamps  # [B, K] — 0 for newest, larger for older

        decay = self.decay  # scalar, always positive
        recency_bias = -decay * age  # [B, K] — more negative for older entries

        # 3. Combined score = motion relevance + recency bias
        combined = motion_scores + recency_bias  # [B, K]

        # 4. Softmax → weights (all entries contribute, but weighted)
        weights = F.softmax(combined, dim=-1)  # [B, K]

        # 5. Weighted VLM features
        weighted_vlm = stored_vlm * weights.unsqueeze(-1).unsqueeze(-1)
        # [B, K, S, D] * [B, K, 1, 1] → [B, K, S, D]

        return weighted_vlm, weights


# ============================================================================
# 4. Memory Injector (Cross-Attention)
# ============================================================================

class MemoryInjector(nn.Module):
    """Inject retrieved memory into Understanding Expert via cross-attention.

    Architecture:
      - Understanding tokens serve as Query
      - Retrieved memory VLM features serve as Key and Value
      - Output is added as a residual to the understanding tokens

    This module is designed to be lightweight (few parameters) and
    inserted at each MoT layer.
    """

    def __init__(self, und_dim: int = 512, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.und_dim = und_dim
        self.num_heads = num_heads
        self.head_dim = und_dim // num_heads

        assert und_dim % num_heads == 0, f"und_dim ({und_dim}) must be divisible by num_heads ({num_heads})"

        # Cross-attention projections
        self.q_proj = nn.Linear(und_dim, und_dim)
        self.k_proj = nn.Linear(und_dim, und_dim)
        self.v_proj = nn.Linear(und_dim, und_dim)
        self.out_proj = nn.Linear(und_dim, und_dim)

        # Layer norm for stability
        self.norm_q = nn.LayerNorm(und_dim)
        self.norm_memory = nn.LayerNorm(und_dim)

        # Gating mechanism (learnable gate for residual)
        self.gate = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        und_tokens: torch.Tensor,
        memory_features: torch.Tensor,
    ) -> torch.Tensor:
        """Inject memory into understanding tokens via cross-attention.

        Args:
            und_tokens: [B, seq_len, und_dim] current understanding tokens
            memory_features: [B, K, seq_len, und_dim] retrieved memory
                (K = number of retrieved entries, each with seq_len tokens)

        Returns:
            enhanced_und_tokens: [B, seq_len, und_dim]
        """
        B, S, D = und_tokens.shape
        K = memory_features.shape[1]

        # Flatten memory: [B, K, S, D] -> [B, K*S, D]
        memory_flat = memory_features.reshape(B, K * S, D)

        # Normalize
        q = self.norm_q(und_tokens)
        memory_normed = self.norm_memory(memory_flat)

        # Project
        Q = self.q_proj(q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]
        K_proj = self.k_proj(memory_normed).view(B, K * S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K*S, head_dim]
        V = self.v_proj(memory_normed).view(B, K * S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K*S, head_dim]

        # Cross-attention
        attn_output = F.scaled_dot_product_attention(
            Q, K_proj, V,
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # [B, H, S, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D]
        output = self.out_proj(attn_output)  # [B, S, D]

        # Gated residual
        enhanced = und_tokens + torch.sigmoid(self.gate) * output

        return enhanced


# ============================================================================
# 5. Convenience: extract motion features from a batch of frames
# ============================================================================

@torch.no_grad()
def extract_motion_features_batch(
    frames: torch.Tensor,
    motion_extractor: MotionFeatureExtractor,
) -> torch.Tensor:
    """Extract motion features from a sequence of frames.

    Args:
        frames: [B, T, C, H, W] video frames in [0, 1]
        motion_extractor: MotionFeatureExtractor instance

    Returns:
        motion_features: [B, T, feature_dim] (first frame has zero motion)
    """
    B, T, C, H, W = frames.shape
    device = frames.device
    dtype = frames.dtype

    motion_features = torch.zeros(
        B, T, motion_extractor.feature_dim,
        device=device, dtype=dtype,
    )

    for t in range(1, T):
        motion_features[:, t] = motion_extractor(
            frames[:, t], frames[:, t - 1]
        )

    return motion_features  # [B, T, feature_dim]


# ============================================================================
# 6. Convenience: extract VLM features for multiple frames
# ============================================================================

@torch.no_grad()
def extract_vlm_features_batch(
    frames: List[torch.Tensor],
    vlm_model,
    und_expert,
) -> torch.Tensor:
    """Extract VLM features for a list of frames.

    Args:
        frames: list of [B, C, H, W] frames in [0, 1]
        vlm_model: frozen VLM model
        und_expert: Understanding Expert with vlm_adapter

    Returns:
        vlm_features: [B, num_frames, seq_len, und_dim]
    """
    features_list = []
    for frame in frames:
        # This requires vlm_inputs, so we need a different approach
        # For probing, we'll pre-extract and store in memory bank
        pass

    return torch.stack(features_list, dim=1) if features_list else None
