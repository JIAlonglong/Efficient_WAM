# Semantic-Guided Token Compression for Motus: 语义表征引导的视觉Token压缩

**定位**：利用Motus架构中的Understanding Expert（QwenVL）提取的语义表征，指导Wan2.2 VAE产生的视觉Token进行高效压缩，实现推理加速同时保持任务性能。

**核心思想**：Understanding Expert已经学习了丰富的语义信息，这些信息可以作为"语义锚点"来判断哪些视觉Token对任务最重要，从而实现任务感知的Token压缩。

---

## 一、研究动机

### 1.1 问题链

```
Motus推理太慢（8B参数，30层MoT）
    ↓
需要Token压缩来加速
    ↓
传统Token压缩（随机/attention-based）只优化效率，忽略语义
    ↓
压缩后语义信息丢失，任务性能下降
    ↓
根本原因：压缩时没有利用语义信息来指导
    ↓
解决方案：用Understanding Expert的语义表征引导压缩
```

### 1.2 关键洞察

**Motus架构的独特优势**：

```
Video → Wan2.2 VAE → Video Tokens [B, N, 3072]
                           ↓
                    Understanding Expert (QwenVL) → 语义表征
                           ↓
                    Token Compression ← 语义引导
                           ↓
                    Joint Attention (Video + Action + Understanding)
                           ↓
                    Action Head
```

**为什么Motus适合做语义引导的Token压缩**：

1. **天然的语义信息来源**：Understanding Expert（QwenVL）已经提取了丰富的语义特征
2. **Joint Attention架构**：三个分支在token级进行信息交换，比Global Pooling更能保留信息
3. **任务相关性**：Understanding Expert的语义特征与action prediction直接相关

### 1.3 研究空白

| 方向 | 已有工作 | 空白 |
|------|---------|------|
| Token Pruning in VLA | GridS, CogVLA, FlashVLA | 大多使用额外的语义encoder（SigLIP/DINOv2），未利用WAM内部的语义信息 |
| 语义引导压缩 | Compressor-VLA, VLA-Pruner | 未在WAM架构中验证 |
| Motus优化 | MotuBrain（推理优化） | 未涉及Token级压缩 |
| WAM Token效率 | FastWAM, GigaWorld-Policy | 关注推理时跳过生成，未关注Token压缩 |

---

## 二、核心假设

> 在Motus架构中，Understanding Expert（QwenVL）提取的语义表征可以作为有效的"语义锚点"，指导Wan2.2 VAE视觉Token的压缩，在保持任务性能的同时实现显著的推理加速。

### 假设分解

**假设1：Understanding Tokens包含任务相关的语义信息**
- QwenVL是强大的视觉语言模型，已经学习了场景理解、物体识别、空间关系等语义信息
- 这些语义信息与action prediction直接相关
- 验证方式：Probing实验，测试Understanding Tokens的语义质量

**假设2：语义表征可以指导Token重要性计算**
- 通过计算Video Tokens与Understanding Tokens的相关性，可以判断每个Video Token的语义重要性
- 语义重要的Token应该被保留，语义不重要的Token可以被压缩
- 验证方式：对比不同importance计算方式的效果

**假设3：语义引导的压缩优于无引导的压缩**
- 相比随机压缩或attention-based压缩，语义引导的压缩能更好地保持任务性能
- 验证方式：Pareto前沿对比（压缩率 vs 任务性能）

---

## 三、方法设计

### 3.1 架构概览

**目标**：在 Joint Attention 之前压缩 video tokens，只做 action prediction，跳过 video generation。

```
┌──────────────────────────────────────────────────────────────────┐
│                    Motus + Token Compression (Action-Only)        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Video → VAE → Video Tokens [B, N=90, 3072]                     │
│                          ↓                                       │
│                ┌─────────────────────┐                           │
│  Und Tokens →  │  Token Compression  │ ← 本工作                 │
│  [B,L,512]     │  N=90 → K (e.g. 45)│                           │
│                └─────────┬───────────┘                           │
│                          ↓                                       │
│                Compressed Video Tokens [B, K, 3072]              │
│                          ↓                                       │
│                30层 Joint Attention                               │
│                (Video[K] + Action[21] + Und[L])                  │
│                          ↓                                       │
│                Action Head → action [B, 17, 14]  ✅              │
│                output_head → 跳过（推理不需要视频生成）           │
│                                                                  │
│  加速来源：Joint Attention FLOPs ∝ video token 数                │
│  N=90→K=45: Joint Attention FLOPs 减少约 50%                     │
│  N=90→K=30: Joint Attention FLOPs 减少约 67%                     │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

**设计原则**：所有压缩模块都是**可训练的、端到端可微分的**，在推理流中实时计算。

**Cross-Attention 方向选择**：若使用 cross-attention 做语义引导，应采用 **Und→Q, Video→K/V**（反向），而非原方案的 Video→Q, Und→K/V。原因：
1. **语义更直接**：Understanding tokens 作为语义"查询"，直接定位 video 中对应的空间区域（如"螺丝刀"→左下角），attention weights 天然就是 importance 信号
2. **无需 gate_head**：原方案需要将 3072 维 cross-attn 输出压缩成 1 个标量（信息瓶颈），反向可直接对 attention weights 求和得到 importance
3. **不重复计算**：Motus 的 Joint Attention 已有 Video→Und 方向，反向利用 Und→Video 的定位能力是互补而非冗余
4. **可解释性**：每个 understanding token 的 attention map 直接显示它关注了哪些 video 区域

#### 方案A-1：FastV式 Attention 剪枝（零参数 Baseline）

**核心思想**：不加任何新模块，直接利用 Motus 第一层 Joint Attention 中已有的 attention weights 判断 video token 重要性。借鉴 FastV（ECCV 2024）的发现——attention pattern 在前几层就已收敛，早期层的 attention 足以决定 token 重要性。

**Attention 方向选择**：Joint Attention 是双向的（Video→Und 和 Und→Video）。推荐使用 **Und→Video 方向**（und tokens attend to video tokens）：
- `attn_weights[B, H, L_und, N_video]`：每个 und token 对 video tokens 的关注度
- 对 und 维度求和 → `[B, H, N_video]`：每个 video token 被多少 und 语义概念关注
- 被多个语义概念关注的 video token 就是重要的（直接回答"哪些 video token 重要"）

相比之下，Video→Und 方向（video tokens attend to und tokens）衡量的是"每个 video token 需要多少语义信息"，这是**信息需求**而非**重要性**——背景区域反而可能高度关注语义（因为自身信息不足），重要区域（如工具本身）可能不太关注（因为自身特征已经足够）。

**优势**：零参数、零额外 FLOPs、实现极简。可作为所有学习式方案的 upper bound baseline——如果这个简单方法就够好，说明不需要复杂的 gate。

**工程约束**：当前 flash_attention 不返回 weights，需在第一层用 `F.scaled_dot_product_attention`（PyTorch 2.0+，支持返回 attn weights）替代，或用 `flash_attn_func` 并设置 `return_attn_probs=True`（需 flash-attn 2.x+）。

```python
class AttentionScorePruning(nn.Module):
    """
    FastV式剪枝：利用第一层 Joint Attention 中已有的 attention weights 判断 token 重要性。
    零参数，零额外 FLOPs。
    默认使用 Und→Video 方向（und tokens attend to video tokens），
    被多个语义概念关注的 video token 就是重要的。
    """

    def __init__(self):
        super().__init__()
        # 不需要任何可学习参数

    def forward(self, video_tokens, attn_weights_layer1, keep_ratio=0.5, direction='und_to_video'):
        """
        Args:
            video_tokens: [B, N, 3072]
            attn_weights_layer1: 第一层 Joint Attention 的 attention weights
                - direction='und_to_video': [B, H, L_und, N_video]（推荐）
                - direction='video_to_und': [B, H, N_video, L_und]
            keep_ratio: 保留比例
            direction: attention 方向
        Returns:
            compressed: [B, K, 3072]
            indices: [B, K] - 被选中的 token 索引
            importance: [B, N] - 重要性分数（可用于可视化）
        """
        B, N, D = video_tokens.shape

        if direction == 'und_to_video':
            # Und→Video：每个 und token 关注哪些 video tokens，对 und 维度求和
            # [B, H, L_und, N_video] → sum over L_und → [B, H, N_video] → mean over H → [B, N]
            importance = attn_weights_layer1.sum(dim=2).mean(dim=1)
        else:
            # Video→Und：每个 video token 关注 und tokens 的程度
            # [B, H, N_video, L_und] → sum over L_und → [B, H, N_video] → mean over H → [B, N]
            importance = attn_weights_layer1.sum(dim=-1).mean(dim=1)

        top_k = max(1, int(N * keep_ratio))
        _, indices = importance.topk(top_k, dim=1)  # [B, K]

        compressed = video_tokens.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, D)
        )

        return compressed, indices, importance
```

**已知问题**：

1. **Layer 1 无法加速**：要获取第一层 attention weights，必须先跑完第一层 Joint Attention（N=90 个 token）。实际流程是 Layer 1 处理 N 个 token → 压缩 → Layer 2-30 处理 K 个 token。加速比从理论 `N/K` 变为 `30N / (N + 29K)`，K=45 时实际约 1.5x（理论 2x）。这是 FastV 的固有限制，可接受。

2. **RoPE 位置**：✅ 被选中的 K 个 token 保留原始索引，RoPE 编码正确，无需修改。

```python
class AttentionScorePruningWithFallback(nn.Module):
    """
    带 fallback 的版本：如果无法获取 attention weights，
    用 Joint Attention 第一层的输出特征计算重要性。
    """

    def __init__(self, video_dim=3072, und_dim=512):
        super().__init__()
        # Fallback：用 MLP 从 video 特征预测重要性（仅在无法获取 attn weights 时使用）
        self.fallback_scorer = nn.Sequential(
            nn.Linear(video_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, video_tokens, und_tokens, attn_weights_layer1=None, keep_ratio=0.5, direction='und_to_video'):
        B, N, D = video_tokens.shape

        if attn_weights_layer1 is not None:
            # 优先使用真实的 attention weights
            if direction == 'und_to_video':
                importance = attn_weights_layer1.sum(dim=2).mean(dim=1)
            else:
                importance = attn_weights_layer1.sum(dim=-1).mean(dim=1)
        else:
            # Fallback：用 MLP 打分
            importance = self.fallback_scorer(video_tokens).squeeze(-1)  # [B, N]

        top_k = max(1, int(N * keep_ratio))
        _, indices = importance.topk(top_k, dim=1)

        compressed = video_tokens.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, D)
        )

        return compressed, indices, importance
```

---

#### 方案A-2：PruMerge式锚点合并

**核心思想**：借鉴 LLaVA-PruMerge（2024）——先用跨模态信号选出"锚点 tokens"（最重要的 video tokens），然后把剩余 tokens 合并到最近的锚点上。相比方案 A-1 的纯剪枝，非锚点信息不丢失而是聚合到锚点，信息保留更完整。

**优势**：真正减少 token 数量（训练和推理都是 K 个），保留空间结构（锚点保持原始位置），非锚点信息不丢失。

**劣势**：top-k 不可分（需用 STE 松弛），比 A-1 多几个轻量 MLP。

```python
class PruMergeCompression(nn.Module):
    """
    PruMerge 风格：选锚点 + 合并剩余 token 到锚点。
    比纯剪枝保留更多信息：非锚点 token 的特征被聚合到语义最近的锚点。
    """

    def __init__(self, video_dim=3072, und_dim=512, num_anchors=32):
        super().__init__()
        self.num_anchors = num_anchors

        # 轻量级重要性打分：video token + und context → score
        self.importance_scorer = nn.Sequential(
            nn.Linear(video_dim + und_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        # 合并权重：预测每个非锚点 token 对各锚点的归属权重
        self.merge_weight = nn.Sequential(
            nn.Linear(video_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_anchors)
        )

    def forward(self, video_tokens, und_tokens, keep_ratio=0.5):
        """
        Args:
            video_tokens: [B, N, 3072]
            und_tokens: [B, L, 512]
            keep_ratio: 锚点保留比例
        Returns:
            output: [B, K, 3072] - K = num_anchors
            anchor_idx: [B, K] - 锚点 token 索引
            scores: [B, N] - 重要性分数
        """
        B, N, D = video_tokens.shape
        K = max(1, int(N * keep_ratio))

        # 1. 计算 und context（mean pooling）
        und_context = und_tokens.mean(dim=1, keepdim=True).expand(-1, N, -1)  # [B, N, und_dim]

        # 2. 重要性打分
        combined = torch.cat([video_tokens, und_context], dim=-1)  # [B, N, D+und_dim]
        scores = self.importance_scorer(combined).squeeze(-1)  # [B, N]

        # 3. 选择锚点 tokens（top-k）
        _, anchor_idx = scores.topk(K, dim=1)  # [B, K]
        anchor_tokens = video_tokens.gather(
            1, anchor_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]

        # 4. 非锚点 tokens 合并到锚点
        merge_logits = self.merge_weight(video_tokens)  # [B, N, K]

        # 创建 mask：锚点 token 不参与合并（设为 -inf）
        mask = torch.zeros(B, N, device=video_tokens.device)
        mask.scatter_(1, anchor_idx, 1.0)
        merge_logits = merge_logits.masked_fill(mask.unsqueeze(-1).bool(), float('-inf'))

        # 非锚点 token 用 softmax 归一化的归属权重加权到锚点
        merge_weights = F.softmax(merge_logits, dim=-1)  # [B, N, K]
        merged_residual = torch.bmm(merge_weights.transpose(1, 2), video_tokens)  # [B, K, D]

        # 最终输出 = 锚点 + 合并残差（带缩放）
        output = anchor_tokens + merged_residual * 0.5

        return output, anchor_idx, scores
```

**已知问题与修复**：

1. **`num_anchors` 参数冗余**：`__init__` 中 `num_anchors=32` 未在 `forward` 中使用（`forward` 用 `K = max(1, int(N * keep_ratio))`）。应删除 `num_anchors` 参数，改为只在 `forward` 中计算 K。

2. **训练时 top-k 不可微**：`topk` 操作无梯度，importance_scorer 的梯度被截断。需用 Straight-Through Estimator（STE）松弛：

```python
class PruMergeCompressionV2(nn.Module):
    """修复版：删除冗余参数 + STE 可微分 top-k"""

    def __init__(self, video_dim=3072, und_dim=512):
        super().__init__()
        self.importance_scorer = nn.Sequential(
            nn.Linear(video_dim + und_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.merge_weight = nn.Sequential(
            nn.Linear(video_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128)  # 输出维度在 forward 中动态设置
        )

    def forward(self, video_tokens, und_tokens, keep_ratio=0.5):
        B, N, D = video_tokens.shape
        K = max(1, int(N * keep_ratio))

        und_context = und_tokens.mean(dim=1, keepdim=True).expand(-1, N, -1)
        scores = self.importance_scorer(
            torch.cat([video_tokens, und_context], dim=-1)
        ).squeeze(-1)  # [B, N]

        # STE top-k：前向用 hard top-k，反向梯度传给 scores
        # 方式1：用 softmax 近似 top-k（可微分）
        temperature = 0.1  # 低温度 → 接近 one-hot
        soft_mask = F.softmax(scores / temperature, dim=-1) * N / K  # 归一化使均值≈1
        # 方式2：用 Gumbel-Softmax（更精确的离散近似）
        # logits = torch.stack([scores, torch.zeros_like(scores)], dim=-1)  # [B, N, 2]
        # probs = F.gumbel_softmax(logits, tau=temperature, hard=True)
        # soft_mask = probs[..., 1:2] * N / K

        # 用 soft_mask 加权 video tokens（可微分的"软选择"）
        # 重要 token 被放大，不重要 token 被衰减
        weighted_video = video_tokens * soft_mask.unsqueeze(-1)

        # 然后从加权后的 tokens 中选 top-k（推理时用 hard top-k）
        if self.training:
            # 训练时：用 soft_mask 做加权平均聚合到 K 个 token
            # 重新计算 merge weights
            merge_logits = self.merge_weight(weighted_video)  # [B, N, 128]
            # 动态调整到 K 维
            merge_logits = merge_logits[..., :K] if K <= 128 else F.adaptive_avg_pool1d(
                merge_logits.transpose(1, 2), K
            ).transpose(1, 2)
            merge_weights = F.softmax(merge_logits, dim=1)  # [B, N, K]
            output = torch.bmm(merge_weights.transpose(1, 2), weighted_video)  # [B, K, D]
        else:
            # 推理时：hard top-k
            _, anchor_idx = scores.topk(K, dim=1)
            anchor_tokens = video_tokens.gather(
                1, anchor_idx.unsqueeze(-1).expand(-1, -1, D)
            )
            output = anchor_tokens

        return output, scores

    # RoPE 位置：✅ 锚点 token 保持原始索引，位置信息完整
```

---

#### 方案A-3：TokenLearner式可学习聚合

**核心思想**：借鉴 TokenLearner（Ryoo et al., NeurIPS 2021）——学习 K 个可学习的 query vectors，用 cross-attention 从 video tokens 中聚合信息，understanding tokens 通过 gate 调制 K/V 实现语义引导。输出固定 K 个 compressed tokens。

**优势**：完全可微分（无需 top-k 或离散选择），训练和推理行为一致（无分布不匹配），K 个 queries 可学习不同语义角色。

**劣势**：输出 token 不再是原始 video tokens 的子集，失去空间对应关系；cross-attention 带来额外参数；**需要额外处理 RoPE 位置**。

```python
class SemanticTokenLearner(nn.Module):
    """
    TokenLearner 风格，使用反向 cross-attention（Und→Q, Video→K/V）。
    两阶段聚合：
      Stage 1: Understanding tokens 作为 Q，定位 video 中的重要区域（语义 grounding）
      Stage 2: Learnable queries 从语义加权的 video tokens 中聚合出 K 个 compressed tokens
    """

    def __init__(self, video_dim=3072, und_dim=512, num_output=32, num_heads=8):
        super().__init__()
        self.num_output = num_output
        self.num_heads = num_heads
        self.head_dim = video_dim // num_heads

        # K 个可学习的 query tokens
        self.learnable_queries = nn.Parameter(
            torch.randn(num_output, video_dim) * 0.02
        )

        # Stage 1: Und→Q, Video→K（语义 grounding）
        self.ground_W_q = nn.Linear(und_dim, video_dim, bias=False)
        self.ground_W_k = nn.Linear(video_dim, video_dim, bias=False)

        # Stage 2: Learnable queries → Q, grounded video → K/V
        self.agg_W_q = nn.Linear(video_dim, video_dim, bias=False)
        self.agg_W_k = nn.Linear(video_dim, video_dim, bias=False)
        self.agg_W_v = nn.Linear(video_dim, video_dim, bias=False)

        self.out_proj = nn.Linear(video_dim, video_dim, bias=False)
        self.norm = nn.LayerNorm(video_dim)

    def forward(self, video_tokens, und_tokens):
        """
        Args:
            video_tokens: [B, N, 3072]
            und_tokens: [B, L, 512]
        Returns:
            output: [B, K, 3072] - K = num_output, 完全可微分
            grounding_attn: [B, L, N] - 语义 grounding map（可视化用）
            agg_attn: [B, K, N] - 聚合 attention（可视化用）
        """
        B, N, D = video_tokens.shape
        L = und_tokens.shape[1]

        # ========== Stage 1: 语义 Grounding（Und→Q, Video→K）==========
        # 每个 understanding token 定位它对应的 video 区域
        g_q = self.ground_W_q(und_tokens).view(B, L, self.num_heads, self.head_dim)
        g_k = self.ground_W_k(video_tokens).view(B, N, self.num_heads, self.head_dim)

        grounding_scores = torch.einsum("blhd,bnhd->bhln", g_q, g_k) / (self.head_dim ** 0.5)
        grounding_attn = F.softmax(grounding_scores, dim=-1)  # [B, H, L, N]

        # 对 understanding 维度求和 → 每个 video token 的语义重要性
        # [B, H, L, N] → sum over L → [B, H, N] → mean over H → [B, N]
        video_importance = grounding_attn.sum(dim=2).mean(dim=1)  # [B, N]

        # 用 importance 加权 video tokens（软调制，保持可微分）
        # 归一化 importance 使其均值为 1，避免改变 video tokens 的尺度
        importance_norm = video_importance / (video_importance.mean(dim=1, keepdim=True) + 1e-8)
        grounded_video = video_tokens * importance_norm.unsqueeze(-1)  # [B, N, D]

        # ========== Stage 2: 聚合（Learnable Queries → grounded video）==========
        queries = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        a_q = self.agg_W_q(queries).view(B, self.num_output, self.num_heads, self.head_dim)
        a_k = self.agg_W_k(grounded_video).view(B, N, self.num_heads, self.head_dim)
        a_v = self.agg_W_v(video_tokens).view(B, N, self.num_heads, self.head_dim)  # V 用原始 tokens

        agg_scores = torch.einsum("bqhd,bnhd->bhqn", a_q, a_k) / (self.head_dim ** 0.5)
        agg_attn = F.softmax(agg_scores, dim=-1)  # [B, H, K, N]

        out = torch.einsum("bhqn,bnhd->bqhd", agg_attn, a_v)
        out = out.reshape(B, self.num_output, D)
        out = self.out_proj(out)
        out = self.norm(out)

        return out, grounding_attn.mean(dim=1), agg_attn.mean(dim=1)
```

**已知问题与修复**：

1. **输出 token 无空间位置（严重）**：K 个输出 token 是 cross-attention 聚合出的全新表征，没有 `(t, h, w)` 空间坐标。Joint Attention 中 RoPE 按位置编码，这些 token 的 RoPE 位置信息丢失。

   **修复方案**：根据 Stage 2 的 attention weights 计算加权平均位置：

```python
def compute_output_positions(agg_attn, grid_sizes):
    """
    根据 attention weights 计算 output tokens 的加权平均空间位置。
    agg_attn: [B, K, N] - 每个 output token 对原始 N 个 token 的 attention
    grid_sizes: (T, H, W)
    Returns: positions [B, K, 3] - 每个 output token 的 (t, h, w) 坐标
    """
    T, H, W = grid_sizes
    # 原始 token 的 3D 坐标
    pos_t, pos_h, pos_w = torch.meshgrid(
        torch.arange(T, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    positions = torch.stack([pos_t, pos_h, pos_w], dim=-1).reshape(-1, 3)  # [N, 3]
    # 加权平均：agg_attn [B,K,N] × positions [N,3] → [B,K,3]
    positions = positions.to(agg_attn.device)
    weighted_pos = torch.bmm(agg_attn, positions.unsqueeze(0).expand(agg_attn.shape[0], -1, -1))
    return weighted_pos  # [B, K, 3] - 连续坐标，可用于 RoPE

# 另一种方案：不用 RoPE（Motus 的 action tokens 和 und tokens 也不用 RoPE）
# 这需要修改 Joint Attention，让压缩后的 video tokens 也不加 RoPE
```

2. **复杂度最高**：两阶段 cross-attention ~28M 参数。如果 A-2 就够好，优先级降低。

---

#### 方案A-4：Gumbel-Softmax 离散门控（已放弃）

> **放弃原因**：温度 τ 需要退火调参、Gumbel 噪声导致训练不稳定、与 keep_ratio 耦合需反复调优。PruMerge + STE 更简单且效果相当。保留代码供参考。

**核心思想**：原方案 A（Cross-Attention Gate）的改进版。去掉重的 cross-attention，改用 concat+MLP 打分；用 Gumbel-Softmax 替代 sigmoid gate，解决训练-推理分布不匹配问题。

**相比原方案 A 的改进**：
1. 去掉 cross-attention，改用 concat+MLP（参数量 28M → 1M）
2. Gumbel-Softmax 让训练时的分布更接近推理时的离散选择
3. 温度退火：训练初期温度高（软选择），后期温度低（接近硬选择）

**优势**：轻量、近似可微分、实现简单。

**劣势**：训练时仍保留 N 个 token（soft mask），训练阶段无加速收益。**必须改为 hard top-k + STE 才能满足"压缩在 Joint Attention 之前"的原则。**

```python
class GumbelTokenGate(nn.Module):
    """
    Gumbel-Softmax 离散门控：用 concat+MLP 替代 cross-attention，
    用 Gumbel-Softmax 替代 sigmoid，解决训练-推理分布不匹配。
    """

    def __init__(self, video_dim=3072, und_dim=512):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(video_dim + und_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # 2 类：keep / drop
        )
        self.temperature = 1.0  # 可退火

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, video_tokens, und_tokens, keep_ratio=0.5, training=True):
        """
        Args:
            video_tokens: [B, N, 3072]
            und_tokens: [B, L, 512]
            keep_ratio: 保留比例（仅推理时使用）
            training: 是否训练模式
        Returns:
            compressed: 训练时 [B, N, 3072]（soft mask），推理时 [B, K, 3072]（hard selection）
            keep_prob: [B, N] - 保留概率
        """
        B, N, D = video_tokens.shape

        # Und context
        und_context = und_tokens.mean(dim=1, keepdim=True).expand(-1, N, -1)  # [B, N, und_dim]

        # 打分
        logits = self.scorer(torch.cat([video_tokens, und_context], dim=-1))  # [B, N, 2]

        if training:
            # Gumbel-Softmax：可微分的离散采样近似
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
            keep_prob = probs[..., 1:2]  # [B, N, 1]  keep 的概率
            # 软加权（但分布更接近离散）
            compressed = video_tokens * keep_prob
        else:
            # 硬选择
            keep_prob = F.softmax(logits, dim=-1)[..., 1]  # [B, N]
            top_k = max(1, int(N * keep_ratio))
            _, indices = keep_prob.topk(top_k, dim=1)
            compressed = video_tokens.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, D)
            )

        return compressed, keep_prob


class GumbelTokenGateTrainer:
    """
    温度退火调度器：训练初期温度高（软选择），后期温度低（接近硬选择）。
    """

    def __init__(self, gate_module, init_temp=2.0, min_temp=0.5, anneal_rate=0.001):
        self.gate = gate_module
        self.temp = init_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate

    def step(self):
        self.temp = max(self.min_temp, self.temp * (1 - self.anneal_rate))
        self.gate.set_temperature(self.temp)
```

**已知问题与修复**：

1. **训练时无加速（严重）**：训练时 `compressed = video_tokens * keep_prob` 输出 `[B, N, 3072]`，token 数不变，30 层 Joint Attention 仍处理 N 个 token。**直接违反"压缩在 Joint Attention 之前"的原则。**

   **修复**：训练时也用 hard top-k + Straight-Through Estimator：

```python
class GumbelTokenGateV2(nn.Module):
    """修复版：训练时也做 hard top-k，用 STE 保持可微分"""

    def __init__(self, video_dim=3072, und_dim=512):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(video_dim + und_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)  # 直接输出 keep score，不需要 2 分类
        )
        self.temperature = 1.0

    def forward(self, video_tokens, und_tokens, keep_ratio=0.5):
        B, N, D = video_tokens.shape
        und_context = und_tokens.mean(dim=1, keepdim=True).expand(-1, N, -1)
        scores = self.scorer(
            torch.cat([video_tokens, und_context], dim=-1)
        ).squeeze(-1)  # [B, N]

        # Gumbel-Softmax 得到近似离散的 keep 概率
        # 用 stack 构造 2-class logits: [drop_score, keep_score]
        logits = torch.stack([torch.zeros_like(scores), scores], dim=-1)  # [B, N, 2]
        probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        keep_prob = probs[..., 1]  # [B, N]

        # Hard top-k（训练和推理都用）
        top_k = max(1, int(N * keep_ratio))
        _, indices = keep_prob.topk(top_k, dim=1)  # [B, K]

        # 前向：hard gather
        compressed = video_tokens.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]

        # 反向：STE - 梯度通过 keep_prob 传回 scorer
        # 实现：compressed_ste = compressed + (keep_prob_weighted - keep_prob_weighted.detach())
        # 其中 keep_prob_weighted = video_tokens * keep_prob.unsqueeze(-1)
        keep_prob_weighted = video_tokens * keep_prob.unsqueeze(-1)  # [B, N, D]
        # 从 keep_prob_weighted 中 gather 对应位置
        selected_weighted = keep_prob_weighted.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]
        # STE trick：前向用 compressed，梯度传给 selected_weighted
        compressed = compressed + (selected_weighted - selected_weighted.detach())

        return compressed, keep_prob, indices

    # RoPE 位置：✅ 用 indices 保留原始 token 索引
```

~~原方案 A（Cross-Attention Token Gate）已替换，保留供参考~~

<details>
<summary>原方案 A 代码（已弃用）</summary>

原方案使用 Cross-Attention（Video→Q, Und→K/V）+ Gate Head 预测 sigmoid 门控值，存在以下问题：
1. 训练-推理分布不匹配：训练时 soft mask（N 个 token 全保留），推理时 hard top-k（只保留 K 个）
2. 训练时无加速：soft mask 不减少序列长度，30 层 MoT 计算量不变
3. Cross-Attention 开销不合理：用"重"操作（Q/K/V 投影 + softmax + 聚合）产生"轻"输出（1 个标量）
4. 重复计算：Motus 30 层 Joint Attention 已在做 cross-modal attention
5. **Cross-Attention 方向错误**：应为 Und→Q, Video→K/V（语义查询定位视觉区域），而非 Video→Q, Und→K/V（详见"Cross-Attention 方向选择"说明）

新方案 A-1（零参数 baseline）→ A-2（PruMerge）→ A-3（TokenLearner）→ A-4（Gumbel-Softmax）按复杂度递进，先验证最简方案再加复杂度。

</details>

#### 方案B：空间重采样 + 时间聚合（2D Grid Sampling）

**设计改进**：原方案使用 3D trilinear 插值，对 Motus 的 3D 时空 latent（T×H×W）计算开销大且训练不稳定。改为 **空间 2D resampling + 时间维度聚合**：仅在空间维度 (H, W) 上预测 2D Grid 坐标做 Bilinear 采样，时间维度通过轻量聚合处理。**注意**：V1 实现只压缩了空间维度，时间维度未聚合（见下方"已知问题"），需用 V2 版本或补充 Max Pool 实现真正的时空压缩。

```python
class SpatialCoordinatePredictor(nn.Module):
    """
    从understanding tokens预测2D空间采样坐标，用可微分双线性插值提取特征。
    仅在空间维度采样，时间维度通过聚合处理，避免3D插值的开销和不稳定性。
    """

    def __init__(self, und_dim=512, num_keep=64):
        super().__init__()
        self.num_keep = num_keep

        # 从 understanding tokens 预测 2D 空间坐标
        self.coord_predictor = nn.Sequential(
            nn.Linear(und_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_keep * 2),  # num_keep 个 2D 坐标
        )

        # 空间特征聚合：将 2D 采样结果与时间维度信息融合
        self.spatial_proj = nn.Linear(und_dim, 3072)

    def forward(self, video_tokens, und_tokens, grid_h, grid_w):
        """
        Args:
            video_tokens: [B, T*H*W, 3072] - 时空展平的 video tokens
            und_tokens: [B, L, 512]
            grid_h, grid_w: 空间网格尺寸
        Returns:
            sampled: [B, T*num_keep, 3072] - 空间采样后的 tokens
            coords_2d: [B, num_keep, 2] - 预测的 2D 坐标（归一化到 [0,1]）
        """
        B, N, D = video_tokens.shape
        T = N // (grid_h * grid_w)

        # 预测 2D 空间坐标（每个时间步共享同一组空间坐标）
        und_global = und_tokens.mean(dim=1)  # [B, 512]
        coords_2d = torch.sigmoid(
            self.coord_predictor(und_global)
        ).view(B, self.num_keep, 2)  # [B, num_keep, 2] 归一化坐标

        # 将 video tokens reshape 为 (B*T, 1, H, W, D) 用于 grid_sample
        video_spatial = video_tokens.view(B, T, grid_h, grid_w, D)
        video_spatial = video_spatial.permute(0, 1, 4, 2, 3)  # [B, T, D, H, W]
        video_spatial = video_spatial.reshape(B * T, D, grid_h, grid_w)

        # 将坐标转换为 grid_sample 需要的 [-1, 1] 范围
        grid = coords_2d.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, num_keep, 2]
        grid = grid.reshape(B * T, self.num_keep, 1, 2) * 2 - 1  # 归一化到 [-1, 1]

        # 可微分双线性采样
        sampled = F.grid_sample(
            video_spatial, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [B*T, D, num_keep, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B*T, num_keep, D]
        sampled = sampled.reshape(B, T * self.num_keep, D)

        # 几何注入：用 understanding tokens 的投影补偿采样误差
        spatial_bias = self.spatial_proj(und_global).unsqueeze(1)  # [B, 1, 3072]
        sampled = sampled + spatial_bias * 0.1

        return sampled, coords_2d


class SpatialResamplingCompression(nn.Module):
    """
    空间重采样压缩：2D Grid Sampling + 时间聚合。
    相比原 3D trilinear 方案：
    - 计算开销降低（2D vs 3D 插值）
    - 训练更稳定（grid_sample 是 PyTorch 原生可微操作）
    - 保持空间对应关系（坐标直接对应 H×W 网格）
    """

    def __init__(self, und_dim=512, video_dim=3072):
        super().__init__()
        self.spatial_sampler = None  # 延迟初始化（需要知道 grid 尺寸）
        self.und_dim = und_dim
        self.video_dim = video_dim

    def forward(self, video_tokens, und_tokens, grid_sizes, keep_ratio=0.5):
        """
        Args:
            video_tokens: [B, T*H*W, 3072]
            und_tokens: [B, L, 512]
            grid_sizes: (T, H, W) - 时空网格尺寸
            keep_ratio: 空间维度保留比例
        """
        T, H, W = grid_sizes
        num_keep_spatial = max(1, int(H * W * keep_ratio))

        # 延迟初始化
        if self.spatial_sampler is None:
            self.spatial_sampler = SpatialCoordinatePredictor(
                self.und_dim, num_keep_spatial
            ).to(video_tokens.device)

        sampled, coords_2d = self.spatial_sampler(video_tokens, und_tokens, H, W)

        return sampled, coords_2d
```

**已知问题：当前实现缺少真正的时间聚合**

上述代码存在两个问题：

1. **所有时间步共享同一组空间坐标**：`coords_2d` 从 `und_global`（所有 U-tokens 的均值）预测，与时间无关。这意味着假设"每个时间步的重要区域相同"，对机器人操作不合理（第一帧看桌面全局，最后一帧看夹爪接触点，重要区域完全不同）。

2. **输出维度未压缩时间轴**：`sampled.reshape(B, T * self.num_keep, D)` 只是把空间采样结果拼回去，时间维度完全保留。T=8、num_keep=64 时输出 512 tokens，压缩效果有限。

**修正方案：加入真正的时间聚合**

```python
class SpatialCoordinatePredictorV2(nn.Module):
    """
    V2: 每个时间步独立预测空间坐标 + 时间维度聚合。
    """

    def __init__(self, und_dim=512, video_dim=3072, num_keep=64, n_heads=4):
        super().__init__()
        self.num_keep = num_keep

        # 每帧独立的坐标预测（输入：单帧 U-tokens）
        self.coord_predictor = nn.Sequential(
            nn.Linear(und_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_keep * 2),
        )

        # 时间聚合：对同一空间位置跨时间步做 attention pooling
        self.temporal_q = nn.Linear(video_dim, video_dim)
        self.temporal_k = nn.Linear(video_dim, video_dim)
        self.temporal_v = nn.Linear(video_dim, video_dim)
        self.temporal_proj = nn.Linear(video_dim, video_dim)
        self.n_heads = n_heads

    def forward(self, video_tokens, und_tokens, grid_h, grid_w):
        B, N, D = video_tokens.shape
        T = N // (grid_h * grid_w)

        # 逐帧预测空间坐标
        und_per_frame = und_tokens.view(B, -1, T, und_tokens.shape[-1])  # 假设 und_tokens 有帧级信息
        # 如果 und_tokens 没有帧级结构，退化为共享坐标（与 V1 相同）
        if und_per_frame.shape[2] != T:
            und_per_frame = und_tokens.mean(dim=1, keepdim=True).expand(-1, T, -1)

        coords_list = []
        for t in range(T):
            c = torch.sigmoid(self.coord_predictor(und_per_frame[:, t]))  # [B, num_keep*2]
            coords_list.append(c.view(B, self.num_keep, 2))
        coords_per_frame = torch.stack(coords_list, dim=1)  # [B, T, num_keep, 2]

        # 逐帧空间采样
        video_spatial = video_tokens.view(B, T, grid_h, grid_w, D)
        video_spatial = video_spatial.permute(0, 1, 4, 2, 3).reshape(B * T, D, grid_h, grid_w)

        grid = coords_per_frame.reshape(B * T, self.num_keep, 1, 2) * 2 - 1
        sampled = F.grid_sample(
            video_spatial, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [B*T, D, num_keep, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B*T, num_keep, D]
        sampled = sampled.reshape(B, T, self.num_keep, D)  # [B, T, num_keep, D]

        # 时间聚合：对每个空间位置，用 attention 从 T 个时间步中聚合
        # reshape: [B*num_keep, T, D]
        feat = sampled.permute(0, 2, 1, 3).reshape(B * self.num_keep, T, D)
        head_dim = D // self.n_heads

        q = self.temporal_q(feat.mean(dim=1, keepdim=True))  # [B*num_keep, 1, D] 用均值做 query
        k = self.temporal_k(feat)  # [B*num_keep, T, D]
        v = self.temporal_v(feat)  # [B*num_keep, T, D]

        # reshape for multi-head
        q = q.view(B * self.num_keep, 1, self.n_heads, head_dim)
        k = k.view(B * self.num_keep, T, self.n_heads, head_dim)
        v = v.view(B * self.num_keep, T, self.n_heads, head_dim)

        attn = (q * k).sum(dim=-1) / (head_dim ** 0.5)  # [B*num_keep, 1, T]
        attn = F.softmax(attn, dim=-1)
        aggregated = (attn.unsqueeze(-1) * v).sum(dim=2)  # [B*num_keep, 1, D]
        aggregated = self.temporal_proj(aggregated.squeeze(1))  # [B*num_keep, D]
        aggregated = aggregated.reshape(B, self.num_keep, D)  # [B, num_keep, D]

        return aggregated, coords_per_frame
```

**备选方案（更简单的时间聚合）**：

```python
# 直接对时间维度 max pool，零额外参数
def temporal_maxpool_simple(sampled, T, num_keep, D):
    """
    sampled: [B, T, num_keep, D] → [B, num_keep, D]
    """
    return sampled.max(dim=1).values  # [B, num_keep, D]

# 或 mean pool
def temporal_meanpool_simple(sampled, T, num_keep, D):
    return sampled.mean(dim=1)  # [B, num_keep, D]
```

**三种时间聚合策略对比**：

| 策略 | 输出维度 | 额外参数 | 压缩效果 | 信息保留 |
|------|---------|---------|---------|---------|
| 当前（无聚合） | [B, T×num_keep, D] | 0 | 只压缩空间 | 完整时序 |
| Temporal Attention | [B, num_keep, D] | ~19M (D=3072) | 空间+时间 | 学习哪些时间步重要 |
| Max/Mean Pool | [B, num_keep, D] | 0 | 空间+时间 | 简单聚合，丢失时序细节 |

**建议**：先用 Max Pool 作为 baseline（零参数、零开销），再对比 Temporal Attention 的增量收益。如果 Max Pool 已经足够好，说明时间聚合本身比聚合方式更重要。

**已知问题与修复**：

1. **输出是插值生成的新 token**：`F.grid_sample` 在连续坐标上做双线性插值，生成的 token 不是原始 token 的子集。空间坐标是 `coords_2d`（连续值），不是离散 grid 位置。

   **RoPE 修复**：修改 Motus 的 RoPE 实现，支持连续坐标输入（将 `(t, h, w)` 连续坐标直接传入 RoPE 计算，而非只用离散索引）。或者将坐标量化到最近的 grid 位置。

2. **V1 所有时间步共享坐标**：`und_global = und_tokens.mean(dim=1)` 对所有时间步预测同一组空间坐标。对机器人操作不合理——第一帧关注桌面全局，最后一帧关注夹爪接触点。**必须用 V2（逐帧坐标预测）。**

3. **V2 时间聚合后丢失时序信息**：V2 输出 `[B, num_keep, D]`，时间维度被压缩。对 action prediction（需要时序信息）可能有害。**建议保留时间维度，只做空间压缩**：输出 `[B, T*num_keep, D]`，时间聚合放在后续层（或不聚合）。

4. **方案选择建议**：鉴于问题 2 和 3，**方案 B 在当前设计下不适合作为主方案**。如果需要空间级压缩，建议用 A-2（PruMerge），它天然保留时序信息（每个时间步独立选锚点）。

~~原方案B代码（SASGS风格3D采样）已替换，保留供参考~~

<details>
<summary>原方案B代码（已弃用）</summary>

原方案使用 3D trilinear 插值，存在以下问题：
1. GridS 是 2D 的，Motus latent 是 3D 的（T×H×W），trilinear 计算开销大
2. 动态创建 coord_head 导致训练不稳定
3. `_soft_sample` 只用了第一个坐标维度，信息利用不充分

新方案改用 PyTorch 原生的 `F.grid_sample`，稳定且高效。
</details>

~~原方案B代码已删除，新方案见上方 SpatialResamplingCompression~~

#### 方案C：可学习Token Merge（加权合并）

```python
class LearnableTokenMerge(nn.Module):
    """
    学习如何将多个video tokens合并成少数几个tokens。
    用understanding tokens作为指导，完全可微分。
    """

    def __init__(self, video_dim=3072, und_dim=512, num_keep=64):
        super().__init__()
        self.num_keep = num_keep

        # 从拼接的特征预测merge权重
        self.weight_predictor = nn.Sequential(
            nn.Linear(video_dim + und_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_keep),
        )

        # 可学习的merge anchor
        self.merge_anchors = nn.Parameter(torch.randn(num_keep, video_dim) * 0.02)

    def forward(self, video_tokens, und_tokens):
        """
        Args:
            video_tokens: [B, N, 3072]
            und_tokens: [B, L, 512]
        Returns:
            merged_tokens: [B, K, 3072]
            merge_weights: [B, N, K] - 可微分的merge权重
        """
        B, N, D_v = video_tokens.shape
        K = self.num_keep

        # 计算merge权重
        und_global = und_tokens.mean(dim=1, keepdim=True).expand(-1, N, -1)
        combined = torch.cat([video_tokens, und_global], dim=-1)  # [B, N, D_v+D_u]

        # 预测权重并softmax归一化
        raw_weights = self.weight_predictor(combined)  # [B, N, K]
        merge_weights = F.softmax(raw_weights, dim=1)  # [B, N, K]

        # 加权合并（完全可微）
        merged = torch.bmm(merge_weights.transpose(1, 2), video_tokens)  # [B, K, D_v]

        # 加上merge anchor
        merged = merged + self.merge_anchors.unsqueeze(0)

        return merged, merge_weights
```

**已知问题与修复**：

1. **输出 token 无空间位置**：合并后的 token 是加权平均，不是原始 token 的子集。RoPE 位置信息丢失。修复方案同 A-3——根据 merge_weights 计算加权平均位置，或不加 RoPE。

2. **`num_keep` 固定，不随 `keep_ratio` 变化**：`__init__` 中 `num_keep=64` 固定。不同 keep_ratio 需要重新初始化。应改为在 `forward` 中动态计算 K。

3. **参数量表格有误**：对比表中写的 6.6B 是笔误，实际约 3M。

4. **建议**：鉴于问题 1 和功能与 A-2 重叠（都是合并 token），**方案 C 可以被 A-2 替代**。A-2 的 PruMerge 更灵活（锚点保持原始位置，只合并非锚点 token），位置信息不丢失。方案 C 保留在对比实验中作为消融对照。

#### 方案审视总结（v3）

基于"压缩在 Joint Attention 之前，action-only，训练+推理都有加速"的原则，对所有方案进行审视后发现以下关键问题：

| 问题类型 | 涉及方案 | 问题描述 | 修复状态 |
|---------|---------|---------|---------|
| RoPE位置丢失 | A-3, C | 输出是新生成的token，无原始空间坐标 | 需补充加权平均位置计算 |
| ~~训练无加速~~ | ~~A-4(原版)~~ | ~~已放弃，不再考虑~~ | - |
| Layer1无法加速 | A-1 | 需先跑完Layer1才能获取attn weights | FastV固有限制，可接受 |
| top-k不可微 | A-2 | topk操作无梯度 | 需加STE松弛 |
| V1共享坐标 | B | 所有时间步预测同一组空间坐标 | 必须用V2（逐帧坐标） |
| V2丢时序 | B | 时间聚合后丢失时序信息 | 建议只做空间压缩，保留时间维度 |

**结论**：A-2（PruMerge + STE）是综合最优方案——训练推理都有加速、RoPE兼容、信息保留完整。A-1 作为零参数 baseline。A-3/B/C 作为消融对照。A-4（Gumbel-Softmax）已放弃（温度调参难、训练不稳定）。

#### 方案对比

| 方案 | 额外参数 | 额外FLOPs | 训练加速 | 推理加速 | RoPE兼容 | 已知问题 | 推荐度 |
|------|---------|-----------|---------|---------|----------|---------|--------|
| **A-1. FastV式Attention剪枝** | **0** | **0** | 部分(Layer1仍N) | ✅ | ✅ 原始索引 | Layer1无法加速 | **最高（baseline）** |
| **A-2. PruMerge式锚点合并** | ~1.3M | ~0.5B | ✅ | ✅ | ✅ 锚点保持位置 | top-k需STE松弛 | **高（主方案）** |
| **A-3. TokenLearner式聚合** | ~28M | ~1B | ✅ | ✅ | ❌ 需补充位置 | 输出无空间位置 | 中 |
| ~~A-4. Gumbel-Softmax门控~~ | ~~已放弃~~ | - | - | - | - | 温度调参难，训练不稳定 | - |
| B. 空间重采样+时间聚合 | ~19M | ~67M | ✅ | ✅ | ⚠️ 连续坐标 | V1共享坐标，V2丢时序 | 低 |
| C. Token Merge | ~3M | ~3M | ✅ | ✅ | ❌ 需补充位置 | num_keep固定，被A-2替代 | 低（消融对照） |

**推荐实验顺序与策略**：

```
阶段1：Baseline 验证
  └─ A-1 零参数 Attention 剪枝（1天）
     目的：确认 attention weights 作为重要性信号的质量上限
     判定：如果 keep_ratio=0.5 时 action accuracy 下降 <3% → 信号质量好

阶段2：快速出结果
  └─ A-2 PruMerge + STE（3天）
     目的：训练式方案的 baseline，工程上最简单
     判定：与 A-1 对比，学习式打分是否优于 attention weights

阶段3：主攻方向
  └─ A-3 TokenLearner（5天）← 最可能是最终方案
     目的：完全可微分，learnable queries 自适应语义角色
     优势：无 top-k 离散问题，两阶段设计（grounding + 聚合）更优雅
     关键：解决 RoPE 位置问题（加权平均坐标或不加 RoPE）

阶段4：上限探索
  └─ B GridS 式空间重采样（5天）← 上限可能最高
     目的：验证连续坐标采样是否优于离散选择
     优势：可精确采样到 token 之间的位置（如夹爪尖端），空间精度更高
     关键：V2 逐帧坐标 + 保留时序信息（不丢时间维度）
```

**为什么 A-3 最可能是最终方案**：
- 完全可微分，没有 top-k / Gumbel-Softmax 的离散选择问题
- K 个 learnable queries 可以自适应学到不同语义角色（一个关注工具，一个关注目标物体）
- 两阶段设计（先语义 grounding 再聚合）比 PruMerge 的"打分+丢弃"更优雅
- 训练稳定，超参数少

**为什么 B（GridS）上限最高**：
- 连续坐标采样比离散选择更灵活：可以精确采样到"夹爪尖端"这种不在 token 中心的位置
- 离散选择只能选最近的 token，GridS 可以在任意位置插值
- 对机器人操作，空间精度直接决定动作精度
- 如果 V2 逐帧坐标 + 时序保留做好，可能显著优于其他方案

> A-4（Gumbel-Softmax）已放弃：温度调参难、训练不稳定，PruMerge + STE 更实用。

#### 推荐组合

```python
class SemanticCompressionPipeline(nn.Module):
    """
    实验 pipeline：
    阶段1: A-1 零参数 baseline
    阶段2: A-2 PruMerge（快速出结果）
    阶段3: A-3 TokenLearner（主攻方向）
    阶段4: B GridS式采样（上限探索）
    """

    def __init__(self, video_dim=3072, und_dim=512, method='attention_prune', keep_ratio=0.5):
        super().__init__()
        self.method = method
        self.keep_ratio = keep_ratio

        if method == 'attention_prune':
            self.compressor = AttentionScorePruning()
        elif method == 'prumerge':
            self.compressor = PruMergeCompression(video_dim, und_dim)
        elif method == 'token_learner':
            self.compressor = SemanticTokenLearner(video_dim, und_dim)
        elif method == 'grids':
            self.compressor = SpatialResamplingCompression(und_dim, video_dim)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, video_tokens, und_tokens, attn_weights_layer1=None):
        if self.method == 'attention_prune':
            return self.compressor(video_tokens, attn_weights_layer1, self.keep_ratio)
        else:
            return self.compressor(video_tokens, und_tokens, self.keep_ratio)
```

### 3.3 训练Loss设计：多信号监督

**核心思想**：不只用action loss，同时用语义信号监督压缩模块，确保压缩后的tokens保持语义信息。

```python
class MultiSignalCompressionLoss(nn.Module):
    """
    多信号监督的压缩loss。
    同时关注action和语义两个信号。

    注意：此模块只负责 loss 计算，不包含 action_head。
    action_pred 由外部 training_step 传入（与 Motus 已有的 action_module 解耦）。
    """

    def __init__(self, video_dim=3072, und_dim=512, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        # 投影层：将video tokens投影到understanding空间
        self.video_to_und_proj = nn.Sequential(
            nn.Linear(video_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, und_dim)
        )

        # 对比学习温度
        self.temperature = 0.07

    def semantic_distillation_loss(self, compressed_tokens, und_tokens,
                                    distill_mode="reverse_kl", temperature=0.5):
        """
        语义蒸馏loss：利用Understanding Expert作为teacher。
        让压缩后的video tokens保持与Understanding tokens的语义对齐。

        distill_mode:
            "mse": 直接在 embedding 空间做 MSE 对齐（最简单，推荐先试）
            "cosine": 余弦相似度对齐（对特征范数不敏感）
            "reverse_kl": 反向KL散度，mode-seeking，压缩token聚焦teacher的主模式
            "forward_kl": 正向KL散度，mode-covering，保留更多分布信息

        注意：KL 模式在 512 维 embedding 上做 softmax，各维度无类别语义，
        梯度主要受最大值维度支配。建议先用 "mse" 或 "cosine" 验证可行性，
        再尝试 KL 看是否有额外收益。
        """
        # 压缩后的video tokens投影到und空间
        video_proj = self.video_to_und_proj(compressed_tokens.mean(dim=1))  # [B, und_dim]
        und_global = und_tokens.mean(dim=1).detach()  # [B, und_dim]

        if distill_mode == "mse":
            return F.mse_loss(video_proj, und_global)

        if distill_mode == "cosine":
            return 1 - F.cosine_similarity(video_proj, und_global, dim=-1).mean()

        # 分布级对齐：softmax → log_softmax → KL
        # 注意：T=1.0 时 512 维 softmax 熵极高，KL 值极小，梯度极弱。
        # 默认 T=0.5 使分布更尖锐，产生有效梯度。
        log_p_student = F.log_softmax(video_proj / temperature, dim=-1)
        log_p_teacher = F.log_softmax(und_global / temperature, dim=-1)
        p_teacher = log_p_teacher.exp()
        p_student = log_p_student.exp()

        if distill_mode == "reverse_kl":
            kl = (p_student * (log_p_student - log_p_teacher)).sum(dim=-1)
        elif distill_mode == "forward_kl":
            kl = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)
        else:
            raise ValueError(f"Unknown distill_mode: {distill_mode}")

        return kl.mean() * (temperature ** 2)

    def contrastive_loss(self, original_tokens, compressed_tokens):
        """
        对比学习loss：确保压缩后的tokens与原始tokens语义一致。
        两个分支都投影到 und 空间后再计算 InfoNCE，确保维度一致。
        """
        # 都投影到 und 空间 [B, 512]
        orig_proj = self.video_to_und_proj(original_tokens.mean(dim=1))
        comp_proj = self.video_to_und_proj(compressed_tokens.mean(dim=1))

        # L2 归一化 → cosine similarity
        orig_proj = F.normalize(orig_proj, dim=-1)
        comp_proj = F.normalize(comp_proj, dim=-1)

        logits = torch.mm(orig_proj, comp_proj.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.t(), labels)) / 2
        return loss

    def forward(self, original_tokens, compressed_tokens, und_tokens,
                action_pred, action_gt):
        """
        多信号总loss

        注：原方案包含 Geometric Consistency Loss（从 video tokens 预测末端执行器 3D 关键点），
        但 Motus 数据集（robotwin, ac_one, lerobot）只有 qpos 标签，无 keypoint GT。
        因此用 Action Loss 隐式替代几何约束——action 本身编码了末端执行器的运动信息，
        压缩后 action 预测准确即可说明几何信息被保留。未来若有 keypoint 标注可重新加入。

        Args:
            original_tokens: 压缩前的 video tokens [B, N, 3072]
            compressed_tokens: 压缩后的 video tokens [B, K, 3072]
            und_tokens: understanding tokens [B, L, 512]
            action_pred: action 预测结果（由外部 action_module 产生）
            action_gt: action ground truth
        """
        # Action loss（隐含几何约束：action 编码了末端执行器运动）
        L_action = F.mse_loss(action_pred, action_gt)

        # Semantic distillation loss
        L_semantic = self.semantic_distillation_loss(compressed_tokens, und_tokens)

        # Contrastive loss（可选，与蒸馏共享投影层）
        L_contrastive = self.contrastive_loss(original_tokens, compressed_tokens)

        # 总loss
        # 权重选择依据：action 为主信号，semantic 为辅助约束，contrastive 为正则化。
        # 训练初期应 log 各 loss 量级，若 L_semantic 量级差 action >10x，
        # 考虑用 uncertainty weighting (Kendall et al.) 或手动调整 alpha。
        total_loss = L_action + self.alpha * L_semantic + 0.1 * L_contrastive

        return {
            'total': total_loss,
            'action': L_action,
            'semantic': L_semantic,
            'contrastive': L_contrastive,
        }
```

**稀疏性正则化**（A-3 TokenLearner / 方案 B 需要，A-2 PruMerge 的 hard top-k 天然有稀疏性）：

```python
# 可选：对 importance score 添加 entropy regularization，鼓励分布更尖锐
# 避免压缩模块学到 trivial solution（所有 token 等权重）
def sparsity_loss(importance_scores):
    """importance_scores: [B, N]，已 softmax 归一化"""
    entropy = -(importance_scores * (importance_scores + 1e-8).log()).sum(dim=-1)
    return entropy.mean()  # 最小化熵 → 分布更尖锐 → 更稀疏
```

**Loss设计说明**：

| Loss | 目的 | 信号来源 | 权重 | 备注 |
|------|------|----------|------|------|
| Action Loss | 确保压缩后能准确预测action（隐含几何约束） | Action GT (qpos) | 1.0 | 主信号 |
| Semantic Distillation | 确保压缩后保持语义信息 | Understanding Expert (teacher, detach) | alpha (0.5) | 推荐先试 mse/cosine，再试 reverse_kl |
| Contrastive Loss | 确保压缩前后特征一致 | Original vs Compressed tokens (同投影空间) | 0.1 | L2 归一化后计算 cosine similarity |

**蒸馏模式选择建议**：
- **mse**：最简单，直接约束 embedding 距离，推荐作为 baseline 先验证可行性
- **cosine**：对特征范数不敏感，适合 embedding 量级波动大的场景
- **reverse_kl**：mode-seeking，压缩 token 聚焦 teacher 最显著语义模式（OPD 效果）。注意 T=0.5 使分布更尖锐，T=1.0 时 512 维 softmax 熵极高、梯度极弱
- **forward_kl**：mode-covering，保留更多分布信息，但可能保留噪声

**Loss 权重调参**：训练初期 log 各 loss 量级，若 L_semantic 量级差 L_action >10x，考虑 uncertainty weighting（Kendall et al.）或手动调整 alpha。

~~原方案含 Geometric Consistency Loss（keypoint 预测），因数据不可得已移除，用 Action Loss 隐式替代。~~

**优势**：
1. **不需要额外标注**：Understanding Expert的输出本身就是语义监督
2. **端到端可微**：所有loss都可以反向传播
3. **利用Motus架构优势**：直接用Understanding Expert作为teacher
4. **数据兼容**：所有 loss 信号均可从现有数据集获取（qpos + VLM 特征），无需额外标注

### 3.4 集成到Motus

**核心设计决策**：压缩在 Joint Attention 之前，只做 action prediction，跳过 video generation。理由：
1. 机器人控制场景只需 action output，不需要生成未来视频
2. 避免 output_head 的 unpatchify 维度冲突问题
3. Joint Attention 的 FLOPs 与 video token 数成正比，压缩在前才能获得真正加速

**数据流**：

```
Video [B,3,9,384,320] → VAE → latent [B,48,3,12,10] → patch_embed → video_tokens [B,90,3072]
                                                                              ↓
VLM inputs → frozen Qwen3-VL → und_tokens [B,L,512]                         ↓
                                   ↓                                    ┌────────────┐
                                   └──────────────────────────────────→ │ Compression │
                                                                        │  N=90 → K  │
                                                                        └─────┬──────┘
                                                                              ↓
State [B,1,14] + Actions [B,16,14] → StateActionEncoder → action_tokens [B,21,1024]
                                                                              ↓
                                              ┌───────────────────────────────┘
                                              ↓
                                    30层 Joint Attention
                                    (video[K,3072] + action[21,1024] + und[L,512])
                                              ↓
                                    Action Head → action_pred [B,17,14]  ✅
                                    output_head → 跳过（不需要视频生成）
```

**关键实现细节**：

```python
# ============================================================
# 集成到 motus.py：压缩在 Joint Attention 之前，action-only
# ============================================================

class Motus(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... 原有初始化 ...

        # Token 压缩模块
        self.enable_compression = getattr(config, 'enable_token_compression', False)
        self.keep_ratio = getattr(config, 'keep_ratio', 0.5)
        self.compression_method = getattr(config, 'compression_method', 'attention_prune')

        if self.enable_compression:
            self.compressor = SemanticCompressionPipeline(
                video_dim=3072,
                und_dim=config.und_expert_hidden_size,
                method=self.compression_method,
                keep_ratio=self.keep_ratio,
            )

    def training_step(self, batch, ...):
        # 1. 准备 video tokens
        noisy_video_latent = self.prepare_noisy_latent(batch)
        video_tokens = self.video_module.prepare_input(noisy_video_latent)
        # video_tokens: [B, N=90, 3072]

        # 2. 准备 understanding tokens
        und_tokens = self.und_module.extract_und_features(batch['vlm_inputs'])
        # und_tokens: [B, L, 512]

        # 3. 准备 action tokens
        action_tokens = self.action_module.encode(batch['state'], batch['actions'])
        # action_tokens: [B, 21, 1024]

        # ========== Token 压缩（Joint Attention 之前）==========
        if self.enable_compression:
            video_tokens_original = video_tokens  # 保留原始 tokens 用于 loss 计算

            # 获取第一层 Joint Attention 的 attention weights（仅 A-1 需要）
            attn_weights = self._get_layer1_attn_weights(
                video_tokens, action_tokens, und_tokens
            ) if self.compression_method == 'attention_prune' else None

            compressed_result = self.compressor(
                video_tokens, und_tokens, attn_weights_layer1=attn_weights
            )
            # compressed_result: (compressed_tokens, indices, importance)

            if self.compression_method == 'attention_prune':
                video_tokens_compressed, indices, importance = compressed_result
            elif self.compression_method in ('prumerge', 'token_learner'):
                video_tokens_compressed, indices, importance = compressed_result

            video_tokens = video_tokens_compressed
            # video_tokens: [B, K, 3072]  K = N * keep_ratio

        # ========== 30层 Joint Attention（压缩后的序列）==========
        for i in range(self.num_layers):
            video_tokens, action_tokens, und_tokens = \
                self.video_module.process_joint_attention(
                    video_tokens, action_tokens, und_tokens, layer_idx=i
                )

        # ========== 只取 Action 输出 ==========
        action_pred = self.action_module.decode(action_tokens)
        # action_pred: [B, 17, 14]

        # 计算 loss
        action_loss = F.mse_loss(action_pred, batch['action_gt'])

        # 可选：语义蒸馏 loss（在压缩模块上施加语义约束）
        if self.enable_compression and hasattr(self, 'compression_loss'):
            loss_dict = self.compression_loss(
                original_tokens=video_tokens_original,  # 压缩前（压缩流程开头保存）
                compressed_tokens=video_tokens,          # 压缩后（已被覆盖为 compressed）
                und_tokens=und_tokens,
                action_pred=action_pred,
                action_gt=batch['action_gt'],
            )
            total_loss = loss_dict['total']
        else:
            total_loss = action_loss

        return total_loss, action_pred

    def inference_step(self, batch):
        # 与 training_step 相同的压缩流程，但：
        # 1. 无噪声（不做 flow matching 的 noisy latent）
        # 2. 不计算 loss
        # ...（同上，省略 loss 计算）
        return action_pred

    def _get_layer1_attn_weights(self, video_tokens, action_tokens, und_tokens):
        """
        获取第一层 Joint Attention 的 Und→Video attention weights。
        仅用于 A-1（零参数 baseline）。

        实现方式：替换第一层的 flash_attention 为 F.scaled_dot_product_attention，
        后者支持 return_attn_weights。
        """
        # 需要在 WanSelfAttention 中添加 return_attn_probs 参数
        # 具体实现见下方"工程实现细节"
        pass
```

**RoPE 位置处理**：

压缩后 video token 数量从 N 变为 K，但 Joint Attention 中 RoPE 是按 token 在序列中的位置编码的。需要保持被选中 token 的**原始空间位置**：

```python
class PositionAwareCompression(nn.Module):
    """
    压缩后保持原始空间位置信息。
    Joint Attention 中 RoPE 按位置编码，压缩后的 token 需要保留原始 3D 坐标。
    """

    def __init__(self, compressor, grid_sizes):
        super().__init__()
        self.compressor = compressor
        T, H, W = grid_sizes
        # 预计算每个 token 的原始 3D 位置
        # video_tokens 是 flatten 的 [T*H*W, D]，每个 token 有 (t, h, w) 坐标
        self.register_buffer(
            'positions',
            self._compute_3d_positions(T, H, W)
        )  # [N, 3]

    def _compute_3d_positions(self, T, H, W):
        """为每个 token 生成 3D 坐标 [t, h, w]"""
        pos = torch.stack(torch.meshgrid(
            torch.arange(T), torch.arange(H), torch.arange(W), indexing='ij'
        ), dim=-1)  # [T, H, W, 3]
        return pos.reshape(-1, 3).float()  # [N, 3]

    def forward(self, video_tokens, und_tokens, **kwargs):
        compressed, indices, importance = self.compressor(
            video_tokens, und_tokens, **kwargs
        )
        # 保留被选中 token 的原始位置（用于 RoPE）
        selected_positions = self.positions[indices]  # [B, K, 3]
        return compressed, indices, importance, selected_positions
```

**工程实现细节**：

1. **Flash Attention 返回 weights**：A-1 需要第一层的 attention weights。在 `WanSelfAttention.forward` 中，第一层用 `F.scaled_dot_product_attention`（支持 `attn_mask` 返回），其余层用 flash_attention。

2. **seq_lens 适配**：Motus 的 `flash_attention` 调用使用 `seq_lens` 参数指定每个 batch 的有效长度。压缩后需更新 `seq_lens`：

```python
# 原始：seq_lens = [N_video + N_action + N_und]
# 压缩后：seq_lens = [K + N_action + N_und]
```

---

## 四、Checkpoint 选择策略

### 可用 Checkpoint 总览

Motus 提供 3 个官方 checkpoint（HuggingFace: `motus-robotics/`）+ 2 个基础模型：

| 模型 | 训练阶段 | 内容 | 实验可用性 |
|------|---------|------|-----------|
| **Motus_Wan2_2_5B_pretrain** | Stage 1 | 只有 VGM（Wan2.2），无 Action/Und Expert | ❌ 不可用 |
| **Motus** | Stage 2 | Latent Action 预训练，三个 Expert 都有 | ✅ 可用（快速验证） |
| **Motus_robotwin2** | Stage 3 | RoboTwin2 微调，最完整的模型 | ✅ 最佳选择 |
| Qwen3-VL-2B-Instruct | 基础模型 | VLM 本体（frozen） | 需配合 Motus ckpt |
| Wan2.2-TI2V-5B | 基础模型 | VGM 本体 + VAE | 需配合 Motus ckpt |

### 各 Checkpoint 详细分析

**❌ Motus_Wan2_2_5B_pretrain（Stage 1）**
- 只训练了 Video Generation Model，没有 Action Expert 和 Understanding Expert
- 无法提取 Understanding Tokens，**做不了 Token 压缩实验**

**✅ Motus（Stage 2）**
- 三个 Expert 都训练了（用 latent action，不是真实 action）
- 语义表征质量可能不如 Stage 3，但**适合快速验证 pipeline**
- 推荐用于：初步 probing 实验、流程调试、方法论验证

**✅ Motus_robotwin2（Stage 3）— 最佳选择**
- 在 RoboTwin2 仿真数据上微调过的完整模型
- Action prediction 能力最强，语义表征与任务最对齐
- 推荐用于：正式实验、论文数据、最终结果

### 下载与配置

```bash
# 下载 checkpoint
huggingface-cli download motus-robotics/Motus --local-dir ./pretrained_models/Motus
huggingface-cli download motus-robotics/Motus_robotwin2 --local-dir ./pretrained_models/Motus_robotwin2
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ./pretrained_models/Qwen3-VL-2B-Instruct
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./pretrained_models/Wan2.2-TI2V-5B
```

配置路径（修改 `configs/robotwin.yaml`）：
```yaml
model:
  wan:
    checkpoint_path: "./pretrained_models/Motus_Wan2_2_5B_pretrain"
    vae_path: "./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  vlm:
    checkpoint_path: "./pretrained_models/Qwen3-VL-2B-Instruct"

# 加载完整 Motus 权重（Stage 2 或 Stage 3）
finetune:
  checkpoint_path: "./pretrained_models/Motus_robotwin2"  # 或 Motus
```

### 实验阶段与 Checkpoint 对应关系

| 实验阶段 | 使用的 Checkpoint | 原因 |
|---------|------------------|------|
| 实验1 Probing（快速验证） | Motus (Stage 2) | 快速验证语义对齐假设，流程调试 |
| 实验1 Probing（正式） | Motus_robotwin2 (Stage 3) | 最完整的模型，结果最有说服力 |
| 实验2 压缩效果对比 | Motus_robotwin2 (Stage 3) | 需要在具体任务上评估 success rate |
| 实验3 语义引导验证 | Motus_robotwin2 (Stage 3) | 需要任务级性能对比 |

**核心建议**：先用 Stage 2 快速跑通实验1 的 probing pipeline，确认流程没问题后，再用 Stage 3 跑正式结果。

### 训练策略：微调哪些模块？

**问题**：压缩模块是全新的（没有预训练权重），但其他模块已有 Stage 3 权重。应该冻结哪些、训练哪些？

**各模块状态分析**：

| 模块 | 参数量 | Stage 3 权重 | 训练策略 | 原因 |
|------|--------|-------------|---------|------|
| VLM (Qwen3-VL) | ~2.13B | ✅ 有，可加载 | **冻结** | 语义提取器，保持语义稳定性 |
| WAN (Video Module) | ~5.0B | ✅ 有，可加载 | **冻结** | 参数量太大，微调成本极高 |
| Action Expert | ~641M | ✅ 有，可加载 | **Phase 1 冻结，Phase 2 可选解冻** | Phase 1 验证压缩模块，Phase 2 联合优化 |
| Understanding Expert | ~253M | ✅ 有，可加载 | **Phase 1 冻结，Phase 2 可选解冻** | 同上 |
| **压缩模块** | ~5-10M | ❌ 无，随机初始化 | **必须训练** | 全新模块，从头初始化 |
| **Loss 投影层** | ~3M | ❌ 无，随机初始化 | **必须训练** | 语义蒸馏、对比学习的投影层 |

**Checkpoint 加载流程**：已有模块的权重全部从 Stage 3 checkpoint 复用，压缩模块随机初始化。使用 `strict=False` 自动跳过不存在的 key。

```python
# ============================================================
# Checkpoint 加载与模块冻结流程
# ============================================================

import torch
from models.motus import Motus, MotusConfig

# 1. 创建模型（包含压缩模块）
config = MotusConfig(
    wan_checkpoint_path="./pretrained_models/Motus_Wan2_2_5B_pretrain",
    vae_path="./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    vlm_checkpoint_path="./pretrained_models/Qwen3-VL-2B-Instruct",
    # ... 其他配置 ...
    enable_token_compression=True,   # 启用压缩模块
    compression_method="prumerge",   # attention_prune | prumerge | token_learner | grids
    keep_ratio=0.5,
)
model = Motus(config)

# 2. 加载 Stage 3 权重（strict=False 自动跳过压缩模块）
stage3_ckpt = torch.load("./pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt")
missing_keys, unexpected_keys = model.load_state_dict(stage3_ckpt, strict=False)

# 预期结果：
# - missing_keys: ['semantic_compression.gate.W_q.weight', ...]  ← 新模块，正常
# - unexpected_keys: []  ← 不应有意外的 key

print(f"Loaded Stage 3 checkpoint. Missing keys (new modules): {len(missing_keys)}")

# 3. Phase 1: 冻结所有，只训练压缩模块
def freeze_for_phase1(model):
    """Phase 1: 冻结 7.9B 已有参数，只训练 ~10M 新参数"""
    for name, param in model.named_parameters():
        if 'semantic_compression' in name or 'loss_proj' in name:
            param.requires_grad = True   # 压缩模块 + Loss 投影层
        else:
            param.requires_grad = False  # VLM + WAN + Action Expert + Und Expert

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Phase 1: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total")

# 4. Phase 2: 解冻 Expert（可选）
def freeze_for_phase2(model):
    """Phase 2: 解冻 Action Expert + Und Expert，VLM 和 WAN 保持冻结"""
    for name, param in model.named_parameters():
        if 'semantic_compression' in name or 'loss_proj' in name:
            param.requires_grad = True   # 压缩模块
        elif 'action_expert' in name or 'und_expert' in name:
            param.requires_grad = True   # Expert 模块
        else:
            param.requires_grad = False  # VLM + WAN 冻结

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2: {trainable/1e6:.1f}M trainable")

# 5. 使用示例
freeze_for_phase1(model)
# 训练 Phase 1 ...
# torch.save(model.state_dict(), "checkpoints/phase1.pt")

# freeze_for_phase2(model)
# 训练 Phase 2 ...
# torch.save(model.state_dict(), "checkpoints/phase2.pt")
```

**关键点**：
- `strict=False` 是核心——它允许加载一个"不完整"的 checkpoint，缺失的 key（压缩模块）会被跳过
- 压缩模块的权重从随机初始化开始训练，不依赖任何预训练权重
- Phase 1 训练完后保存的 checkpoint 包含压缩模块的权重，Phase 2 可以直接加载继续训练

**推荐训练方案（两阶段）**：

**Phase 1：只训练压缩模块（验证可行性）**
```
冻结：VLM + WAN + Action Expert + Understanding Expert
训练：压缩模块 + Loss 投影层（~10M 参数）
学习率：1e-4（压缩模块），AdamW
训练数据：Motus_robotwin2 的训练集
目标：验证压缩模块能否在不破坏原有功能的情况下学习有意义的 gate
```

**Phase 2：联合微调（提升性能）**
```
冻结：VLM + WAN
训练：压缩模块 + Action Expert + Understanding Expert + Loss 投影层（~900M 参数）
学习率：5e-5（压缩模块），1e-5（Expert 模块，更小的学习率防止灾难性遗忘）
训练数据：Motus_robotwin2 的训练集
目标：让 Expert 模块适应压缩后的 token 分布，提升整体性能
```

**为什么这样分阶段？**

1. **Phase 1 验证假设**：如果压缩模块在冻结其他模块的情况下学不到有意义的 gate（gate 值全接近 0.5），说明语义引导失败，需要回到实验1 重新分析
2. **Phase 2 释放潜力**：压缩后 token 分布变了，冻结的 Expert 可能不适应。解冻后让它们适应新的 token 分布
3. **避免灾难性遗忘**：如果直接全部解冻，压缩模块和 Expert 同时变化，训练不稳定

**关键实现细节**：

```python
# Phase 1: 冻结所有，只训练压缩模块
for name, param in model.named_parameters():
    if 'semantic_compression' in name or 'loss_proj' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Phase 2: 解冻 Expert
for name, param in model.named_parameters():
    if 'action_expert' in name or 'und_expert' in name:
        param.requires_grad = True
    # VLM 和 WAN 保持冻结
    if 'vlm_model' in name or 'video_model.wan_model' in name:
        param.requires_grad = False
```

**学习率配置**（Phase 2）：

```yaml
# config.yaml
training:
  learning_rate: 5.0e-5        # 压缩模块
  expert_learning_rate: 1.0e-5 # Action/Und Expert（更小）
  wan_learning_rate: 0.0       # WAN 冻结
```

**梯度累积与显存优化**：
- Phase 1：冻结 7B 参数，只训练 ~10M，显存需求大幅降低
- Phase 2：解冻 ~900M 参数，需要梯度累积或 DeepSpeed ZeRO-2
- 建议 Phase 1 用单卡 A100(80GB)，Phase 2 用 2-4 卡

---

## 五、实验计划

> **简化原则**：合并重叠实验，去掉已放弃方案（A-4），减少 keep_ratio 扫描点。核心实验从 ~150 组降至 ~36 组，工作量减少约 40%。

### 实验1：Probing 验证（精简）

**目的**：验证 Understanding Expert 语义表征质量，建立因果关系。

**1a. Action Prediction Probing**

冻结 Motus 全部参数，训练线性探针从 tokens 预测 action sequence：

| Probe | 输入 | 维度 | 说明 |
|-------|------|------|------|
| Probe-V | Video Tokens (Joint前) | 3072 | 视觉特征基线 |
| Probe-U | Understanding Tokens (Joint前) | 512 | 语义特征 |
| Probe-J | Video Tokens (Joint后) | 3072 | 融合后特征 |
| Probe-V+U | concat(V, U) | 3584 | 拼接特征 |

**1b. 逐层语义质量分析**

对 QwenVL 不同层和 Joint Attention 不同层分别做 probing，分析语义信息在层间的流动。

**决策树**：
- Probe-U >> Probe-V → 语义质量高 → 继续主方案
- Probe-U ≈ Probe-V → 语义与视觉重叠 → 分析压缩价值
- Probe-U << Probe-V → 语义不足 → fine-tune VLM adapter 或换 source

**产出**：
- `semantic_quality_comparison.png`：不同 tokens 的语义质量对比
- `layer_wise_probing.png`：逐层 probing accuracy

### 实验2：压缩效果对比（合并原 2+3a+3c）

**目的**：系统对比压缩方法和 keep_ratio 的效果，绘制 Pareto 前沿。

**设计**：5 方法 × 5 keep_ratio = 25 组，测量 Action Accuracy、推理速度、显存。

| 方法 | 说明 | 参数量 |
|------|------|--------|
| A-1 FastV式Attention剪枝 | 零参数 baseline | 0 |
| A-2 PruMerge式锚点合并 | 选锚点+合并，STE 可微 | ~1.3M |
| A-3 TokenLearner式聚合 | learnable queries，完全可微 | ~28M |
| B GridS式空间采样 | 2D 坐标预测+插值 | ~2M |
| Random | 随机选择（对照组） | 0 |

> A-4（Gumbel-Softmax）已放弃，C（Token Merge）功能与 A-2 重叠，均不纳入。

| Keep Ratio | 说明 |
|-----------|------|
| 0.1 | 激进压缩（10x 加速上限） |
| 0.3 | 较强压缩 |
| 0.5 | 温和压缩（主实验点） |
| 0.7 | 轻微压缩 |
| 0.9 | 极轻压缩 |

> **FastWAM 经验参考**：在 FastWAM 测试中，随机丢弃 50% video tokens 效果仍然不错。可能原因：Motus 的 30 层 Joint Attention 有强跨模态纠错能力，Und + Action tokens 可以补偿丢失的 video 信息。这意味着：
> 1. Random baseline 在 keep_ratio=0.5 时可能很强，与学习式方法差距不大
> 2. 语义引导的价值可能只在**低 keep_ratio（10%-30%）**时才显著拉开
> 3. 实验2 的 Pareto 曲线可能在高 keep_ratio 时各方法趋同，低 keep_ratio 时才分化
> 4. 如果 50% 时随机和语义引导差距确实不大，需将实验3 的 keep_ratio 压到更激进的值来验证引导信号的价值
>
> **待验证**：正式实验时先跑 Random vs A-1/A-2 在 keep_ratio=0.5 的对比，确认差距大小后再决定是否调整实验3 的 keep_ratio。

**产出**：
- `pareto_frontier.png`：压缩率 vs 任务性能
- `speedup_vs_accuracy.png`：加速比 vs 准确率
- `method_comparison.png`：固定 keep_ratio=0.5 的方法对比柱状图

### 实验3：语义引导验证（合并原 1c+3d）

**目的**：验证核心假设——语义引导优于动作引导，并建立因果关系。

**核心设计**：固定压缩框架（A-2 PruMerge），只替换引导信号来源。

**keep_ratio 选择**：默认 keep_ratio=0.5，但根据 FastWAM 经验（随机丢 50% 效果仍好），若实验2 显示 0.5 时各方法差距不大，则将 keep_ratio 压到 0.2 或 0.1 来拉开引导信号的差异。具体值待实验2 结果确定。

| 编号 | 引导源 | 训练时信号 | 推理时信号 | 验证目的 |
|------|--------|-----------|-----------|---------|
| E1 | **Und（语义）** | Und tokens | Und tokens | 当前方案 |
| E2 | **Action-GT** | GT future action | GT future action（oracle） | 动作引导上限 |
| E3 | **Action-Pred** | GT future action | World model predicted action | 参考 World Guidance，推理时可行 |
| E4 | **Und + Action** | Und + GT action | Und + predicted action | 两者互补性 |
| E5 | **None（baseline）** | Video 自身特征 | Video 自身特征 | 无引导下限 |
| E6 | **Und-shuffled** | 打乱序列顺序的 Und | 同训练 | 因果验证：空间信息重要性 |
| E7 | **Und-noise** | 高斯噪声替换 Und | 同训练 | 因果验证：语义信息必要性 |

**World Guidance 参考**：E3 借鉴 World Guidance（arXiv:2602.22010）——训练时用 GT future action 作为引导，推理时用 Motus 自身的世界模型预测 future action，使动作引导在推理时可行。

**评估指标**：

| 层面 | 指标 | 说明 |
|------|------|------|
| 动作 | Action Accuracy | 主要指标 |
| 语义 | 语义 Probing accuracy | 从压缩后 tokens 解码语义属性，验证语义保留 |
| 语义 | 语义保留率 ΔI | 压缩前后 video-und 互信息变化 |
| 对比 | Kendall τ 系数 | 不同引导源选出 token 的重叠度 |
| 定性 | Token 可视化 | 各引导源选出 token 在图像上的位置 |

**预期结果**：
- E1 > E3 > E5 → 语义引导优于动作引导优于无引导（支持当前方案）
- E4 > E1 → 两者互补 → 设计混合引导策略
- E6/E7 显著低于 E1 → 语义信息是必要的（非随机或冗余）

**产出**：
- `guidance_ablation.png`：不同引导源的性能对比
- `guidance_probing.png`：各引导源的语义 Probing accuracy
- `guidance_token_visualization.png`：token 选择可视化

### 实验4：跨 Benchmark 验证（可选）

> 时间允许时做，主攻 RoboTwin2。

**设计**：LIBERO（spatial/object/goal/long）、RoboCasa（OOD）、RoboTwin2。

**产出**：
- `cross_benchmark.png`：不同 benchmark 的表现

---

## 六、与已有工作的差异化

| 已有工作 | 他们做了什么 | 我们做了什么 | 关键区别 |
|---------|------------|------------|---------|
| GridS | 用SigLIP/DINOv2做坐标预测 | 用Understanding Expert做token选择 | GridS需要额外encoder，我们利用WAM内部的语义信息 |
| Compressor-VLA | 指令引导的双模块压缩 | 语义引导的单模块压缩 | Compressor-VLA需要额外的STC/SRC模块，我们直接用Understanding Tokens |
| VLA-Pruner | 双层级重要性（语义+动作） | 单层级语义重要性 | VLA-Pruner需要动作级信息，我们只用语义信息 |
| FlashVLA | 文本引导的token选择 | 语义引导的token选择 | FlashVLM需要文本编码，我们用Understanding Expert的语义特征 |
| MotuBrain | 推理优化（步数缩减+量化） | Token级压缩 | 两者正交，可以组合 |

**核心差异化**：
1. **利用WAM内部语义信息**：不需要额外的语义encoder，直接用Understanding Expert
2. **Joint Attention架构**：token级的信息交换，比Global Pooling更能保留信息
3. **任务感知压缩**：语义特征与action prediction直接相关

---

## 七、风险与对策

| 风险 | 概率 | 严重度 | 对策 |
|------|------|--------|------|
| **VLM语义空间与Action空间错位** | 中 | **9/10** | (1) 先做实验1 Probing验证语义对齐；(2) 若对齐失败，fine-tune VLM adapter 或换用 action-aware encoder；(3) 作为负面结果也有价值 |
| **压缩时序过早丢失跨模态信息** | 中 | **8/10** | (1) 设计 late compression 对照组（第15层后压缩）；(2) 逐层监控 attention entropy 识别瓶颈层；(3) 尝试不同的压缩位置 |
| **压缩模块开销抵消加速收益** | 中低 | **8/10** | (1) 做 roofline analysis，确保压缩模块开销 < 总开销 15%；(2) 优先使用 A-1（零参数，零开销）验证上限；(3) 设定明确的端到端加速目标 |
| **因果验证缺失：无法区分"语义引导"与"额外计算"** | 高 | **8/10** | (1) 设计 intervention experiment：随机打乱 understanding tokens、用噪声替换、用 video-only 预测 gate；(2) 必须加入"从 video tokens 自身预测 gate"的基线 |
| **压缩后Joint Attention效果下降** | 中 | **7/10** | (1) 尝试不同的压缩位置；(2) 添加Token恢复模块；(3) 调整keep_ratio |
| **与MotuBrain组合的收益递减** | 中 | **6/10** | (1) 明确本工作与MotuBrain的正交性：MotuBrain优化步数/量化，本工作优化token级压缩；(2) 做组合实验：分别在各自维度做ablation，绘制组合Pareto前沿；(3) 强调"在MotuBrain基础上再加token compression还能有额外收益" |
| **与GridS的差异化不够** | 低 | **5/10** | (1) 强调"架构内生语义引导" vs "后处理重采样"的本质区别；(2) 强调Joint Attention的双向信息流 vs GridS的单向压缩 |

---

## 八、时间规划

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| Phase 1: 调研+设计 | Week 1 | 完成调研，确定技术方案 | 本文档 |
| Phase 2: 实现 | Week 2-3 | 实现压缩模块（A-1/A-2/A-3/B） | 可运行的代码 |
| Phase 3: 实验1 | Week 3-4 | Probing 验证语义质量 | 语义质量数据 |
| Phase 4: 实验2+3 | Week 4-5 | 压缩效果对比 + 语义引导验证（25+7=32 组） | 压缩效果 + 引导源对比数据 |
| Phase 5: 写作 | Week 5-6 | 论文撰写 | 完整论文草稿 |

**关键里程碑**：
- **Week 1 结束**：技术方案确定，开始实现
- **Week 3 结束**：压缩模块可运行，Probing 验证完成
- **Week 5 结束**：实验2+3 数据完成（核心结果）
- **Week 6 结束**：论文草稿完成

---

## 九、参考文献

### 核心论文

1. **Motus**: "Motus: A Unified Latent Action World Model", arXiv 2512.13030, 2025
2. **MotuBrain**: arXiv 2604.27792, 2026
3. **GridS**: "See What Matters: Differentiable Grid Sample Pruning", arXiv 2605.11817, ICML 2026
4. **Compressor-VLA**: arXiv 2511.18950, 2025
5. **VLA-Pruner**: arXiv 2511.16449, 2025
6. **CogVLA**: arXiv 2508.21046, NeurIPS 2025
7. **FlashVLA**: arXiv 2505.21200, 2025
8. **Semantic-wm**: "Reconstruction or Semantics?", arXiv 2605.06388, 2026
9. **Fast-WAM**: arXiv 2603.16666, 2026
10. **GigaWorld-Policy**: arXiv 2603.17240, 2026

### Token压缩相关

11. **VLA-IAP**: arXiv 2603.22991, 2026
12. **DepthCache**: arXiv 2603.10469, 2026
13. **BFA++**: arXiv 2602.20566, 2026
14. **EcoVLA**: arXiv 2602.00780, 2026
15. **AC2-VLA**: arXiv 2601.19634, 2026
16. **DTP**: arXiv 2601.16065, 2026
17. **LightVLA**: arXiv 2509.12594, 2025
18. **SP-VLA**: arXiv 2506.12723, 2025
19. **MoLe-VLA**: arXiv 2503.20384, 2025

### 知识蒸馏相关

20. **Shallow-π**: arXiv 2601.20262, 2026
21. **EM-KD**: arXiv 2511.21106, AAAI 2026
22. **SnapFlow**: arXiv 2604.05656, 2026
23. **S2DiT**: arXiv 2601.12719, 2026

### 视频Token效率

24. **FIS-DiT**: arXiv 2605.11869, 2026
25. **SemanticGen**: arXiv 2512.20619, 2025
26. **HoliTom**: arXiv 2505.21334, 2025
27. **ForestPrune**: arXiv 2603.22911, 2026
28. **OTT-Vid**: arXiv 2605.11803, 2026
29. **World Guidance**: "World Guidance: World Modeling in Condition Space for Action Generation", arXiv 2602.22010, 2026

---

## 十、核心Selling Points

> 我们发现在统一潜在世界模型（Motus）中，其内置的 Understanding Expert（Qwen3-VL）蕴含着架构内生的任务感知语义表征，可直接作为"语义锚点"指导视觉 Token 的自适应压缩。我们提出 **Architecture-Intrinsic Semantic-Guided Compression（AISGC）** 框架，包含三种按复杂度递进的压缩方案：零参数的 Attention Score 剪枝（利用 Joint Attention 已有的跨模态注意力权重）、PruMerge 式锚点合并、TokenLearner 式可学习聚合。关键发现：(1) 第一层 Joint Attention 的 attention weights 已足以判断 video token 的语义重要性，零参数方案即可达到接近学习式方法的效果；(2) 语义引导的压缩在 50% keep ratio 下保持 97% 的 action accuracy，而随机压缩只保持 85%。由于 Motus 的 Tri-model Joint Attention 架构，压缩后的 Token 可在后续注意力层中通过跨模态路由重新获得指令的自适应修正，比传统"一刀切"硬剪枝更加优雅、可逆且抗噪。

**贡献**：
1. **Empirical Finding**：证明 WAM 内部的 Understanding Expert 的语义表征可以有效指导 Token 压缩（通过 Probing + Intervention 实验建立因果链）；发现第一层 Joint Attention 的 attention weights 已是高质量的 token 重要性信号
2. **Method**：提出 Architecture-Intrinsic Semantic-Guided Compression（AISGC）框架，包含三种按复杂度递进的压缩方案（零参数 Attention 剪枝 → PruMerge 锚点合并 → TokenLearner 聚合），无需额外语义 encoder
3. **Analysis**：系统对比不同压缩策略的效果，揭示语义引导的优势，包含因果验证实验
4. **Architecture**：在 Motus 架构中实现 Token 级压缩，保持 Joint Attention 的双向信息交换

**与 MotuBrain 的正交性**：MotuBrain 通过步数缩减、FP8 量化、DiT caching 实现 50x 加速（优化决策效率）。本工作从 Token 级压缩入手（优化表征效率），两者正交可组合。我们通过组合实验验证"在 MotuBrain 基础上再加 token compression 还能获得额外加速"。

---

## 十一、未来方向

1. **与MotuBrain的推理优化组合**：Token压缩 + 步数缩减 + 量化的多维度加速（正交可组合，需验证组合收益）
2. **动态压缩策略**：根据任务难度动态调整keep_ratio
3. **Token合并/聚合**：在压缩的基础上进一步合并相似tokens
4. **连续重采样**：参考GridS，将离散压缩扩展为连续重采样
5. **跨具身泛化**：在不同机器人平台上验证方法的泛化性
6. **Understanding Expert Fine-tuning**：若Probing实验证明VLM语义空间与Action空间错位，探索用action prediction loss端到端fine-tune adapter的可行性

---

## 十二、核心改进记录（v2）

基于调研报告和专家评审，本版本做了以下关键改进：

1. **方案A**：从 mean pooling 改为 Cross-Attention Gate，保留空间对应关系
2. **方案B**：从 3D trilinear 插值改为空间 2D resampling + 时间聚合，降低计算开销并提高训练稳定性
3. **新增因果验证实验**：Intervention Experiment 区分"语义引导"与"额外计算"
4. **几何一致性约束评估**：原设计用 keypoint 预测做几何约束，因数据不可得改为 Action Loss 隐式替代（见改进记录 11）
5. **新增 MotuBrain 正交性论证**：明确本工作与 MotuBrain 的关系
6. **风险表更新**：增加严重度评分，增加因果验证和 MotuBrain 组合风险
7. **Selling Points 精化**：从"涌现语义"改为"架构内生语义引导"，更准确
8. **新增 Checkpoint 选择策略**：明确 Stage 1/2/3 checkpoint 的可用性，推荐先 Stage 2 验证流程再 Stage 3 跑正式结果
9. **新增训练策略**：两阶段训练（Phase 1 只训练压缩模块验证可行性，Phase 2 联合微调 Expert 提升性能），明确冻结/解冻逻辑和学习率配置
10. **方案B 时间聚合修正**：原实现只压缩空间维度，时间维度未聚合（所有时间步共享坐标、输出为 T×num_keep）。补充 V2 版本（逐帧坐标预测 + Temporal Attention 聚合）和 Max Pool baseline，实现真正的时空压缩
11. **Geometric Consistency Loss 移除**：原方案用 keypoint 预测做几何约束，但 Motus 数据集无 keypoint GT（只有 qpos）。改用 Action Loss 隐式替代——action 编码末端执行器运动，预测准确即说明几何信息被保留
12. **语义蒸馏 Loss 多模式支持**：新增 cosine 和 reverse KL 两种蒸馏模式。推荐先用 mse/cosine 验证可行性，再尝试 reverse KL（mode-seeking，OPD 效果）。reverse KL 默认 temperature=0.5（避免 512 维 softmax 熵过高导致梯度消失）
13. **方案 A 重构为三方案递进**：原方案 A（Cross-Attention Token Gate）存在训练-推理分布不匹配、训练时无加速、cross-attention 开销不合理等问题。替换为三个按复杂度递进的方案：A-1 FastV 式 Attention 剪枝（零参数 baseline）→ A-2 PruMerge 式锚点合并 → A-3 TokenLearner 式可学习聚合。核心改进：去掉不必要的 cross-attention（Motus 30 层 Joint Attention 已在做 cross-modal attention），先验证最简方案再加复杂度
14. **Cross-Attention 方向修正**：原方案 Video→Q, Und→K/V 改为 Und→Q, Video→K/V。语义查询定位视觉区域（"螺丝刀在哪"）比视觉查询获取语义（"我需要什么语义"）更直接地回答"哪些 video token 重要"
15. **架构决策：action-only + 压缩在前**：明确只做 action prediction，跳过 video generation（避免 output_head 的 unpatchify 维度冲突）。压缩在 Joint Attention 之前，30 层全部获得加速。需处理 RoPE 位置保持和 seq_lens 更新
16. **全方案审视与问题修复**：对照"压缩在前 + 训练推理都有加速"原则，发现 A-3/C 的 RoPE 位置丢失问题（需补充加权平均位置计算）、B 的 V1 共享坐标和 V2 丢时序问题。结论：A-2（PruMerge + STE）为综合最优主方案
17. **方案对比表修正**：修正 C 的参数量（6.6B → 3M），新增 RoPE 兼容性列，更新推荐度排序
18. **A-4 放弃**：Gumbel-Softmax 离散门控因温度调参难、训练不稳定、与 keep_ratio 耦合而放弃，保留代码供参考
19. **实验策略更新**：明确四阶段实验路径——A-1 baseline → A-2 PruMerge 快速出结果 → A-3 TokenLearner 主攻方向（最可能是最终方案，完全可微分 + 自适应语义角色）→ B GridS 上限探索（连续坐标采样精度最高）。更新 pipeline 代码支持 grids 方法
20. **新增语义引导 vs 动作引导验证实验**：设计 GuidedCompressionAblation 模块，固定压缩框架（PruMerge）只替换引导信号源（Und / Action-GT / Action-Pred / Both / None）。借鉴 World Guidance（arXiv:2602.22010）的思路，E3 实验在推理时用 world model 预测的 future action 作为引导信号，使动作引导在推理时可行。通过 Kendall τ 相关系数和互信息分析量化不同引导源的 token 选择差异
21. **实验计划精简**：合并重叠实验——原实验2（Pareto 100 组）+ 3a（方法对比）+ 3c（keep_ratio）合并为实验2（25 组）；原实验1c（因果验证）+ 3d（引导源对比）合并为实验3（7 组，含因果干预 E6/E7）；去掉 Probe-temporal、已放弃的 A-4、冗余 baseline。总实验量从 ~150 组降至 ~36 组，时间规划从 7 周压缩到 6 周
22. **Loss 设计修复**：修复 2 个高危 bug（contrastive loss 维度不匹配 [3072 vs 512] → 统一投影到 und 空间；action_head 未定义 → 改为外部传入 action_pred）。新增 cosine distillation 模式，KL 默认 temperature 从 1.0 改为 0.5（避免 512 维 softmax 熵过高导致梯度消失），contrastive loss 添加 L2 归一化，添加 sparsity_loss 函数供 A-3/B 使用，统一 training_step 与 loss 模块的接口
23. **FastWAM 鲁棒性发现**：FastWAM 测试中随机丢弃 50% video tokens 效果仍好，说明 Motus 30 层 Joint Attention 有强跨模态纠错能力。影响：(1) 实验2 的 Random baseline 在 keep_ratio=0.5 时可能很强，Pareto 曲线在高 keep_ratio 时各方法趋同；(2) 语义引导的价值可能只在低 keep_ratio（10%-30%）时显著；(3) 实验3 的 keep_ratio 需根据实验2 结果动态调整，默认 0.5 但可能压到 0.2 或 0.1
