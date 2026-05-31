# 静态语义 vs. 时空动态：Token 压缩中的对齐问题分析与验证计划

**问题**：Motus 的 Understanding Expert（QwenVL）只接收 first_frame + text instruction，输出静态语义 tokens。这些 tokens 如何有效引导编码了连续帧间时空变化的 Video Tokens 的压缩？
**日期**：2026-05-22

---

## 一、问题精确定位

### 1.1 架构事实

```
Understanding 路径（静态）：
  first_frame [B,3,384,320] + text instruction
  → frozen Qwen3-VL → last hidden states [B, seq_len, 2048]
  → vlm_adapter (MLP 2048→512) → understanding tokens [B, L, 512]

Video 路径（动态）：
  video_frames [B,9,384,320] (3 condition + 6 target)
  → Wan2.2 VAE → latent [B, 48, 3, 12, 10]
  → patch_embedding (Conv3d 48→3072) → video tokens [B, 90, 3072]
  （含 3D RoPE 编码的 (t, h, w) 位置信息）
```

### 1.2 核心矛盾

| 维度 | Understanding Tokens | Video Tokens |
|------|---------------------|--------------|
| 输入 | 单帧图像 + 文本 | 多帧视频（3+6 帧） |
| 时间信息 | **零**（纯静态 snapshot） | **丰富**（3D RoPE 编码时空位置） |
| 语义信息 | **丰富**（物体识别、空间关系、指令理解） | **弱**（原始视觉特征） |
| 维度 | 512 | 3072 |
| 数量 | L（可变） | 90（固定：3×10×3） |

**矛盾**：用不含时间信息的 understanding tokens 去决定哪些含时间信息的 video tokens 重要。

### 1.3 三个具体问题

**问题1：时间盲区**

Understanding tokens 知道"螺丝刀在左上角"、"夹爪在右下角"，但不知道：
- 夹爪在第几帧开始移动
- 哪一帧发生了关键接触
- 运动轨迹的时间顺序

**问题2：压缩位置的时序不对称**

压缩设计在 Joint Attention 之前（Layer 0）。但 Joint Attention 的核心功能就是让 understanding tokens 和 video tokens 做跨模态对齐——在对齐发生之前就做裁剪，等于用未对齐的信号做决策。

**问题3：RoPE 与语义引导的冲突**

Video tokens 有 3D RoPE 编码 `(t, h, w)`。语义引导选择的依据是 "哪些 token 被 understanding tokens 关注"——这是空间维度的信号，不是时空维度的。

---

## 二、验证指标体系

### 指标1：Single Frame Bias Score（SFB）— 最优先

**来源**：Lei et al., ACL 2023 "Revealing Single Frame Bias"

**目的**：量化单帧静态信息能解释多少下游任务性能。

**公式**：
```
SFB = Accuracy(单帧输入) / Accuracy(完整视频输入)
```

**实验设计**：
```
Baseline:  完整 90 token 输入（3帧 × 30 空间 token）
Ablation:  只保留 first_frame 的 30 个 token（重复或 padding 到 90）
```

**判断标准**：
| SFB 值 | 含义 | 行动 |
|--------|------|------|
| > 0.9 | 静态信息足够，temporal gap 不重要 | 当前方案可行，继续推进 |
| 0.8 - 0.9 | 灰色地带 | 需要指标 2-4 进一步判断 |
| < 0.8 | temporal 信息重要 | 需要方案 A/B/C/D/E |

**实现**：
```python
def compute_sfb(model, dataset, device):
    """
    计算 Single Frame Bias Score。
    """
    # Baseline: full input
    acc_full = evaluate(model, dataset, mode='full', device=device)
    
    # Ablation: single frame only
    acc_single = evaluate(model, dataset, mode='single_frame', device=device)
    
    sfb = acc_single / acc_full
    return sfb, acc_full, acc_single

def evaluate(model, dataset, mode='full', device='cuda'):
    """评估模型在不同输入模式下的 accuracy。"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=32):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if mode == 'single_frame':
                # 只保留 first_frame，替换 video_frames
                batch['video_frames'] = batch['first_frame'].unsqueeze(1).expand(
                    -1, batch['video_frames'].shape[1], -1, -1, -1
                )
            
            action_pred = model.inference_step(batch)
            # 计算 accuracy（根据具体任务定义）
            correct += compute_correct(action_pred, batch['action_gt'])
            total += batch['action_gt'].shape[0]
    
    return correct / total
```

**输出**：
- `sfb_score.png`: SFB 值
- `sfb_per_task.png`: 每个任务/数据集的 SFB

---

### 指标2：Temporal Gain Profile — 量化哪帧最重要

**来源**：综合 FastV (ECCV 2024) + StreamingTOM (CVPR 2026)

**目的**：找出 temporal 信息在哪些帧最重要，验证 understanding tokens 是否能识别这些帧。

**公式**：
```
TG(frame_t) = Accuracy(all_frames) - Accuracy(all_frames_except_t)
```

**实验设计**：
```
对 T=3 帧分别做 leave-one-out：
  TG(frame_0) = Acc(full) - Acc(去掉 frame_0)
  TG(frame_1) = Acc(full) - Acc(去掉 frame_1)
  TG(frame_2) = Acc(full) - Acc(去掉 frame_2)
```

**判断标准**：
| TG 分布 | 含义 |
|---------|------|
| 均匀（TG_0 ≈ TG_1 ≈ TG_2） | temporal 信息均匀分布，静态语义无法区分 |
| 某帧尖峰（TG_i >> others） | 该帧是关键帧，静态语义如果能识别它就好 |
| 全部接近 0 | temporal 信息不重要 |

**实现**：
```python
def temporal_gain_profile(model, dataset, device='cuda'):
    """计算每帧的 Temporal Gain。"""
    # Baseline
    acc_full = evaluate(model, dataset, mode='full', device=device)
    
    tg_profile = []
    for t in range(3):  # 3 帧
        acc_without_t = evaluate(model, dataset, mode=f'leave_out_frame_{t}', device=device)
        tg_t = acc_full - acc_without_t
        tg_profile.append(tg_t)
    
    return tg_profile  # [TG_0, TG_1, TG_2]
```

**可视化**：
```python
import matplotlib.pyplot as plt

def plot_tg_profile(tg_profile, save_path='tg_profile.png'):
    frames = ['Frame 0\n(Condition)', 'Frame 1\n(Mid)', 'Frame 2\n(Final)']
    colors = ['#2ecc71' if tg > 0.01 else '#e74c3c' for tg in tg_profile]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(frames, tg_profile, color=colors, edgecolor='black', linewidth=1.2)
    plt.ylabel('Temporal Gain (Δ Accuracy)', fontsize=12)
    plt.title('Per-Frame Temporal Importance', fontsize=14)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    
    for bar, tg in zip(bars, tg_profile):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{tg:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
```

---

### 指标3：Cross-Frame Semantic Redundancy + Kendall τ — 核心验证

**来源**：FrameFusion (ICCV 2025) + ForestPrune (2026)

**目的**：衡量 understanding tokens 的语义 importance 与 video tokens 的 temporal novelty 是否相关。

**公式**：
```
# Temporal novelty (无需参数)
R(v_i^t) = mean_{t' ≠ t} max_j cosine_sim(v_i^t, v_j^{t'})

# Semantic importance (来自 understanding tokens)
S_i = attention_weight(understanding_tokens → video_token_i)

# 相关性
Kendall τ = kendalltau(S, R)
```

**判断标准**：
| Kendall τ | 含义 | 行动 |
|-----------|------|------|
| > 0.5 | 语义 importance 与 temporal novelty 正相关 | 静态语义能捕捉 temporal 信息 |
| ≈ 0 | 两者无关 | 静态语义无法指导 temporal 压缩 |
| < 0 | 负相关 | 静态语义会错误地保留冗余 token |

**实现**：
```python
def compute_temporal_redundancy(video_tokens, grid_sizes):
    """
    计算每个 video token 的 cross-frame redundancy。
    
    Args:
        video_tokens: [B, N, 3072]
        grid_sizes: (T, H, W)
    Returns:
        redundancy: [B, N] — 每个 token 的 temporal novelty (越低 = 越冗余)
    """
    T, H, W = grid_sizes
    B, N, D = video_tokens.shape
    tokens_per_frame = H * W
    
    video_reshaped = video_tokens.view(B, T, tokens_per_frame, D)
    
    redundancy = torch.zeros(B, T, tokens_per_frame, device=video_tokens.device)
    
    for b in range(B):
        for t in range(T):
            for i in range(tokens_per_frame):
                sims = []
                for t2 in range(T):
                    if t2 != t:
                        # 与另一帧所有 token 的最大 cosine similarity
                        cos_sim = F.cosine_similarity(
                            video_reshaped[b, t, i].unsqueeze(0),  # [1, D]
                            video_reshaped[b, t2],                  # [H*W, D]
                            dim=-1
                        )  # [H*W]
                        sims.append(cos_sim.max())
                redundancy[b, t, i] = torch.stack(sims).mean()
    
    return redundancy.view(B, N)  # [B, N]


def compute_semantic_importance(model, batch):
    """
    从 understanding tokens 对 video tokens 的 attention 计算语义 importance。
    
    注意：需要修改 WanSelfAttention 以返回 attention weights。
    """
    # 方法1：用 Joint Attention 第一层的 attention weights
    # 需要设置 return_attn_weights=True
    
    # 方法2：简化版 — cosine similarity
    und_tokens = model.und_module.extract_und_features(batch['vlm_inputs'])
    video_tokens = model.video_module.prepare_input(batch['noisy_video_latent'])
    
    und_global = und_tokens.mean(dim=1)  # [B, 512]
    
    # 投影到相同维度
    und_proj = model.compressor.video_to_und_proj(video_tokens.mean(dim=1))  # [B, 512]
    
    # cosine similarity 作为 importance proxy
    importance = F.cosine_similarity(und_proj, und_global, dim=-1)  # [B]
    importance = importance.unsqueeze(1).expand(-1, video_tokens.shape[1])  # [B, N]
    
    return importance


def kendall_tau_analysis(video_tokens, semantic_imp, grid_sizes):
    """
    计算语义 importance 与 temporal novelty 的 Kendall τ。
    """
    from scipy.stats import kendalltau
    
    redundancy = compute_temporal_redundancy(video_tokens, grid_sizes)
    
    B = video_tokens.shape[0]
    taus = []
    
    for b in range(B):
        tau, p_value = kendalltau(
            semantic_imp[b].cpu().numpy(),
            redundancy[b].cpu().numpy()
        )
        taus.append({
            'tau': tau,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    mean_tau = np.mean([t['tau'] for t in taus])
    std_tau = np.std([t['tau'] for t in taus])
    sig_ratio = np.mean([t['significant'] for t in taus])
    
    return {
        'mean_tau': mean_tau,
        'std_tau': std_tau,
        'significant_ratio': sig_ratio,
        'per_sample': taus
    }
```

**可视化**：
```python
def plot_kendall_scatter(semantic_imp, redundancy, save_path='kendall_scatter.png'):
    """散点图：semantic importance vs temporal novelty。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：散点 + 拟合线
    ax = axes[0]
    for b in range(min(semantic_imp.shape[0], 10)):  # 前10个batch
        ax.scatter(semantic_imp[b].cpu().numpy(), 
                   redundancy[b].cpu().numpy(), 
                   alpha=0.3, s=10)
    ax.set_xlabel('Semantic Importance (from Understanding Tokens)')
    ax.set_ylabel('Temporal Novelty (Cross-frame Redundancy)')
    ax.set_title('Semantic vs Temporal Importance')
    ax.axhline(y=redundancy.mean(), color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=semantic_imp.mean(), color='blue', linestyle='--', alpha=0.5)
    
    # 右图：Kendall τ 分布
    ax = axes[1]
    # ... histogram of tau values
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
```

---

### 指标4：Pareto Curve 对比 — 最终验证

**来源**：FastV (ECCV 2024)

**目的**：对比不同压缩引导信号的效率-准确率权衡。

**实验设计**：
```
4种引导信号 × 5个 keep_ratio = 20 组实验

引导信号：
  1. Random（baseline）
  2. Semantic（当前方案：understanding tokens attention）
  3. Temporal（帧间差异）
  4. Hybrid（α × Semantic + (1-α) × Temporal）

keep_ratio: [0.1, 0.3, 0.5, 0.7, 0.9]
```

**输出**：
- `pareto_comparison.png`: 4 条 Pareto 曲线
- `speedup_vs_accuracy.png`: 加速比 vs 准确率

**判断标准**：
| 结果 | 含义 |
|------|------|
| Semantic >> Temporal | 静态语义足够 |
| Temporal >> Semantic | 需要 temporal 信号 |
| Hybrid >> 两者 | 需要混合信号（最优方案） |
| 所有方法 ≈ Random | 问题不在引导信号，在于压缩本身 |

---

## 三、如果静态效果不好：解决方案

### 方案A：Temporal Adapters（推荐首选）

**来源**：Frozen BiLM / Temporal Adapters (2024)

**核心思想**：在 frozen QwenVL 中插入轻量 temporal adapter，让 VLM 输出的 understanding tokens 带有 temporal 信息。

**架构**：
```
First Frame + Text → frozen QwenVL → [Temporal Adapter] → temporal-aware understanding tokens
                      (不动)           (可训练)
```

**实现**：
```python
class TemporalAdapter(nn.Module):
    """
    轻量 temporal shift adapter，插入 VLM 中间层。
    参数量 < 1% of VLM。
    """
    def __init__(self, hidden_dim=2048, num_frames=3, bottleneck=256):
        super().__init__()
        # Bottleneck adapter
        self.down_proj = nn.Linear(hidden_dim, bottleneck)
        self.temporal_conv = nn.Conv1d(bottleneck, bottleneck, 
                                        kernel_size=num_frames, 
                                        padding=num_frames//2, 
                                        groups=bottleneck)
        self.up_proj = nn.Linear(bottleneck, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
    def forward(self, hidden_states, num_frames=3):
        """
        Args:
            hidden_states: [B, seq_len, 2048]
            num_frames: 时序帧数
        Returns:
            output: [B, seq_len, 2048] (temporal-aware)
        """
        B, L, D = hidden_states.shape
        tokens_per_frame = L // num_frames
        
        # Reshape 为帧级
        hidden_reshaped = hidden_states.view(B * tokens_per_frame, D, num_frames)
        
        # Bottleneck
        bottleneck = self.activation(self.down_proj(hidden_reshaped.permute(0, 2, 1)))
        
        # Temporal convolution (跨帧信息融合)
        temporal_out = self.temporal_conv(bottleneck)
        
        # 上投影 + 残差
        out = self.up_proj(temporal_out.permute(0, 2, 1))
        out = out.view(B, L, D)
        
        return self.norm(hidden_states + out)
```

**集成到 VLM**：
```python
# 在 UndModule.extract_und_features 中
class UndModule:
    def __init__(self, ...):
        ...
        # Temporal adapter (插入 VLM 中间层)
        self.temporal_adapter = TemporalAdapter(
            hidden_dim=2048, 
            num_frames=3,
            bottleneck=256
        )
    
    def extract_und_features(self, vlm_inputs):
        # ... 原有 VLM forward ...
        
        # 在中间层插入 temporal adapter
        hidden_states = vlm_output.hidden_states[14]  # 第14层
        hidden_states = self.temporal_adapter(hidden_states, num_frames=3)
        
        # 继续 forward 到最后一层
        # ...
        
        return adapted_features
```

**优势**：
- 参数量 < 1% of VLM (~20M)
- 只训练 adapter，frozen 权重不变
- 直接让 understanding tokens 带有 temporal 信息

**训练策略**：
```
Phase 1: 冻结 VLM + WAN，只训练 temporal adapter + 压缩模块
Phase 2: 解冻 temporal adapter + 压缩模块 + Expert modules
```

---

### 方案B：Multi-Frame VLM Input（最直接）

**来源**：LLaVA-Video (SlowFast), LLaVA-OneVision (Temporal Spacer)

**核心思想**：直接把多帧图像喂给 QwenVL，而不是只喂 first_frame。

**实现**：
```python
def preprocess_vlm_messages_multi_frame(text_instruction, frames_pil_list, processor):
    """
    输入多帧图像，而不是单帧。
    
    Args:
        text_instruction: 任务指令
        frames_pil_list: [first_frame, mid_frame, last_frame] 或 [first_frame, last_frame]
        processor: QwenVL processor
    """
    content = []
    for i, frame_pil in enumerate(frames_pil_list):
        content.append({"type": "image", "image": frame_pil})
        if i < len(frames_pil_list) - 1:
            # Temporal spacer token
            content.append({"type": "text", "text": f"[Frame {i+1}]"})
    content.append({"type": "text", "text": text_instruction})
    
    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                      padding=True, return_tensors="pt")
    return inputs
```

**修改数据集**：
```python
# robotwin_agilex_dataset.py
def __getitem__(self, index):
    # ... 原有代码 ...
    
    # 多帧 VLM 输入
    frames_to_send = [first_frame_pil]  # 至少包含 first_frame
    
    # 添加中间帧和最后一帧
    if num_frames >= 3:
        mid_idx = len(video_frames) // 2
        frames_to_send.append(tensor_to_pil(video_frames[mid_idx]))
    frames_to_send.append(tensor_to_pil(video_frames[-1]))
    
    vlm_inputs = preprocess_vlm_messages_multi_frame(
        text_instruction, frames_to_send, self.vlm_processor
    )
    
    return {
        'first_frame': first_frame,
        'video_frames': video_frames,
        'vlm_inputs': vlm_inputs,
        # ...
    }
```

**优势**：
- 零架构修改（只改数据输入）
- QwenVL 本身就能处理多帧
- understanding tokens 自然获得 temporal 信息

**劣势**：
- VLM 计算量增加（多帧 = 多倍视觉 tokens）
- understanding tokens 序列变长，可能影响后续 Joint Attention

---

### 方案C：Hybrid Importance Signal（最小改动）

**核心思想**：不改 VLM，在压缩模块中加入 temporal 信号。

**实现**：
```python
class HybridImportance(nn.Module):
    """
    结合 spatial semantic + temporal dynamics 的混合重要性。
    零 VLM 修改，最小工程量。
    """
    def __init__(self, video_dim=3072, und_dim=512):
        super().__init__()
        # Semantic branch
        self.semantic_scorer = nn.Linear(und_dim, 1)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, video_tokens, und_tokens, grid_sizes):
        T, H, W = grid_sizes
        B, N, D = video_tokens.shape
        tokens_per_frame = H * W
        
        # 1. Semantic importance (来自 understanding tokens)
        und_global = und_tokens.mean(dim=1)  # [B, 512]
        semantic_imp = self.semantic_scorer(und_global).squeeze(-1)  # [B]
        semantic_imp = semantic_imp.unsqueeze(1).expand(-1, N)  # [B, N]
        
        # 2. Temporal importance (帧间差异，无需参数)
        video_reshaped = video_tokens.view(B, T, tokens_per_frame, D)
        temporal_diff = (video_reshaped[:, 1:] - video_reshaped[:, :-1]).norm(dim=-1)
        temporal_imp = temporal_diff.mean(dim=1)  # [B, H*W]
        temporal_imp = torch.cat([temporal_imp[:, :1], temporal_imp], dim=1)
        temporal_imp = temporal_imp.reshape(B, N)
        
        # 3. Normalize
        semantic_imp = (semantic_imp - semantic_imp.mean(dim=1, keepdim=True)) / \
                       (semantic_imp.std(dim=1, keepdim=True) + 1e-8)
        temporal_imp = (temporal_imp - temporal_imp.mean(dim=1, keepdim=True)) / \
                       (temporal_imp.std(dim=1, keepdim=True) + 1e-8)
        
        # 4. Fusion
        combined = torch.stack([semantic_imp, temporal_imp], dim=-1)  # [B, N, 2]
        importance = self.fusion(combined).squeeze(-1)  # [B, N]
        
        return importance
```

---

### 方案D：Keyframe-Aware Compression

**来源**：CogVLM2-Video, StreamingTOM

**核心思想**：先识别关键帧，然后在关键帧上做更细粒度的压缩。

**实现**：
```python
class KeyframeAwareCompression(nn.Module):
    """关键帧感知压缩：动态分配不同帧的 token 预算。"""
    
    def __init__(self, video_dim=3072, und_dim=512):
        super().__init__()
        # 关键帧 importance 评分器
        self.frame_scorer = nn.Sequential(
            nn.Linear(video_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, video_tokens, und_tokens, grid_sizes, total_budget=45):
        T, H, W = grid_sizes
        B, N, D = video_tokens.shape
        tokens_per_frame = H * W
        
        video_reshaped = video_tokens.view(B, T, tokens_per_frame, D)
        
        # 1. Temporal importance (帧间差异)
        frame_diff = (video_reshaped[:, 1:] - video_reshaped[:, :-1]).norm(dim=-1).mean(dim=-1)
        frame_importance = torch.cat([frame_diff[:, :1], frame_diff], dim=1)  # [B, T]
        
        # 2. Semantic importance
        semantic_score = self.frame_scorer(video_reshaped.mean(dim=2)).squeeze(-1)  # [B, T]
        
        # 3. 混合并分配预算
        combined = F.softmax(frame_importance + semantic_score, dim=-1)
        budget_per_frame = (combined * total_budget).int().clamp(min=2)
        
        # 4. 每帧内选 top-k
        selected = []
        for t in range(T):
            frame_tokens = video_reshaped[:, t]  # [B, H*W, D]
            for b in range(B):
                k = budget_per_frame[b, t].item()
                _, idx = frame_tokens[b].norm(dim=-1).topk(k)
                selected.append(frame_tokens[b, idx])
        
        return torch.cat(selected, dim=1)  # [B, K, D]
```

---

### 方案E：SlowFast Dual Pathway

**来源**：SlowFast Networks (2019), SlowFast-LLaVA (2024)

**核心思想**：将 video tokens 分为 Slow（语义）和 Fast（运动），分别处理后融合。

**架构**：
```
Video Tokens [B, 90, 3072]
    │
    ├── Slow Pathway: 关键帧的完整空间 token
    │   → 取 frame_0 和 frame_2 的全部 30 token = 60 token
    │   → Slow Tokens [B, 60, 3072]
    │
    └── Fast Pathway: 所有帧的 motion-relevant token
        → 每帧只取帧间差异最大的 10 个 token
        → Fast Tokens [B, 30, 3072]
    
    → Concat → [B, 90, 3072] (token 数不变)
    → Understanding tokens 与 Slow 分支做语义 attention
    → Understanding tokens 与 Fast 分支做 motion attention
```

---

## 四、方案对比

| 方案 | 参数量 | VLM 修改 | 训练复杂度 | 预期效果 | 推荐度 |
|------|--------|---------|-----------|---------|--------|
| **A. Temporal Adapters** | ~20M | 中（插入层） | 低 | 高（系统性解决） | **首选** |
| **B. Multi-Frame Input** | 0 | 无 | 极低 | 中（依赖 VLM 能力） | 快速验证 |
| **C. Hybrid Signal** | ~1M | 无 | 低 | 中（temporal 信号简单） | 最小改动 |
| **D. Keyframe-Aware** | ~2M | 无 | 低 | 中高（动态预算分配） | 灵活 |
| **E. SlowFast** | ~5M | 无 | 中 | 高（双路径互补） | 架构优雅 |

---

## 五、实验路径

```
Phase 1: 诊断（1-2天）
├── Step 1: SFB Score → 量化 temporal 信息的重要性
├── Step 2: TG Profile → 找出哪帧最重要
└── Step 3: Kendall τ → 语义与 temporal novelty 的相关性

Phase 2: 决策（基于 Phase 1 结果）
├── 如果 SFB > 0.9 → 当前方案可行，继续推进原计划
├── 如果 SFB < 0.8 → 需要方案 A/B/C/D/E
└── 如果 0.8 < SFB < 0.9 → 用 Pareto Curve 进一步判断

Phase 3: 实现（如果需要改进）
├── 快速验证: 方案 B（Multi-Frame Input）— 1天
├── 最小改动: 方案 C（Hybrid Signal）— 2天
├── 首选方案: 方案 A（Temporal Adapters）— 3-5天
└── 架构方案: 方案 E（SlowFast）— 5-7天
```

---

## 六、文献支撑

### 核心论文

| 论文 | 年份 | 关键发现 | 对本问题的启示 |
|------|------|---------|--------------|
| **Revealing Single Frame Bias** (Lei et al.) | ACL 2023 | 很多 video 任务 SFB > 0.85 | 静态信息可能足够 |
| **FastV** (Chen et al.) | ECCV 2024 | attention 在前几层收敛 | Layer 1 attention 可做 importance 判断 |
| **OTT-Vid** | 2026 | OT 分配 temporal budget | 直接解决 static-temporal gap |
| **ForestPrune** | 2026 | token forest 捕捉跨帧关系 | 语义相似性可识别 temporal novelty |
| **StreamingTOM** | CVPR 2026 | 帧间变化驱动压缩 | temporal novelty 指导 token 选择 |
| **LLaVA-OneVision** | 2024 | Temporal Spacer + attention | 冻结 VLM 也能做好 video understanding |
| **SlowFast** (Feichtenhofer et al.) | 2019 | Slow pathway 单独就很强 | 静态语义对大多数任务可能足够 |
| **CogVideoX** | ICLR 2025 | Expert Transformer 融合 static/dynamic | adaptive LayerNorm 做跨模态对齐 |
| **VideoLLaMA 2** | 2024 | STC connector 桥接视觉和语言 | 专用对齐模块可解决 gap |
| **PVC** | 2024 | 渐进式编码统一 image/video | 同一机制处理静态和动态 |

### 关键发现总结

1. **静态特征比想象中强大**：SlowFast 的 Slow pathway 单独就很强，很多 video 任务 85%+ 性能来自单帧
2. **Attention 天然补偿 temporal gap**：LLaVA-OneVision 证明冻结 VLM + attention 就能做好 video understanding
3. **OTT-Vid 直接解决**：用 optimal transport 计算帧间语义距离，动态分配 token budget
4. **训练免费方法有效**：ForestPrune、OTT-Vid 等无需额外训练就能实现 temporal-aware 压缩

---

## 七、风险与对策

| 风险 | 概率 | 对策 |
|------|------|------|
| SFB > 0.9，但压缩后性能仍下降 | 中 | 检查是否是 RoPE 位置丢失问题，而非 temporal gap |
| Temporal adapter 训练不稳定 | 中低 | 用 LoRA 替代 full adapter，降低学习率 |
| Multi-frame VLM 输入导致 understanding tokens 过长 | 高 | 用 temporal spacer + 空间 pooling 控制长度 |
| Hybrid signal 的 α 调参困难 | 中 | 用 uncertainty weighting (Kendall et al.) 自动学习 |
| 方案间对比不公平 | 中 | 统一总 token 预算，只改变引导信号 |
