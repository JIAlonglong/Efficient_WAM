# 实验1：Understanding Tokens 语义质量验证——详细设计

**参考实现**：`/root/intern/jialongliu/projects/semantic-wm/`（arXiv 2605.06388）
**日期**：2026-05-20

---

## 一、核心问题

计划中的实验1假设数据集有"空间理解、物体识别、指令理解"等语义标签，但 Motus 实际使用的数据集（robotwin, ac_one, lerobot）**只有原始关节位置（qpos）作为标签**，没有物体位置、空间关系、动作类型等语义标注。因此需要重新设计。

---

## 二、semantic-wm 的核心思路

semantic-wm 的比较框架：
1. 用不同 encoder（VAE/Cosmos/Qwen/DINOv2 等）提取特征
2. 在冻结的 encoder 特征上训练 probe，做二分类（成功/失败）
3. 对比不同 encoder 的 probe accuracy 和 AUC

4 种 probe 架构（定义在 `src/evaluation/probe.py`）：

| Probe | 架构 | 输入 | 适用场景 |
|-------|------|------|----------|
| `LinearProbe` | mean pool → Linear | (B,T,H,W,C) | 最基础的线性探针 |
| `TemporalProbe` | CLS + 1-layer Transformer | (B,T,H,W,C) | 捕捉时序依赖 |
| `SpatiotemporalProbe` | T×S tokens + 2-layer Transformer | (B,T,H,W,C) | 最精细的时空分析 |
| `ProgressRegressor` | per-frame Linear → MSE | (B,T,H,W,C) | 回归任务（预测时间步） |

**关键差异**：semantic-wm 做的是**二分类（success/failure）**，Motus 实验1 需要做的是**回归（预测 action sequence）**。但 probe 架构和训练框架可以高度复用。

---

## 三、Motus 适配方案

### Step 1：Token 提取（替代 semantic-wm 的 `extract_features`）

```python
# src/probe/extract_motus_tokens.py

@torch.no_grad()
def extract_motus_tokens(motus_model, batch):
    """
    从 Motus 中提取冻结的 tokens。
    参考 semantic-wm 的 extract_features()，但输入是 Motus 的多模态数据。

    Returns:
        dict with keys:
            "video_before": [B, N, 3072]  - Joint Attention 前的 video tokens
            "video_after":  [B, N, 3072]  - Joint Attention 后的 video tokens
            "und_before":   [B, L, 512]   - Joint Attention 前的 understanding tokens
            "und_after":    [B, L, 512]   - Joint Attention 后的 understanding tokens
    """
    # VAE encode + patch embedding
    video_latent = motus_model.video_module.encode_vae(batch["video_frames"])
    video_tokens_before = motus_model.video_module.prepare_input(video_latent)

    # Understanding Expert
    und_tokens_before = motus_model.und_module.extract_und_features(batch["vlm_inputs"])

    # Action Expert
    action_tokens = motus_model.action_expert.input_encoder(
        state_tokens, noisy_actions, registers
    )

    # Joint Attention（30层）
    video_tokens = video_tokens_before.clone()
    und_tokens = und_tokens_before.clone()
    for layer in motus_model.video_module.layers:
        video_tokens, action_tokens, und_tokens = layer(
            video_tokens, action_tokens, und_tokens
        )

    return {
        "video_before": video_tokens_before,   # [B, N, 3072]
        "video_after": video_tokens,            # [B, N, 3072]
        "und_before": und_tokens_before,        # [B, L, 512]
        "und_after": und_tokens,                # [B, L, 512]
    }
```

### Step 2：Probe 架构（复用 semantic-wm，适配维度）

semantic-wm 的 probe 输入是 `(B, T, H, W, C)` 的 patch 级特征。Motus 的 token 是 `(B, N, D)` 的序列，需要适配：

```python
# src/probe/motus_probe.py

class MotusLinearProbe(nn.Module):
    """
    适配 Motus token 的线性探针。
    参考 semantic-wm 的 LinearProbe (probe.py:94)，但输入是 (B, N, D) 而非 (B, T, H, W, C)。
    """
    def __init__(self, feature_dim, action_dim=14, chunk_size=16, pool_mode="mean"):
        super().__init__()
        self.pool_mode = pool_mode
        self.head = nn.Linear(feature_dim, action_dim * chunk_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def forward(self, x):
        """
        x: [B, N, D] 或 [B, T, N, D]（多帧时）
        """
        if x.dim() == 4:
            # 多帧: [B, T, N, D] → mean over N → [B, T, D]
            x = x.mean(dim=2)
        if self.pool_mode == "mean":
            x = x.mean(dim=1)  # [B, D]
        return self.head(x).view(-1, self.chunk_size, self.action_dim)


class MotusTemporalProbe(nn.Module):
    """
    适配多帧输入的时序探针。
    参考 semantic-wm 的 TemporalProbe (probe.py:127)。
    输入: 多帧的 token 序列 (B, T, N, D)
    """
    def __init__(self, feature_dim, n_frames, action_dim=14, chunk_size=16, n_heads=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=n_heads,
            dim_feedforward=feature_dim * 4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(feature_dim, action_dim * chunk_size)

    def forward(self, x):
        """
        x: [B, T, N, D] → mean pool over N → [B, T, D]
        """
        x = x.mean(dim=2)  # [B, T, D]
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, D]
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0]).view(-1, self.chunk_size, self.action_dim)


class MotusSpatiotemporalProbe(nn.Module):
    """
    时空探针：每个 spatial patch 是独立 token。
    参考 semantic-wm 的 SpatiotemporalProbe (probe.py:187)。
    """
    def __init__(self, feature_dim, n_frames, n_patches=64,
                 action_dim=14, chunk_size=16, n_heads=8):
        super().__init__()
        self.n_patches = n_patches
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.temporal_embed = nn.Parameter(torch.randn(1, n_frames, 1, feature_dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, n_patches, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=n_heads,
            dim_feedforward=feature_dim * 4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(feature_dim, action_dim * chunk_size)

    def forward(self, x):
        """
        x: [B, T, N, D] → reshape to [B, T, n_patches, D] → flatten → Transformer
        """
        B, T, N, D = x.shape
        # 假设 N 已经是 n_patches（或需要 adaptive pooling）
        x = x + self.temporal_embed + self.spatial_embed
        x = x.reshape(B, T * N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0]).view(-1, self.chunk_size, self.action_dim)
```

### Step 3：Dataset（参考 semantic-wm 的 `TrajectoryProbeDataset`）

semantic-wm 的 `TrajectoryProbeDataset`（`probe_dataset.py`）加载 MP4 + npz，返回 `(frames, actions, success, length)`。Motus 的 dataset 已有类似结构，适配如下：

```python
# src/probe/motus_probe_dataset.py

class MotusProbeDataset(Dataset):
    """
    从 Motus 训练数据中提取 token + action 标签。
    参考 semantic-wm 的 TrajectoryProbeDataset，但：
    - 不需要 success label（Motus 做回归）
    - 直接缓存提取的 tokens（避免重复前向传播）
    """
    def __init__(self, motus_model, dataloader, device, cache_dir="probe_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self.cache_dir / "tokens.pt"
        if cache_path.exists():
            data = torch.load(cache_path)
            self.video_before = data["video_before"]
            self.video_after = data["video_after"]
            self.und_before = data["und_before"]
            self.und_after = data["und_after"]
            self.actions = data["actions"]
        else:
            self._extract_and_cache(motus_model, dataloader, device, cache_path)

    def _extract_and_cache(self, model, loader, device, cache_path):
        video_before, video_after = [], []
        und_before, und_after = [], []
        actions = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting tokens"):
                tokens = extract_motus_tokens(model, batch)
                video_before.append(tokens["video_before"].cpu())
                video_after.append(tokens["video_after"].cpu())
                und_before.append(tokens["und_before"].cpu())
                und_after.append(tokens["und_after"].cpu())
                actions.append(batch["action_sequence"].cpu())

        self.video_before = torch.cat(video_before)
        self.video_after = torch.cat(video_after)
        self.und_before = torch.cat(und_before)
        self.und_after = torch.cat(und_after)
        self.actions = torch.cat(actions)

        torch.save({
            "video_before": self.video_before,
            "video_after": self.video_after,
            "und_before": self.und_before,
            "und_after": self.und_after,
            "actions": self.actions,
        }, cache_path)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "video_before": self.video_before[idx],
            "video_after": self.video_after[idx],
            "und_before": self.und_before[idx],
            "und_after": self.und_after[idx],
            "action": self.actions[idx],
        }
```

### Step 4：训练循环（参考 semantic-wm 的 `train_success_probe`）

semantic-wm 的训练循环核心结构（`train_probe.py:27`）：

```python
# 参考 train_probe.py 的 train_success_probe()，改为回归任务
def train_action_probe(args):
    """
    训练 action prediction probe。
    参考 semantic-wm 的 train_success_probe()，但：
    - loss 从 BCEWithLogitsLoss 改为 MSELoss（回归）
    - 评估指标从 accuracy/AUC 改为 MSE/Pearson correlation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载冻结的 Motus
    motus = load_frozen_motus(args)
    motus.eval()
    motus.requires_grad_(False)

    # 2. 提取 tokens
    train_dataset = MotusProbeDataset(motus, train_loader, device, "probe_cache/train")
    test_dataset = MotusProbeDataset(motus, test_loader, device, "probe_cache/test")

    train_loader_probe = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader_probe = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. 定义 probe 配置矩阵
    probe_configs = {
        # 输入空间对比
        "V_before_linear": MotusLinearProbe(3072),      # Video (Joint前)
        "U_before_linear": MotusLinearProbe(512),        # Und (Joint前)
        "V_after_linear": MotusLinearProbe(3072),        # Video (Joint后)
        "U_after_linear": MotusLinearProbe(512),         # Und (Joint后)
        "VU_before_linear": MotusLinearProbe(3072+512),  # concat(V, U)
        # Probe 架构对比
        "V_before_temporal": MotusTemporalProbe(3072, n_frames=8),
        "U_before_temporal": MotusTemporalProbe(512, n_frames=8),
    }

    # 4. 训练每个 probe
    results = {}
    for name, probe in probe_configs.items():
        probe = probe.to(device)
        optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.MSELoss()

        best_mse = float("inf")
        for epoch in range(50):
            probe.train()
            for batch in train_loader_probe:
                feature_key = "video_before" if "V_before" in name else \
                              "und_before" if "U_before" in name else \
                              "video_after" if "V_after" in name else \
                              "und_after" if "U_after" in name else None

                if feature_key is None and "VU" in name:
                    x = torch.cat([batch["video_before"], batch["und_before"]], dim=-1)
                else:
                    x = batch[feature_key]

                pred = probe(x.to(device))
                loss = criterion(pred, batch["action"].to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            # 评估
            test_mse = evaluate_probe_mse(probe, test_loader_probe, feature_key, device)
            if test_mse < best_mse:
                best_mse = test_mse
                torch.save(probe.state_dict(), f"probe_checkpoints/{name}_best.pt")

        results[name] = {"best_mse": best_mse}

    return results
```

### Step 5：评估指标

参考 semantic-wm 的评估方式，但改为回归指标：

```python
def evaluate_probe_mse(probe, loader, feature_key, device):
    """评估 probe 的 MSE 和 Pearson 相关系数。"""
    probe.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch[feature_key].to(device)
            pred = probe(x)
            all_preds.append(pred.cpu())
            all_targets.append(batch["action"])

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse = F.mse_loss(preds, targets).item()

    # Per-dimension Pearson correlation
    correlations = []
    for d in range(14):  # action_dim
        corr = np.corrcoef(preds[:, :, d].flatten().numpy(),
                           targets[:, :, d].flatten().numpy())[0, 1]
        correlations.append(corr)

    return {"mse": mse, "correlations": correlations}
```

---

## 四、完整实验矩阵

| Probe | 输入 | Pool | 目标 | 对应 semantic-wm |
|-------|------|------|------|-----------------|
| V-before-linear | Video Tokens (Joint前) | mean | Action | LinearProbe |
| U-before-linear | Und Tokens (Joint前) | mean | Action | LinearProbe |
| V-after-linear | Video Tokens (Joint后) | mean | Action | LinearProbe |
| U-after-linear | Und Tokens (Joint后) | mean | Action | LinearProbe |
| V+U-before | concat(V, U) | mean | Action | — |
| V-before-temporal | Video Tokens (多帧) | — | Action | TemporalProbe |
| U-before-temporal | Und Tokens (多帧) | — | Action | TemporalProbe |
| V-before-spatiotemporal | Video Tokens (多帧) | super_patch | Action | SpatiotemporalProbe |

### 消融对照组

| 对照 | 目的 | 实现 |
|------|------|------|
| 随机 gate | 排除 gate 机制本身的贡献 | 用随机向量替换 understanding tokens |
| 打乱顺序 | 验证空间信息的重要性 | 随机打乱 understanding tokens 的序列顺序 |
| Video-only gate | 区分"语义引导"与"额外计算" | 从 video tokens 自身预测 gate |

---

## 五、产出物

参考 semantic-wm 的可视化方式：

1. **`probe_accuracy_comparison.png`**：柱状图对比所有 probe 的 MSE
2. **`layer_wise_information.png`**：类似 semantic-wm 的 horizon curves，展示不同层的 probing accuracy
3. **`token_importance_heatmap.png`**：可视化哪些 token 被 probe 重视（参考 semantic-wm 的 PCK 空间分析）
4. **`per_dimension_correlation.png`**：14 个 action 维度的 Pearson 相关系数

---

## 六、决策树

```
实验1完成后：
├── Probe-U >> Probe-V → Understanding Tokens 语义质量高 → 继续主方案
├── Probe-U ≈ Probe-V → 语义信息与视觉信息重叠 → 需分析是否仍有压缩价值
├── Probe-U << Probe-V → Understanding Tokens 语义不足 → 需 fine-tune VLM 或换 semantic source
└── Probe-J >> Probe-U + Probe-V → Joint Attention 融合效果显著 → 考虑在 Joint Attention 之后压缩
```

---

## 七、预估时间

| 步骤 | 时间 | 资源 | 备注 |
|------|------|------|------|
| Token 提取 + 缓存 | 2-4h | 1x A100 | 一次性，离线提取 |
| 8 个 probe 训练 | 2-3h | 1x GPU | 每个 ~20min |
| 逐层分析 | 2-3h | 1x GPU | QwenVL 28层 + Joint 30层 |
| 互信息估算 | 2-3h | 1x GPU | MINE / InfoNCE |
| 可视化 | 1h | CPU | |
| **总计** | **~1 天** | | |

---

## 八、关键借鉴点总结

| semantic-wm 的做法 | Motus 适配 |
|-------------------|-----------|
| `extract_features()` 统一提取 | `extract_motus_tokens()` 统一提取三种 token |
| `create_probe()` 工厂模式 (probe.py:283) | 同样用工厂模式管理多种 probe |
| `train_success_probe()` 二分类 (train_probe.py:27) | 改为 `train_action_probe()` 回归 |
| `evaluate_probe_on_generated()` 对比真实/生成 | 可扩展为对比 Joint 前后 |
| `TrajectoryProbeDataset` (probe_dataset.py) | 复用 Motus 已有 dataset + token 缓存 |
| `LinearProbe` / `TemporalProbe` / `SpatiotemporalProbe` | 适配 (B,N,D) 输入维度 |

---

## 九、文件结构

```
Motus/
  src/
    probe/
      __init__.py
      extract_motus_tokens.py    # Token 提取
      motus_probe.py             # Probe 架构（Linear, Temporal, Spatiotemporal）
      motus_probe_dataset.py     # Dataset + 缓存
      train_action_probe.py      # 训练循环
      evaluate_probe.py          # 评估指标
      visualize_probe.py         # 可视化
  configs/
    probe.yaml                   # Probe 实验配置
  scripts/
    run_probe_experiment.sh      # 一键运行脚本
```
