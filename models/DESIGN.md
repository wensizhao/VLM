# models/ 模块设计文档（v2）

## 1. 架构总览：七层模块 + Text Memory Bank

```
Video (B,T,C,H,W)               Raw Prompt Strings
       │                                │
       ▼                                ▼
┌──────────────────┐          ┌──────────────────────┐
│ 1. Vision Encoder│          │ 2. Prompt Processor  │  prompts/processor.py
│    backbone.py    │          │    template expand   │
│    encode_video() │          │    ensemble / cache  │
└────────┬─────────┘          │    hard negatives    │
         │ (B,T,D)            └──────────┬───────────┘
         │  帧级，不池化                 │ list[str] (K prompts)
         │                               ▼
         │                    ┌──────────────────────┐
         │                    │    Text Encoder      │  backbone.py
         │                    │    encode_text()     │
         │                    └──────────┬───────────┘
         │                               │ (K, L, D)
         │                               │  token 级，不取 EOS
         └───────────────┬───────────────┘
                         ▼
              ┌─────────────────────────┐
              │ 3. Semantic Alignment   │  models/alignment.py
              │    vis_proj + txt_proj  │
              │    → shared space D_a   │
              └────────────┬────────────┘
                           │
              AlignmentOutput {
                visual_feature:   (B, T, D)
                visual_embedding: (B, T, D_a)
                text_feature:     (K, L, D)
                text_embedding:   (K, L, D_a)
              }
                           │
                           ▼
              ┌─────────────────────────┐
              │ 4. Matcher              │  models/matcher.py
              │    cosine / dot / temp  │
              └────────────┬────────────┘
                           │
              MatchOutput {
                similarity:  (B, K)
                logits:      (B, K)
              }
                           │
                           ▼
              ┌─────────────────────────┐
              │ 5. Fusion               │  models/fusion.py
              │    Text Memory context  │
              │    NO cartesian product │
              │    input: raw|emb|concat│
              └────────────┬────────────┘
                           │ (B, T, D_f)  ← B 不膨胀！
              FusionOutput {
                fused:       (B, T, D_f)
                attention:   (B, T, L_total) | None
                gate_values: (B, T, D) | None
              }
                           ▼
              ┌─────────────────────────┐
              │ 6. Temporal             │  models/temporal.py
              │    Identity / Transf.   │
              └────────────┬────────────┘
                           │ (B, T, D_f)
              TemporalOutput {
                features: (B, T, D_f)
              }
                           ▼
              ┌─────────────────────────┐
              │ 7. Head                 │  models/head.py
              │    frame_score (B,T)    │
              │    clip_score  (B,)     │
              │    embedding   (B,D_f)  │
              └────────────┬────────────┘
                           │
              HeadOutput {
                frame_score:  (B, T)
                clip_score:   (B,)
                embedding:    (B, D_f)
              }
                           │
                           ▼
              ┌─────────────────────────┐
              │  Matcher (again)        │  models/matcher.py
              │  embedding @ txt_pooled │
              │  → per-prompt anomaly   │
              │    scores (B, K)        │
              └─────────────────────────┘
```

**关键设计原则：**

1. **任何模块都不做 Cartesian Product** — Fusion 处理 `(B,T,D)`，K 个 prompt 作为 Text Memory Bank 存在
2. **B 在整个 pipeline 中不变** — `vis` 永远不复制 K 份，只在最后的 Matcher 做 dot product `(B,D_f) @ (K,D_f)^T → (B,K)`
3. **T 维全链路保留** — Backbone → Alignment → Fusion → Temporal → Head，帧级信息不丢失
4. **全部输出为 frozen dataclass** — 类型安全、debug 友好、训练 loss 可访问任意中间层
5. **Matcher 从 Alignment 和 Head 完全解耦** — 它只是一个可插拔的相似度计算器

---

## 2. 文件清单

| 文件 | 职责 | 核心类 |
|---|---|---|
| `models/outputs.py` | 统一的输出 dataclass 定义 | `AlignmentOutput`, `MatchOutput`, `FusionOutput`, `TemporalOutput`, `HeadOutput`, `ModelOutput` |
| `models/backbone.py` | Vision + Text 编码器 | `CLIPBackbone` |
| `prompts/processor.py` | Prompt 预处理管线 | `PromptProcessor` |
| `models/alignment.py` | 视觉-语言语义对齐 | `SemanticAlignment` |
| `models/matcher.py` | 相似度计算与匹配 | `Matcher` |
| `models/fusion.py` | 特征融合（Text Memory 模式） | `ConcatFusion`, `GatedFusion`, `CrossAttnFusion` |
| `models/temporal.py` | 时序建模 | `TemporalIdentity`, `TemporalTransformer` |
| `models/head.py` | 异常分数预测 | `AnomalyHead` |
| `models/__init__.py` | 顶层编排 | `VLMModel` |

---

## 3. models/outputs.py — 输出 dataclass

### 3.1 设计目标

- 所有模块返回统一、不可变的 dataclass，类型明确
- 训练 loss 函数可以直接访问中间层输出（不用传 dict key 字符串）
- None 字段表示该模块不产出此信息（如 ConcatFusion 没有 attention weights）

### 3.2 定义

```python
from dataclasses import dataclass
from torch import Tensor

@dataclass(frozen=True)
class AlignmentOutput:
    """SemanticAlignment 模块输出。"""
    visual_feature: Tensor       # (B, T, D)    — backbone 原始视觉特征
    visual_embedding: Tensor     # (B, T, D_a)  — 投影到共享空间的视觉特征
    text_feature: Tensor         # (K, L, D)    — backbone 原始文本特征
    text_embedding: Tensor       # (K, L, D_a)  — 投影到共享空间的文本特征


@dataclass(frozen=True)
class MatchOutput:
    """Matcher 模块输出。"""
    similarity: Tensor           # (B, K)       — 余弦相似度矩阵 [0, 1]
    logits: Tensor               # (B, K)       — temperature-scaled logits
    temperature: float           # 当前使用的温度值


@dataclass(frozen=True)
class FusionOutput:
    """Fusion 模块输出 — B 不膨胀，T 保留。"""
    fused: Tensor                # (B, T, D_f)  — 融合后特征
    attention: Tensor | None     # (B, T, L_total) — CrossAttn attention weights
    gate_values: Tensor | None   # (B, T, D)    — GatedFusion 门控系数


@dataclass(frozen=True)
class TemporalOutput:
    """Temporal 模块输出。"""
    features: Tensor             # (B, T, D_f)  — 时序建模后特征


@dataclass(frozen=True)
class HeadOutput:
    """Head 模块输出 — frame_score 是真正的帧级预测。"""
    frame_score: Tensor          # (B, T)       — 帧级异常分数 [0, 1]
    clip_score: Tensor           # (B,)         — clip 级 = amax(frame_score)
    embedding: Tensor            # (B, D_f)     — clip 级全局特征


@dataclass(frozen=True)
class ModelOutput:
    """VLMModel 顶层输出 — 包含所有中间模块输出。"""
    frame_score: Tensor          # (B, T)       — 最终帧级异常分数
    clip_score: Tensor           # (B,)         — 最终 clip 级异常分数
    embedding: Tensor            # (B, D_f)     — 全局特征
    anomaly_scores: Tensor       # (B, K)       — 每个 prompt 的异常匹配分数
    alignment: AlignmentOutput   # 对齐模块输出（contrastive loss 用）
    fusion: FusionOutput         # 融合模块输出（可视化用）
    temporal: TemporalOutput     # 时序模块输出
    head: HeadOutput             # 预测头输出
```

---

## 4. backbone.py — 编码器

### 4.1 设计目标

与 v1 基本一致，只有一处微调：`encode_text` 接受 `list[str]`，不再内部做 tokenizer 相关判断。Prompt 预处理由 `PromptProcessor` 完成后再传入。

### 4.2 接口

```python
class CLIPBackbone(nn.Module):
    def encode_video(self, video: Tensor) -> Tensor:
        """video: (B, T, C, H, W) → (B, T, D) 帧级 L2 归一化"""

    def encode_text(self, prompts: list[str]) -> Tensor:
        """prompts: list[str] → (K, L, D) token 级 L2 归一化"""

    @property
    def dim(self) -> int:
        """共享嵌入维度 D"""
```

### 4.3 实现要点

- `encode_video`: `(B,T,C,H,W) → (B*T,C,H,W) → model.encode_image() → (B*T,D) → (B,T,D)`
- `encode_text`: 复刻 CLIP 内部流程（token_embed → +pos → transformer → ln_final），在 `text_global_pool` 之前停下
- 如果存在 `text_projection`，逐 token 投影
- 两种特征都做 L2 normalize

---

## 5. prompts/processor.py — Prompt 处理器

### 5.1 设计目标

Prompt 管理不只是"替换模板字符串"。作为 Open-Vocabulary VAD 的入口，它需要：

- **Template Expansion**: 一个模板 → 多种变体（"a person {action}" → ["running", "fighting", ...]）
- **Ensemble**: 同时使用多种 prompt 类型（label + scene + contrast）
- **Hard Negatives**: 训练时生成易混淆的负样本
- **Cache**: 避免重复 tokenize 相同 prompt
- **Future**: 支持 Learnable Prompt / Prefix Tuning

### 5.2 接口

```python
class PromptProcessor:
    def __init__(
        self,
        templates: dict[str, str] | None = None,
        expansions: dict[str, list[str]] | None = None,
    ):
        """templates: {"label": "a person {action}", ...}
           expansions: {"action": ["running", "fighting", ...], ...}"""

    def process(
        self,
        types: list[str] | None = None,
        expand: bool = True,
        hard_negatives: bool = False,
    ) -> list[str]:
        """返回处理后的 prompt 列表。

        Args:
            types: 使用哪些模板类型，None = 全部
            expand: 是否用 expansions 展开占位符
            hard_negatives: 是否加入 hard negative prompts
        Returns:
            list[str] — K 个处理好的 prompt 字符串
        """

    @property
    def num_prompts(self) -> int:
        """process() 当前会返回多少个 prompt。"""
```

### 5.3 与旧 PromptManager 的关系

`PromptManager`（`prompts/__init__.py`）保留为底层模板存储，`PromptProcessor` 在其上构建扩展能力。

---

## 6. models/alignment.py — 语义对齐

### 6.1 设计目标

**这是 Open-Vocabulary VAD 最核心的模块。** 它把视觉和文本特征投影到共享语义空间，使得：

- "一张有人在打架的视频帧" 与 "fighting" 在空间中接近
- "正常行走" 与 "fighting" 在空间中远离
- 对齐质量直接影响零样本泛化能力——如果对齐不好，未见过的 prompt 无法匹配

### 6.2 接口

```python
class SemanticAlignment(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        aligned_dim: int = 512,
        proj_type: str = "linear",  # "linear" | "mlp"
    ):
        self.vis_proj = ...  # 视觉投影层
        self.txt_proj = ...  # 文本投影层

    def forward(
        self,
        visual_feature: Tensor,    # (B, T, D)
        text_feature: Tensor,      # (K, L, D)
    ) -> AlignmentOutput:
        """
        1. vis_proj(visual_feature) → visual_embedding (B, T, D_a)
        2. txt_proj(text_feature)   → text_embedding   (K, L, D_a)
        3. 保留 T 和 L，不做池化（池化是 Matcher 的事）
        """
```

### 6.3 设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 保留 T/L | ✅ | 下游 Fusion 需要帧级 + token 级特征 |
| 不计算 similarity | ✅ | similarity 是 Matcher 的职责 |
| proj_type = "linear" (MVP) | ✅ | 简单投影 + LayerNorm，后续可换 MLP |
| 不引入对比 loss | ✅ | loss 在外部，由 trainer 从 AlignmentOutput 中取值计算 |

### 6.4 实现

```python
class SemanticAlignment(nn.Module):
    def __init__(self, in_dim=512, aligned_dim=512, proj_type="linear"):
        super().__init__()
        if proj_type == "linear":
            self.vis_proj = nn.Sequential(
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )
            self.txt_proj = nn.Sequential(
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )
        elif proj_type == "mlp":
            self.vis_proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )
            self.txt_proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )

    def forward(self, visual_feature, text_feature):
        visual_embedding = self.vis_proj(visual_feature)    # (B, T, D_a)
        text_embedding = self.txt_proj(text_feature)        # (K, L, D_a)
        return AlignmentOutput(
            visual_feature=visual_feature,
            visual_embedding=visual_embedding,
            text_feature=text_feature,
            text_embedding=text_embedding,
        )
```

---

## 7. models/matcher.py — 相似度匹配

### 7.1 设计目标

**完全从 Alignment 和 Head 解耦。** Matcher 只是一个接受两组池化向量、返回相似度矩阵的函数模块。

可以在两个位置调用：
1. **对齐阶段**：`Matcher(vis_pooled, txt_pooled)` → 对比学习 loss 的输入
2. **推理阶段**：`Matcher(head_embedding, prompt_embeddings)` → per-prompt 异常分数

### 7.2 接口

```python
class Matcher(nn.Module):
    def __init__(
        self,
        strategy: str = "cosine",     # "cosine" | "dot" | "learnable"
        temperature: float = 0.07,
        learnable_temp: bool = True,
    ):
        """strategy 决定 similarity 计算方式，temperature 控制 logits 分布。"""

    def forward(
        self,
        query: Tensor,               # (B, D) — pooled visual / embedding
        key: Tensor,                  # (K, D) — pooled text / prompt embeddings
    ) -> MatchOutput:
        """
        similarity = cos(query, key)           # (B, K) in [−1, 1] or [0, 1]
        logits     = similarity / temperature  # (B, K) for cross-entropy
        """
```

### 7.3 策略对比

| strategy | 公式 | 适用 |
|---|---|---|
| `cosine` | `cos(q, k)` | 标准对比学习 (CLIP 风格) |
| `dot` | `q @ k.T` | 特征已 L2 归一化时等价 cosine |
| `learnable` | `q @ W @ k.T` | 学习模态间映射 |

### 7.4 实现

```python
class Matcher(nn.Module):
    def __init__(self, strategy="cosine", temperature=0.07, learnable_temp=True):
        super().__init__()
        if strategy not in ("cosine", "dot", "learnable"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy

        if strategy == "learnable":
            self.W = nn.Parameter(torch.randn(in_dim, in_dim) * 0.02)

        if learnable_temp:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * math.log(1 / temperature)
            )
        else:
            self.register_buffer(
                "logit_scale", torch.ones([]) * math.log(1 / temperature)
            )

    def forward(self, query, key):
        # 输入池化
        q = F.normalize(query, dim=-1) if self.strategy != "dot" else query
        k = F.normalize(key, dim=-1) if self.strategy != "dot" else key

        if self.strategy == "learnable":
            similarity = q @ self.W @ k.T
        else:
            similarity = q @ k.T                      # (B, K)

        logits = similarity * self.logit_scale.exp()  # (B, K)
        return MatchOutput(similarity=similarity, logits=logits,
                          temperature=1.0 / self.logit_scale.exp().item())
```

---

## 8. models/fusion.py — 特征融合（Text Memory 模式）

### 8.1 核心变更：Text Memory Bank

**不再做 Cartesian Product `(B,K,T,D)`。** 所有 K 个 prompt 的 token 序列拼接成一个 Text Memory Bank：

```
txt_embedding (K, L, D_a)
       │ reshape / flatten
       ▼
text_memory (1, K*L, D_a)  ← 当作一个大的 Key/Value 池
       │ expand to batch
       ▼
text_memory (B, K*L, D_a)
```

每个视频帧通过 Cross-Attention attend 到整个 Text Memory，产生一个 text context vector。Fusion 输出 `(B, T, D_f)`，**B 不膨胀**。

```
vis (B, T, D)  ──→  CrossAttn(q=vis, kv=text_memory)  ──→  text_context (B, T, D)
                         │
                  attn_weights (B, T, K*L)   ← 可用于可解释性
```

### 8.2 fusion_input 配置

```python
fusion = ConcatFusion(in_dim=D, out_dim=512, fusion_input="embedding")  # default
fusion = ConcatFusion(in_dim=D, out_dim=512, fusion_input="raw")
fusion = ConcatFusion(in_dim=D*2, out_dim=512, fusion_input="concat")  # concat raw + emb
```

| fusion_input | 含义 | in_dim |
|---|---|---|
| `"embedding"` | 使用 Alignment 投影后的特征 `(D_a)` | `aligned_dim` |
| `"raw"` | 使用 Backbone 原始特征 `(D)` | `backbone.dim` |
| `"concat"` | 拼接两者 `(D + D_a)` | `backbone.dim + aligned_dim` |

### 8.3 统一接口

所有 Fusion 实现相同的 forward 签名：

```python
def forward(
    self,
    visual: Tensor,              # (B, T, D_in) — 由 fusion_input 决定
    text_memory: Tensor,         # (K, L, D_in) — 所有 prompt 的 token 序列
) -> FusionOutput:
```

### 8.4 ConcatFusion

```python
class ConcatFusion(nn.Module):
    """先 CrossAttn 获取 text context → 逐帧 concat 融合。"""

    def __init__(self, in_dim, out_dim, dropout=0.1, fusion_input="embedding"):
        super().__init__()
        self.fusion_input = fusion_input
        # 轻量 cross-attn：pool text memory to context per frame
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=4, dropout=dropout, batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, visual, text_memory):
        B, T, D = visual.shape
        K, L, D2 = text_memory.shape  # D2 == D

        # text_memory: (K, L, D) → (B, K*L, D)
        tm = text_memory.view(1, K * L, D).expand(B, -1, -1)

        # 每帧 attend 到整个 text memory
        text_context, attn_weights = self.cross_attn(
            query=visual, key=tm, value=tm,
        )  # → (B, T, D), (B, T, K*L)

        fused = torch.cat([visual, text_context], dim=-1)  # (B, T, 2*D)
        return FusionOutput(
            fused=self.proj(fused),          # (B, T, D_f)
            attention=attn_weights,          # (B, T, K*L)
            gate_values=None,
        )
```

### 8.5 GatedFusion

```python
class GatedFusion(nn.Module):
    """先 CrossAttn 获取 text context → 逐帧门控融合。"""

    def __init__(self, in_dim, fusion_input="embedding"):
        super().__init__()
        self.fusion_input = fusion_input
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=4, batch_first=True,
        )
        self.W_v = nn.Linear(in_dim, in_dim)
        self.W_t = nn.Linear(in_dim, in_dim)

    def forward(self, visual, text_memory):
        B, T, D = visual.shape
        K, L, _ = text_memory.shape
        tm = text_memory.view(1, K * L, D).expand(B, -1, -1)

        text_context, attn_weights = self.cross_attn(
            query=visual, key=tm, value=tm,
        )  # → (B, T, D)

        gate = torch.sigmoid(self.W_v(visual) + self.W_t(text_context))
        fused = gate * visual + (1 - gate) * text_context  # (B, T, D)
        return FusionOutput(
            fused=fused,
            attention=attn_weights,          # (B, T, K*L)
            gate_values=gate,                # (B, T, D)
        )
```

### 8.6 CrossAttnFusion

```python
class CrossAttnFusion(nn.Module):
    """标准 Cross-Attention：Q=vis, K/V=text_memory → proj。"""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, fusion_input="embedding"):
        super().__init__()
        self.fusion_input = fusion_input
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, visual, text_memory):
        B, T, D = visual.shape
        K, L, _ = text_memory.shape
        tm = text_memory.view(1, K * L, D).expand(B, -1, -1)

        attn_out, attn_weights = self.attn(query=visual, key=tm, value=tm)
        out = self.norm(visual + attn_out)   # residual + norm
        return FusionOutput(
            fused=self.proj(out),            # (B, T, D_f)
            attention=attn_weights,          # (B, T, K*L)
            gate_values=None,
        )
```

---

## 9. models/temporal.py — 时序建模

### 9.1 设计目标

Fusion 输出 `(B, T, D_f)` 后，在 T 维上建模帧间依赖。baseline 用 Identity（零开销透传），后续可升级为 Transformer / GRU。

### 9.2 接口

```python
class TemporalIdentity(nn.Module):
    def forward(self, x: Tensor) -> TemporalOutput:
        return TemporalOutput(features=x)

class TemporalTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1):
        ...

    def forward(self, x: Tensor) -> TemporalOutput:
        # x: (B, T, D_f) → self-attn over T → (B, T, D_f)
        return TemporalOutput(features=out)
```

---

## 10. models/head.py — 异常检测头

### 10.1 设计目标

- 输入 `(B, T, D_f)`，T 维完整保留
- 先得到帧级分数，再聚合为 clip 级
- 输出 embedding 供 Matcher 做 per-prompt 匹配
- **不负责 similarity 计算**——那是 Matcher 的事

### 10.2 接口

```python
class AnomalyHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        self.frame_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused: Tensor) -> HeadOutput:
        """
        fused: (B, T, D_f)
        → frame_score  (B, T)    真正的帧级预测
        → clip_score   (B,)      amax(frame_score)
        → embedding    (B, D_f)  mean_pool(fused)
        """
```

---

## 11. models/__init__.py — VLMModel 顶层

### 11.1 完整 forward 流程

```
1. prompts = prompt_processor.process(types, expand, hard_negatives)
2. vis_feat = backbone.encode_video(video)                # (B, T, D)
3. txt_feat = backbone.encode_text(prompts)               # (K, L, D)
4. aligned  = alignment(vis_feat, txt_feat)               # AlignmentOutput

5. # 对齐阶段 matching — 用于对比学习 loss
   vis_pooled = aligned.visual_embedding.mean(dim=1)      # (B, D_a)
   txt_pooled = aligned.text_embedding.mean(dim=1)        # (K, D_a)
   alignment_match = matcher(vis_pooled, txt_pooled)      # MatchOutput

6. # Fusion — Text Memory 模式，B 不膨胀
   fusion_in = select(aligned, fusion.fusion_input)       # (B,T,D_in) + (K,L,D_in)
   fused_out = fusion(*fusion_in)                         # FusionOutput (B,T,D_f)

7. temp_out  = temporal(fused_out.fused)                  # TemporalOutput (B,T,D_f)

8. head_out  = head(temp_out.features)                    # HeadOutput

9. # 推理阶段 matching — per-prompt 异常分数
   prompt_embs = aligned.text_embedding.mean(dim=1)       # (K, D_a)
   anomaly_match = matcher(head_out.embedding, prompt_embs)  # MatchOutput (B,K)

10. return ModelOutput(
        frame_score=head_out.frame_score,                 # (B, T)
        clip_score=head_out.clip_score,                   # (B,)
        embedding=head_out.embedding,                     # (B, D_f)
        anomaly_scores=anomaly_match.similarity,          # (B, K)
        alignment=aligned,
        fusion=fused_out,
        temporal=temp_out,
        head=head_out,
    )
```

### 11.2 模块组合

```python
class VLMModel(nn.Module):
    def __init__(
        self,
        backbone: CLIPBackbone,
        prompt_processor: PromptProcessor,
        alignment: SemanticAlignment,
        matcher: Matcher,
        fusion: ConcatFusion | GatedFusion | CrossAttnFusion,
        temporal: TemporalIdentity | TemporalTransformer,
        head: AnomalyHead,
    ):
        ...

    def forward(self, video: Tensor) -> ModelOutput:
        """video: (B, T, C, H, W) → ModelOutput"""
```

### 11.3 三种典型配置

```python
# Baseline — 快速跑通
model = VLMModel(
    backbone=CLIPBackbone("ViT-B-32"),
    prompt_processor=PromptProcessor(DEFAULT_TEMPLATES, DEFAULT_EXPANSIONS),
    alignment=SemanticAlignment(512, 512, proj_type="linear"),
    matcher=Matcher(strategy="cosine", temperature=0.07),
    fusion=ConcatFusion(512, 512, fusion_input="embedding"),
    temporal=TemporalIdentity(),
    head=AnomalyHead(512, 256),
)

# 轻量微调 — Gated + Transformer
model = VLMModel(
    backbone=backbone,
    prompt_processor=prompt_processor,
    alignment=SemanticAlignment(512, 512, proj_type="mlp"),
    matcher=Matcher(strategy="cosine", temperature=0.07),
    fusion=GatedFusion(512, fusion_input="embedding"),
    temporal=TemporalTransformer(512, num_heads=4, num_layers=2),
    head=AnomalyHead(512, 256),
)

# 最高精度 — CrossAttn + Transformer + concat input
model = VLMModel(
    backbone=backbone,
    prompt_processor=prompt_processor,
    alignment=SemanticAlignment(512, 512, proj_type="mlp"),
    matcher=Matcher(strategy="cosine", temperature=0.07),
    fusion=CrossAttnFusion(1024, 512, fusion_input="concat"),
    temporal=TemporalTransformer(512, num_heads=8, num_layers=4),
    head=AnomalyHead(512, 256),
)
```

---

## 12. 数据流完整示例

```python
from data import build_avenue_dataloader
from models import VLMModel
from models.backbone import CLIPBackbone
from models.alignment import SemanticAlignment
from models.matcher import Matcher
from models.fusion import ConcatFusion
from models.temporal import TemporalIdentity
from models.head import AnomalyHead
from prompts import PromptManager
from prompts.processor import PromptProcessor

# 1. 数据加载
loader = build_avenue_dataloader(root="./data", split="training", batch_size=8)
batch = next(iter(loader))
video = batch["video"]  # (8, 16, 3, 224, 224)

# 2. 构建模型
backbone = CLIPBackbone("ViT-B-32")
pm = PromptManager()
pp = PromptProcessor(
    templates=pm.to_dict(),
    expansions={"action": ["running", "walking", "fighting", "loitering"]},
)
model = VLMModel(
    backbone=backbone,
    prompt_processor=pp,
    alignment=SemanticAlignment(512, 512),
    matcher=Matcher(strategy="cosine"),
    fusion=ConcatFusion(512, 512, fusion_input="embedding"),
    temporal=TemporalIdentity(),
    head=AnomalyHead(512, 256),
)

# 3. Forward — 一步完成
output = model(video)

# 4. 输出使用
output.frame_score      # (8, 16)  — 帧级异常分数
output.clip_score       # (8,)     — clip 级
output.anomaly_scores   # (8, 40)  — B=8, K=40 (4 actions × 10 templates)
output.alignment.visual_embedding  # (8, 16, 512) — 对齐后特征
output.fusion.attention            # (8, 16, K*L) — 可解释注意力

# 5. Loss 计算（外部）
contrastive_loss = cross_entropy(
    matcher(output.alignment.visual_embedding.mean(1),
            output.alignment.text_embedding.mean(1)).logits,
    labels
)
anomaly_loss = bce(output.frame_score, batch["frame_label"])
total_loss = contrastive_loss * 0.5 + anomaly_loss * 0.5
```

---

## 13. 形状约定总表

| 中间变量 | 形状 | 产生模块 |
|---|---|---|
| `vis_feat` | `(B, T, D)` | Backbone.encode_video |
| `txt_feat` | `(K, L, D)` | Backbone.encode_text |
| `vis_aligned` | `(B, T, D_a)` | Alignment |
| `txt_aligned` | `(K, L, D_a)` | Alignment |
| `similarity` | `(B, K)` | Matcher（对齐阶段） |
| `fused` | `(B, T, D_f)` | Fusion — **B 不膨胀** |
| `attn_weights` | `(B, T, K*L)` | Fusion (CrossAttn) |
| `temporal_feat` | `(B, T, D_f)` | Temporal |
| `frame_score` | `(B, T)` | Head |
| `clip_score` | `(B,)` | Head |
| `embedding` | `(B, D_f)` | Head |
| `anomaly_scores` | `(B, K)` | Matcher（推理阶段） |

---

## 14. 不做什么

| 不做 | 原因 |
|---|---|
| Backbone 里做 pooling | 帧级/Token 级信息对下游不可恢复 |
| Fusion 做 Cartesian Product | K 增大时 OOM，Text Memory 模式 O(K) safe |
| Alignment 里做 similarity | Matcher 独立模块，职责分离 |
| Head 里做 per-prompt 匹配 | Matcher 统一处理 |
| `explanation` 字段 | 先跑通基础 pipeline |
| 在 VLMModel 里做 `if/else` | 所有模块输入输出类型一致 |
| Learnable Prompt / Prefix Tuning | PromptProcessor 预留接口，后续迭代 |
