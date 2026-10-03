# VLM-VAD 实验结果

> 本文档记录**实际运行**得到的真实指标（非占位符）。所有数字可用文末命令复现。
> ⚠️ 当前标签为 `motion_diff` 运动伪标签，**不能作为学术结论**，仅用于验证框架。

## 1. 实验设置

| 项目 | 配置 |
|---|---|
| 数据集 | CUHK Avenue，testing split（21 个视频 / 15,324 帧） |
| 帧标签 | `motion_diff` 伪标签（threshold = 3.0，基于相邻帧平均绝对差） |
| 视觉编码器 | 冻结 CLIP **ViT-B/32**（`laion2b_s34b_b79k`） |
| 文本编码器 | 冻结 CLIP Text Transformer（token 平均池化，与训练侧一致） |
| 异常分数 | `mean(sim(异常prompt组)) − mean(sim(正常prompt组))` |
| 特征路径 | 预提取 CLIP 特征（`data/features/`，约 30× 加速） |
| 硬件 | NVIDIA RTX 4060 Ti，PyTorch 2.3.1 + CUDA 12.1 |

**Prompt 配置（31 个，极性 +16 / −15）**
- `label`（14）：`a person {fighting|…|walking|…}`
- `scene`（9）：`a {abnormal activity|…|normal street|…} scene`
- `contrast`（8）：`a video of {abnormal behavior|…|normal behavior}`

## 2. Zero-shot 结果

| 指标 | 数值 |
|---|---|
| Frame AUC | 0.4226 |
| Frame AP | 0.0417 |
| Clip AUC | 0.4660 |
| Clip AP | 0.1602 |
| Video AUC | 0.8000 |
| Video AP | 0.9897 |
| 推理耗时（原始视频） | 132.4 s |
| 推理耗时（特征缓存） | 4.6 s |

## 3. Prompt 消融实验

| 实验 | #prompts | Frame AUC | Frame AP | Clip AUC | Video AUC | Video AP |
|---|---|---|---|---|---|---|
| label_only | 14 | 0.4226 | 0.0417 | 0.4660 | 0.8000 | 0.9897 |
| **scene_only** | 9 | **0.5111** | 0.0525 | 0.5520 | 0.9000 | 0.9951 |
| contrast_only | 8 | 0.3443 | 0.0391 | 0.4281 | 0.9000 | 0.9951 |
| all_types | 31 | 0.4338 | 0.0428 | 0.4787 | 0.9000 | 0.9951 |

**观察**
1. `scene_only` 帧级排序最好（0.5111），`contrast_only` 最差（0.3443）。
2. 视频级 `scene_only` / `contrast_only` / `all_types` 均达 0.90 AUC。
3. Prompt 数量与性能**不成正比**：`all_types`（31 个）反而低于 `scene_only`（9 个），说明更多 prompt 引入了语义噪声。

## 4. 微调（fine-tuned）结果

在相同数据/标签/特征路径下，解冻下游模块（alignment + fusion + head，共 19 个可训练参数张量，
backbone 保持冻结）训练 10 个 epoch：

| 指标 | Zero-shot（冻结） | Fine-tuned（best @ epoch 3） |
|---|---|---|
| Frame AUC | 0.4226 | **0.8422** |
| Frame AP | 0.0417 | **0.3870** |
| Clip AUC | 0.4660 | **0.8411** |
| Clip AP | 0.1602 | **0.5781** |
| Video AUC | 0.8000 | **1.0000** |

**结论**：微调后帧级 / clip 级 / 视频级全部大幅提升（帧 AUC +0.42）。
说明冻结 CLIP 特征本身可线性分离，瓶颈在于 zero-shot 的语义 prompt 与
`motion_diff` 伪标签不一致；一旦用 BCE 监督下游头，性能即恢复正常。

产物：`checkpoints/exp01/{best,last}.pt`、`results/exp01/{metrics,explanations}.json`、
`results/exp01/plots/*.png`。

## 5. 可解释性示例（真实输出）

对 testing 视频 `01` 的最异常片段（起始帧 544）：

```json
{
  "start_frame": 544,
  "clip_score": 0.0038,
  "top_prompts": [
    {"prompt": "a person standing", "polarity": -1, "similarity": 0.1885},
    {"prompt": "a person sitting",  "polarity": -1, "similarity": 0.1877},
    {"prompt": "a person fighting", "polarity": 1,  "similarity": 0.1871},
    {"prompt": "a person talking",  "polarity": -1, "similarity": 0.1866}
  ],
  "explanation": "该视频最异常的片段起始于第 544 帧，与以下 prompt 语义最接近：'a person standing'(sim=0.19) / …"
}
```

可视化产物：`results/zero_shot_label/zero_shot_plots/{video}_heatmap.png` 与 `{video}_prompts.png`（21 个视频）。

## 6. 关键发现与局限

### 6.1 Frame AUC（zero-shot）< 0.5 的原因（重要）
`motion_diff` 把"高运动量"定义为异常，而 CLIP 的 `fighting/theft` 等语义与"运动量"并不正相关，
因此在**帧级**排序上低于随机水平。这**不是代码缺陷**，而是伪标签与语义目标的错配。
我们验证过两种文本池化策略：

| 文本池化 | Frame AUC | Video AUC |
|---|---|---|
| token 平均（默认，与训练一致） | 0.4226 | 0.8000 |
| CLIP 官方 EOS | 0.3695 | 0.5000 |

即使用 CLIP 更"标准"的 EOS 池化也无法改善，印证瓶颈在标签而非实现。

### 6.2 结论
- 视频级零样本检测可用（AUC 0.80–0.90），**帧级零样本定位不可用**；
  微调下游头后帧级 AUC 恢复至 0.84。
- 该结果仅证明**框架可运行、指标真实**，学术结论必须等真实二值 mask。

### 6.3 局限
1. 伪标签：`motion_diff` 非真实异常语义，指标绝对值不可比论文。
2. 域差异：Avenue 为固定监控视角，CLIP 泛化受限。
3. backbone 冻结：未解冻 CLIP 视觉/文本编码器（`freeze_vision/text: true`）。

## 7. 复现命令

```bash
# 0) 环境（GPU 容器）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate

# 1) 预提取 CLIP 特征（约 30× 加速）
python tools/extract_video_features.py --root data --split training --out data/features/training --device cuda
python tools/extract_video_features.py --root data --split testing  --out data/features/testing  --device cuda

# 2) 运动伪标签
python tools/extract_motion_labels.py --root data --split training --threshold 3.0 --out data/labels
python tools/extract_motion_labels.py --root data --split testing  --threshold 3.0 --out data/labels

# 3) Zero-shot 测试
python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts label

# 4) Prompt 消融
python scripts/prompt_comparison.py --config configs/experiment.yaml

# 5) 微调（下游 alignment + fusion + head）
python train.py --config configs/experiment.yaml

# 6) 用 checkpoint 单独评估
python eval.py --config configs/experiment.yaml --ckpt checkpoints/exp01/best.pt

# 7) 续训（从 last.pt 的 epoch 之后继续）
python train.py --config configs/experiment.yaml --resume checkpoints/exp01/last.pt
```

## 8. 待办（下一步）

- [ ] 获取 Avenue 官方二值 mask，切换 `frame_label_mode: pixel` 重新评估。
- [ ] 解冻 CLIP backbone 做真正的端到端微调（`model.backbone.freeze_vision/text: false`）。
- [ ] 与 motion-based / 传统方法在**相同标签**下对比。
- [ ] 扩展 UCF-Crime（视频级标签）验证泛化。
- [ ] Fusion 消融：`concat` / `gated` / `crossattn`；Temporal：`identity` / `transformer`。
