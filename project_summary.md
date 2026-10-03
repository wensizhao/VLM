# VLM-VAD 项目进展总结

> 本文档记录**实际可运行**的状态。实验结果见 [experiment_results_template.md](experiment_results_template.md)。

## 项目目标

基于视觉语言模型（VLM）+ Prompt Engineering 的视频异常检测与可解释性研究。

```
Vision-Language Model + Prompt Engineering → Video Anomaly Detection + Explainability
```

## 当前状态：端到端闭环已跑通

| 模块 | 状态 | 证据 |
|---|---|---|
| 环境（Docker + GPU） | ✅ RTX 4060 Ti / CUDA 12.1 可用 | 容器内 `torch.cuda.is_available() == True` |
| 数据集 | ✅ Avenue 已就位（16 train / 21 test） | `data/Avenue_Dataset/` |
| 标签 | ⚠️ `motion_diff` 运动伪标签（非真实 mask） | `data/labels/` |
| 配置契约 | ✅ 与代码完全对齐 | `configs/{experiment,prompts}.yaml` |
| 模型前向 | ✅ backbone→alignment→fusion→head | `models/` |
| Zero-shot | ✅ 真实指标 | `results/zero_shot_test.json` |
| Prompt 消融 | ✅ 4 组实验对比 | `results/prompt_comparison.{json,md}` |
| 微调训练 | ✅ 10 epoch，含 checkpoint / 续训 | `checkpoints/exp01/` |
| 评估 + 可解释性 | ✅ metrics + explanations + 图 | `results/exp01/` |

## 核心结果（真实测量）

| 指标 | Zero-shot（冻结 CLIP） | Fine-tuned（10 epoch） |
|---|---|---|
| Frame AUC | 0.4226 | **0.8422** |
| Clip AUC | 0.4660 | **0.8411** |
| Video AUC | 0.8000 | **1.0000** |

Prompt 消融中 `scene_only` 帧级最好（0.5111），`all_types`（31 prompt）反而更低。

> ⚠️ 上述均为 `motion_diff` 伪标签下的结果，用于验证框架，**不能作为学术结论**。
> 详见实验结果文档 §6「关键发现与局限」。

## 已验证的工程修复

1. **配置 schema 不匹配**：旧 config 缺少 `data.root` / `paths.run_name` / `model.alignment` 等，`load_experiment_config` 直接崩溃 → 已按代码实际键重写。
2. **scripts 语法错误 + 假指标**：`zero_shot_test.py` 有语法错误，两个脚本硬编码返回 `0.5` → 已基于真实模型重写。
3. **GPU 确定性 bug**：`set_seed` 未设 `CUBLAS_WORKSPACE_CONFIG`，所有 CUDA 矩阵乘报错 → 已修复。
4. **续训 bug**：`Trainer.fit()` 忽略已训练 epoch，续训从头开始 → 已修复（`start_epoch`）。
5. **可复现性**：`configs/*.yaml` 曾被 gitignore，且运行不保存配置 → 已取消忽略并自动快照到 `results/{run}/config.yaml`。

## 复现（一条命令链）

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate

python tools/extract_video_features.py --root data --split training --out data/features/training --device cuda
python tools/extract_video_features.py --root data --split testing  --out data/features/testing  --device cuda
python tools/extract_motion_labels.py  --root data --split training --threshold 3.0 --out data/labels
python tools/extract_motion_labels.py  --root data --split testing  --threshold 3.0 --out data/labels

python scripts/zero_shot_test.py  --config configs/experiment.yaml --prompts label
python scripts/prompt_comparison.py --config configs/experiment.yaml
python train.py --config configs/experiment.yaml
python eval.py  --config configs/experiment.yaml --ckpt checkpoints/exp01/best.pt
```

## 下一步（按优先级）

1. **获取 Avenue 官方二值 mask** → 切 `frame_label_mode: pixel` 重新评估（当前首要阻塞）。
2. 解冻 CLIP backbone 做真正端到端微调。
3. Fusion / Temporal 消融（`concat|gated|crossattn` × `identity|transformer`）。
4. 扩展 UCF-Crime（视频级标签）验证泛化。

## 交付文件

- 模型：`models/`（config-driven factory）
- 数据：`datasets/`（VideoDataset / FeatureDataset 自动切换）
- Prompt：`prompts/`（模板 + 扩展 + 极性）
- 训练：`train/`、`train.py`
- 评估：`eval/`（`inference.py` 微调评估、`zero_shot.py` 零样本）、`eval.py`
- 脚本：`scripts/{zero_shot_test,prompt_comparison}.py`
- 工具：`tools/extract_*.py`、`tools/annotation_tool/`
- 结果：`results/`、`experiment_results_template.md`
