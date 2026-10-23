# VLM-WS: Vision-Language Model for Video Anomaly Detection

基于视觉语言模型（VLM）+ Prompt Engineering 的视频异常检测与可解释性研究项目。

> **核心思想：** Vision-Language Model + Prompt Engineering → Video Anomaly Detection + Explainability

当前开发进度、已修复问题和未完成的研究任务见 [项目审计报告](docs/project_audit.md)。
默认 v2 流程将 16 个训练视频按 seed 划分为 12 个训练、4 个验证视频；21 个测试视频
只用于最终评估。旧 `exp01` 属于历史工程结果，不能作为独立测试集上的官方成绩。

官方标注已经接入：`data/labels_official/testing` 共 21 视频、15,324 帧、3,712 异常帧。
完整结果与当前限制见 [官方评估报告](docs/official_evaluation.md)。
下列命令在项目容器内执行；已有缓存可以直接评估，重复实验请使用新结果目录：

```bash
# 首次转换；验证原始 ZIP、视频 ID、帧数、二值 mask 和文件校验值
python tools/prepare_official_labels.py --archive data/ground_truth_demo.zip --root data --out data/labels_official
# 固定的四组 prompt 比较，EOS 文本池化，不使用测试标签选优
python scripts/prompt_comparison.py --config configs/official_eval.yaml --output results/my_official_zero_shot/summary.json --out-dir results/my_official_zero_shot
# 仅从训练视频建立正常特征库；留出训练视频校准，测试真值只用于计分
python scripts/normality_baseline.py --out-dir results/my_normality --checkpoint checkpoints/my_normality/memory.pt
# 原有六个模型的真实标签复评，保留原训练协议
python scripts/reevaluate_official.py --source-sweep results/stage2_verified_sweep/summary.json --out-dir results/my_official_recheck
```

`configs/official_eval.yaml` 专用于官方测试评估，不能用于训练；压缩包没有训练标注。
默认 `configs/experiment.yaml` 仍是运动伪标签的工程训练协议。
报告采用全部测试帧的原始分数计算 Frame AUC/AP，不做逐视频分数归一化。
测试视频均含异常，Video ROC-AUC 显示 `null`；全异常视频的 Frame ROC-AUC 同样未定义。

开始新实验前生成 v2 标签和特征（已有 v2 缓存时可跳过）：

```bash
python tools/extract_motion_labels.py --root data --split training --out data/labels_v2
python tools/extract_motion_labels.py --root data --split testing --out data/labels_v2
python tools/extract_video_features.py --root data --split training --out data/features_v2/training
python tools/extract_video_features.py --root data --split testing --out data/features_v2/testing
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --output results/stage2_verified_sweep
```

`configs/sweep.yaml` 自动运行 prompt × fusion × backbone × seed 矩阵；当前默认是
3 种 prompt × 2 种 fusion × 1 个 backbone × 1 个 seed，每组训练 1 轮，用于验证实验流程。
失败组记录日志且整体返回非零。重复运行使用新的 sweep name，避免覆盖已存在的 checkpoint。
单独训练使用 `python train.py --config configs/experiment.yaml --run-name my_run`。

`data.use_feature: auto` 自动选缓存，`true` 强制缓存，`false` 强制像素路径。
新缓存必须有匹配 encoder / pretrained / preprocessing 的 manifest；旧缓存需重新提取。
在线与离线编码共用 RGB resize/crop/normalize 路径。非 CLIP 编码器可通过
`models.registry.backbones.register(name, factory)` 接入，再配置 `model.backbone.provider`。

## 前置要求

| 依赖 | 说明 |
|---|---|
| Docker Engine 24+ | 容器运行时 |
| Docker Compose 插件 2.24+ | 开发容器编排 |
| NVIDIA Container Toolkit | **可选**，仅 GPU 训练需要 |

## 快速启动

```bash
# 1. 克隆仓库
git clone <repo-url>
cd vlm_ws

# 2. CPU 开发：拉取预构建镜像，自动创建 .env 并匹配宿主 UID/GID
./scripts/dev.sh cpu pull
./scripts/dev.sh cpu up -d
./scripts/dev.sh cpu check

# 3. 进入开发环境
./scripts/dev.sh cpu shell
```

Intel/AMD 共用 amd64 镜像，ARM64 使用同一 `:cpu` 标签中的 arm64 镜像。
不同 Linux 发行版共用同一套配置；NVIDIA x86-64 主机将命令中的 `cpu` 换成 `gpu`，
使用单独的 `:cuda12.1` 镜像，并先安装驱动和 NVIDIA Container Toolkit。
VS Code 的 `Reopen in Container` 可选择 CPU 或 NVIDIA CUDA 配置。
首次打开会先复用本地镜像，没有时自动拉取；拉取失败则自动本地构建，成功后继续打开。

预构建镜像发布在 [Docker Hub](https://hub.docker.com/r/zgzdocker/vlm-ws)，首次使用无需本地构建。
更新已有镜像时，先运行 `./scripts/dev.sh cpu pull` 或 `gpu pull`，再执行对应的 `up -d`。
修改 `Dockerfile` 或依赖后，可本地构建：

```bash
./scripts/dev.sh cpu build       # NVIDIA 使用 gpu build
./scripts/dev.sh cpu up -d
```

## 异常视频检测完整流程

本节描述从原始视频到「异常分数 + 可解释性报告」的完整链路。所有命令都在容器内、
项目根目录执行。建议按阶段 0 → 6 顺序操作。

### 数据流总览

```
Avenue 视频 (B,T,C,H,W) ──┐
                          │  CLIPBackbone.encode_video
                          ▼
              视觉特征 vis_feat (B,T,D)          prompt 文本 (K,)
                          │                        │ CLIPBackbone.encode_text
                          │                        ▼
                          │              文本 token 特征 (K,L,D)
                          ▼                        │
                 SemanticAlignment（对齐到共享维度 D_a）
                          │                        │
                          └────────┬───────────────┘
                                   ▼
        Fusion（concat / gated / crossattn）与文本上下文融合 (B,T,D_f)
                                   ▼
        Temporal（identity / transformer）时序建模 (B,T,D_f)
                                   ▼
        AnomalyHead ──► frame_logits (B,T) / frame_score (B,T) / clip_score (B)
                                   ▼
     滑窗遍历全部 clip → 按帧坐标等权平均 → 稠密逐帧分数 (N,)
                                   ▼
     Frame / Clip / Video 三级 AUC & AP
                                   ▼
     Matcher(embedding, text) → top-k prompt → template 解释 + 热力图
```

关键设计：
- **训练用 16 帧 clip，评估用滑窗覆盖整段视频**，得到真正逐帧的 `(N,)` 异常曲线，
  而不是每个 clip 一个标量（见 `eval/inference.py`）。
- **可训练模块只有 alignment / fusion / temporal / head**；CLIP backbone 默认冻结，
  因此支持「先离线提特征、再快速迭代」。
- **Prompt 是配置项而非硬编码**：模型内部通过 `PromptProcessor` 读取
  `configs/prompts.yaml`，训练与推理使用同一套 prompt 集合。

---

### 阶段 0 · 环境准备

```bash
./scripts/dev.sh cpu up -d    # NVIDIA 主机使用 gpu
./scripts/dev.sh cpu check
./scripts/dev.sh cpu shell
```

CPU 与 CUDA 镜像共用研究依赖，分别安装 CPU-only / CUDA PyTorch。
修改 `Dockerfile` / `requirements/` 后，使用对应的 `./scripts/dev.sh cpu build`
或 `./scripts/dev.sh gpu build` 重建。完整平台范围、迁移方式和权限说明见
[容器文档](docs/container.md)。

Web UI 使用 `./scripts/dev.sh cpu web up -d`，NVIDIA 主机改用 `gpu`，
浏览器打开 `http://127.0.0.1:8000`。首次启动会自动拉取所选后端的镜像。
页面功能和迁移方式见 [Web Studio](docs/webapp.md)，后续同步上游见 [分支说明](docs/branches.md)。

### 阶段 1 · 数据与标注

1. 按「[数据集](#数据集)」一节把 CUHK Avenue 放入 `data/Avenue_Dataset/`。
2. 确保 GT mask 文件存在于对应目录（如 `data/Avenue_Dataset/ground_truth/testing_label_mask`），
   并确认标注模式（**重要**，见下方 ⚠️ 说明）：
   - 持有**官方二值 mask**（变量名 `volLabel`）→ 保持默认 `data.frame_label_mode: pixel`；
   - 只有降采样灰度 `vol` 文件 → 可设 `data.frame_label_mode: motion_diff`，用运动伪标签跑通流程。

`motion_diff` / `official` 模式需要先生成 / 准备帧标签缓存（一次性，训练与评估会从
`data/labels_v2/{split}/` 读取）：

```bash
python tools/extract_motion_labels.py --root ./data --split training --out ./data/labels_v2 --threshold 3.0
python tools/extract_motion_labels.py --root ./data --split testing  --out ./data/labels_v2 --threshold 3.0
```

### 阶段 2 · Zero-shot baseline（无需训练，秒级）

在训练前先建立可信的无训练基线：冻结 CLIP，直接比较「异常 prompt 组平均相似度 −
正常 prompt 组平均相似度」作为逐帧异常分数
（`s_t = mean_{k∈A} cos(v_t,t_k) − mean_{k∈N} cos(v_t,t_k)`）。

```bash
# 单次 zero-shot（--prompts: label | scene | contrast | all）
python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts scene

# 多 prompt 批量消融（一次跑多组，输出 json + markdown 表）
python scripts/prompt_comparison.py --config configs/experiment.yaml \
    --experiments label_only scene_only contrast_only all_types
```

产物：每次运行的所有结果都落在 `results/zero_shot_{prompts}/`（`summary.json` +
`zero_shot_metrics.json` / `zero_shot_explanations.json` / `zero_shot_plots/` 热力图），
以及 `results/prompt_comparison.{json,md}` 与各实验子目录。

> 该路径完全跳过 `VLMModel`，与微调结果使用**相同的滑窗聚合与三级指标**，
> 因此两者可直接对比。`eval.zero_shot` 还返回 AGENTS.md §4 要求的
> `anomaly_score / frame_score / embedding / explanation` 契约。

### 阶段 3 · 离线特征提取（可选，冻结 backbone 时提速 ~10×）

backbone 冻结时，视觉特征与输入无关的部分可预先算好，训练时直接读 `.pt`：

```bash
python tools/extract_video_features.py --root ./data --split training \
    --model ViT-B-32 --pretrained laion2b_s34b_b79k \
    --out ./data/features_v2/training --batch-size 64

python tools/extract_video_features.py --root ./data --split testing \
    --out ./data/features_v2/testing --batch-size 64

# （可选）预编码 prompt 文本特征
python tools/extract_text_features.py --config configs/prompts.yaml \
    --out ./data/features/text/prompts.pt
```

提取完成后无需改代码：`datasets/builders.py` 在 auto 模式检测 `data/features_v2/{split}/`
并校验 manifest 后走 `FeatureDataset`（也可显式设 `data.use_feature: true`）。
此时 batch 中的 key 由 `video` 变为 `vis_feat`，Trainer/推理会自动选择
`model.forward_from_visual()` 路径。

### 阶段 4 · 训练

```bash
# 训练 + 自动视频级评估（一条命令闭环）
python train.py --config configs/experiment.yaml

# 冒烟测试（只跑 1 个 epoch，验证链路）
python train.py --config configs/experiment.yaml --epochs 1 --run-name smoke

# 续训（模型/优化器/调度器/epoch 全量恢复）
python train.py --config configs/experiment.yaml \
    --resume ./checkpoints/baseline_v2/last.pt
```

每个 batch 的闭环为：`forward → VLMVADLoss(BCE + 对比对齐) → backward →
grad_clip → optimizer.step → scheduler.step → 日志`（见 `train/trainer.py`）。
`train.py` 在训练结束后自动加载 `best.pt`，对测试集执行 `evaluate_videos()`。

损失函数（`train/losses.py`）：
- **BCE**：逐帧异常分类，权重 `train.loss.w_bce`；
- **对比对齐**：视觉 embedding 与 prompt 文本的跨模态对比学习，权重
  `train.loss.w_contrastive`；两者共同决定最终 `loss`。

### 阶段 5 · 评估与可解释性

```bash
# 只评估已有 checkpoint（不重新训练）
python eval.py --config configs/experiment.yaml \
    --ckpt ./checkpoints/baseline_v2/best.pt

# 在训练集上评估（可选）
python eval.py --config configs/experiment.yaml \
    --ckpt ./checkpoints/baseline_v2/best.pt --eval-split training
```

评估输出到 `results/{run_name}/`：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 全局 Frame/Clip/Video AUC & AP + 每视频逐帧指标 |
| `explanations.json` | 每个视频最异常 clip 的 top-k prompt、异常帧排名、自然语言解释 |
| `plots/{id}_heatmap.png` | 时序异常热力图（帧分数曲线 + GT 色带） |
| `plots/{id}_prompts.png` | 该 clip 对每个 prompt 的相似度条形图 |
| `config.yaml` | 自包含配置快照，后续外部 prompt 文件变化不会改写本次配置 |
| `frame_scores.npz` / `frame_labels.npz` | 完整视频逐帧分数与标签 |
| `split_manifest.json` / `environment.json` | 训练/验证视频划分、运行依赖和设备 |

解释文本回答「为什么这个视频是异常的」：取相似度最高的异常 prompt 作为原因，
例如 **“该视频段（起始帧 512）被判定为异常，因为画面语义与 `a person fighting`
(sim=0.31) 最接近……”**。底层模型输出的张量契约见 `models/outputs.py`：
`frame_score (B,T)`、`anomaly_score (B,)`（兼容 `clip_score`）、`embedding (B,D_f)`、
`explanation`（模型前向可为 None，评估时生成模板解释）；`output.to_dict()` 保留梯度。
zero-shot 路径
（`eval/zero_shot.py`）额外按 AGENTS.md §4 汇总 `anomaly_score` / `frame_score` /
`embedding` / `explanation` 四个字段。

### 阶段 5b · 可视化演示（把检测效果变成能直接看的 MP4）

上面的 `plots/*.png` 是静态曲线，不利于直观展示。用 `scripts/visualize_demo.py`
可以把**真实画面 + 异常高亮 + 完整时间轴**渲染成可直接播放的 H.264 MP4：

```bash
python scripts/visualize_demo.py --config configs/experiment.yaml \
    --ckpt checkpoints/baseline_v2/best.pt --videos 01 06 18 --out results/demo
```

| 产物（`results/demo/`） | 内容 |
|---|---|
| `{id}_demo.mp4` | 1280×720 动画：画面 + 异常帧红框 + 分数条 + 右侧 top 异常 prompt + 底部整段分数时间轴（GT 红带 + 播放头 + 阈值线） |
| `{id}_demo.gif` | 最异常片段短动图（可直接贴 PPT） |
| `{id}_top_frames.png` | 最异常 K 帧缩略图网格（带帧号/分数/GT） |
| `timeline_{id}.png` | **整段视频**逐帧分数曲线 + GT 色带 + 阈值线 |
| `index.md` | 汇总表 + 中文解释 + 文件链接（一个入口看全部） |

说明：
- 逐帧分数直接复用 `eval.inference.evaluate_videos`（不重复实现推理），
  所以 MP4 与 `metrics.json` 中的数字完全一致。
- 红框阈值默认取该视频 frame score 的 **90 分位**（展示用相对阈值，非模型校准决策边界），
  可用 `--threshold 0.5` 覆盖。
- 视频上文字为英文（容器内无中文字体），中文解释写入 `index.md`。
- 解释现在**只从异常极性 prompt 中选 top-k**（`polarity=+1`），不会再出现
  “用 `a peaceful plaza scene` 解释异常”的问题。

### 阶段 6 · 实验对比（Prompt / Fusion / Backbone）

全部**改 yaml 即可**，无需改代码：

| 实验维度 | 改动位置 | 可选值 |
|---|---|---|
| Prompt 类型消融 | `configs/prompts.yaml: experiments` + `scripts/prompt_comparison.py` | label / scene / contrast / 组合 |
| Prompt 词表 | `configs/prompts.yaml: expansions` | 任意词汇（保持正常/异常词互斥） |
| Fusion 对比 | `configs/experiment.yaml: model.fusion.type` | `concat` / `gated` / `crossattn` |
| Backbone 互换 | `model.backbone.name` + `pretrained` | 任意 open_clip 模型（如 ViT-L-14） |
| 时序建模 | `model.temporal.type` | `identity` / `transformer` |
| 文本池化 | `eval.text_pool` | `mean`（与训练一致）/ `eos`（CLIP 官方） |

```bash
# 对比 concat vs cross-attn：分别改 yaml 的 model.fusion.type，各自跑一遍
python train.py --config configs/experiment.yaml --run-name fusion_concat
# 修改为 crossattn 后
python train.py --config configs/experiment.yaml --run-name fusion_crossattn
```

`scripts/prompt_comparison.py` 会把多组 prompt 实验汇总成 markdown 表，当前仓库
历史 v1 示例（zero-shot, frozen CLIP, Avenue testing，运动伪标签；不能作为官方异常检测成绩）：

| experiment | #prompts | Frame AUC | Video AUC |
|---|---|---|---|
| label_only | 14 | 0.4226 | 0.8000 |
| scene_only | 9 | 0.5111 | 0.9000 |
| contrast_only | 8 | 0.3443 | 0.9000 |
| all_types | 31 | 0.4338 | 0.9000 |

### 产物目录与复现性

```
checkpoints/{run_name}/best.pt   # 选优指标 = 验证集 clip_auc
checkpoints/{run_name}/last.pt   # 最新 epoch（续训用）
logs/{run_name}/                 # TensorBoard events
results/{run_name}/              # metrics / explanations / plots / config.yaml
```

固定随机性由 `utils/reproducibility.set_seed(seed)` 统一完成（Python / NumPy /
PyTorch / dataloader shuffle），seed 来自 `configs/experiment.yaml: seed`。
每次运行都会把最终配置写入 `results/{run_name}/config.yaml`，保证实验可追溯。

### ⚠️ 关于数据集路径与 GT 标注（重要）

数据路径由 `configs/experiment.yaml` 的 `data.video_path` / `data.mask_path` 显式控制，
不再写死 `training_videos` / `testing_videos`：

- **`video_path`（必填）**：视频目录（按 split 显式声明，见 `data.train` / `data.test`）。
  未填写 → 直接报错。
- **`mask_path`（可选）**：GT 二值 mask 目录（如 `ground_truth/testing_label_mask`）。
  **未填写 → 该目录下所有视频视为正常视频**（帧标签全 0，配合 loss 的
  `skip_bce_when_no_pos` 做无监督训练）。
- mask 文件命名约定：与视频 stem 匹配，优先 `{stem}_label.mat`，其次 `{stem}.mat`；
  mat 文件内变量名为 **`volLabel`**（不是 `vol`），形状为 `(1, N)` 的 object 数组，
  每格一帧 `(H, W)` 二值 mask。
- **`frame_label_mode`（默认 `pixel`）**：`pixel` 直接用二值 mask 派生帧标签；
  `motion_diff` 用帧间差运动伪标签（VAD 经典 baseline，**学术结论请使用真实 mask**）；
  `official` 读取官方 mat 标注派生的帧标签缓存。
- pixel 模式发现灰度视频帧伪装成 mask 会直接报错，防止生成全部为异常的错误标签。
  特征/标签缓存路径严格遵守 frame_label_mode，pixel 模式不会读取运动伪标签缓存。

### GPU 主机

```bash
./scripts/dev.sh gpu up -d
./scripts/dev.sh gpu check
```

### CPU / 核显主机

默认使用 CPU-only 镜像，支持 Intel/AMD x86-64 与 ARM64：

```bash
./scripts/dev.sh cpu up -d
./scripts/dev.sh cpu check
```

## 数据集

项目使用 [CUHK Avenue Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) 进行视频异常检测实验。

下载后将数据集放入 `data/` 目录，结构如下：

```
data/
├── __init__.py
├── avenue_dataset.py
├── dataloader.py
└── Avenue_Dataset/          # 从官网下载后解压到这里
    ├── training_videos/     # 16 个训练视频 (.avi)
    ├── testing_videos/      # 21 个测试视频 (.avi)
    ├── training_vol/        # (可选) 16 个 vol (.mat)，本仓库附件实际为灰度帧体
    ├── testing_vol/         # (可选) 21 个 vol (.mat)，不能作为二值异常真值
    └── ground_truth/
        ├── testing_label_mask/   # 21 个官方 GT 二值 mask ({id}_label.mat, 变量 volLabel)
        └── ...
```

> 数据集文件被 `.gitignore` 排除，不会提交到仓库。

## 项目结构

```
vlm_ws/
├── models/                  # 模型模块
│   ├── DESIGN.md            # 模型架构设计文档
│   ├── __init__.py          # VLMModel（backbone + fusion + head）
│   ├── factory.py           # 配置驱动模型工厂（backbone swap / fusion 对比）
│   ├── backbone.py          # CLIP 视觉 + 文本编码器
│   ├── alignment.py         # 视觉-语言语义对齐投影
│   ├── matcher.py           # 相似度匹配（对比学习 / per-prompt 异常）
│   ├── fusion.py            # ConcatFusion / GatedFusion / CrossAttnFusion
│   ├── temporal.py          # TemporalIdentity / TemporalTransformer
│   ├── head.py              # 异常检测头（frame_logits + frame_score）
│   └── outputs.py           # 各模块统一的 frozen dataclass 输出
├── data/                    # 数据存放目录；__init__.py 保留兼容导出
├── datasets/                # 新数据模块（推荐）
│   ├── video_dataset.py     # 原始像素 Dataset（官方 GT volLabel mask）
│   ├── feature_dataset.py   # 预提取特征 Dataset（训练提速 ~10×）
│   ├── dataloader.py        # DataLoader 工厂函数
│   └── builders.py          # 配置驱动：自动选 Video/Feature 路径
├── prompts/                 # Prompt 模板管理
│   ├── __init__.py          # PromptManager：label / scene / contrast 三种模板
│   └── processor.py         # PromptProcessor：展开 / ensemble / hard negatives / 缓存
├── train/                   # 训练模块
│   ├── losses.py            # VLMVADLoss = 帧级 BCE + 对比对齐（含数学注释）
│   └── trainer.py           # Trainer：optimizer / scheduler / AMP / checkpoint
├── eval/                    # 评估模块
│   ├── metrics.py           # frame / clip / video 级 AUC & AP
│   └── inference.py         # 视频级滑窗推理 + template-based 可解释性
├── utils/                   # 工具模块
│   ├── reproducibility.py   # set_seed（固定所有随机源）
│   ├── logging.py           # get_logger / MetricLogger / TBLogger
│   ├── io.py                # checkpoint / JSON 读写
│   ├── config.py            # 配置加载（experiment + prompts 合并）
│   └── visualization.py     # 时序热力图 / prompt 相似度 / attention
├── tools/                   # 离线特征提取脚本
│   ├── extract_video_features.py
│   └── extract_text_features.py
├── train.py                 # 训练入口（配置驱动，一条命令闭环）
├── eval.py                  # 评估入口（加载 checkpoint 出指标+解释+图）
├── run_web.py               # Web Studio 入口（数据集、训练、验证、对比）
├── webapp/                  # 模块化 Web UI 与任务管理
├── configs/                 # 实验配置文件
│   ├── experiment.yaml      # 训练/模型/数据/路径
│   └── prompts.yaml         # prompt 词汇表与极性关键词
├── checkpoints/             # 模型权重（运行时生成）
├── logs/                    # TensorBoard 日志（运行时生成）
├── results/                 # 实验结果（运行时生成）
├── requirements/            # Python 依赖
│   ├── base.txt             # 核心依赖（含标注工具 flask）
│   ├── torch.txt            # CPU/CUDA 共用的 PyTorch 版本
│   ├── dev.txt              # 开发依赖（Jupyter、pytest 等）
│   └── ci.txt               # CI 依赖（base + pytest，不含 Jupyter）
├── scripts/
│   ├── dev.sh               # CPU/GPU 统一启动、构建、检查入口
│   ├── zero_shot_test.py    # zero-shot 基线入口
│   ├── prompt_comparison.py # prompt 消融实验
│   ├── visualize_demo.py    # 检测效果 → MP4/GIF 可视化演示
│   └── docker/              # 容器入口脚本与 shell 模版
├── tests/                   # 冒烟测试（配置 / 极性 / 指标 / forward / loss）
├── pytest.ini               # pytest 配置（pythonpath = 项目根）
├── .github/workflows/ci.yml # CI：push / PR 自动跑 pytest
├── .github/workflows/docker-publish.yml # 镜像自动构建并推送 Docker Hub
├── Dockerfile               # 镜像构建文件
├── docker-compose.yml       # 通用 CPU 开发运行配置
├── docker-compose.gpu.yml   # CUDA 镜像和 NVIDIA 设备覆盖配置
├── docker-compose.host-tools.yml # 可选个人工具配置挂载
├── .env.example             # 环境变量模板
├── .devcontainer/           # VS Code Dev Container 配置
├── docs/container.md        # 容器环境详细文档
└── AGENTS.md                # AI Agent 编码规范
```

## 测试

最小冒烟测试不需要数据集、不下载 CLIP 权重，CPU 上数秒跑完：

```bash
pytest -q
```

GPU 与本地 Avenue 缓存的额外检查（不会在普通 CI 中运行）：

```bash
VLM_REAL_INTEGRATION=1 HF_HUB_OFFLINE=1 python -m pytest -q tests/test_real_pipeline.py
```

新增回归覆盖视频隔离、matcher 参数更新、全正常 batch 分类梯度、标签模式隔离、
运动标签跨 chunk 边界、真实 frame_indices 聚合、续训 RNG 恢复、实验矩阵失败状态。
GPU 检查验证像素/缓存特征一致及 AMP 训练→验证→断点恢复。

覆盖点（`tests/test_smoke.py`）：

| 测试 | 验证内容 |
|---|---|
| `test_load_experiment_config*` | experiment.yaml + prompts.yaml 合并、配置快照落盘 |
| `test_prompt_polarity` | prompt → {+1 异常, -1 正常} 极性映射正确 |
| `test_*_metrics` | Frame/Clip/Video AUC & AP 数值与单类防御 |
| `test_model_forward*_shapes` | `ModelOutput` 契约：`frame_score (B,T)` / `clip_score (B,)` / `embedding (B,D_f)` |
| `test_vlmvad_loss_backward` | BCE + 对比对齐 loss 有限且可反向传播 |

模型 forward 用 `FakeCLIPBackbone` 替身（接口/形状与 `CLIPBackbone` 一致），
因此 CI 无需下载预训练权重。GitHub Actions 见 `.github/workflows/ci.yml`
（push / PR 自动执行 `pytest -q`）。

## 容器内开发

```bash
# Jupyter Lab（端口 8888）
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser

# TensorBoard（端口 6006）
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLM_CPU_IMAGE` | `zgzdocker/vlm-ws:cpu` | CPU 多架构镜像；可固定为提交标签或 digest |
| `VLM_CUDA_IMAGE` | `zgzdocker/vlm-ws:cuda12.1` | NVIDIA x86-64 镜像 |
| `CPU_BASE_IMAGE` | `python:3.10-slim-bookworm` | CPU 本地构建基础镜像 |
| `CUDA_BASE_IMAGE` | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` | CUDA 本地构建基础镜像 |
| `LOCAL_UID` / `LOCAL_GID` | 启动脚本匹配宿主用户 | 容器内用户 ID，避免输出文件归 root |
| `SHM_SIZE` | `2gb` | 容器共享内存大小 |
| `BIND_ADDRESS` | `127.0.0.1` | Jupyter / TensorBoard 绑定地址 |
| `JUPYTER_PORT` | `8888` | Jupyter 端口映射 |
| `TENSORBOARD_PORT` | `6006` | TensorBoard 端口映射 |
| `INSTALL_OPTIONAL_SHELL_TOOLS` | `0` | 本地构建时是否安装个人 CLI 工具 |

在 `.env` 文件中修改即可。

## 容器内 opencode

团队默认容器不依赖个人 CLI、配置或凭据。需要 opencode 时，以
`INSTALL_OPTIONAL_SHELL_TOOLS=1` 构建个人镜像，再显式启用个人挂载；操作见
[容器文档](docs/container.md#可选个人开发工具)。会话数据库仍保存在容器 HOME 卷内。

## 提醒

- **框架：** PyTorch 2.3, OpenCLIP, HuggingFace Transformers
- **视频处理：** OpenCV；Decord 为可选后端
- **语言模型：** CLIP (ViT + Text Transformer)
- **容器：** Docker, Docker Compose
- **依赖注入：** OmegaConf, PyYAML

## 更多文档

- [容器环境详细说明](docs/container.md)
- [模型架构设计](models/DESIGN.md)
- [AI Agent 编码规范](AGENTS.md)
