# VLM-WS: Vision-Language Model for Video Anomaly Detection

基于视觉语言模型（VLM）+ Prompt Engineering 的视频异常检测与可解释性研究项目。

> **核心思想：** Vision-Language Model + Prompt Engineering → Video Anomaly Detection + Explainability

## 前置要求

| 依赖 | 说明 |
|---|---|
| Docker Engine 24+ | 容器运行时 |
| Docker Compose v2 | 多容器编排 |
| NVIDIA Container Toolkit | **可选**，仅 GPU 训练需要 |

## 快速启动

```bash
# 1. 克隆仓库
git clone <repo-url>
cd vlm_ws

# 2. 创建环境变量文件
cp .env.example .env

# 3. 构建镜像（首次约 5-10 分钟）
docker compose build

# 4. 启动容器
docker compose up -d

# 5. 进入开发环境
docker compose exec vlm-dev zsh
```

## 训练 / 评估（一条命令闭环）

```bash
# 训练 + 视频级评估（自动保存 checkpoint / 指标 / 解释 / 可视化图）
python train.py --config configs/experiment.yaml

# 只评估已训练好的 checkpoint
python eval.py --config configs/experiment.yaml --ckpt ./checkpoints/exp01/best.pt

# 续训
python train.py --config configs/experiment.yaml --resume ./checkpoints/exp01/last.pt
```

**快速迭代路径**（backbone 冻结时推荐，训练提速 ~10×）：

```bash
# 1. 离线提取所有视频帧的 CLIP 特征（一次性）
python tools/extract_video_features.py --video-dir ./data/Avenue_Dataset/testing_videos \
    --mask-dir ./data/Avenue_Dataset/ground_truth/testing_label_mask \
    --out ./data/features/testing

# 2. 在 config 里设置 use_feature: true 即可自动走 FeatureDataset
```

> 实验对比（prompt ablation / fusion 对比 / backbone 互换）**全部改 yaml 即可**，无需改代码：
> 见 `configs/prompts.yaml`（prompt 词汇表）与 `configs/experiment.yaml`（`model.fusion.type` / `model.backbone.name`）。

### ⚠️ 关于数据集路径与 GT 标注（重要）

数据路径由 `configs/experiment.yaml` 的 `data.video_path` / `data.mask_path` 显式控制，
不再写死 `training_videos` / `testing_videos`：

- **`video_path`（必填）**：视频目录。未填写 → 直接报错。
- **`mask_path`（可选）**：GT 二值 mask 目录（如 `ground_truth/testing_label_mask`）。
  **未填写 → 该目录下所有视频视为正常视频**（帧标签全 0，配合 loss 的
  `skip_bce_when_no_pos` 做无监督训练）。
- mask 文件命名约定：与视频 stem 匹配，优先 `{stem}_label.mat`，其次 `{stem}.mat`；
  mat 文件内变量名为 **`volLabel`**（不是 `vol`），形状为 `(1, N)` 的 object 数组，
  每格一帧 `(H, W)` 二值 mask。

### GPU 主机

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### CPU 主机

在 `.env` 中切换为 CPU 镜像：

```bash
sed -i 's|^BASE_IMAGE=.*|BASE_IMAGE=pytorch/pytorch:2.3.1-cpu|' .env
docker compose build
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
├── data/                    # 旧数据模块（向后兼容 re-export）
│   ├── avenue_dataset.py    # Avenue 数据集类
│   └── dataloader.py        # DataLoader 工厂函数
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
├── configs/                 # 实验配置文件
│   ├── experiment.yaml      # 训练/模型/数据/路径
│   └── prompts.yaml         # prompt 词汇表与极性关键词
├── checkpoints/             # 模型权重（运行时生成）
├── logs/                    # TensorBoard 日志（运行时生成）
├── results/                 # 实验结果（运行时生成）
├── requirements/            # Python 依赖
│   ├── base.txt             # 核心依赖
│   └── dev.txt              # 开发依赖（Jupyter、pytest 等）
├── scripts/docker/          # 容器入口脚本与 shell 模版
├── Dockerfile               # 镜像构建文件
├── docker-compose.yml       # 开发容器编排
├── docker-compose.gpu.yml   # GPU 覆盖配置
├── .env.example             # 环境变量模板
├── .devcontainer/           # VS Code Dev Container 配置
├── docs/container.md        # 容器环境详细文档
└── AGENTS.md                # AI Agent 编码规范
```

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
| `BASE_IMAGE` | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` | 基础镜像 |
| `LOCAL_UID:GID` | `1000:1000` | 容器内用户 ID |
| `SHM_SIZE` | `16gb` | 共享内存大小 |
| `JUPYTER_PORT` | `8888` | Jupyter 端口映射 |
| `TENSORBOARD_PORT` | `6006` | TensorBoard 端口映射 |

在 `.env` 文件中修改即可。

## 提醒

- **框架：** PyTorch 2.3, OpenCLIP, HuggingFace Transformers
- **视频处理：** OpenCV, Decord
- **语言模型：** CLIP (ViT + Text Transformer)
- **容器：** Docker, Docker Compose
- **依赖注入：** OmegaConf, PyYAML

## 更多文档

- [容器环境详细说明](docs/container.md)
- [模型架构设计](models/DESIGN.md)
- [AI Agent 编码规范](AGENTS.md)
