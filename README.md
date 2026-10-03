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
cp .env.example .env          # 首次
docker compose build          # 首次约 5-10 分钟
docker compose up -d
docker compose exec vlm-dev zsh
```

GPU 主机使用 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`；
CPU 主机把 `.env` 的 `BASE_IMAGE` 切到 `pytorch/pytorch:2.3.1-cpu` 后重建。

### 阶段 1 · 数据与标注

1. 按「[数据集](#数据集)」一节把 CUHK Avenue 放入 `data/Avenue_Dataset/`。
2. 确认标注模式（**重要**，见下方 ⚠️ 说明）：
   - 持有**官方二值 mask** → `configs/experiment.yaml` 设 `data.frame_label_mode: pixel`；
   - 使用本仓库自带降采样灰度 `vol` → 保持默认 `data.frame_label_mode: motion_diff`。

`motion_diff` 模式需要先生成运动伪标签（一次性，训练/评估会自动从
`data/labels/{split}/` 读取）：

```bash
python tools/extract_motion_labels.py --root ./data --split training --out ./data/labels --threshold 3.0
python tools/extract_motion_labels.py --root ./data --split testing  --out ./data/labels --threshold 3.0
```

### 阶段 2 · Zero-shot baseline（无需训练，秒级）

在训练前先建立可信的无训练基线：冻结 CLIP，直接比较「异常 prompt 组平均相似度 −
正常 prompt 组平均相似度」作为逐帧异常分数
（`s_t = mean_{k∈A} cos(v_t,t_k) − mean_{k∈N} cos(v_t,t_k)`）。

```bash
# 单次 zero-shot（--prompts: label | scene | contrast | all）
python scripts/zero_shot_test.py --config configs/experiment.yaml \
    --prompts scene --output results/zero_shot_test.json

# 多 prompt 批量消融（一次跑多组，输出 json + markdown 表）
python scripts/prompt_comparison.py --config configs/experiment.yaml \
    --experiments label_only scene_only contrast_only all_types
```

产物：`results/zero_shot_test.json`、`results/prompt_comparison.{json,md}`，
以及每个实验目录下的 `zero_shot_metrics.json` / `zero_shot_explanations.json` / 热力图。

> 该路径完全跳过 `VLMModel`，与微调结果使用**相同的滑窗聚合与三级指标**，
> 因此两者可直接对比。`eval.zero_shot` 还返回 AGENTS.md §4 要求的
> `anomaly_score / frame_score / embedding / explanation` 契约。

### 阶段 3 · 离线特征提取（可选，冻结 backbone 时提速 ~10×）

backbone 冻结时，视觉特征与输入无关的部分可预先算好，训练时直接读 `.pt`：

```bash
python tools/extract_video_features.py --root ./data --split training \
    --model ViT-B-32 --pretrained laion2b_s34b_b79k \
    --out ./data/features/training --batch-size 64

python tools/extract_video_features.py --root ./data --split testing \
    --out ./data/features/testing --batch-size 64

# （可选）预编码 prompt 文本特征
python tools/extract_text_features.py --config configs/prompts.yaml \
    --out ./data/features/text/prompts.pt
```

提取完成后无需改代码：`datasets/builders.py` 检测到 `data/features/{split}/`
存在就会自动走 `FeatureDataset`（也可显式设 `data.use_feature: true`）。
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
    --resume ./checkpoints/exp01/last.pt
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
    --ckpt ./checkpoints/exp01/best.pt

# 在训练集上评估（可选）
python eval.py --config configs/experiment.yaml \
    --ckpt ./checkpoints/exp01/best.pt --eval-split training
```

评估输出到 `results/{run_name}/`：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 全局 Frame/Clip/Video AUC & AP + 每视频逐帧指标 |
| `explanations.json` | 每个视频最异常 clip 的 top-k prompt、异常帧排名、自然语言解释 |
| `plots/{id}_heatmap.png` | 时序异常热力图（帧分数曲线 + GT 色带） |
| `plots/{id}_prompts.png` | 该 clip 对每个 prompt 的相似度条形图 |
| `config.yaml` | 本次运行的完整配置快照（可复现性，AGENTS.md §7） |

解释文本回答「为什么这个视频是异常的」：取相似度最高的异常 prompt 作为原因，
例如 **“该视频段（起始帧 512）被判定为异常，因为画面语义与 `a person fighting`
(sim=0.31) 最接近……”**。底层模型输出的张量契约见 `models/outputs.py`：
`frame_score (B,T)`、`clip_score (B,)`、`embedding (B,D_f)`；zero-shot 路径
（`eval/zero_shot.py`）额外按 AGENTS.md §4 汇总 `anomaly_score` / `frame_score` /
`embedding` / `explanation` 四个字段。

### 阶段 5b · 可视化演示（把检测效果变成能直接看的 MP4）

上面的 `plots/*.png` 是静态曲线，不利于直观展示。用 `scripts/visualize_demo.py`
可以把**真实画面 + 异常高亮 + 完整时间轴**渲染成可直接播放的 H.264 MP4：

```bash
python scripts/visualize_demo.py --config configs/experiment.yaml \
    --ckpt checkpoints/exp01/best.pt --videos 01 06 18 --out results/demo
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
示例（zero-shot, frozen CLIP, Avenue testing）：

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

### ⚠️ 关于 Avenue 数据集标注（重要）

本仓库 `data/Avenue_Dataset/*_vol/*.mat` 中的 `vol` 变量经检测**实际是降采样的灰度视频帧，
而不是二值异常 mask**（与视频帧相关系数 ≈ 0.999）。因此：

- **本仓库默认** `frame_label_mode: motion_diff`（`configs/experiment.yaml`）：基于帧间差的
  运动伪标签，VAD 经典 baseline 之一，配合自带数据即可跑通流程，**学术结论请使用真实 mask**。
- 若你持有**官方二值 mask**：把 `configs/experiment.yaml` 改为 `frame_label_mode: pixel`
  后再训练/评估，无需其他改动。
- 代码会自动检测可疑 mask 并在日志里提醒（`datasets/video_dataset.py: mask_looks_like_frames`）。

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
    ├── training_vol/        # 16 个像素级标注 (.mat)
    └── testing_vol/         # 21 个像素级标注 (.mat)
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
│   ├── video_dataset.py     # 原始像素 Dataset（支持 pixel / motion_diff 标签）
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
├── tools/                   # 离线特征与标签提取脚本
│   ├── extract_video_features.py
│   ├── extract_text_features.py
│   └── extract_motion_labels.py
├── train.py                 # 训练入口（配置驱动，一条命令闭环）
├── eval.py                  # 评估入口（加载 checkpoint 出指标+解释+图）
├── configs/                 # 实验配置文件
│   ├── experiment.yaml      # 训练/模型/数据/路径
│   └── prompts.yaml         # prompt 词汇表与极性关键词
├── checkpoints/             # 模型权重（运行时生成）
├── logs/                    # TensorBoard 日志（运行时生成）
├── results/                 # 实验结果（运行时生成）
├── requirements/            # Python 依赖
│   ├── base.txt             # 核心依赖（含标注工具 flask）
│   ├── dev.txt              # 开发依赖（Jupyter、pytest 等）
│   └── ci.txt               # CI 依赖（base + pytest，不含 Jupyter）
├── scripts/
│   ├── zero_shot_test.py    # zero-shot 基线入口
│   ├── prompt_comparison.py # prompt 消融实验
│   ├── visualize_demo.py    # 检测效果 → MP4/GIF 可视化演示
│   └── docker/              # 容器入口脚本与 shell 模版
├── tests/                   # 冒烟测试（配置 / 极性 / 指标 / forward / loss）
├── pytest.ini               # pytest 配置（pythonpath = 项目根）
├── .github/workflows/ci.yml # CI：push / PR 自动跑 pytest
├── Dockerfile               # 镜像构建文件
├── docker-compose.yml       # 开发容器编排
├── docker-compose.gpu.yml   # GPU 覆盖配置
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
| `BASE_IMAGE` | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` | 基础镜像 |
| `LOCAL_UID:GID` | `1000:1000` | 容器内用户 ID |
| `SHM_SIZE` | `16gb` | 共享内存大小 |
| `JUPYTER_PORT` | `8888` | Jupyter 端口映射 |
| `TENSORBOARD_PORT` | `6006` | TensorBoard 端口映射 |
| `HOST_OPENCODE_CONFIG` | `$HOME/.config/opencode` | 宿主机 opencode 配置挂载路径 |
| `HOST_OPENCODE_AUTH` | `$HOME/.local/share/opencode/auth.json` | 宿主机 opencode 认证文件挂载路径 |
| `HOST_AGENTS_DIR` | `$HOME/.agents` | 宿主机全局 Agent skills 挂载路径 |

在 `.env` 文件中修改即可。

## 容器内 opencode

镜像会随 `INSTALL_OPTIONAL_SHELL_TOOLS=1` 自动安装 `opencode-ai`，并把宿主机的
`~/.config/opencode`、`~/.local/share/opencode/auth.json` 和 `~/.agents` 绑定挂载进
容器（`auth.json` 先只读挂载到 `/opt/opencode-host/`，由入口脚本复制到容器 HOME）。
因此容器内直接执行 `opencode` 即可复用宿主机的模型 /provider 配置与登录态；会话
数据库保存在容器卷内（`/tmp/devhome`），不会与宿主机的 SQLite 冲突。

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
