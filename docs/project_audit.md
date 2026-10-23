# 项目审计与下一阶段进度

本轮已完成一次实际开发与验证：工程链路可以训练、恢复、评估并导出可视化，
官方标注接入后已完成独立的真实帧级评估，详见 [官方评估报告](official_evaluation.md)。
工程链路可运行，但研究目标尚未全部完成，原运动伪标签成绩不能当作真实异常检测成绩。

## 项目当前结构

`datasets/` 加载视频或逐帧特征；`data/` 保留旧接口。
`prompts/` 管理配置化 label/scene/contrast 模板。
`models/` 组合编码器、语义对齐、concat/gated/crossattn 融合、时序模块和分类头。
`train/` 完成 BCE + 视觉文本对比对齐、优化器更新、日志和 checkpoint。
`eval/` 完成滑窗聚合、逐帧/clip/视频指标、异常候选描述与可视化。

输入 video `(B,T,3,H,W)` 或缓存 `(B,T,D)`，输出
`anomaly_score(B), frame_score(B,T), embedding(B,D_f), explanation(Optional[str])`。
可通过 `output.to_dict()` 获取统一字典，保留现有 dataclass 属性接口与 autograd 梯度。
解释在评估时生成，模型内不写 prompt 字符串。

## 排查结果与本轮修复

| 问题 | 修复与影响 |
|---|---|
| testing 每轮参与最优模型选择 | 训练视频按 seed 分为训练和验证，测试集只在训练结束后使用 |
| matcher 有梯度却不在优化器里 | 加入参数组、梯度清零与裁剪；恢复 best 模型时一并恢复 matcher |
| 全正常 batch 跳过分类训练 | 默认对全部 batch 执行 BCE，保持正常类别的分类信号 |
| AMP 配置只有标志、无实际作用 | 实现 autocast、GradScaler、unscale、溢出时不推进 scheduler |
| 断点恢复缺失随机状态与配置检查 | 保存 Python/NumPy/Torch/CUDA/DataLoader RNG、scaler、prompt、完整配置；不兼容恢复直接报错 |
| CLIP 输入未按预训练参数归一化 | 统一原始 RGB resize/crop/normalize 路径；输入内存布局和数值确定性一致 |
| 特征缓存换模型后可被误用 | manifest 校验 encoder、pretrained、预处理版本；拒绝未知旧缓存和未冻结视觉编码器 |
| 特征路径忽略 frame_label_mode | pixel 不读取 motion 标签，motion 模式要求独立缓存；校验标签长度与二值性 |
| 灰度视频被当成二值 mask | pixel 模式明确拒绝该数据，避免得到全部为异常的标签 |
| 运动标签在 clip/chunk 边界不一致 | 按原始视频前一帧计算；提取 chunk 包含边界前驱帧 |
| stride 聚合写错帧坐标、未覆盖帧被当正常 | 两条评估路径共用真实 frame_indices 聚合；缺失覆盖明确报错 |
| 解释在未训练的 head/text 空间匹配 | 使用对比训练的 aligned visual/text 空间，可支持不同 fusion 输出维度 |
| 异常帧排名仅给 clip 内局部坐标 | 输出视频绝对帧号；保存完整分数、标签、完整视频热力图 |
| 缺少自动 fusion/backbone 训练对比 | 增加实验矩阵入口、独立配置/目录、失败状态与非零退出，保护已有 checkpoint |
| MP4 时间轴背景未显示、伪标签称为 GT | 修复画布坐标与背景写入；显示 PSEUDO 标签与相对 HIGH/LOW SCORE |
| 低分、负相似度也被解释成确认异常 | 使用“候选描述”及未校准决策状态，区分检索语义与确认的事件类别 |

设计顺序与接口约定见 [next_stage_design.md](next_stage_design.md)。
旧 `exp01` 模型、缓存和实验文件均保留，新产物使用独立目录。

## 实际验证证据

- 在现有 GPU 容器使用 Python 3.10.14、Torch 2.3.1、open_clip 2.26.1、CUDA 12.1、RTX 4060 Ti。
- 最终完整测试 **32 项通过**（含 2 项真实 GPU 检查）；普通 CI 默认跳过真实数据检查。
  默认回归验证输出、损失、matcher 更新、视频隔离、标签与缓存拒绝路径、stride、配置快照、矩阵失败状态和恢复随机性。
- 额外 GPU 测试使用真实 CLIP 权重和 Avenue 视频：在线/离线特征一致；AMP 训练与验证成功；恢复后下一轮下游权重与连续训练逐值一致。
- 重建 16 个训练视频、21 个测试视频的 v2 特征与标签缓存，共 30,652 帧。
- 执行 3 种 prompt × 2 种 fusion 的六组真实视频训练，每组 1 epoch、seed=42。
- 固定验证视频为 training 的 06/07/08/10，另 12 个视频训练；testing 独立评估。
- 运行独立 eval 入口、zero-shot 入口，以及测试视频 18 的 MP4/GIF/时间轴导出。

证据文件：

- `results/verification/junit.xml`：本轮最终回归与 GPU 检查记录。
- `results/stage2_verified_sweep/summary.json`、`summary.md`：六组运行状态、验证选优与测试指标。
- `results/stage2_verified_*/config.yaml`、`split_manifest.json`、`environment.json`：复现依据。
- `results/stage2_verified_*/frame_scores.npz`、`frame_labels.npz`：完整逐帧输出。
- `results/demo_v2/index.md`、`18_demo.mp4`：最终演示。

以下是**运动伪标签的一轮工程验证**，不是官方真实异常识别准确率，也不是收敛或模型优劣结论：

| Prompt | Fusion | 测试 Frame AUC | 测试 Frame AP |
|---|---|---:|---:|
| label | concat | 0.7156 | 0.1558 |
| label | crossattn | 0.7086 | 0.1858 |
| scene | concat | 0.7197 | 0.1680 |
| scene | crossattn | 0.7114 | 0.1849 |
| contrast | concat | 0.7152 | 0.1660 |
| contrast | crossattn | 0.7140 | 0.1897 |

历史 exp01 的 0.8422 Frame AUC 使用不同预处理且通过测试集选优，不能直接与新实验比较。
当前视频峰值分数也不能用未经校准的阈值解释为可靠异常概率。

## 还剩下什么：按优先级推进

1. **官方异常真值已完成接入。** 用户提供压缩包，已验证 MATLAB `volLabel` cell、
   完整二值性、视频 ID、帧数与分辨率；原 ZIP 保留，生成带校验值的独立标签缓存。
   已运行四组零样本基线、六组原模型复评及正常特征库基线。
2. **正常数据学习协议仍需发展。** [官方数据集说明](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
   描述训练视频以正常场景为主、测试包含异常。因此本项目还需正常性建模或明确的弱监督方案；
   不能把运动伪标签分类器当成官方监督式异常分类基线。这是根据数据协议作出的研究方法判断。
   已固定 zero-shot + EOS 文本池化基线，并从 12 个训练视频建立正常特征库，
   用 4 个留出训练视频校准；该基线帧级 AUC=0.5744，仍需适合异常语义的时序/区域表征与学习方法。
3. **Prompt 语义与解释验证。** 当前 violent/shooting 等描述与 Avenue 的异常种类未必匹配。
   需场景词表、token padding mask/池化对比、正常/异常候选比较和人工核验。
   检索命中描述不是事件事实，更不是模型因果解释。
4. **正式实验规模。** 多 seed、第二个真实 backbone、时序模块与 loss 权重对比、完整训练曲线。
   本轮真实运行了一个 backbone；第二个 backbone 尚未下载/训练，非 CLIP 注册接口只做了契约验证。
   使用不同编码器自己的特征缓存，报告均值/方差、耗时、参数量；不能按测试结果反复挑参数。
5. **指标与决策校准。** 正确处理单类 AUC 的“不可定义”状态；验证集校准阈值，报告召回/误报与 AP。
   Avenue 官方还涉及空间定位评估；当前模型是帧级检测，视频红框不代表像素区域定位。
6. **最终复现交付。** 在真实标注和协议固定后保存完整环境、数据版本、实验配置与论文结果表。
   目前结果足以验证工程运行，尚不足以宣称准确率目标或研究结论达成。

## 复现入口

```bash
# 默认不下载权重的 CPU 回归
python -m pytest -q

# 已有真实权重/数据的 GPU 验证
VLM_REAL_INTEGRATION=1 HF_HUB_OFFLINE=1 python -m pytest -q tests/test_real_pipeline.py

# 修改 sweep.name 后执行新实验；已存在的实验会拒绝覆盖
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --output results/my_sweep

# 恢复被中断的 v2 训练：保持模型、数据、prompt、seed、训练总步数相同
python train.py --config results/my_run/config.yaml --resume checkpoints/my_run/last.pt
```

CLIP 预处理依据已安装版本的 [官方 transform 源码](https://github.com/mlfoundations/open_clip/blob/v2.26.1/src/open_clip/transform.py)
核对。Context7 在本会话未提供可调用工具，未假称已通过该工具查询。
