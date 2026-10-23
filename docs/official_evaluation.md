# 官方真值接入与独立评估报告

官方真实帧级评估已跑通，工程链路具备可核验的标签与结果。当前性能仍偏低，尚不能称为已实现高质量异常识别。

## 标注核验

- 原始文件：`data/ground_truth_demo.zip`，CRC 完整。SHA256：`60fec1728ec8f73a58aad3aeb5729d70a805a47e0b8eb4bf91ab67ef06386d77`。
- 全部 21 个 testing 视频，15,324 帧，其中 3,712 帧异常。
- 逐视频帧数、360×640 mask 分辨率与视频一致。实际变量是 MATLAB `volLabel` cell，Python 第 0 帧对应第 1 个 cell。
- 每帧任一标注像素非零即为异常；输出 `(N,)` 二值标签。原始 ZIP 保留，未覆盖原来的灰度 vol 文件。
- 官方标签独立存于 `data/labels_official/testing`，缓存带源 ZIP、MAT、标签文件校验值。像素/特征路径共用相同标签。
- 包内没有 training 真值。测试标签不用于训练、阈值校准或模型选择。

## 指标协议

使用全部 15,324 个测试帧的原始分数计算 pooled Frame ROC-AUC / AP；滑窗重叠均值聚合，按实际 frame_indices 对齐，不做逐视频 min-max 归一化。未评估空间定位，不与官方空间检测/定位指标直接等同。

21 个视频都含异常，因此 Video ROC-AUC 为 `null`。Video AP=1 仅反映全正标签，不能说明模型好。08 和 18 视频全为异常，其各自 Frame ROC-AUC 也为 `null`。零正样本时 AP 为 `null`，不把未定义指标伪装为 0 分。

## 固定 zero-shot 基线

冻结 ViT-B-32 / laion2b_s34b_b79k，EOS 文本池化。异常 prompt 平均余弦相似度减正常 prompt 平均相似度。配置化四组 prompt 同时报告，没有按测试结果选择或改写 prompt。

| Prompt | Frame AUC | Frame AP |
|---|---:|---:|
| label_only | 0.5806 | 0.3332 |
| scene_only | 0.6263 | 0.3799 |
| contrast_only | 0.4831 | 0.2416 |
| all_types | 0.6080 | 0.3419 |

结果：`results/official_zero_shot_verified/summary.json`、`summary.md`，每组包含完整分数、真值、候选解释、热力图和 prompt/来源记录。

## 原有六组模型复评

这些 checkpoint 都只训练过 1 epoch，训练目标是 motion_diff 伪标签，选优依据是留出 training 视频。此次只更换测试标签，保留全部模型、prompt 和训练协议。它们不是用官方异常真值训练的模型。

| Prompt | Fusion | Frame AUC | Frame AP |
|---|---|---:|---:|
| label | concat | 0.5822 | 0.2862 |
| label | crossattn | 0.5837 | 0.2880 |
| scene | concat | 0.5857 | 0.2881 |
| scene | crossattn | 0.5855 | 0.2893 |
| contrast | concat | 0.5850 | 0.2884 |
| contrast | crossattn | 0.5867 | 0.2912 |

结果：`results/official_recheck_verified/summary.json`；逐组保存 config、逐视频指标、训练来源、标签来源、密集 NPZ、解释和日志。批量子进程曾遇到容器 MKL/OpenMP 冲突，设置 GNU threading 后六组全部成功；初次失败日志保留在 `results/official_recheck`。

## 正常特征库基线

新增非参数正常性模型：从 12 个 training 视频的冻结 CLIP 帧特征中按 seed=42 抽取最多 4096 个参考帧，查询 5 个最近邻的平均余弦距离。仅在其余 training 视频 06/07/08/10 上用 99% 分位数固定阈值。

真实 Frame AUC=0.5744，Frame AP=0.2923。该结果不是性能提升；其价值是建立了不使用运动标签或测试真值拟合的正常性参照。训练帧被假设为以正常为主，没有逐帧 training 真值，仍可能包含离群帧。

结果：`results/official_normality_verified/metrics.json`；模型：`checkpoints/official_normality_verified/memory.pt`。保存拟合/校准视频划分、抽样索引、参数、正常校准分数、阈值、完整测试分数/真值、帧排名及热力图。

## 验证与可视化

- 最终全部 **51 项测试通过**，包括 3 项真实数据/GPU 检查；记录为 `results/verification/official_final_junit.xml`。
  覆盖真实 GPU 编码/AMP 恢复与官方 MAT/缓存/像素标签对齐。
- 自动拒绝灰度伪 mask、错误帧数/分辨率、重复或缺失视频、伪标签来源混用、文件校验失败和不完整特征缓存。
- 正常特征库测试验证距离几何、标准输出维度、确定性抽样、checkpoint 重载及完全不依赖 training 标注的输入路径。
- `results/demo_official/index.md`、06/18 的 MP4/GIF/帧排名图：显示官方 GT 与模型相对高低分，候选描述仍不代表已确认事件。
- 修复固定 0–1 纵轴的问题；时间轴按原始分数范围显示，真值背景使用独立坐标，不会压扁低分或截掉负分。

## 尚未完成的研究工作

工程可运行与异常检测质量是两个不同的验收层次。当前真实成绩尚不能支持良好的异常识别结论。下一步重点应是训练数据上的区域/时序正常性表征，以及符合 Avenue 场景的文本对齐；需要独立训练验证机制，再开展多 seed、第二 backbone、prompt/fusion/loss 对比。测试标签只能用于固定实验的最终报告，不能反复用于挑参。

目前已完成可信标签、真实评估和正常性参照，尚未完成语义解释人工验证、有效时序训练、多 backbone 正式实验和可用性能门槛验收。
