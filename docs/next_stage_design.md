# 下一阶段设计：可信、可复现的实验闭环

本阶段先修正数据流与实验协议，再做模型优化。保留已有 exp01 产物。

## 模块与数据流

- `datasets/labels.py`：严格区分 pixel 真值与 motion_diff 伪标签；校验缓存标签。
- `datasets/splits.py`：训练视频按 seed 划分 train/validation，测试视频仅最终评估。
- `models/backbone.py`：RGB [0,1] 输入按模型预处理参数归一化；缓存记录编码器身份。
- `models/outputs.py`：兼容现有 dataclass，并提供 anomaly_score/frame_score/embedding/explanation 字典接口，保留梯度。
- `train/trainer.py`：模型和 matcher 都更新、AMP 实际执行、保存 RNG/配置/数据划分以恢复训练。
- `eval/aggregation.py`：按真实 frame_indices 聚合，缺失覆盖时报错，避免生成虚假的正常帧。
- `train/experiments.py` + `scripts/run_experiments.py`：配置驱动 prompt/fusion/backbone 实验矩阵，隔离产物、失败返回非零、汇总结果。

张量流：video `(B,T,3,H,W)` → backbone `(B,T,D)`；text `(K,L,D)` →
alignment `(B,T,D_a)/(K,L,D_a)` → fusion `(B,T,D_f)` → temporal →
head `frame_score(B,T), anomaly_score(B), embedding(B,D_f)`。
特征路径从 `(B,T,D)` 开始，与像素路径共用后续模块。

## 验收

1. 无下载 CPU 回归：正负标签、视频隔离、matcher 更新、恢复随机性、输出字典、stride 聚合。
2. 已有容器与真实 Avenue 视频：重建新缓存，不覆盖旧缓存；跑有限轮训练、独立测试评估。
3. 自动运行 label/scene/contrast 与 concat/crossattn，对比记录完整配置。
4. 伪标签结果明确注明，不当作官方异常检测准确率。

## 后续研究边界

真实官方异常标注与正常训练集对应的学习协议仍需单独验证。运动伪标签只能验证工程流程。
取得真值后需要 normal-only/weakly-supervised 方法、独立验证方案、多个 seed 与 backbone 对比、
解释语义的人工核验，才能支撑论文结论。
