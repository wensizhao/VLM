# 官方真值接入：模块设计与验收

本阶段目标：检查用户下载的原始压缩包，保留文件校验值；将官方异常标注转换为
逐视频 `(N,)` 二值标签，确认与视频帧一一对应后执行独立测试评估。

模块划分：

- `datasets/annotations.py`：官方 MAT 标注解析，明确识别变量、时间轴、MATLAB cell 数组，验证二值性。
- `tools/prepare_official_labels.py`：检查压缩包完整性、选择标注成员、核对全部测试视频帧数，生成标签和 provenance manifest。
- `datasets/labels.py` / `datasets/cache.py`：校验官方标签来源、视频 ID、帧数及 manifest，避免误用运动标签。
- `datasets/video_dataset.py` / `datasets/feature_dataset.py`：增加显式 `official` 模式，两条输入路径使用同一真值。
- 配置与评估：独立 official 配置与结果目录，固定 zero-shot EOS 基线，并复评已有模型；原模型仍标明运动伪标签训练协议。

视频张量流不变：像素 `(B,T,3,H,W)` 或特征 `(B,T,D)` → 模型。
标注流为原始每帧 mask → `any(nonzero)` → 标签 `(N,)` → 按真实 frame_indices 切为 `(T,)`。
仅有帧标签时不伪造像素定位能力；训练视频的正常性假设与测试标注来源分别记录。

验收必须覆盖：完整包/错误包，二值 mask/灰度帧拒绝，cell/3D 时间轴，视频 ID/帧数一致性，
缺失标注失败，伪标签混用失败，在线与缓存路径相同标签，以及真实测试集逐帧 AUC/AP。
正式性能结论需后续协议与多次实验支持。
