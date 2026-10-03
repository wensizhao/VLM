# 数据处理流程详解（第一性原理入门）

本文档从零开始，逐步骤解释 `data/` 目录下的数据处理管线。适合想理解"代码为什么这样写"的初学者。

---

## 目录

1. [全景图：一条数据从磁盘到 GPU 的旅程](#1-全景图一条数据从磁盘到-gpu-的旅程)
2. [Step 0：路径解析 — 让程序"找到"数据](#2-step-0路径解析--让程序找到数据)
3. [Step 1：扫描目录 — 建立视频与标注的对应关系](#3-step-1扫描目录--建立视频与标注的对应关系)
4. [Step 2：读取元数据 — 不打开整个视频就知道它有多长](#4-step-2读取元数据--不打开整个视频就知道它有多长)
5. [Step 3：切分 Clip — 把一个长视频切成很多小片段](#5-step-3切分-clip--把一个长视频切成很多小片段)
6. [Step 4：读取像素 — 真正把图像数据加载到内存](#6-step-4读取像素--真正把图像数据加载到内存)
7. [Step 5：加载标注 Mask — 告诉模型"哪里出问题了"](#7-step-5加载标注-mask--告诉模型哪里出问题了)
8. [Step 6：组装样本 — `__getitem__` 的完整流程](#8-step-6组装样本--__getitem__-的完整流程)
9. [Step 7：构建 DataLoader — 批量、打乱、并行加载](#9-step-7构建-dataloader--批量打乱并行加载)
10. [核心概念速查表](#10-核心概念速查表)

---

## 1. 全景图：一条数据从磁盘到 GPU 的旅程

```
磁盘上的文件                         Python 对象                    GPU 上的张量
═══════════                        ════════════                  ════════════
                                                          
  training_videos/01.avi  ─┐                                    
  training_vol/vol01.mat  ─┤     AvenueVideoRecord              
                            │    ├─ video_path                  
                            ├──► ├─ mask_path                   
                            │    ├─ num_frames                  
  training_videos/02.avi  ─┤    └─ frame_height/width           
  training_vol/vol02.mat  ─┘                                    
                                       │                        
                                       ▼                        
                               AvenueClipRecord         ┌──────────────────┐
                               ├─ video_index           │ video: (T,C,H,W) │
                               └─ start_frame           │ mask:  (T,H,W)   │──► model ◄── prompts
                                       │                │ label: (T,)      │         (from PromptManager)
                                       ▼                └──────────────────┘
                                  __getitem__()                │
                                  读取帧 + mask                ▼
                                  + 缩放 + 归一化         DataLoader
                                                        batch_size=8
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │ video: (B,T,C,H,W)│
                                                    │ mask:  (B,T,H,W)  │──► GPU
                                                    │ label: (B,T)      │
                                                    └──────────────────┘
```

**核心思想：** 数据管线做的一直是同一件事——**把真实世界的数据（视频文件、标注文件）转换成模型能理解的张量（float32, 归一化, 固定形状）**。

> **注意：** Dataset 只返回视频和标签，**不包含 prompt 文本**。Prompt 由 `prompts/` 模块独立管理（`PromptManager.get_batch()`）。这样设计的原因是一个视频 clip 在 CLIP/SigLIP 推理时通常需要同时与**多个** prompt 比较，而不是只绑定一个固定的 prompt。详见本文末尾的 [Prompt 解耦说明](#11-prompt-解耦说明)。

---

## 2. Step 0：路径解析 — 让程序"找到"数据

### 2.1 代码：`_resolve_avenue_root()`

```python
def _resolve_avenue_root(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    if (root_path / "training_videos").is_dir():
        return root_path
    nested_root = root_path / "Avenue_Dataset"
    if (nested_root / "training_videos").is_dir():
        return nested_root
    raise FileNotFoundError(...)
```

### 2.2 每一步在做什么

| 操作 | 语法 | 效果 |
|---|---|---|
| `Path(root)` | `pathlib.Path` | 把字符串转成路径对象，跨平台兼容（Windows `\` vs Linux `/`） |
| `.expanduser()` | Path 方法 | 把 `~` 展开成 `/home/vscode` |
| `.resolve()` | Path 方法 | 把相对路径转成绝对路径，例如 `./data` → `/workspace/vlm_ws/data` |
| `/` 运算符 | Path 重载的 `/` | 拼接路径：`root_path / "training_videos"` |
| `.is_dir()` | Path 方法 | 检查这个路径是否存在且是目录 |

### 2.3 为什么要这样设计

数据集的目录结构有两种常见形态：

```
# 形态 A：用户直接指向数据集根目录
data/Avenue_Dataset/
    ├── training_videos/
    └── testing_videos/

# 形态 B：用户指向了上级目录
data/
    └── Avenue_Dataset/
        ├── training_videos/
        └── testing_videos/
```

这个函数自动尝试两种可能，无论用户传 `./data` 还是 `./data/Avenue_Dataset` 都能正确找到。这种"容忍多种输入格式"的设计称为 **Robustness 原则**。

### 2.4 同类设计：`_normalize_split()`

```python
def _normalize_split(split: str) -> Split:
    normalized = split.strip().lower()
    if normalized in {"train", "training"}:
        return "training"
    if normalized in {"test", "testing", "eval", "evaluation"}:
        return "testing"
    raise ValueError(f"Unsupported split: {split!r}")
```

无论用户写 `"Train"`、`"TRAINING"`、`" train "` 都能正确识别。`.strip()` 去空格，`.lower()` 转小写，然后匹配到一个标准值。

**第一性原理：** 用户输入是不可靠的。在系统入口处做一次标准化（canonicalization），后面的代码就只需要处理 `"training"` 和 `"testing"` 两种情况。

---

## 3. Step 1：扫描目录 — 建立视频与标注的对应关系

### 3.1 代码：`_build_video_records()`

```python
def _build_video_records(self) -> list[AvenueVideoRecord]:
    video_dir = self.root / f"{self.split}_videos"
    mask_dir  = self.root / f"{self.split}_vol"
    video_paths = sorted(video_dir.glob("*.avi"))
    
    records: list[AvenueVideoRecord] = []
    for video_path in video_paths:
        video_id = video_path.stem       # 文件名去掉后缀
        mask_path = mask_dir / f"vol{video_id}.mat"
        ...
        records.append(AvenueVideoRecord(...))
    return records
```

### 3.2 每一步在做什么

| 操作 | 语法 | 效果 |
|---|---|---|
| `f"{self.split}_videos"` | f-string | 生成 `"training_videos"` 或 `"testing_videos"` |
| `.glob("*.avi")` | Path 方法 | 匹配目录下所有 `.avi` 文件，返回生成器 |
| `sorted(...)` | 内置函数 | 排序，保证每次扫描顺序一致（**可复现性的第一步**） |
| `video_path.stem` | Path 属性 | `01.avi` → `"01"`，即去掉后缀的文件名 |
| `f"vol{video_id}.mat"` | f-string | 拼接标注文件名：`"vol01.mat"` |

### 3.3 `AvenueVideoRecord` — 为什么用 `@dataclass(frozen=True)`

```python
@dataclass(frozen=True)
class AvenueVideoRecord:
    video_id: str
    split: Split
    video_path: Path
    mask_path: Path
    num_frames: int
    fps: float
    frame_height: int
    frame_width: int
    mask_height: int
    mask_width: int
```

| 概念 | 说明 |
|---|---|
| `@dataclass` | 自动生成 `__init__`、`__repr__` 等，减少样板代码 |
| `frozen=True` | 实例创建后**不可修改**。元数据不应该被中途改掉，这避免了"某处意外修改导致难以排查的 bug" |

这是一个**纯数据容器**——只保存信息，不包含行为。数据与逻辑分离是良好的工程实践。

### 3.4 视频与标注的对应关系

```
training_videos/01.avi  ←→  training_vol/vol01.mat
training_videos/02.avi  ←→  training_vol/vol02.mat
        ...                          ...
```

Avenue 数据集的命名约定：视频文件叫 `<id>.avi`，对应的像素级标注文件叫 `vol<id>.mat`。代码利用这个约定自动配对。

---

## 4. Step 2：读取元数据 — 不打开整个视频就知道它有多长

### 4.1 什么是元数据

元数据是"关于数据的数据"。一个视频文件的元数据包括：

| 元数据 | 含义 | 用途 |
|---|---|---|
| `num_frames` | 总帧数 | 决定能切多少个 clip |
| `fps` | 每秒帧数（如 25 fps） | 时间相关的分析 |
| `frame_height` | 帧高度（像素） | 缩放决策 |
| `frame_width` | 帧宽度（像素） | 缩放决策 |

### 4.2 代码：`_read_video_metadata()`

```python
def _read_video_metadata(video_path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        num_frames   = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    return num_frames, fps, frame_height, frame_width
```

### 4.3 `cv2.VideoCapture` 的工作原理

```python
capture = cv2.VideoCapture(str(video_path))
```

OpenCV 的 `VideoCapture` 不是把整个视频加载到内存，而是打开一个**到文件的"指针"**（类似文件句柄）。你可以通过它：
- 查询元数据（不消耗大量内存）
- 按帧索引跳转（`capture.set(cv2.CAP_PROP_POS_FRAMES, 100)`）

类比：`VideoCapture` 就像一个图书管理员，你问他"这本书有多少页？"（`CAP_PROP_FRAME_COUNT`），他翻一下目录就告诉你，不需要把整本书复印一遍。

### 4.4 `try/finally` 的重要性

```python
try:
    ...  # 读取元数据
finally:
    capture.release()  # 无论如何都要释放
```

视频文件是操作系统资源（文件句柄）。如果不 `release()`，资源会泄漏——积累多了程序会崩溃。`finally` 保证即使在 `try` 中抛出异常，`release()` 也一定会执行。

### 4.5 读取 Mask 元数据：`_read_mask_metadata()`

```python
def _read_mask_metadata(mask_path: Path) -> tuple[int, int, int]:
    variables = whosmat(mask_path)
    for name, shape, _dtype in variables:
        if name == "vol":
            mask_height, mask_width, num_frames = shape
            return int(mask_height), int(mask_width), int(num_frames)
```

这里用了 `scipy.io.whosmat()` 而不是 `loadmat()`。

| 函数 | 行为 | 内存占用 |
|---|---|---|
| `loadmat()` | 读取 **整个** `.mat` 文件到内存 | 大（几十 MB） |
| `whosmat()` | 只读取变量名和形状 | 几乎为零 |

**为什么不直接 `loadmat`：** 我们只想确认 mask 的尺寸是否和视频一致，不需要完整内容。这是一种**懒加载（lazy loading）**思想——不到真正需要的时候就不加载。

---

## 5. Step 3：切分 Clip — 把一个长视频切成很多小片段

### 5.1 为什么要切 Clip

一个视频可能有几百到上千帧。但模型一次只能处理固定长度的片段。原因：

1. **Transformer/CLIP 的输入长度有限**：CLIP ViT 处理单张图，我们要做的是多帧 → 聚合
2. **梯度传播限制**：太长的序列会导致梯度消失/爆炸
3. **显存限制**：一帧 224×224×3 ≈ 150KB (float32)，1000 帧就是 150MB，batch 后直接爆显存
4. **数据增强**：一个视频切成多个 clip，相当于**样本数量扩大了数十倍**

### 5.2 三个关键参数

```python
clip_length = 16    # 每个 clip 采样多少帧
clip_stride = 1     # clip 内部相邻帧的间隔
clip_step   = 16    # 相邻两个 clip 起点之间的间隔
```

用具体例子理解（视频共 100 帧）：

```
参数: clip_length=16, clip_stride=2, clip_step=8

clip 跨度 = (16 - 1) × 2 + 1 = 31 帧  ← 一个 clip 覆盖的时间范围

Clip 0: 帧  0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
Clip 1: 帧  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38
Clip 2: 帧 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46
...
```

```
clip_stride = 1  →  采样连续帧：帧 0,1,2,...,15
clip_stride = 2  →  隔一帧采样：帧 0,2,4,...,30  （时间跨度更大，但采样帧数相同）
clip_step 较小   →  clip 之间有重叠（数据增强效果）
clip_step = clip_length →  不重叠（样本独立）
```

### 5.3 代码：`_compute_clip_starts()`

```python
def _compute_clip_starts(num_frames, clip_span, clip_step,
                         include_last_clip, pad_short_clips) -> list[int]:
    if num_frames < clip_span:
        return [0] if pad_short_clips else []

    starts = list(range(0, num_frames - clip_span + 1, clip_step))
    last_start = num_frames - clip_span
    if include_last_clip and starts and starts[-1] != last_start:
        starts.append(last_start)
    return starts
```

**核心数学：** 

| 变量 | 公式 |
|---|---|
| `clip_span` | `(clip_length - 1) × clip_stride + 1` |
| 第一个起点 | `0` |
| 最后一个合法起点 | `num_frames - clip_span` |
| 按步长生成 | `range(0, num_frames - clip_span + 1, clip_step)` |

**`include_last_clip` 的作用：**

```
num_frames=100, clip_span=31, clip_step=16

range(0, 100-31+1, 16) = range(0, 70, 16) = [0, 16, 32, 48, 64]
最后一个可能的起点 = 100 - 31 = 69

include_last_clip=False → 起点列表: [0, 16, 32, 48, 64]          (69 不在列表里)
include_last_clip=True  → 起点列表: [0, 16, 32, 48, 64, 69]     (手动补上 69)
```

为什么要补最后一个 clip？如果不补，视频末尾的那一段帧就永远不会被用到了——**数据利用率**问题。

### 5.4 `_build_frame_indices()` — 从起点到具体帧号

```python
def _build_frame_indices(start_frame, clip_length, clip_stride, num_frames):
    frame_indices = start_frame + np.arange(clip_length) * clip_stride
    return np.clip(frame_indices, 0, num_frames - 1)
```

例：`start_frame=10, clip_length=16, clip_stride=1`

```
np.arange(16)        → [0, 1, 2, ..., 15]
             * 1     → [0, 1, 2, ..., 15]
       + 10          → [10, 11, 12, ..., 25]
```

`np.clip(..., 0, num_frames-1)` 把越界的帧号限制在合法范围。这是防御性设计——如果真的越界了，宁可重复读一帧也不要崩溃。

---

## 6. Step 4：读取像素 — 真正把图像数据加载到内存

### 6.1 两种后端的对比

| 特性 | OpenCV (`cv2`) | Decord |
|---|---|---|
| 读取方式 | 逐帧 seek + read | 批量获取 `get_batch()` |
| 速度 | 慢（seek 开销大） | 快（内部优化了批量读取） |
| 健壮性 | 高（几乎支持所有格式） | 有时对某些编码失败 |
| 安装 | 轻量 | 需要额外依赖 |

**策略：** `"auto"` 模式优先尝试 decord，失败了静默回退到 OpenCV。这就是**优雅降级**（graceful degradation）。

### 6.2 Decord 批量读取

```python
def _read_frames_with_decord(video_path, frame_indices):
    reader = VideoReader(str(video_path), num_threads=1)
    frames = reader.get_batch(frame_indices.tolist()).asnumpy()
    return frames  # shape: (T, H, W, C), dtype: uint8, RGB
```

| 步骤 | 含义 |
|---|---|
| `VideoReader(path)` | 打开视频文件，建立索引 |
| `.get_batch([0,5,10,15])` | 一次性跳跃读取 4 帧，内部优化了 seek |
| `.asnumpy()` | 从 decord 的 NDArray 转成 NumPy 数组 |
| `num_threads=1` | 单线程——和 PyTorch DataLoader 的多进程协作更好 |

### 6.3 OpenCV 逐帧读取

```python
def _read_frames_with_opencv(video_path, frame_indices):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    current_position = -1

    try:
        for index in frame_indices:
            if current_position != int(index):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_position = int(index) + 1
    finally:
        capture.release()

    return np.stack(frames, axis=0)
```

**关键细节：**

- **`cv2.COLOR_BGR2RGB`**：OpenCV 默认用 BGR 色彩顺序（历史原因），但 PyTorch 和所有主流视觉模型都用 RGB。这里做转换，避免后续颜色通道错乱。
- **`current_position` 缓存**：如果下一帧刚好是连续的（`index == current_position`），就不需要 `capture.set()`，节省一次 seek 操作。
- **`np.stack(frames, axis=0)`**：把帧列表堆叠成 `(T, H, W, C)` 的 4D 数组。`axis=0` 表示在第一个维度（时间）堆叠。

### 6.4 BGR vs RGB 的历史

```
OpenCV 诞生于 2000 年，当时 BGR 是某些摄像头和图像格式的默认像素排列。
RGB 后来成了 Web 和深度学习的主流。两者只是通道顺序不同，像素数据完全一样。
```

---

## 7. Step 5：加载标注 Mask — 告诉模型"哪里出问题了"

### 7.1 什么是 Pixel-level Anomaly Mask

Avenue 数据集为**测试集**的每一帧提供了像素级标注：

```
正常帧：所有像素都是 0
异常帧：异常区域的像素是 1（或 255）
```

这是二值图像（binary mask）。类比：给一张照片盖上红色薄膜，红色区域就是"这里发生了异常"。

### 7.2 `.mat` 文件格式

`.mat` 是 MATLAB 的数据文件格式。Avenue 用 `vol` 变量存储 mask 数据，形状为 `(H, W, T)` （高度 × 宽度 × 帧数）。

```python
# mat 文件内部：
# vol: [H=360, W=640, T=1000] 的 uint8 数组
```

### 7.3 加载代码：`_load_mask_volume()`

```python
@lru_cache(maxsize=8)
def _load_mask_volume(mask_path: str) -> np.ndarray:
    data = loadmat(mask_path)
    volume = np.asarray(data["vol"])                  # (H, W, T)
    binary_volume = (volume > 0).astype(np.float32)   # 二值化
    return np.transpose(binary_volume, (2, 0, 1))     # (T, H, W)
```

### 7.4 逐步解析

**① `loadmat(mask_path)`：**

从 `.mat` 文件读取所有变量，返回一个字典：`{"vol": array(...), "__header__": ..., "__version__": ...}`。

**② `(volume > 0).astype(np.float32)`：**

```python
volume > 0  ：
# [[0, 0, 255, 0],     [[False, False, True,  False],
#  [0, 255, 0, 0],  →   [False, True,  False, False],
#  [0, 0, 0, 0]]         [False, False, False, False]]

.astype(np.float32) ：
# [[0., 0., 1., 0.],
#  [0., 1., 0., 0.],
#  [0., 0., 0., 0.]]
```

这样做是为了：
- **统一标签**：不管原始值是 1 还是 255，都变成 `0.0` 和 `1.0`
- **类型匹配**：PyTorch 模型输出是 `float32`，mask 也得是 `float32`，否则计算 loss 时报错

**③ `np.transpose(..., (2, 0, 1))`：**

```
MATLAB 格式: (H, W, T) = (360, 640, 1000)
                     ↓ transpose
Python 格式:  (T, H, W) = (1000, 360, 640)
```

为什么？因为我们的视频帧是 `(T, C, H, W)`，mask 用 `(T, H, W)` 后，时间维度对齐，操作更自然。

### 7.5 `@lru_cache(maxsize=8)` — 为什么要缓存

```python
@lru_cache(maxsize=8)
def _load_mask_volume(mask_path: str) -> np.ndarray:
    ...
```

| 概念 | 解释 |
|---|---|
| LRU | Least Recently Used，最近最少使用淘汰策略 |
| maxsize=8 | 最多缓存 8 个 mask 文件 |

**为什么需要缓存：** 一个视频可能被切成几十个 clip，每个 clip 都调用一次 `__getitem__`。如果没有缓存，同一个 `.mat` 文件会被反复读取几十次。有了缓存，第一次读取后后续调用直接返回缓存的数组。

类比：你去图书馆查资料，第一次去找书，后面每次直接翻笔记本上的摘抄。

---

## 8. Step 6：组装样本 — `__getitem__` 的完整流程

这是整个数据处理管线最核心的方法。

### 8.1 完整代码（带注释）

```python
def __getitem__(self, index: int) -> dict[str, Any]:
    # ① 找到这个 index 对应的 clip 和视频
    clip_record  = self.clip_records[index]
    video_record = self.video_records[clip_record.video_index]

    # ② 计算要读取哪些帧
    frame_indices = _build_frame_indices(
        start_frame=clip_record.start_frame,
        clip_length=self.clip_length,
        clip_stride=self.clip_stride,
        num_frames=video_record.num_frames,
    )

    # ③ 读取原始视频帧
    raw_frames = self._read_video_frames(video_record.video_path, frame_indices)
    # raw_frames: (T, H, W, C), uint8, [0, 255]

    # ④ NumPy → PyTorch 张量，并调整维度顺序
    video = torch.from_numpy(raw_frames.copy()) \
                 .permute(0, 3, 1, 2) \
                 .contiguous()
    # 现在: (T, C, H, W)

    # ⑤ 转为 float32 并归一化
    video = video.to(dtype=torch.float32) / 255.0
    # 现在: (T, C, H, W), float32, [0.0, 1.0]

    # ⑥ 缩放（如果需要）
    if self.image_size is not None:
        video = F.interpolate(
            video,
            size=self.image_size,     # 例如 (224, 224)
            mode="bilinear",
            align_corners=False,
        )

    # ⑦ 加载对应的 mask
    mask_volume = _load_mask_volume(str(video_record.mask_path))  # (T, H, W)
    clip_masks  = torch.from_numpy(mask_volume[frame_indices].copy()).to(torch.float32)
    # clip_masks: (clip_length, H, W)

    # ⑧ 计算标签
    frame_label = (clip_masks.flatten(start_dim=1).amax(dim=1) > 0).to(torch.long)
    # frame_label: (clip_length,) — 每帧是否有异常
    clip_label  = frame_label.amax()
    # clip_label: 标量 — 这个 clip 是否包含异常

    # ⑨ 返回（不含 prompt — prompt 由 prompts/ 模块独立管理）
    return {
        "video":       video,          # (T, C, H, W)
        "frame_label": frame_label,    # (T,)
        "pixel_mask":  clip_masks,     # (T, H, W)
        "clip_label":  clip_label,     # 标量
        "video_id":    video_record.video_id,
        "start_frame": clip_record.start_frame,
    }
```

### 8.2 关键操作深入解析

#### 8.2.1 `.permute(0, 3, 1, 2)` — 维度重排

```python
video = torch.from_numpy(raw_frames.copy()).permute(0, 3, 1, 2)
#       raw_frames: (T, H, W, C) = (0, 1, 2, 3)
#                         ↓ permute(0, 3, 1, 2)
#       video:      (T, C, H, W) = (0, 3, 1, 2)
```

| 库 | 默认通道位置 | 原因 |
|---|---|---|
| NumPy / OpenCV / decord | `(T, H, W, C)` - Channels Last | 图像存储和 I/O 的自然格式 |
| PyTorch | `(T, C, H, W)` - Channels First | GPU 卷积运算更高效（内存对齐） |

**为什么 PyTorch 用 Channels First：** 卷积核在通道维度上做点积。当 C 维在 H 和 W 前面时，同一个空间位置的所有通道值在内存中是连续的，GPU 可以一次读取。

#### 8.2.2 `.contiguous()` — 内存连续性

`permute()` 只改变了"如何解释内存中的字节"，不移动数据。之后如果要做某些操作（比如 `view()`），需要连续的内存布局。`.contiguous()` 保证这一点，如果已经是连续的就零开销。

#### 8.2.3 `video.to(torch.float32) / 255.0` — 归一化

```
原始像素值: uint8, [0, 255]
归一化后:   float32, [0.0, 1.0]
```

**为什么要归一化：**

1. **数值稳定性**：神经网络权重通常初始化为很小的值（如均值 0，方差 0.01）。如果输入是 0~255，第一层的输出会非常大，梯度爆炸
2. **统一尺度**：不同来源的数据（图像、文本 embedding）都归一化后，模型更容易学习
3. **激活函数**：Sigmoid 在输入接近 0 时梯度最大，输入 ±255 时梯度几乎为 0（饱和区）

#### 8.2.4 `F.interpolate` — 图像缩放

```python
video = F.interpolate(video, size=(224, 224), mode="bilinear", align_corners=False)
```

| 参数 | 含义 |
|---|---|
| `size=(224, 224)` | 目标高度和宽度 |
| `mode="bilinear"` | 双线性插值——取周围 4 个像素的加权平均 |
| `align_corners=False` | 像素对齐方式（PyTorch 推荐默认值，行为与大多数学术工作一致） |

**为什么缩放：**
- CLIP ViT 接受固定输入 224×224（或 336×336）
- 不同视频分辨率不同，缩放统一尺寸
- 降低显存消耗

**为什么 mask 用 `mode="nearest"`：**

```python
clip_masks = F.interpolate(clip_masks.unsqueeze(1), size=self.image_size, mode="nearest").squeeze(1)
```

Mask 是二值的（0 或 1）。bilinear 插值会产生 0.3、0.7 这样的中间值，破坏了二值性。最近邻插值只取最近的像素，保持 0 和 1。

#### 8.2.5 标签计算

```python
frame_label = (clip_masks.flatten(start_dim=1).amax(dim=1) > 0).to(torch.long)
```

分解这个链式调用：

```
clip_masks: (T, H, W)           例如 (16, 224, 224)

.flatten(start_dim=1)
  → (T, H×W)                    例如 (16, 50176)
    保留 dim=0（时间），压缩后面所有维度

.amax(dim=1)
  → (T,)                        例如 [0., 0., 1., 0., ...]
    对每一帧，取所有像素的最大值
    如果该帧有任何异常像素 → 1.0；全正常 → 0.0

> 0
  → (T,)                        [False, False, True, False, ...]
    bool 张量

.to(torch.long)
  → (T,)                        [0, 0, 1, 0, ...]
    转为整数标签
```

**设计逻辑：** 只要一帧中有任意一个像素被标为异常，这一帧就是异常帧。

```python
clip_label = frame_label.amax()  # 任一帧异常 → clip 异常
```

同样的逻辑扩展到 clip 级别。

---

## 9. Step 7：构建 DataLoader — 批量、打乱、并行加载

### 9.1 代码：`build_avenue_dataloader()`

```python
def build_avenue_dataloader(
    root, split="training", batch_size=4, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=False,
    seed=None, **dataset_kwargs
) -> DataLoader:
    
    dataset = build_avenue_dataset(root=root, split=split, **dataset_kwargs)
    
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = seed_worker
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
```

### 9.2 DataLoader 的核心参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `batch_size` | 4 | 每次给模型喂几个样本。太小 → 训练慢，太大 → 显存不够 |
| `shuffle` | True | 每个 epoch 打乱顺序。否则模型记住顺序而不是学习特征 |
| `num_workers` | 0 | `0` = 主进程读数据；`>0` = 开子进程并行读。GPU 训练建议 4-8 |
| `pin_memory` | True | 把数据锁在 CPU 的固定内存区域，GPU 拷贝更快（DMA 传输） |
| `drop_last` | False | 最后一个 batch 不够 `batch_size` 时是否丢弃 |
| `persistent_workers` | 自动 | worker 进程在 epoch 之间常驻，减少创建/销毁开销 |

### 9.3 为什么需要 `num_workers`

```
num_workers=0:
  GPU 等待 → CPU 读帧 → GPU 计算 → GPU 等待 → CPU 读帧 → ...
  50% 时间 GPU 在摸鱼

num_workers=4:
  Worker 1: 读 batch 1  ─┐
  Worker 2: 读 batch 2   │ 并行读取
  Worker 3: 读 batch 3   │
  Worker 4: 读 batch 4  ─┘
  GPU: 一直在计算，几乎不等待
```

类比：4 个备菜员（workers）给 1 个大厨（GPU）备料，大厨不用等。

### 9.4 `pin_memory=True` 的底层原理

```
普通内存 → GPU 显存：CPU 先把数据拷到临时缓冲区 → 再 DMA 到 GPU
Pinned 内存 → GPU 显存：直接 DMA 到 GPU（跳过 CPU 临时缓冲区）

DMA = Direct Memory Access，硬件直接传输，不经过 CPU
```

代价是 pinned 内存不能太大（操作系统限制），但对 batch 大小的数据完全不是问题。

### 9.5 可复现性：`seed_worker`

```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

每次 PyTorch 创建一个 worker 进程，会给它分配一个内部种子。`torch.initial_seed()` 读取这个种子，然后同步给 NumPy 和 Python random。

**为什么需要这个：** 如果不用 `seed_worker`，每个 worker 里的随机行为（比如数据增强的随机裁剪）会使用系统时钟作为种子 → 每次运行结果不同 → 实验不可复现。

---

## 10. 核心概念速查表

### 10.1 张量形状约定

| 数据 | 形状 | 含义 |
|---|---|---|
| 原始视频帧 (NumPy) | `(T, H, W, C)` | 时间, 高, 宽, 通道 |
| 视频帧 (PyTorch) | `(T, C, H, W)` | Channels First |
| Batch 视频 | `(B, T, C, H, W)` | 加了 Batch 维度 |
| Mask 体 | `(T, H, W)` | 无通道维（二值图） |
| 帧级标签 | `(T,)` | 每帧一个 0/1 |
| Clip 级标签 | 标量 | 整个 clip 一个 0/1 |

### 10.2 数据类型约定

| 数据类型 | 用途 | 范围 |
|---|---|---|
| `uint8` | 原始像素 | 0 ~ 255 |
| `float32` | 模型输入/输出、loss 计算 | 通常是 [0, 1] 或归一化后的值 |
| `long` (int64) | 分类标签 | 0 或 1 |

### 10.3 `tuple` vs `list` 返回值

```python
def _read_video_metadata(...) -> tuple[int, float, int, int]:
    return num_frames, fps, frame_height, frame_width  # 隐式打包成 tuple
```

轻量级的多个返回值用 `tuple`（不可变，暗示"这是一组固定的属性"），复杂的数据用 `@dataclass` 或字典。

### 10.4 `sorted()` 的重要性

```python
video_paths = sorted(video_dir.glob("*.avi"))
```

不加 `sorted()`，`glob()` 返回的顺序依赖于文件系统（在不同 OS 甚至同一 OS 不同文件系统上都可能不同）。排序后保证每次运行 `video_paths[0]` 都是 `01.avi` → 可复现。

### 10.5 为什么要 `.copy()` 再转 Tensor

```python
raw_frames = self._read_video_frames(...)       # NumPy 数组
video = torch.from_numpy(raw_frames.copy())      # 先 copy 再转
```

如果 NumPy 数组和 Tensor 共享同一块内存，修改 Tensor 会同时修改 NumPy 数组。`.copy()` 断开这个联系，创建独立副本。代价是复制一次数据（很小，可以忽略）。

---

## 附录：完整数据流追踪

```
用户调用:
  dataloader = build_avenue_dataloader(
      root="./data", split="training", batch_size=8, clip_length=16
  )

内部执行:
  ① build_avenue_dataset() → AvenueDataset.__init__()
     ② _resolve_avenue_root()           → self.root = Path(".../Avenue_Dataset")
     ③ _normalize_split("training")     → self.split = "training"
     ④ _build_video_records()           → 16 个 AvenueVideoRecord
        每个记录: video_path, mask_path, num_frames, fps, 分辨率...
     ⑤ _build_clip_records()            → ~数千个 AvenueClipRecord
        每个记录: video_index, start_frame

  ⑥ build_avenue_dataloader() → DataLoader(dataset, batch_size=8, ...)

训练循环中:
  for batch in dataloader:
      ┌─ DataLoader 调用 dataset.__getitem__(i) 8 次 ─────────────┐
      │ ⑦ frame_indices = _build_frame_indices(start_frame, ...)  │
      │ ⑧ raw_frames = _read_video_frames(video_path, indices)     │
      │ ⑨ video = 转 Tensor → permute → 归一化 → 缩放             │
      │ ⑩ mask = 从 .mat 加载 → 二值化 → transpose → 取对应帧    │
      │ ⑪ label = mask.amax() > 0                                 │
      │ ⑫ return {video, mask, label, video_id, start_frame}      │
      └────────────────────────────────────────────────────────────┘
      → 8 个样本堆叠成 batch
      → {video: (B,T,C,H,W), frame_label: (B,T), ...}
      → 送入 GPU (pin_memory 加速)

  # Prompt 从独立模块获取，与视频 batch 一起传给模型：
  from prompts import PromptManager
  pm = PromptManager()
  prompts = pm.get_batch()  # ["a person running", "a surveillance scene...", ...]
  output = model(batch["video"], prompts)
```

这就是一条 Avenue 视频数据从磁盘到模型输入的全过程（prompt 由独立的 `prompts/` 模块管理）。

---

## 11. Prompt 解耦说明

### 11.1 为什么 Dataset 不返回 prompt

本项目的 Dataset 设计刻意**不包含 prompt 文本**。原因在于 CLIP / SigLIP 视频异常检测的工作方式：

```
传统分类：                          CLIP 异常检测：
图片 → [猫, 狗, 鸟] → 选一个        视频 → [正常走路, 奔跑, 打架, 车辆行驶, ...] → 相似度矩阵
                                            │
                                    一个 clip 同时与 K 个 prompt 比较
                                    而不是只绑定一个 prompt
```

**具体场景：** 一个包含"打架"的视频段：
- 它与 `"a person fighting"` 的相似度应该很高
- 它与 `"a person walking normally"` 的相似度应该很低
- 异常分数 = 与异常行为 prompt 的相似度 − 与正常行为 prompt 的相似度

这需要 Dataset 之外有一个独立的 Prompt 管理器，一次性提供**一组** prompt 给模型。

### 11.2 当前设计

```
┌─────────────────────┐        ┌──────────────────────┐
│ data/               │        │ prompts/             │
│ ├─ AvenueDataset    │        │ ├─ PromptManager     │
│ │  返回: video,     │        │ │  ├─ get_batch()    │──► ["prompt1", "prompt2", ...]
│ │        mask,      │        │ │  └─ get()          │──► "a person running"
│ │        label      │        │ └─ PromptType        │
│ └───────────────────┘        │    DEFAULT_PROMPT_TEMPLATES
         │                    └──────────────────────┘
         │ video (B,T,C,H,W)              │ prompts: list[str] (K 个)
         │                                │
         ▼                                ▼
    ┌──────────────────────────────────────────┐
    │              Model.forward()              │
    │  video_feat = encode_video(video)         │  → (B, D)
    │  text_feat  = encode_text(prompts)        │  → (K, D)
    │  sim_matrix = video_feat @ text_feat.T    │  → (B, K)
    │  anomaly_score = f(sim_matrix)            │  → (B,)
    └──────────────────────────────────────────┘
```

### 11.3 使用示例

```python
from data import build_avenue_dataloader
from prompts import PromptManager

# Dataset 只返回视频和标签
loader = build_avenue_dataloader(root="./data", split="training", batch_size=8)
batch = next(iter(loader))
# batch 包含: video, frame_label, pixel_mask, clip_label, video_id, start_frame
# batch 不包含: prompt_text, prompt_type  ← 已移除

# Prompt 独立管理
pm = PromptManager()
pm.register("normal_walking", "a person walking normally in a corridor")
pm.register("running", "a person running fast")
pm.register("fighting", "two people fighting aggressively")

# 推理时：一个视频 batch 同时对比所有 prompt
prompts = pm.get_batch()  
# ["a person walking normally...", "a person running fast", ...]

output = model(batch["video"], prompts)
# output["similarity"]: (B, K) — 每个 clip 与每个 prompt 的相似度
```

