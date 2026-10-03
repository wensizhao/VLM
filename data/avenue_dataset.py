# 这个文件提供 Avenue 数据集的读取、切片和标注处理工具。
"""Avenue dataset utilities for clip-based video anomaly detection."""

# 让类型注解在运行时以字符串形式延迟解析，减少前向引用问题。
from __future__ import annotations

# dataclass 可以快速定义只存数据的类。
from dataclasses import dataclass
# lru_cache 用来缓存函数结果，避免重复读取同一个 mask 文件。
from functools import lru_cache
# Path 用更安全、跨平台的方式处理文件路径。
from pathlib import Path
# Any 表示任意类型，Literal 表示只能取固定的几个字符串值。
from typing import Any, Literal

# OpenCV 用于读取视频信息和视频帧。
import cv2
# NumPy 用于数组计算和帧索引处理。
import numpy as np
# PyTorch 是本项目的核心深度学习框架。
import torch
# F 里提供了插值缩放等常用张量函数。
import torch.nn.functional as F
# decord 是另一个视频读取后端，通常批量读取速度更快。
from decord import VideoReader
# loadmat 用于读取 .mat 标注文件，whosmat 用于只查看 mat 文件中的变量信息。
from scipy.io import loadmat, whosmat
# Dataset 是 PyTorch 数据集基类。
from torch.utils.data import Dataset

# VideoBackend 限制视频读取后端只能是下面三种之一。
VideoBackend = Literal["auto", "decord", "opencv"]
# Split 限制数据集划分只能是 training 或 testing。
Split = Literal["training", "testing"]


# frozen=True 表示实例创建后字段不可修改，适合保存元数据。
@dataclass(frozen=True)
class AvenueVideoRecord:
    """Metadata for a single Avenue video and its pixel-level annotation."""

    video_id: str  # 视频编号，通常来自文件名去掉后缀后的部分。
    split: Split  # 该视频属于 training 还是 testing。
    video_path: Path  # 视频文件的完整路径。
    mask_path: Path  # 对应像素级标注 .mat 文件的完整路径。
    num_frames: int  # 视频总帧数。
    fps: float  # 视频帧率。
    frame_height: int  # 视频帧高度。
    frame_width: int  # 视频帧宽度。
    mask_height: int  # 标注 mask 的高度。
    mask_width: int  # 标注 mask 的宽度。


# 这个 dataclass 用来表示从一个视频中切出的某个时间片段 clip。
@dataclass(frozen=True)
class AvenueClipRecord:
    """A temporal clip extracted from a single Avenue video."""

    video_index: int  # 这个 clip 来自 self.video_records 中的第几个视频。
    start_frame: int  # 这个 clip 的起始帧下标。


# 统一把各种 split 写法转换成标准写法。
def _normalize_split(split: str) -> Split:
    normalized = split.strip().lower()  # 去掉首尾空格并转成小写，方便兼容不同输入。
    if normalized in {"train", "training"}:  # 如果用户传入的是训练集相关写法。
        return "training"  # 统一返回 training。
    if normalized in {"test", "testing", "eval", "evaluation"}:  # 如果用户传入的是测试/评估相关写法。
        return "testing"  # 统一返回 testing。
    raise ValueError(f"Unsupported split: {split!r}")  # 其他写法直接报错，避免默默出问题。


# 根据用户传入的 root，自动定位真正的 Avenue 数据集根目录。
def _resolve_avenue_root(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()  # 支持 ~ 并转成绝对路径。
    if (root_path / "training_videos").is_dir():  # 如果当前目录下已经能看到 training_videos。
        return root_path  # 说明这里就是数据集根目录，直接返回。

    nested_root = root_path / "Avenue_Dataset"  # 有些下载后的目录结构会多包一层 Avenue_Dataset。
    if (nested_root / "training_videos").is_dir():  # 检查这种嵌套目录结构是否存在。
        return nested_root  # 如果存在，就返回里面这一层。

    raise FileNotFoundError(  # 两种常见结构都找不到时，给出清晰错误信息。
        "Could not locate Avenue dataset directories under "
        f"{root_path}. Expected training_videos/testing_videos."
    )


# 读取视频的基础元数据，不真正把所有帧加载到内存中。
def _read_video_metadata(video_path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(video_path))  # 用 OpenCV 打开视频文件。
    if not capture.isOpened():  # 如果视频打不开。
        raise RuntimeError(f"Failed to open video: {video_path}")  # 直接报错。

    try:  # try/finally 保证后面一定会释放视频句柄。
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 读取总帧数。
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 读取帧宽度。
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 读取帧高度。
        fps = float(capture.get(cv2.CAP_PROP_FPS))  # 读取帧率。
    finally:
        capture.release()  # 无论前面成功还是失败，都释放资源。

    if num_frames <= 0:  # 如果视频帧数异常。
        raise RuntimeError(f"Video has no readable frames: {video_path}")  # 报错提示视频不可用。

    return num_frames, fps, frame_height, frame_width  # 返回视频元数据。


# 只读取 mat 文件中的变量形状信息，用来拿到 mask 的尺寸和帧数。
def _read_mask_metadata(mask_path: Path) -> tuple[int, int, int]:
    variables = whosmat(mask_path)  # 查看 mat 文件里有哪些变量以及它们的形状。
    for name, shape, _dtype in variables:  # 逐个检查变量。
        if name == "vol":  # Avenue 的像素级标注通常保存在 vol 变量中。
            if len(shape) != 3:  # 我们期望它是 3 维数组: H, W, T。
                raise RuntimeError(
                    f"Expected vol to have 3 dimensions in {mask_path}, got {shape}"
                )
            mask_height, mask_width, num_frames = shape  # 拆出高、宽、帧数。
            return int(mask_height), int(mask_width), int(num_frames)  # 返回整数形式的元数据。
    raise KeyError(f"Could not find 'vol' variable in {mask_path}")  # 没找到 vol 就报错。


# 缓存最近读取过的 mask，避免一个视频的多个 clip 重复加载同一份 mat 文件。
@lru_cache(maxsize=8)
def _load_mask_volume(mask_path: str) -> np.ndarray:
    data = loadmat(mask_path)  # 真正读取 .mat 文件内容。
    if "vol" not in data:  # 如果缺少核心变量 vol。
        raise KeyError(f"Could not find 'vol' variable in {mask_path}")  # 直接报错。

    volume = np.asarray(data["vol"])  # 转成 NumPy 数组，便于后续处理。
    if volume.ndim != 3:  # 仍然要求它必须是三维。
        raise RuntimeError(
            f"Expected vol to have 3 dimensions in {mask_path}, got {volume.shape}"
        )

    binary_volume = (volume > 0).astype(np.float32)  # 把所有非零像素转成 1，得到二值 mask。
    return np.transpose(binary_volume, (2, 0, 1))  # 从 (H, W, T) 转成更常用的 (T, H, W)。


# 根据视频长度和 clip 配置，计算每个 clip 的起始帧位置。
def _compute_clip_starts(
    num_frames: int,  # 视频总帧数。
    clip_span: int,  # 一个 clip 实际覆盖的总跨度，例如长度 16、步长 2 时跨度更大。
    clip_step: int,  # 相邻 clip 起点之间的移动步长。
    include_last_clip: bool,  # 是否强制把最后一个能对齐末尾的 clip 加进去。
    pad_short_clips: bool,  # 当视频过短时，是否仍然从第 0 帧构造一个 clip。
) -> list[int]:  # 返回所有 clip 的起始帧列表。
    if num_frames < clip_span:  # 如果视频长度不足以放下一个完整 clip。
        return [0] if pad_short_clips else []  # 允许补短 clip 就返回 [0]，否则返回空列表。

    starts = list(range(0, num_frames - clip_span + 1, clip_step))  # 按固定步长生成所有合法起点。
    last_start = num_frames - clip_span  # 计算“最后一个刚好贴住视频末尾”的起点。
    if include_last_clip and starts and starts[-1] != last_start:  # 如果需要包含最后一个 clip，且当前最后一个起点还没覆盖它。
        starts.append(last_start)  # 手动补上最后一个起点。
    return starts  # 返回最终的起点列表。


# 根据起始帧、clip 长度和采样步长，生成当前 clip 对应的帧下标。
def _build_frame_indices(
    start_frame: int,  # clip 的起始帧。
    clip_length: int,  # clip 中采样多少帧。
    clip_stride: int,  # clip 内部相邻两帧之间的间隔。
    num_frames: int,  # 整个视频的总帧数，用于边界裁剪。
) -> np.ndarray:  # 返回形状为 (clip_length,) 的帧下标数组。
    frame_indices = start_frame + np.arange(clip_length, dtype=np.int64) * clip_stride  # 例如起点 10、步长 2，会得到 10,12,14...
    if num_frames <= 0:  # 额外检查视频帧数是否合法。
        raise RuntimeError("num_frames must be positive")  # 非法时直接报错。
    return np.clip(frame_indices, 0, num_frames - 1)  # 把下标限制在合法范围内，避免越界。


# 使用 decord 按给定下标一次性批量读取多帧。
def _read_frames_with_decord(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    reader = VideoReader(str(video_path), num_threads=1)  # 创建 decord 视频读取器。
    frames = reader.get_batch(frame_indices.tolist()).asnumpy()  # 批量取帧，并转成 NumPy 数组。
    return frames  # 返回形状通常为 (T, H, W, C) 的 RGB 帧数据。


# 使用 OpenCV 按给定下标逐帧读取，多数情况下作为 decord 的回退方案。
def _read_frames_with_opencv(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))  # 打开视频文件。
    if not capture.isOpened():  # 如果打不开视频。
        raise RuntimeError(f"Failed to open video: {video_path}")  # 直接报错。

    frames: list[np.ndarray] = []  # 用列表暂存读取到的每一帧。
    current_position = -1  # 记录当前 OpenCV 读到了哪一帧，减少重复 seek。

    try:  # 确保最后一定释放视频句柄。
        for index in frame_indices:  # 按需要的帧下标逐一读取。
            if current_position != int(index):  # 如果当前游标不在目标帧位置。
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))  # 就手动跳转到指定帧。
            ok, frame = capture.read()  # 读取当前帧。
            if not ok or frame is None:  # 读取失败时。
                raise RuntimeError(
                    f"Failed to read frame {int(index)} from {video_path}"
                )

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV 默认是 BGR，这里转成更常用的 RGB。
            current_position = int(index) + 1  # 更新当前游标到下一帧位置。
    finally:
        capture.release()  # 释放视频资源。

    return np.stack(frames, axis=0)  # 把帧列表堆叠成 (T, H, W, C) 数组。


# 这是最终给训练和推理使用的数据集类。
class AvenueDataset(Dataset[dict[str, Any]]):
    """Clip-based dataset for the CUHK Avenue anomaly detection benchmark."""

    # 初始化函数负责接收配置、检查参数、建立视频索引和 clip 索引。
    def __init__(
        self,
        root: str | Path,  # 数据集根目录。
        split: str = "training",  # 使用训练集还是测试集。
        clip_length: int = 16,  # 每个 clip 采样多少帧。
        clip_stride: int = 1,  # clip 内相邻采样帧之间的间隔。
        clip_step: int | None = None,  # 相邻两个 clip 的起始位置间隔，默认等于 clip_length。
        image_size: tuple[int, int] | None = (224, 224),  # 是否把视频帧缩放到固定大小，格式为 (H, W)。
        video_backend: VideoBackend = "auto",  # 选择视频读取后端，auto 会优先尝试 decord。
        resize_mask_to_video: bool = False,  # 是否把像素级 mask 也缩放到和视频帧一样大。
        include_last_clip: bool = True,  # 是否保证最后一个 clip 覆盖到视频末尾。
        pad_short_clips: bool = False,  # 视频过短时是否仍然构造一个 clip。
    ) -> None:
        if clip_length <= 0:  # clip 长度必须大于 0。
            raise ValueError("clip_length must be positive")
        if clip_stride <= 0:  # clip 内部采样步长必须大于 0。
            raise ValueError("clip_stride must be positive")
        if clip_step is not None and clip_step <= 0:  # 如果显式传了 clip_step，也必须大于 0。
            raise ValueError("clip_step must be positive when provided")
        if image_size is not None and (  # 只要设置了 image_size，就要求它是合法的二元组。
            len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0
        ):
            raise ValueError("image_size must be a (height, width) tuple")
        if video_backend not in {"auto", "decord", "opencv"}:  # 限制后端必须是支持的选项。
            raise ValueError(f"Unsupported video_backend: {video_backend!r}")

        self.root = _resolve_avenue_root(root)  # 解析并保存真实数据集根目录。
        self.split = _normalize_split(split)  # 标准化 split 名称。
        self.clip_length = clip_length  # 保存 clip 长度配置。
        self.clip_stride = clip_stride  # 保存 clip 内采样步长。
        self.clip_step = clip_step if clip_step is not None else clip_length  # 如果没传 clip_step，就默认按不重叠 clip 的方式移动。
        self.image_size = image_size  # 保存输出图像大小配置。
        self.video_backend = video_backend  # 保存视频读取后端配置。
        self.resize_mask_to_video = resize_mask_to_video  # 保存是否缩放 mask 的配置。
        self.include_last_clip = include_last_clip  # 保存是否补最后一个 clip 的配置。
        self.pad_short_clips = pad_short_clips  # 保存短视频是否补 clip 的配置。

        self.video_records = self._build_video_records()  # 扫描所有视频，建立视频级元数据列表。
        self.clip_records = self._build_clip_records()  # 基于视频元数据，建立 clip 级索引列表。

        if not self.clip_records:  # 如果一个 clip 都没生成出来。
            raise RuntimeError(
                "No clips were generated. Check clip_length/clip_stride/clip_step "
                "against the dataset frame counts."
            )  # 给出明确提示，让用户检查切片配置。

    # 扫描目录中的所有视频文件，并与对应标注文件一一配对。
    def _build_video_records(self) -> list[AvenueVideoRecord]:
        video_dir = self.root / f"{self.split}_videos"  # 例如 training_videos 或 testing_videos。
        mask_dir = self.root / f"{self.split}_vol"  # 例如 training_vol 或 testing_vol。
        video_paths = sorted(video_dir.glob("*.avi"))  # 找到所有 avi 视频并排序，保证可复现。
        if not video_paths:  # 如果一个视频都没找到。
            raise FileNotFoundError(f"No AVI videos found in {video_dir}")

        records: list[AvenueVideoRecord] = []  # 用来保存每个视频的元数据记录。
        for video_path in video_paths:  # 逐个处理视频文件。
            video_id = video_path.stem  # 取文件名（不含后缀）作为视频 id。
            mask_path = mask_dir / f"vol{video_id}.mat"  # Avenue 的标注文件命名通常是 vol+视频名.mat。
            if not mask_path.is_file():  # 如果对应标注文件不存在。
                raise FileNotFoundError(
                    f"Missing annotation file for {video_path.name}: {mask_path.name}"
                )

            num_frames, fps, frame_height, frame_width = _read_video_metadata(video_path)  # 读取视频的帧数、帧率和分辨率。
            mask_height, mask_width, mask_frames = _read_mask_metadata(mask_path)  # 读取 mask 的高、宽和帧数。
            if num_frames != mask_frames:  # 视频帧数和标注帧数必须一致。
                raise RuntimeError(
                    "Frame count mismatch between video and annotation: "
                    f"{video_path.name} has {num_frames}, {mask_path.name} has {mask_frames}"
                )

            records.append(  # 构造一个视频级记录对象并加入列表。
                AvenueVideoRecord(
                    video_id=video_id,  # 保存视频编号。
                    split=self.split,  # 保存该视频所属的数据划分。
                    video_path=video_path,  # 保存视频文件路径。
                    mask_path=mask_path,  # 保存对应标注文件路径。
                    num_frames=num_frames,  # 保存视频总帧数。
                    fps=fps,  # 保存视频帧率。
                    frame_height=frame_height,  # 保存视频帧高度。
                    frame_width=frame_width,  # 保存视频帧宽度。
                    mask_height=mask_height,  # 保存 mask 高度。
                    mask_width=mask_width,  # 保存 mask 宽度。
                )
            )
        return records  # 返回所有视频记录。

    # 基于每个视频的长度，进一步计算出所有 clip 的起始位置。
    def _build_clip_records(self) -> list[AvenueClipRecord]:
        clip_span = (self.clip_length - 1) * self.clip_stride + 1  # 一个 clip 从第一帧到最后一帧实际跨越多少原视频帧。
        clip_records: list[AvenueClipRecord] = []  # 保存所有 clip 记录

        for video_index, video_record in enumerate(self.video_records):  # 逐个视频生成 clip。
            start_frames = _compute_clip_starts(  # 计算当前视频所有可用 clip 的起始帧。
                num_frames=video_record.num_frames,  # 当前视频的总帧数。
                clip_span=clip_span,  # 当前 clip 的总时间跨度。
                clip_step=self.clip_step,  # 相邻 clip 起点之间的步长。
                include_last_clip=self.include_last_clip,  # 是否强制补最后一个 clip。
                pad_short_clips=self.pad_short_clips,  # 视频太短时是否仍返回一个 clip。
            )
            clip_records.extend(  # 把当前视频的所有起始帧都转换成 AvenueClipRecord。
                AvenueClipRecord(video_index=video_index, start_frame=start_frame)
                for start_frame in start_frames
            )

        return clip_records  # 返回整个数据集的 clip 列表。

    # 根据配置选择视频读取后端。
    def _read_video_frames(
        self,
        video_path: Path,  # 当前视频路径。
        frame_indices: np.ndarray,  # 需要读取的帧下标。
    ) -> np.ndarray:  # 返回读取出来的多帧数组。
        if self.video_backend in {"auto", "decord"}:  # auto 和 decord 都会先尝试 decord。
            try:
                return _read_frames_with_decord(video_path, frame_indices)  # 优先使用批量读取更方便的 decord。
            except Exception:  # decord 失败时进入回退逻辑。
                if self.video_backend == "decord":  # 如果用户强制要求 decord。
                    raise  # 那就把异常继续抛出，不偷偷换后端。

        return _read_frames_with_opencv(video_path, frame_indices)  # 其他情况回退到 OpenCV 读取。

    # 返回数据集长度，即一共有多少个 clip。
    def __len__(self) -> int:
        return len(self.clip_records)

    # 根据下标取出一个样本，这是 Dataset 最核心的方法。
    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_record = self.clip_records[index]  # 先取到当前 clip 的索引信息。
        video_record = self.video_records[clip_record.video_index]  # 再取到该 clip 对应的视频级信息。

        frame_indices = _build_frame_indices(  # 计算当前 clip 需要读取哪些帧。
            start_frame=clip_record.start_frame,  # clip 的起始帧位置。
            clip_length=self.clip_length,  # 一共采样多少帧。
            clip_stride=self.clip_stride,  # 相邻采样帧之间的间隔。
            num_frames=video_record.num_frames,  # 当前视频总帧数，用于防止越界。
        )
        raw_frames = self._read_video_frames(video_record.video_path, frame_indices)  # 按这些下标读取原始视频帧。
        video = torch.from_numpy(raw_frames.copy()).permute(0, 3, 1, 2).contiguous()  # 把 NumPy 数组转成张量，并从 (T,H,W,C) 改成 PyTorch 常用的 (T,C,H,W)。
        video = video.to(dtype=torch.float32) / 255.0  # 转成 float32，并归一化到 0 到 1。
        if self.image_size is not None:  # 如果配置了固定输出尺寸。
            video = F.interpolate(  # 对每一帧做双线性插值缩放。
                video,  # 输入视频张量，形状是 (T, C, H, W)。
                size=self.image_size,  # 目标输出大小 (H, W)。
                mode="bilinear",  # 图像缩放常用的双线性插值方式。
                align_corners=False,  # 双线性插值的常见安全设置。
            )

        mask_volume = _load_mask_volume(str(video_record.mask_path))  # 读取当前视频完整的 mask 体数据，形状是 (T,H,W)。
        clip_masks = torch.from_numpy(mask_volume[frame_indices].copy()).to(  # 取出当前 clip 对应的 mask，并转成 float32 张量。
            dtype=torch.float32  # mask 用 float32 更方便后续插值和计算。
        )
        frame_label = (clip_masks.flatten(start_dim=1).amax(dim=1) > 0).to(torch.long)  # 只要某一帧任意像素为 1，就把这一帧标成异常帧。
        if self.resize_mask_to_video and self.image_size is not None:  # 如果希望 mask 跟视频帧保持同样分辨率。
            clip_masks = F.interpolate(  # 先临时补一个通道维，再用最近邻插值缩放，最后去掉通道维。
                clip_masks.unsqueeze(1),  # 从 (T, H, W) 变成 (T, 1, H, W)。
                size=self.image_size,  # 缩放到与视频相同的大小。
                mode="nearest",  # mask 要用最近邻，避免插值产生中间值。
            ).squeeze(1)  # 再把通道维去掉，恢复成 (T, H, W)。

        clip_label = frame_label.amax()  # 只要 clip 中任意一帧异常，就把整个 clip 标为异常视频片段。

        return {  # 返回一个完整样本字典，供训练/验证代码直接使用。
            "video": video,  # 视频张量，形状通常是 (T, C, H, W)。
            "frame_label": frame_label,  # 帧级标签，形状是 (T,)。
            "pixel_mask": clip_masks,  # 像素级 mask，形状通常是 (T, H, W)。
            "clip_label": clip_label,  # clip 级标签，标量张量。
            "video_id": video_record.video_id,  # 当前样本来自哪个视频。
            "start_frame": clip_record.start_frame,  # 当前 clip 在原视频中的起始帧位置。
        }
