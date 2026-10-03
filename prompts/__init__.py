# 这个文件负责管理和获取 prompt 模板文本。
"""Prompt template management — decoupled from encoding and dataset internals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

# 三种内置 prompt 类型对应的别名，方便下游类型注解。
PromptType = Literal["label", "scene", "contrast"]

# 项目自带的默认模板，开箱即用，也方便其他模块引用。
DEFAULT_PROMPT_TEMPLATES: dict[str, str] = {
    "label": "a person running",
    "scene": "a surveillance scene with abnormal behavior",
    "contrast": "normal vs abnormal behavior comparison",
}


class PromptManager:
    """可扩展的 prompt 模板管理器。

    设计目标：
    - 内置 label / scene / contrast 三种默认模板
    - 支持构造时覆盖或注册新模板
    - 后续实验只需一个小对象，不需要改代码
    """

    def __init__(
        self,
        templates: Mapping[str, str] | None = None,
        default_type: PromptType | str = "scene",
    ) -> None:
        # 从默认模板复制一份，避免修改全局变量
        self._templates = dict(DEFAULT_PROMPT_TEMPLATES)
        if templates is not None:
            self._templates.update(templates)

        if default_type not in self._templates:
            raise KeyError(
                f"Default prompt type {default_type!r} is not in templates. "
                f"Available: {list(self._templates)}"
            )
        self._default_type = default_type

    # ── 核心 API ──────────────────────────────────────────────

    def get(self, prompt_type: str | None = None) -> str:
        """按类型获取单个 prompt 文本，不传则使用默认类型。"""
        key = prompt_type if prompt_type is not None else self._default_type
        if key not in self._templates:
            raise KeyError(
                f"Prompt type {key!r} not found. "
                f"Available: {list(self._templates)}"
            )
        return self._templates[key]

    def get_batch(self, prompt_types: list[str] | None = None) -> list[str]:
        """返回多个 prompt 文本，供模型对同一视频进行多 prompt 对比。

        这是 CLIP / SigLIP + 视频异常检测的推荐用法：
        一个视频 clip 同时与一组 prompt 比较，得到 (B, K) 相似度矩阵，
        再据此判断异常。

        Args:
            prompt_types: 需要的 prompt 类型列表。
                          传 None 则返回全部已注册的 prompt。
        Returns:
            list[str] — 例如 ["a person running", "a surveillance scene..."]
        """
        if prompt_types is None:
            prompt_types = list(self._templates.keys())
        missing = [t for t in prompt_types if t not in self._templates]
        if missing:
            raise KeyError(
                f"Prompt types {missing!r} not found. "
                f"Available: {list(self._templates)}"
            )
        return [self._templates[t] for t in prompt_types]

    def get_batch_with_types(
        self, prompt_types: list[str] | None = None
    ) -> list[tuple[str, str]]:
        """同 :meth:`get_batch`，但同时返回每个 prompt 的类型名。

        返回值中的类型名可用于异常解释：
        "该视频最匹配的 prompt 是 'label' (a person running)"。

        Returns:
            list[tuple[str, str]] — [(type, text), ...]
        """
        if prompt_types is None:
            prompt_types = list(self._templates.keys())
        missing = [t for t in prompt_types if t not in self._templates]
        if missing:
            raise KeyError(
                f"Prompt types {missing!r} not found. "
                f"Available: {list(self._templates)}"
            )
        return [(t, self._templates[t]) for t in prompt_types]

    # ── 扩展 API ──────────────────────────────────────────────

    def register(self, name: str, template: str) -> None:
        """注册一个新的 prompt 模板，方便做 ablation 实验。"""
        self._templates[name] = template

    def remove(self, name: str) -> None:
        """删除某个模板（不能删除最后一个，至少保留一个）。"""
        if name not in self._templates:
            raise KeyError(f"Prompt type {name!r} does not exist.")
        if len(self._templates) <= 1:
            raise RuntimeError("Cannot remove the last remaining prompt template.")
        if name == self._default_type:
            raise RuntimeError(
                f"Cannot remove the current default type {name!r}. "
                "Change default_type first."
            )
        del self._templates[name]

    # ── 查询 API ──────────────────────────────────────────────

    @property
    def default_type(self) -> str:
        return self._default_type

    @default_type.setter
    def default_type(self, value: str) -> None:
        """更换默认 prompt 类型。"""
        if value not in self._templates:
            raise KeyError(
                f"Cannot set default to {value!r} — not in templates. "
                f"Available: {list(self._templates)}"
            )
        self._default_type = value

    def list_types(self) -> list[str]:
        """列出所有当前可用的 prompt 类型名。"""
        return list(self._templates)

    def to_dict(self) -> dict[str, str]:
        """返回当前全部模板的浅拷贝，方便传给 Dataset。"""
        return dict(self._templates)

    def __repr__(self) -> str:
        return (
            f"PromptManager(types={self.list_types()}, "
            f"default={self._default_type!r})"
        )
