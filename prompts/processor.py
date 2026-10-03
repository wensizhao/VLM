# 这个文件负责把原始 prompt 模板扩展为完整的 prompt 列表——Open-Vocabulary 的入口。
"""Prompt processing pipeline for Open-Vocabulary Video Anomaly Detection.

PromptProcessor 在 :class:`PromptManager` 之上构建：
- **Template Expansion** — 一个模板 → 多种变体（"a person {action}" → N prompts）
- **Ensemble**            — 同时使用多种 prompt 类型（label + scene + contrast）
- **Hard Negatives**      — 训练时生成易混淆的负样本
- **Cache**               — 避免重复处理相同 prompt

与旧 PromptManager 的关系：
  PromptManager    = 底层模板存储（get / register / remove）
  PromptProcessor  = 上层处理管线（expand / ensemble / negative / cache）

未来扩展：Learnable Prompt / Prefix Tuning 也可以作为新的处理步骤插入。
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping
from typing import Any

from . import DEFAULT_PROMPT_TEMPLATES, PromptManager


class PromptProcessor:
    """Prompt 处理管线——Open-Vocabulary VAD 的入口。

    典型用法::

        pp = PromptProcessor(
            templates={"label": "a person {action}", "scene": "a {scene} scene"},
            expansions={
                "action": ["running", "fighting", "walking", "loitering"],
                "scene":  ["normal", "abnormal", "crowded", "empty"],
            },
        )
        prompts = pp.process(types=["label", "scene"], expand=True)
        # → ["a person running", "a person fighting", ..., "a normal scene", ...]

        # 训练时加入 hard negatives
        prompts = pp.process(expand=True, hard_negatives=True)
        # → normal prompts + hard negative prompts + labels

    参数化:
        - ``types``: 使用哪些模板类型，None = 全部
        - ``expand``: 是否展开占位符
        - ``hard_negatives``: 是否加入 hard negative prompts
    """

    # 内置的 hard negative 模式——对每个正常 prompt，生成语义相近但异常的变体
    _HARD_NEGATIVE_PATTERNS: tuple[tuple[str, str], ...] = (
        ("normal", "abnormal"),
        ("walking", "running suspiciously"),
        ("standing", "loitering"),
        ("empty", "sudden crowd"),
        ("clean", "messy scattered objects"),
    )

    def __init__(
        self,
        templates: Mapping[str, str] | None = None,
        expansions: Mapping[str, list[str]] | None = None,
        hard_negatives: list[str] | None = None,
        cache_enabled: bool = True,
    ) -> None:
        """初始化 PromptProcessor。

        Args:
            templates:  模板字典，{"type": "template with {placeholder}"}。
                        不传则使用 DEFAULT_PROMPT_TEMPLATES。
            expansions: 占位符展开字典，{"placeholder": ["value1", "value2"]}。
            hard_negatives: 自定义 hard negative prompts。不传则使用内置模式。
            cache_enabled: 是否缓存 process() 结果。
        """
        # 底层模板管理器
        # PromptManager 设计上总是从 DEFAULT_PROMPT_TEMPLATES 起步再 update，
        # 所以当用户显式传 templates 时，需要先切换 default_type 再清理遗留默认模板。
        user_templates = dict(templates) if templates is not None else None
        self._manager = PromptManager(templates=user_templates)
        if user_templates is not None:
            # 把 default_type 切到用户的一个 key 上，否则 remove 会拒绝删除默认类型
            first_user_key = next(iter(user_templates.keys()))
            self._manager.default_type = first_user_key
            # 清理不属于用户的默认模板
            for key in self._manager.list_types():
                if key not in user_templates:
                    try:
                        self._manager.remove(key)
                    except RuntimeError:
                        pass  # 最后一个模板不可删除

        self._expansions = dict(expansions) if expansions is not None else {}

        # Hard negatives
        self._custom_hard_negatives: list[str] = list(hard_negatives) if hard_negatives else []

        # Cache
        self._cache_enabled = cache_enabled
        self._cache: dict[tuple, list[str]] = {}

    # ── 核心 API ──────────────────────────────────────────────────

    def process(
        self,
        types: list[str] | None = None,
        expand: bool = True,
        hard_negatives: bool = False,
    ) -> list[str]:
        """处理并返回完整的 prompt 列表。

        Args:
            types:          使用哪些模板类型，None = 全部已注册模板。
            expand:         是否用 expansions 展开占位符。
            hard_negatives: 是否加入 hard negative prompts（通常仅训练时）。

        Returns:
            list[str] — K 个处理好的 prompt 字符串。
        """
        # 检查缓存
        cache_key = self._make_cache_key(types, expand, hard_negatives)
        if self._cache_enabled and cache_key in self._cache:
            return list(self._cache[cache_key])

        # 1. 收集模板
        template_types = types if types is not None else self._manager.list_types()
        templates = [(t, self._manager.get(t)) for t in template_types]

        # 2. 展开占位符
        if expand and self._expansions:
            prompts = self._expand_templates(templates)
        else:
            prompts = [text for _, text in templates]

        # 3. 去重（保持顺序）
        prompts = list(dict.fromkeys(prompts))

        # 4. Hard negatives（仅训练时）
        if hard_negatives:
            negatives = self._generate_hard_negatives(prompts)
            prompts = prompts + negatives

        # 缓存
        if self._cache_enabled:
            self._cache[cache_key] = list(prompts)

        return prompts

    def process_with_types(
        self,
        types: list[str] | None = None,
        expand: bool = True,
        hard_negatives: bool = False,
    ) -> list[tuple[str, str]]:
        """同 :meth:`process`，但返回 ``[(type, text), ...]``。

        类型信息可用于异常解释："该视频最匹配的 prompt 是 'label' 类型"。
        """
        prompt_texts = self.process(
            types=types, expand=expand, hard_negatives=hard_negatives,
        )
        typed: list[tuple[str, str]] = []
        for text in prompt_texts:
            # 反查类型：如果文本匹配某个模板的展开结果，标记该类型
            ptype = self._guess_type(text)
            typed.append((ptype, text))
        return typed

    # ── 查询 / 管理 API ──────────────────────────────────────────

    @property
    def num_prompts(self) -> int:
        """当前配置下 process() 会返回多少个 prompt。"""
        return len(self.process())

    def get_expansions(self, placeholder: str) -> list[str]:
        """查看某个占位符的所有展开值。"""
        if placeholder not in self._expansions:
            raise KeyError(
                f"Placeholder {placeholder!r} not found. "
                f"Available: {list(self._expansions)}"
            )
        return list(self._expansions[placeholder])

    def register_expansion(self, placeholder: str, values: list[str]) -> None:
        """注册或替换一个占位符的展开值列表。"""
        self._expansions[placeholder] = list(values)
        self._invalidate_cache()

    def register_template(self, name: str, template: str) -> None:
        """注册一个新的 prompt 模板类型。"""
        self._manager.register(name, template)
        self._invalidate_cache()

    def remove_template(self, name: str) -> None:
        """删除一个模板类型。"""
        self._manager.remove(name)
        self._invalidate_cache()

    def list_types(self) -> list[str]:
        """列出所有已注册的模板类型名。"""
        return self._manager.list_types()

    def clear_cache(self) -> None:
        """清空 prompt 缓存（修改 expansions 或 templates 后自动调用）。"""
        self._cache.clear()

    # ── 内部实现 ──────────────────────────────────────────────────

    def _expand_templates(
        self, templates: list[tuple[str, str]]
    ) -> list[str]:
        """展开模板中的 {placeholder} 占位符。

        每个模板中每个占位符独立展开，产生笛卡尔积。
        例如 "a {adj} {scene}" × adj=["normal","abnormal"] × scene=["scene"]
        → 2 个 prompt。
        """
        _PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

        result: list[str] = []
        for _ptype, template in templates:
            # 找出模板中所有的占位符
            placeholders = _PLACEHOLDER_RE.findall(template)
            if not placeholders:
                result.append(template)
                continue

            # 为每个占位符收集可选值（只有全部存在时才能展开）
            choices: list[list[str]] = []
            expandable = True
            for ph in placeholders:
                if ph in self._expansions:
                    choices.append(self._expansions[ph])
                else:
                    expandable = False
                    break

            if not expandable:
                result.append(template)
                continue

            # 笛卡尔积展开
            for combination in itertools.product(*choices):
                text = template
                for ph, val in zip(placeholders, combination):
                    text = text.replace(f"{{{ph}}}", val)
                result.append(text)

        return result

    # 用于按整个单词精确匹配的正则（避免 "abnormal" 匹配到 "normal" 子串）
    _WORD_BOUNDARY = re.compile(r"\b")

    def _generate_hard_negatives(self, prompts: list[str]) -> list[str]:
        """基于内置模式 + 自定义列表生成 hard negative prompts。

        Hard negatives 的直觉：
        - "a person walking" → "a person running suspiciously"
        - "normal scene"       → "abnormal scene"
        语义相近但属于异常类别，迫使模型学习细粒度区分能力。

        使用词边界正则确保只替换整个单词，避免子串匹配错误
        （如 "abnormal" 中的 "normal" 被二次替换）。
        """
        negatives: list[str] = []

        # 内置替换模式
        for prompt in prompts:
            for source, target in self._HARD_NEGATIVE_PATTERNS:
                # 词边界匹配：source 必须是独立单词，不是其他单词的一部分
                pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)
                if pattern.search(prompt):
                    neg = pattern.sub(target, prompt)
                    negatives.append(neg)

        # 自定义 hard negatives
        negatives.extend(self._custom_hard_negatives)

        # 去重
        return list(dict.fromkeys(negatives))

    def _guess_type(self, text: str) -> str:
        """反向推断一个 prompt 属于哪个模板类型。

        策略：检查文本是否匹配某个模板（展开后）的模式。
        失败时返回 "unknown"。
        """
        for ptype in self._manager.list_types():
            template = self._manager.get(ptype)
            # 把占位符替换为正则通配符
            pattern = re.escape(template)
            pattern = re.sub(r"\\\{[^}]+\\\}", r".*", pattern)
            if re.fullmatch(pattern, text):
                return ptype
        return "unknown"

    def _make_cache_key(
        self,
        types: list[str] | None,
        expand: bool,
        hard_negatives: bool,
    ) -> tuple[Any, ...]:
        """生成缓存键——types 转为 tuple 以保证 hashable。"""
        return (
            tuple(sorted(types)) if types is not None else None,
            expand,
            hard_negatives,
        )

    def _invalidate_cache(self) -> None:
        """修改配置后自动清空缓存。"""
        self.clear_cache()

    # ── 序列化 ────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        """导出 Processor 配置（不包含 cache）。"""
        return {
            "templates": self._manager.to_dict(),
            "expansions": dict(self._expansions),
            "hard_negatives": list(self._custom_hard_negatives),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PromptProcessor":
        """从配置恢复 Processor。"""
        return cls(
            templates=state.get("templates"),
            expansions=state.get("expansions"),
            hard_negatives=state.get("hard_negatives"),
        )

    def __repr__(self) -> str:
        n_regular = len(self.process(expand=True, hard_negatives=False))
        n_hard = len(self.process(expand=True, hard_negatives=True)) - n_regular
        return (
            f"PromptProcessor("
            f"types={self.list_types()}, "
            f"expansions={list(self._expansions.keys())}, "
            f"K={n_regular}+{n_hard}neg, "
            f"cached={len(self._cache) > 0})"
        )
