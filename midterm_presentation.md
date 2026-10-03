# VLM-VAD 项目中期答辩文档

> **进展更新（已实现端到端）**：环境/数据/配置/训练/评估/zero-shot/prompt 消融全部跑通。
> 真实指标：zero-shot video AUC 0.80–0.90；微调 10 epoch 后 frame AUC 0.84 / video AUC 1.00。
> 详见 [experiment_results_template.md](experiment_results_template.md) 与 [project_summary.md](project_summary.md)。
> ⚠️ 当前使用 `motion_diff` 伪标签，结果仅用于验证框架。

## 项目概述

### 项目名称
基于视觉语言模型的视频异常检测与可解释性研究

### 项目目标
- **核心目标**：构建基于视觉语言模型（VLM）+ Prompt Engineering的视频异常检测系统
- **技术特点**：支持多模态实验（vision + text prompt），提供可解释的检测结果
- **应用价值**：智能监控、安防检测、异常行为分析

### 核心技术架构
```
Vision-Language Model + Prompt Engineering → Video Anomaly Detection + Explainability
```

## 项目进展

### ✅ 已完成任务

#### 1. 数据集问题解决
- **问题识别**：发现Avenue数据集.mat标注文件实际为灰度视频帧，非二值异常mask
- **解决方案**：
  - 创建了 `dataset_solution.md` 详细文档
  - 提供了三种解决方案：获取真实标注、使用运动伪标签、切换数据集
  - 配置支持 `frame_label_mode: motion_diff` 快速验证
- **当前状态**：运动伪标签模式已配置并完成 zero-shot / prompt 消融 / 微调全流程。

#### 2. Zero版本测试框架
- **完成内容**：
  - 创建了完整的zero-shot测试配置文件
  - 实现了无需训练的模型验证流程
  - 支持冻结CLIP模型的直接推理
- **核心文件**：
  - `configs/experiment.yaml` - 基础实验配置
  - `configs/prompts.yaml` - Prompt模板配置
  - `scripts/zero_shot_test.py` - Zero-shot测试脚本
  - `scripts/prompt_comparison.py` - Prompt对比实验脚本

#### 3. Prompt标注方案
- **三种Prompt类型设计**：
  - **Label类型**：直接描述异常行为（如 "a fighting behavior"）
  - **Scene类型**：描述场景和异常活动（如 "a street with abnormal activity"）
  - **Contrast类型**：对比正常和异常行为（如 "normal vs abnormal behavior"）
- **标注工具开发**：
  - 创建了完整的Web标注界面
  - 支持视频播放、时间轴标注、Prompt选择
  - 包含质量检查和进度跟踪功能
- **标注规范文档**：
  - `prompt_annotation_guide.md` - 详细标注指南
  - `tools/annotation_tool/` - 完整的标注工具实现

#### 4. 实验设计
- **多Prompt对比实验**：
  - 支持不同prompt类型的独立测试
  - 支持prompt组合效果对比
  - 自动生成实验报告和统计分析
- **实验配置**：
  - `label_only` - 仅使用label类型prompt
  - `scene_only` - 仅使用scene类型prompt
  - `contrast_only` - 仅使用contrast类型prompt
  - `all_types` - 使用所有类型prompt组合

### 🔄 进行中任务

#### 5. 项目文档整理
- **架构设计文档**：`models/DESIGN.md`
- **数据处理流程**：`data/README.md`
- **AI Agent编码规范**：`AGENTS.md`
- **数据集解决方案**：`dataset_solution.md`
- **Prompt标注指南**：`prompt_annotation_guide.md`

### 📋 待完成任务

#### 6. 实验结果展示（已完成初版）
- ✅ 已运行 zero-shot 测试并生成真实结果（`results/zero_shot_test.json`）
- ✅ 已完成 prompt 对比实验（`results/prompt_comparison.{json,md}`）
- ✅ 已生成可视化材料（`results/*/plots/*.png`、`results/*/explanation*.json`）
- ⏳ 待补充：真实二值 mask 下的正式结果

## 技术架构详解

### 1. 模块化设计
```
vlm_ws/
├── models/                  # 模型模块
│   ├── backbone.py          # CLIP视觉+文本编码器
│   ├── fusion.py           # 多模态融合（concat/cross-attention）
│   ├── matcher.py          # 相似度匹配
│   ├── temporal.py         # 时序建模
│   └── outputs.py          # 统一输出格式
├── datasets/               # 数据模块
│   ├── video_dataset.py    # 原始视频Dataset
│   ├── feature_dataset.py  # 预提取特征Dataset
│   └── builders.py         # 配置驱动构建
├── prompts/                # Prompt模块
│   ├── processor.py        # Prompt处理和缓存
│   └── __init__.py         # Prompt管理器
├── train/                  # 训练模块
│   ├── trainer.py          # 训练循环
│   └── losses.py           # 损失函数
├── eval/                   # 评估模块
│   ├── inference.py        # 推理和评估
│   └── metrics.py          # 评估指标
└── tools/                  # 工具模块
    └── annotation_tool/    # 人工标注工具
```

### 2. 核心算法流程

#### Zero-shot推理流程
1. **视频特征提取**：使用冻结的CLIP视觉编码器提取帧特征
2. **Prompt编码**：将文本prompt通过CLIP文本编码器转换为特征
3. **相似度计算**：计算视觉特征与文本特征的余弦相似度
4. **异常评分**：相似度越低，异常可能性越高
5. **时序聚合**：对帧级分数进行时序聚合得到视频级结果

#### Prompt处理机制
- **模板展开**：将预设模板展开为具体的prompt文本
- **极性标记**：为正负样本添加极性关键词
- **缓存机制**：避免重复计算相同prompt的特征
- **组合策略**：支持多种prompt的组合使用

### 3. 损失函数设计
```python
VLMVADLoss = α * AnomalyClassificationLoss + β * ContrastiveLoss
```
- **异常分类损失**：二值交叉熵，学习异常/正常分类
- **对比损失**：视觉-文本对齐损失，增强语义匹配
- **权重调节**：可调节两个损失的相对重要性

## 实验设计

### 1. 数据集配置
- **主要数据集**：CUHK Avenue Dataset
- **备选数据集**：UCF-Crime, ShanghaiTech Campus
- **标注模式**：
  - `pixel`：真实二值标注（需要获取真实标注）
  - `motion_diff`：运动差分伪标签（当前使用）

### 2. Prompt实验配置
| 实验名称 | Prompt类型 | 数量 | 预期特点 |
|---------|-----------|------|---------|
| label_only | Label类型 | 10个 | 直接描述行为，语义明确 |
| scene_only | Scene类型 | 10个 | 上下文信息丰富 |
| contrast_only | Contrast类型 | 10个 | 强调对比性 |
| all_types | 混合类型 | 9个 | 信息最全面 |

### 3. 评估指标
- **帧级指标**：AUC, AP (Frame-level)
- **视频级指标**：AUC, AP (Video-level)
- **可解释性**：时序热力图，attention可视化

## 创新点

### 1. Prompt Engineering创新
- **多类型Prompt设计**：针对不同异常特点设计专门的prompt模板
- **自适应Prompt选择**：根据异常类型自动选择最合适的prompt
- **Prompt组合策略**：探索不同prompt组合的协同效应

### 2. Zero-shot检测机制
- **无需训练**：直接使用预训练CLIP进行异常检测
- **可解释性强**：通过prompt提供直观的异常解释
- **灵活性强**：支持快速更换prompt进行实验

### 3. 模块化架构
- **配置驱动**：通过YAML配置文件控制整个实验流程
- **即插即用**：支持不同的backbone、fusion、temporal模块
- **可扩展性**：易于添加新的prompt类型和模型组件

## 应用场景

### 1. 智能监控
- 商场、银行、机场等公共场所的异常行为检测
- 实时报警和异常事件记录

### 2. 交通管理
- 交通违规行为检测
- 异常交通事件识别

### 3. 工业安全
- 工厂车间异常行为监控
- 安全生产违规检测

### 4. 社区安防
- 居民小区安全监控
- 可疑行为识别

## 技术优势

### 1. 相比传统方法
- **无需标注数据**：zero-shot特性减少对标注数据的依赖
- **可解释性强**：提供文本形式的异常解释
- **适应性强**：通过prompt工程适应不同场景

### 2. 相比其他VLM方法
- **专门优化**：针对视频异常检测任务优化
- **时序建模**：考虑视频的时序特性
- **多模态融合**：有效的视觉-语言对齐机制

## 风险评估与应对

### 1. 技术风险
- **数据集质量问题**：已制定解决方案，支持多种数据源
- **模型性能**：zero-shot baseline已建立，可逐步优化
- **计算资源**：支持特征预提取，减少训练时间

### 2. 项目风险
- **进度风险**：模块化设计确保各部分可并行开发
- **质量风险**：完整的测试框架和评估体系
- **人员风险**：详细的文档和工具支持

## 下一步计划

### 1. 短期目标（1-2周）
- [x] 运行zero-shot测试，获取基础结果
- [x] 完成prompt对比实验
- [x] 准备中期答辩材料

### 2. 中期目标（1-2月）
- [ ] 获取真实数据集标注（当前首要阻塞）
- [x] 实现端到端训练（下游 alignment+fusion+head）
- [ ] 优化模型架构（fusion / temporal 消融）

### 3. 长期目标（3-6月）
- [ ] 扩展到多个数据集
- [ ] 实现实时检测系统
- [ ] 发表学术论文

## 预期成果

### 1. 学术成果
- **高水平论文**：目标CVPR/ICCV/ECCV等顶级会议
- **开源项目**：完整的代码库和文档
- **数据集贡献**：标注工具和实验数据

### 2. 技术成果
- **可复现系统**：完整的实验流程和评估体系
- **实用工具**：标注工具和可视化界面
- **性能基准**：zero-shot和fine-tuned的性能基线

### 3. 应用成果
- **原型系统**：可演示的异常检测系统
- **行业应用**：与合作伙伴的实际应用
- **技术转化**：专利和商业应用

## 总结

VLM-VAD项目已经建立了完整的技术框架和实验体系，具备了以下优势：

1. **技术先进性**：基于最新的视觉语言模型技术
2. **实用性**：模块化设计，易于部署和扩展
3. **创新性**：prompt工程在视频异常检测的创新应用
4. **完整性**：从数据处理到模型评估的完整流程

项目已经为中期答辩做好了充分准备，下一步将重点完成实验验证和结果展示，为后续的研究和应用奠定坚实基础。