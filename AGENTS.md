## 1 目标（Agent Objective）
本仓库的AI Agent任务是： 在“视觉语言模型 + 视频异常检测”方向下，生成结构清晰、可训练、可复现、模块化的研究级代码，并支持多模态实验（vision + text prompt）。
**Agent 的输出必须服务于以下目标：**
+ 可运行（not pseudo-code）
+ 可复现（reproducible）
+ 可扩展（research-friendly）
+ 可实验对比（prompt / model / fusion）
## 2 代码生成原则（Code Generation Principles）
### 2.1 不允许“玩具代码”
Agent **不得输出**：
+ 只有函数框架但不可运行
+ 未实现 forward / loss
+ 未定义输入输出维度
+ 空 training loop
**必须输出**：
+ 完整 PyTorch module
+ 明确 tensor shape 流动
+ 可运行 training/inference pipeline
### 2.2 模块化强制要求
所有代码必须拆分为**以下结构**：
models/ → backbone + fusion + head
data/ → dataset + dataloader
prompts/ → prompt templates
train/ → training loop
eval/ → evaluation metrics
utils/ → logging / visualization
**禁止**：
+ 所有逻辑写在一个 .py 文件：hardcode dataset path / prompt text
### 2.3 VLM使用规范
Agent **必须遵守**：
+ vision encoder 与 text encoder 必须可替换
+ 禁止写死 CLIP（除非 baseline）
+ fusion 必须至少支持两种：
concat fusion
cross-attention fusion（推荐）
## 3 Prompt使用规范（Critical）
### 3.1 Prompt必须可配置
所有 prompt 必须来自：
Python
config.prompt_type = "label | scene | contrast"
**禁止**：
在 forward() 内写死 prompt
在模型内部生成 prompt string
### 3.2 Prompt类型要求
**必须支持**：
label prompt
"a person running"
scene prompt
"a surveillance scene with abnormal behavior"
contrast prompt
"normal vs abnormal behavior comparison"
### 3.3 Prompt实验要求
Agent生成实验代码时必须支持：
多prompt batch evaluation
prompt ablation study
prompt switching without code rewrite
## 4 输出规范（Output Specification）
**所有模型 forward 输出必须包含**：
Python
{
  "anomaly_score": Tensor[batch],
  "frame_score": Tensor[batch, T],
  "embedding": Tensor[batch, D],
  "explanation": Optional[str]
}
**禁止**：
+ 只输出 loss
+ 只输出 logits（没有解释或score）
## 5 可解释性要求（Explainability）
Agent 必须优先支持：
### 5.1 基础解释
abnormal frames ranking
temporal anomaly heatmap
### 5.2 文本解释（必须支持至少一种）
template-based explanation
retrieval-based description
LLM-generated explanation（optional）
输出必须能回答：
“为什么这个视频是异常的？”
## 6训练规范（Training Rules）
### 6.1 Loss必须明确
至少包含：
+ anomaly classification loss
+ contrastive loss（vision-text alignment）
**禁止**：
undefined loss function
不解释 loss 作用
### 6.2 训练循环必须完整
必须包含：
+ forward
+ loss computation
+ backward
+ optimizer step
+ logging
## 7实验可复现性（Reproducibility）
必须固定：
+ random seed
+ dataloader shuffle seed
+ model config
+ prompt config
必须输出：
Bash
results/
logs/
checkpoints/
configs/
## 8 实验对比要求（Critical for Research）
**Agent必须自动支持**：
prompt type ablation
fusion method comparison
backbone swap experiments
禁止：
单一模型不可扩展结构
## 9 代码风格规范
Python 3.10+
PyTorch-first
typing recommended
avoid global variables
avoid magic numbers
## 10 Agent行为约束（VERY IMPORTANT）
Agent在生成代码时必须：
必须做到：
+ 先设计模块结构，再写代码
+ 保证输入输出一致
+ 明确每个 tensor shape
+ 写清训练/推理路径
**禁止**：
+ 直接“生成完整项目但无法运行”
+ 忽略数据流
+ 忽略prompt设计
+ 忽略evaluation
## 11 项目核心思想（For Agent Context）
该项目本质是：
Vision-Language Model + Prompt Engineering → Video Anomaly Detection + Explainability