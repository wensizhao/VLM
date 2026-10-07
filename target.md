### 仅利用CLIP和fine-tunning候选prompts  进行候选prompts好坏描述对比实验
1.无需反向传播 只经行fine-tunning prompts的内容来提高CLIP映射的点积正确度好坏
> 核心研究问题：不同 Prompt 构造策略能否提高 CLIP 对异常行为的语义匹配能力？
2.添加LoRA层微调，LoRA是放在CLIP内部的，可以是微量参数的矩阵，来对CLIP语义空间进行视频异常检测任务的特化。这里是有反向传播过程的。
> 核心研究问题：相比单纯 Prompt Engineering，在 CLIP 中引入 LoRA 微调，能否进一步提高异常行为识别准确率？
#### To do list:
+ 对一个视频片段进行帧分组，而CLIP只是图片+prompts处理器，然后一个图片需要计算多个prompts匹配，所以会得到一个similarity matrix,而这组帧里的帧需要进行时间帧聚合方法
> 问题点：prompts分类方法？ 帧聚合方法（平均，最大值，时间加权）？ 
> 目标：不断调整 prompt，让最终 similarity vector 更符合我的 ground truth 
> 理由: CLIP 的 zero-shot 分类性能会受到 wording / phrasing 影响，因此 prompt engineering 确实是 CLIP 的一个重要使用方式
> 策略：进行控制变量实验：最后也是得到一个 result matrix
+ 对于最后的result,我们希望可以得到一个具体的语义陈述,这是对简单的normal/abnormal判断的进阶，我们希望abnormal和normal的判断正确率以及 文本符合的判断正确率
> 问题点:给视频打上怎样的prompts分类标签才能够尽可能符合全面性和客观性，比如以动作？场景？状态词？或者对于这种抽象的：我们使用更加具像化的prompts


实验矩阵| 实验       | Prompt    | CLIP参数 | 多帧聚合 |
| -------- | --------- | ------ | ---- |
| Baseline | 基础Prompt  | 冻结     | Mean |
| Exp 1    | Prompt策略A | 冻结     | Mean |
| Exp 2    | Prompt策略B | 冻结     | Mean |
| Exp 3    | 最优Prompt  | 冻结     | Max  |
| Exp 4    | 最优Prompt  | LoRA   | Mean |
| Exp 5    | 最优Prompt  | LoRA   | Max  |



### 利用现有完全冻结VLM/部分冻结VLM
两路Loss:一路是abnormal score:(0 or 1) ，反向传播,调整linear层；一路是 description:来增加VLM给到的 prompt的正确率，也同样调整linear层。就是先统一冻结VLM，然后只调整VLM的输出高维矩阵线性化，得到最总得分，或者得到一个语义矩阵再转化成一个token输出
第二步再采取LoRA部分参数调整，以及VLM只冻结部分层，