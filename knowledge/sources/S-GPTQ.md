# S-GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

## 本地完整原文

[完整原文阅读](../../references/S-GPTQ/reader.html) · [全文文本](../../references/S-GPTQ/fulltext.txt) · [原始文件](../../references/S-GPTQ/original.pdf)

完整 PDF：16 页，版本 2210.17323v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2023 / v2。

## 来源支持的机制与定义

GPTQ 是训练后权重量化方法，使用校准输入所反映的层输入结构，近似考虑量化误差对层输出的影响。它关注的目标不只是孤立权重元素的舍入误差。

## InfraAgents 如何据此推导

本库用 Δy＝ΔWx 解释输入分布为何影响量化选择，并额外推导 scales、分组大小、解包和执行格式的容量与成本；这些工程账目不等同于论文的实验成绩。

## 阅读范围与出处身份

训练后权重量化、校准与精度代价。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
