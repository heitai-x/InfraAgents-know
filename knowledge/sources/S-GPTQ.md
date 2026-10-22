# S-GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

## 本地完整原文

[完整原文](../../references/S-GPTQ/reader.html) · [全文文本](../../references/S-GPTQ/fulltext.txt)

固定版本：2210.17323v2。来源版本日期：2023-03-22；整理核对日期：2026-10-22。

## 来源支持的机制与定义

GPTQ 是训练后权重量化方法，使用校准输入所反映的层输入结构，近似考虑量化误差对层输出的影响。它关注的目标不只是孤立权重元素的舍入误差。

## InfraAgents 如何据此推导

本库用 Δy＝ΔWx 解释输入分布为何影响量化选择，并额外推导 scales、分组大小、解包和执行格式的容量与成本；这些工程账目不等同于论文的实验成绩。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
