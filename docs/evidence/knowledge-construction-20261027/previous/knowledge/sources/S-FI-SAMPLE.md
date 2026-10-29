# S-FI-SAMPLE：FlashInfer：Sampling

## 本地完整原文

[完整原文](../../references/S-FI-SAMPLE/reader.html) · [全文文本](../../references/S-FI-SAMPLE/fulltext.txt)

固定版本：Git 070fabf082b509419f878c853e5b0114c2496594 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

FlashInfer sampling 接口提供从概率或 logits 的按行采样，以及 top-k、top-p 等相关操作；调用的输入表示和参数决定实际概率处理。

## InfraAgents 如何据此推导

本库先确定 logits 行到请求/位置的映射，再定义变换顺序和随机流；语法 mask 与截断后应使用相容的归一化分布。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
