# S-TORCH-NUMERIC：PyTorch — Numerical accuracy

## 本地完整原文

[完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

固定版本：Git 647b6a2ff9a83a886464932dfad78781ceb61cb6 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。

## InfraAgents 如何据此推导

本库区分逐位一致、容差一致、argmax、概率分布和任务质量。用 logits margin 与首次分歧位置解释为何小误差可能改变后续文本。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
