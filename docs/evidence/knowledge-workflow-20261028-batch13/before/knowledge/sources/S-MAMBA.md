# S-MAMBA：Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## 本地完整原文

[完整原文](../../references/S-MAMBA/reader.html) · [全文文本](../../references/S-MAMBA/fulltext.txt)

固定版本：2312.00752v2。来源版本日期：2024-05-31；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Mamba 使用输入相关的选择性状态空间机制。递归生成保留有限大小的状态，长序列处理可以采用专门的扫描/分块执行方式。

## InfraAgents 如何据此推导

递归状态通常原地推进，不能只缩短 KV 长度来回滚。本库据此推导前缀检查点、投机快照、卷积状态与混合模型的共同恢复位置。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
