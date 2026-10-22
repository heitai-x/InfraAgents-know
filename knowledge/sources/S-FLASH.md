# S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 本地完整原文

[完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

固定版本：2205.14135v2。来源版本日期：2022-06-23；整理核对日期：2026-10-22。

## 来源支持的机制与定义

FlashAttention 通过片上分块和在线 softmax，避免将完整 attention 分数与概率矩阵写入设备主存；目标函数仍是精确 attention 的数学形式。

## InfraAgents 如何据此推导

对一块保留最大值 m、分母 ℓ 与未归一化加权和 u。合并另一块时按新的公共最大值重缩放 ℓ 和 u，最后计算 u/ℓ。本库据此解释 split-K、片上空间、IO 与浮点重排。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
