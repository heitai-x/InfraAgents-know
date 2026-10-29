# S-GQA：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

## 本地完整原文

[完整原文](../../references/S-GQA/reader.html) · [全文文本](../../references/S-GQA/fulltext.txt)

固定版本：2305.13245v3。来源版本日期：2023-12-23；整理核对日期：2026-10-22。

## 来源支持的机制与定义

GQA 让一组 query heads 共享 K/V heads；MQA 是只保留一组 K/V 的特殊形态。输出 query head 数与 KV head 数需要分别描述。

## InfraAgents 如何据此推导

由 2LNHkvdb 计算普通同构 KV 容量。减少 Hkv 不同于等比例减少 Hq 对应的 attention 算术；内核是否复用读取及 TP 是否复制 KV 需另行分析。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
