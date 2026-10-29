# S-TRANSFORMER：Attention Is All You Need

## 本地完整原文

[完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

固定版本：1706.03762v7。来源版本日期：2023-08-02；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。

## InfraAgents 如何据此推导

当 E 与投影权重固定时，K＝EWk、V＝EWv 不依赖 decoder 步数，因而可以缓存。每步 QK 与加权求和仍存在；具体 mask 和位置规则由模型定义。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
