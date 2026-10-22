# S-FI：FlashInfer：Attention kernels

## 本地完整原文

[完整原文](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt)

固定版本：Git 070fabf082b509419f878c853e5b0114c2496594 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

截止日前的 FlashInfer 固定源码区分 query 分段、KV 页表及末页长度。相关 wrapper 的 plan 安排计算与工作区，run 执行 attention；plan 的捕获限制与图模式缓冲须按该版本接口核对。

## InfraAgents 如何据此推导

本库据此表达独立的 Q/KV 几何，推导跨请求源缓存、plan key、固定 metadata 地址和可变有效长度之间的关系。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
