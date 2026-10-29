# S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager

## 本地完整原文

[完整原文](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt)

固定版本：Git 9c61864bf8a911a8369f35d79d538c7f11cf3dc2 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

固定历史设计文档区分 KV hidden bytes、token block size 和物理 page size，讨论异构层分组、统一分配以及各组可恢复前缀的交集。具体功能限制只适用于该文档版本。

## InfraAgents 如何据此推导

本库将公共恢复位置推广为各状态族可恢复集合的交集，并讨论 LCM、联合预留与淘汰。文档中的具体版本功能限制不被外推到 0.24。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
