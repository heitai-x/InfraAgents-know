# S-XGRAMMAR：XGrammar

## 本地完整原文

[完整原文](../../references/S-XGRAMMAR/reader.html) · [全文文本](../../references/S-XGRAMMAR/fulltext.txt)

固定版本：2411.15100v3。来源版本日期：2025-05-12；整理核对日期：2026-10-22。

## 来源支持的机制与定义

XGrammar 以语法状态决定合法 token，并利用可预计算与依赖上下文的部分分离，减少在线合法性判断开销。

## InfraAgents 如何据此推导

本库将 token 展开、解析栈、合法概率质量和采样同步起来，推导投机分支回滚、grammar 缓存身份与 CPU/GPU 重叠条件。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
