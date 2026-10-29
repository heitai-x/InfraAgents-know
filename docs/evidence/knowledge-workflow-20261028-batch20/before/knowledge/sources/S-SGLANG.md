# S-SGLANG：SGLang: Efficient Execution of Structured Language Model Programs

## 本地完整原文

[完整原文](../../references/S-SGLANG/reader.html) · [全文文本](../../references/S-SGLANG/fulltext.txt)

固定版本：2312.07104v2。来源版本日期：2024-06-06；整理核对日期：2026-10-22。

## 来源支持的机制与定义

SGLang 的 RadixAttention 利用 token 前缀树组织复用状态，结构化执行场景可以复用重复前缀计算。

## InfraAgents 如何据此推导

本库进一步区分树路径、物理块粒度、活跃引用和淘汰价值，解释边内分叉与状态恢复。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
