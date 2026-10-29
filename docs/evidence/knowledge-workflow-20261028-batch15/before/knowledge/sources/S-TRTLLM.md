# S-TRTLLM：TensorRT-LLM Architecture Overview

## 本地完整原文

[完整原文](../../references/S-TRTLLM/reader.html) · [全文文本](../../references/S-TRTLLM/fulltext.txt)

固定版本：Git 0343a9d9f2a8f69f9fe783834b042ff7687a28a4 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

已读 TensorRT-LLM 架构页将入口、PyExecutor、scheduler、model engine 与 sampler 连接成后台迭代，并介绍 graph padding 和 overlap 思路。

## InfraAgents 如何据此推导

本库用两代 batch 关系解释 host/设备重叠，分析 padding、在途步与停止处理的成本，不迁移网页中的性能数字。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
