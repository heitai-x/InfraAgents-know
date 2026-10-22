# S-MLPERF-LLM：MLPerf Inference — Llama 2 与 LoadGen（固定历史版本）

## 本地完整原文

[完整原文](../../references/S-MLPERF-LLM/reader.html) · [全文文本](../../references/S-MLPERF-LLM/fulltext.txt)

固定版本：Git 3ecc2e0a0c126503debec8b8213b1be949f04636 (2026-04-22)。来源版本日期：2026-04-22；整理核对日期：2026-10-22。

## 来源支持的机制与定义

MLPerf 的 Llama 2 服务基准将特定任务、到达模型、质量和延迟条件组织成评测协议，服务能力依赖这些共同条件。

## InfraAgents 如何据此推导

本库借鉴联合质量/延迟约束与有效吞吐的定义方式，不迁移其阈值或成绩；按自身 workload 计算 offered、admitted 和合格完成量。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
