# S-MLPERF-LLM：MLCommons — Llama 2 70B Inference Benchmark

## 本地完整原文

[完整原文阅读](../../references/S-MLPERF-LLM/reader.html) · [全文文本](../../references/S-MLPERF-LLM/fulltext.txt) · [原始文件](../../references/S-MLPERF-LLM/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2024 benchmark definition。

## 来源支持的机制与定义

MLPerf 的 Llama 2 服务基准将特定任务、到达模型、质量和延迟条件组织成评测协议，服务能力依赖这些共同条件。

## InfraAgents 如何据此推导

本库借鉴联合质量/延迟约束与有效吞吐的定义方式，不迁移其阈值或成绩；按自身 workload 计算 offered、admitted 和合格完成量。

## 阅读范围与出处身份

Server 到达模型、token 吞吐与延迟约束。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
