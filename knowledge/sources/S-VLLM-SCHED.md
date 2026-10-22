# S-VLLM-SCHED：vLLM — V1 Scheduler source reference

## 本地完整原文

[完整原文](../../references/S-VLLM-SCHED/reader.html) · [全文文本](../../references/S-VLLM-SCHED/fulltext.txt)

固定版本：Git 9c61864bf8a911a8369f35d79d538c7f11cf3dc2 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。

## InfraAgents 如何据此推导

本库从这些责任推导长期请求状态与临时执行视图的分离；某次默认排序策略不被提升为所有 workload 的最优策略。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
