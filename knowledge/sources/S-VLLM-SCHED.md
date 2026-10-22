# S-VLLM-SCHED：vLLM — V1 Scheduler source reference

## 本地完整原文

[完整原文阅读](../../references/S-VLLM-SCHED/reader.html) · [全文文本](../../references/S-VLLM-SCHED/fulltext.txt) · [原始文件](../../references/S-VLLM-SCHED/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：vLLM 0.24.0。

## 来源支持的机制与定义

vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。

## InfraAgents 如何据此推导

本库从这些责任推导长期请求状态与临时执行视图的分离；某次默认排序策略不被提升为所有 workload 的最优策略。

## 阅读范围与出处身份

版本化源码文档中的调度入口与请求状态。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
