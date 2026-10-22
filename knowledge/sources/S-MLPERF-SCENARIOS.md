# S-MLPERF-SCENARIOS：MLCommons — Inference Mobile Scenarios

## 本地完整原文

[完整原文阅读](../../references/S-MLPERF-SCENARIOS/reader.html) · [全文文本](../../references/S-MLPERF-SCENARIOS/fulltext.txt) · [原始文件](../../references/S-MLPERF-SCENARIOS/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：scenario definitions / snapshot 2026-09-20。

## 来源支持的机制与定义

不同推理场景使用不同请求驱动方式，例如前一响应完成后再发与独立到达。相同平均吞吐并不表示相同负载压力。

## InfraAgents 如何据此推导

本库由闭环负反馈解释 coordinated omission，并用稳定窗口的 L＝λW 连接平均并发、到达率和逗留时间。

## 阅读范围与出处身份

单流、响应后再发与独立到达的负载区别；不照搬具体阈值。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
