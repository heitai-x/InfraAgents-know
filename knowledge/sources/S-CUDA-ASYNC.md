# S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution

## 本地完整原文

[完整原文阅读](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt) · [原始文件](../../references/S-CUDA-ASYNC/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online snapshot 2026-09-20。

## 来源支持的机制与定义

CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。

## InfraAgents 如何据此推导

本库据此解释负的“设备开始减 API 结束”、双缓冲时序、传输源的生命周期以及可真正隐藏的 host 工作。

## 阅读范围与出处身份

stream/event、主机返回与设备完成、主机传输条件。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
