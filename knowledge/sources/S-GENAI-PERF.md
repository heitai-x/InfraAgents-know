# S-GENAI-PERF：NVIDIA GenAI-Perf

## 本地完整原文

[完整原文阅读](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt) · [原始文件](../../references/S-GENAI-PERF/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online snapshot 2026-09-20。

## 来源支持的机制与定义

GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。

## InfraAgents 如何据此推导

本库明确 TTFT 起点、TPOT 分母、ITL 聚合顺序和窗口吞吐，连接内部执行与用户收到内容的时间。

## 阅读范围与出处身份

TTFT、逐响应 ITL、吞吐与长度统计口径。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
