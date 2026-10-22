# S-OTEL：OpenTelemetry — Context propagation

## 本地完整原文

[完整原文阅读](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt) · [原始文件](../../references/S-OTEL/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online snapshot 2026-09-20。

## 来源支持的机制与定义

OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。

## InfraAgents 如何据此推导

连续批中的请求与 batch 是多对多关系，本库增加请求、序列、逻辑位置和执行代次的说明；这些关系不能只靠一棵父子 span 树自动表达。

## 阅读范围与出处身份

跨服务上下文关联；不等价于请求/批次多对多映射。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
