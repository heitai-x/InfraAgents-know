# S-FA-IMPL：FlashAttention 官方实现说明

## 本地完整原文

[完整原文阅读](../../references/S-FA-IMPL/reader.html) · [全文文本](../../references/S-FA-IMPL/fulltext.txt) · [原始文件](../../references/S-FA-IMPL/repository.tar.gz)

固定源码提交：edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d；归档范围：entire_repository。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online 2026-09-20。

## 来源支持的机制与定义

FlashAttention 的版本说明记录：2.1 将非等长 Q/K 的 causal 对齐由左上改成右下。示例 Q 长 2、K 长 5 时，右下对齐使两行分别可见前 4、5 个位置。

## InfraAgents 如何据此推导

本库以绝对位置推导期望 mask，再比较 backend 接口，不能只把同名 causal 布尔值直接替换。

## 阅读范围与出处身份

非等长 Q/K 的 causal 对齐及版本变化。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
