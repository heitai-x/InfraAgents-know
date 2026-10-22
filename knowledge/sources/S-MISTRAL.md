# S-MISTRAL：Mistral 7B

## 本地完整原文

[完整原文阅读](../../references/S-MISTRAL/reader.html) · [全文文本](../../references/S-MISTRAL/fulltext.txt) · [原始文件](../../references/S-MISTRAL/original.pdf)

完整 PDF：9 页，版本 2310.06825v1。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2023 paper。

## 来源支持的机制与定义

Mistral 7B 提供滑动窗口 attention 的具体模型例子：局部可见历史与滚动状态可以降低相关层的直接历史成本。

## InfraAgents 如何据此推导

本库按每层可见集合推导窗口缓存、绝对位置、环形物理槽和多层传播；不能从一个 window 参数推断所有层都可丢弃全历史。

## 阅读范围与出处身份

滑动窗口与滚动缓存的模型背景。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
