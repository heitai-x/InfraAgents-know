# S-SGLANG：SGLang: Efficient Execution of Structured Language Model Programs

## 本地完整原文

[完整原文阅读](../../references/S-SGLANG/reader.html) · [全文文本](../../references/S-SGLANG/fulltext.txt) · [原始文件](../../references/S-SGLANG/original.pdf)

完整 PDF：20 页，版本 2312.07104v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2024 / v2。

## 来源支持的机制与定义

SGLang 的 RadixAttention 利用 token 前缀树组织复用状态，结构化执行场景可以复用重复前缀计算。

## InfraAgents 如何据此推导

本库进一步区分树路径、物理块粒度、活跃引用和淘汰价值，解释边内分叉与状态恢复。

## 阅读范围与出处身份

RadixAttention 与结构化程序的运行时复用。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
