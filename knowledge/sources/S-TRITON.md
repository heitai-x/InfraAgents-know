# S-TRITON：Triton：Fused Attention

## 本地完整原文

[完整原文阅读](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt) · [原始文件](../../references/S-TRITON/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：main snapshot 2026-09-20。

## 来源支持的机制与定义

Triton 的 fused attention 示例展示块级张量程序、在线 softmax 归约、tile 划分与相应配置选择。逻辑 tile 和实际线程/寄存器布局有映射关系。

## InfraAgents 如何据此推导

本库从复用、寄存器与片上存储活跃集合推导融合与 tile 权衡，并把 autotune 的冷成本与形状变体纳入服务模型。

## 阅读范围与出处身份

分块注意力、在线归约、tile 与 autotune 机制。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
