# S-TORCH-DYNAMIC：PyTorch — Dynamic Shapes

## 本地完整原文

[完整原文阅读](../../references/S-TORCH-DYNAMIC/reader.html) · [全文文本](../../references/S-TORCH-DYNAMIC/fulltext.txt) · [原始文件](../../references/S-TORCH-DYNAMIC/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：PyTorch 2.14 documentation。

## 来源支持的机制与定义

编译会基于 shape、stride、dtype、常量等条件特化，并用 guard 确定专门程序是否可继续使用；部分维度可以符号化，变化也可能触发重编译。

## InfraAgents 如何据此推导

本库从运行分布推导分桶、padding 和变体成本，分别解释编译程序与 CUDA Graph 的 key，避免把动态 shape 支持等同于任意图重放。

## 阅读范围与出处身份

形状特化、guard 与重编译。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
