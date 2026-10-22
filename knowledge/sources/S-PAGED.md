# S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention

## 本地完整原文

[完整原文阅读](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt) · [原始文件](../../references/S-PAGED/original.pdf)

完整 PDF：16 页，版本 2309.06180v1。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2023 / v1。

## 来源支持的机制与定义

PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。

## InfraAgents 如何据此推导

由层数、KV head 数、通道、dtype 和块长计算每块字节，再计算尾部碎片与共享收益。引用保护和设备完成共同决定块能否重用；分页本身不消除 attention 对可见历史的读取。

## 阅读范围与出处身份

分块 KV 管理和共享机制；未迁移论文性能数值。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
