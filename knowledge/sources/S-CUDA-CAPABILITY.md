# S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities

## 本地完整原文

[完整原文阅读](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt) · [原始文件](../../references/S-CUDA-CAPABILITY/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online snapshot 2026-09-20。

## 来源支持的机制与定义

CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。

## InfraAgents 如何据此推导

本库分别分析存储、乘法、累加和输出精度，以及 tile、对齐、打包格式与真实 dispatch。

## 阅读范围与出处身份

硬件功能按代际区分；不提供本项目测量。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
