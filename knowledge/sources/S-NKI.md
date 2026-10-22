# S-NKI：Trainium2 Architecture Guide for NKI

## 本地完整原文

[完整原文阅读](../../references/S-NKI/reader.html) · [全文文本](../../references/S-NKI/fulltext.txt) · [原始文件](../../references/S-NKI/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：Neuron 2.32.0 documentation snapshot。

## 来源支持的机制与定义

已读 Trainium2 NKI 资料区分 HBM、SBUF、PSUM、DMA 与计算引擎；片上张量的 partition/free 轴和转置路径具有具体硬件意义。

## InfraAgents 如何据此推导

本库用活跃 tile 集合与在线归约推导 attention 数据流，分析双缓冲、矩阵供数和片上存储生存期。

## 阅读范围与出处身份

HBM/SBUF/PSUM、partition/free 轴、DMA 与 Tensor Engine。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
