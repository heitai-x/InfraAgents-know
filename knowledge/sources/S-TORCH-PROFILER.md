# S-TORCH-PROFILER：PyTorch — torch.profiler

## 本地完整原文

[完整原文阅读](../../references/S-TORCH-PROFILER/reader.html) · [全文文本](../../references/S-TORCH-PROFILER/fulltext.txt) · [原始文件](../../references/S-TORCH-PROFILER/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：PyTorch 2.14 documentation。

## 来源支持的机制与定义

框架 profiler 可以采集算子、设备工作、shape、stack 和内存相关信息。更详细的采集会增加运行开销，部分选项也会延长张量引用。

## InfraAgents 如何据此推导

本库使用框架视图连接高层算子与具体 kernel，并将无扰动服务计时和短窗口诊断采集分开解释。

## 阅读范围与出处身份

算子采集、shape/stack 开销与引用保留。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
