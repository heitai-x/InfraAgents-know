# S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs

## 本地完整原文

[完整原文阅读](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt) · [原始文件](../../references/S-CUDA-TOPOLOGY/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online snapshot 2026-09-20。

## 来源支持的机制与定义

多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。

## InfraAgents 如何据此推导

本库用带延迟/带宽的拓扑图解释 TP、PP、EP、状态迁移和 NUMA 亲和，按消息大小与频率选择通信组。

## 阅读范围与出处身份

拓扑、上下文、peer access 与多设备通信。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
