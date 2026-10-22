# S-NCCL-GROUP：NCCL — Group Calls

## 本地完整原文

[完整原文阅读](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt) · [原始文件](../../references/S-NCCL-GROUP/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：NCCL 2.31.2 documentation。

## 来源支持的机制与定义

NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。

## InfraAgents 如何据此推导

本库将通信提交、设备完成和请求控制流连接起来，解释局部跳过计算为何可能破坏其他 ranks 等待的操作。

## 阅读范围与出处身份

组调用与跨设备提交顺序。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
