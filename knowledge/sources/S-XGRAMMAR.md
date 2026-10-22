# S-XGRAMMAR：XGrammar

## 本地完整原文

[完整原文阅读](../../references/S-XGRAMMAR/reader.html) · [全文文本](../../references/S-XGRAMMAR/fulltext.txt) · [原始文件](../../references/S-XGRAMMAR/original.pdf)

完整 PDF：17 页，版本 2411.15100v3。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2024 paper。

## 来源支持的机制与定义

XGrammar 以语法状态决定合法 token，并利用可预计算与依赖上下文的部分分离，减少在线合法性判断开销。

## InfraAgents 如何据此推导

本库将 token 展开、解析栈、合法概率质量和采样同步起来，推导投机分支回滚、grammar 缓存身份与 CPU/GPU 重叠条件。

## 阅读范围与出处身份

语法状态、token 掩码和 CPU/GPU 重叠。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
