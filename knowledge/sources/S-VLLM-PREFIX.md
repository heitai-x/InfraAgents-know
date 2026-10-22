# S-VLLM-PREFIX：vLLM — Automatic Prefix Caching

## 本地完整原文

[完整原文阅读](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt) · [原始文件](../../references/S-VLLM-PREFIX/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：vLLM 0.24.0。

## 来源支持的机制与定义

所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。

## InfraAgents 如何据此推导

本库以计算依赖闭包推导模型/adapter/位置/模态内容的身份，区分逻辑命中、完整物理块与真正可跳过的计算。

## 阅读范围与出处身份

链式块标识、额外身份信息与缓存管理。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
