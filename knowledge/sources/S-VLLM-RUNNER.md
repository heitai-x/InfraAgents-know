# S-VLLM-RUNNER：vLLM — GPUModelRunner source reference

## 本地完整原文

[完整原文阅读](../../references/S-VLLM-RUNNER/reader.html) · [全文文本](../../references/S-VLLM-RUNNER/fulltext.txt) · [原始文件](../../references/S-VLLM-RUNNER/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：vLLM 0.24.0。

## 来源支持的机制与定义

GPUModelRunner 将调度输出映射为持久 batch、压紧输入、attention metadata、模型执行与采样结果。模型状态规格最终映射到物理缓存 tensor。

## InfraAgents 如何据此推导

本库用请求到 batch、query 到状态槽、logits 到输出三段映射定位组合问题；模型结构变化需要贯穿 spec、分配与消费者。

## 阅读范围与出处身份

调度输出到模型执行和采样状态的入口。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
