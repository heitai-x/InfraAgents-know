# S-NXDI：NxD Inference Features Configuration Guide

## 本地完整原文

[完整原文阅读](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt) · [原始文件](../../references/S-NXDI/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：Neuron 2.32.0 documentation snapshot。

## 来源支持的机制与定义

NxD Inference 说明静态形状可执行程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可配置不同桶；持久状态与模型执行共同受配置约束。

## InfraAgents 如何据此推导

本库从真实 shape 联合分布选择有限程序族，保持有效长度、逻辑位置和设备 KV 连续，计入编译和驻留成本。

## 阅读范围与出处身份

静态形状分桶、QKV 融合、KV 与推理功能。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
