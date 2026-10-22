# S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

## 本地完整原文

[完整原文阅读](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt) · [原始文件](../../references/S-SARATHI/original.pdf)

完整 PDF：16 页，版本 2308.16369v1。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2023 / v1。

## 来源支持的机制与定义

SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。

## InfraAgents 如何据此推导

本库从 q 个新 query、h 个历史 token 推导 attention 对数 qh＋q(q＋1)/2；固定 token 块不等于固定时间片，块长需结合目标 ITL、launch 和权重复用选择。

## 阅读范围与出处身份

prefill/decode 差异与分块混合调度。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
