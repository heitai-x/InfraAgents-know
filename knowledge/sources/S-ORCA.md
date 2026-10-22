# S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models

## 本地完整原文

[完整原文阅读](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt) · [原始文件](../../references/S-ORCA/original.pdf)

完整 PDF：19 页，版本 OSDI 2022。

资料形态：按已核对范围整理的本地非代码释义。原文版本：OSDI 2022。

## 来源支持的机制与定义

Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。

## InfraAgents 如何据此推导

把 batch 视为长期请求状态的一轮视图，推导 token 预算、压紧映射和输出归属。不同请求可并行，同一请求的自回归依赖仍保留。

## 阅读范围与出处身份

迭代级调度和选择性批处理。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
