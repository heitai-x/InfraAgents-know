# S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models

## 本地完整原文

[完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

固定版本：OSDI 2022。来源版本日期：2022-07-13；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。

## InfraAgents 如何据此推导

把 batch 视为长期请求状态的一轮视图，推导 token 预算、压紧映射和输出归属。不同请求可并行，同一请求的自回归依赖仍保留。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
