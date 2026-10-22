# S-MOONCAKE：Mooncake: Kimi’s KVCache-centric Architecture for LLM Serving

## 本地完整原文

[完整原文](../../references/S-MOONCAKE/reader.html) · [全文文本](../../references/S-MOONCAKE/fulltext.txt)

固定版本：2407.00079v1。版本坐标日期：2024-06-24；整理日期：2026-10-22。

## 来源支持的机制与定义

§3–6 描述分离式 KV 池、块身份、层间传输、缓存感知调度、热点副本和过载提前拒绝。

## InfraAgents 如何据此推导

将缓存目录、远端读取、队列、目标容量与 P/D 接管放到同一设计链；显式比较命中、重算与路由成本。

## 原文定位与阅读范围

PDF pp.3–11，§3–6.3；§5 的成本比较与 §6 的提前拒绝滞后；剩余全文仍待复核。

## 适用边界

仅使用 2407.00079v1，不引入后续修订。评估使用重放 trace 与 dummy LLaMA2-70B 结构，部分 SLO 是相对倍率；不将论文指标当成真实内容质量或当前服务的绝对 SLO。启发式预测与阈值需目标验证。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
