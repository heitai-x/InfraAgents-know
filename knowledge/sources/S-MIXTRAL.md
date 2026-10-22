# S-MIXTRAL：Mixtral of Experts

## 本地完整原文

[完整原文阅读](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt) · [原始文件](../../references/S-MIXTRAL/original.pdf)

完整 PDF：13 页，版本 2401.04088v1。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2024 / v1。

## 来源支持的机制与定义

Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。

## InfraAgents 如何据此推导

本库把一次专家层拆为路由、dispatch、分组 FFN 和 combine，分别计算权重驻留、有效算术、专家负载偏斜与远端流量。

## 阅读范围与出处身份

稀疏专家路由，激活参数与总参数的区别。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
