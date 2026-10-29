# S-MIXTRAL：Mixtral of Experts

## 本地完整原文

[完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

固定版本：2401.04088v1。来源版本日期：2024-01-08；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。

## InfraAgents 如何据此推导

本库把一次专家层拆为路由、dispatch、分组 FFN 和 combine，分别计算权重驻留、有效算术、专家负载偏斜与远端流量。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
