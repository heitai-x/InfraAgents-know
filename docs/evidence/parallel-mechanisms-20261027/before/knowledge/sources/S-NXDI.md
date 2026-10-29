# S-NXDI：NxD Inference Features Configuration Guide

## 本地完整原文

[完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

固定版本：Git ceca8d914d1c4ca531bd27d8e37581b039dcad2b (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

NxD Inference 说明静态形状可执行程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可配置不同桶；持久状态与模型执行共同受配置约束。

## InfraAgents 如何据此推导

本库从真实 shape 联合分布选择有限程序族，保持有效长度、逻辑位置和设备 KV 连续，计入编译和驻留成本。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
