# S-OTEL：OpenTelemetry — Context propagation

## 本地完整原文

[完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)

固定版本：Git 1138b1fa831f11d16d86a1d65355c56c19cb0cb1 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。

## InfraAgents 如何据此推导

连续批中的请求与 batch 是多对多关系，本库增加请求、序列、逻辑位置和执行代次的说明；这些关系不能只靠一棵父子 span 树自动表达。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
