# S-MLPERF-SCENARIOS：MLCommons — Inference Mobile Scenarios

## 本地完整原文

[完整原文](../../references/S-MLPERF-SCENARIOS/reader.html) · [全文文本](../../references/S-MLPERF-SCENARIOS/fulltext.txt)

固定版本：Git 3ecc2e0a0c126503debec8b8213b1be949f04636 (2026-04-22)。来源版本日期：2026-04-22；整理核对日期：2026-10-22。

## 来源支持的机制与定义

不同推理场景使用不同请求驱动方式，例如前一响应完成后再发与独立到达。相同平均吞吐并不表示相同负载压力。

## InfraAgents 如何据此推导

本库由闭环负反馈解释 coordinated omission，并用稳定窗口的 L＝λW 连接平均并发、到达率和逗留时间。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
