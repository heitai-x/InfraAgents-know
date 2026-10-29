# S-MLPERF-LLM：MLPerf Inference — Llama 2 与 LoadGen（固定历史版本）

## 本地完整原文

[文档原文](../../references/S-MLPERF-LLM/upstream/language/llama2-70b/README.md)

固定版本：Git 3ecc2e0a0c126503debec8b8213b1be949f04636 (2026-04-22)。来源版本日期：2026-04-22；整理核对日期：2026-10-22。

## 联合约束与负载身份

本来源将 Llama 2 的任务说明与 LoadGen 场景、设置和交互图一起归档。模型/数据与质量要求、请求产生方式、完成通知和性能条件共同定义一次比较；把最终 tokens/s 从这些条件中抽离，会丢失其含义。

算法层面先固定输入样本与期望输出的判定方式，再按所选场景产生请求、收集结果和完成事件，最后分别判断质量与时间约束。完成量不能只计成功返回的一部分而忽略其他已提供请求；压测器因等待而降低 offered 压力时也应明确。

不同到达场景的具体定义见[S-MLPERF-SCENARIOS](S-MLPERF-SCENARIOS.md)。本库从中提炼联合质量/延迟和完整观察边界的方法，不照搬特定任务阈值或成绩，见[测量可比性](../workloads/measurement-comparability.md)与[Goodput](../workloads/slo-goodput.md)。实现头文件保留作审计，文档部分保留原文入口；这里没有本项目实测。
