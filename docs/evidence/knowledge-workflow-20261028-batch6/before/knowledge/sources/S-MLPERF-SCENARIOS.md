# S-MLPERF-SCENARIOS：MLCommons — Inference Mobile Scenarios

## 本地完整原文

[文档原文](../../references/S-MLPERF-SCENARIOS/upstream/loadgen/README.md)

固定版本：Git 3ecc2e0a0c126503debec8b8213b1be949f04636 (2026-04-22)。来源版本日期：2026-04-22；整理核对日期：2026-10-22。

## 到达过程为什么改变问题

归档的 LoadGen 场景注释区分闭环与独立到达。SingleStream 每次发一个样本，完成后再发下一个；MultiStream 一次查询包含多个样本，也在前一查询完成后继续，查询延迟由其中最晚样本完成决定。Server 按随机 Poisson 到达过程提供请求，用目标到达率和延迟条件评价；Offline 一次交付整个样本集合，关注完成吞吐。

```pseudocode
闭环：发送查询 → 等待其完成 → 生成下一查询
独立到达：按给定到达过程产生计划时间 → 按计划发出 → 分别记录完成
离线：给出整个输入集合 → 执行 → 按完整完成量与完成时间计吞吐
```

闭环会在系统变慢时自动减少后续压力。由此得到的平均吞吐或延迟，不能直接代表同一名义速率下的独立到达服务。多样本查询的最大完成时间也不等于样本延迟的平均值。

## 范围与可用推导

这些定义来自固定 LoadGen 实现的注释，不替代正式评测规则；原文也说明冲突时规范优先。这里不迁移某项基准的阈值。比较服务时需固定到达模型、观察人口、拒绝/失败和计时边界，再评价[到达与并发](../workloads/arrival-and-concurrency.md)及[Goodput](../workloads/slo-goodput.md)。
