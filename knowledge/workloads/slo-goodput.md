# K-WORK-005：Goodput：联合 SLO、拒绝和容量前沿

服务能力是单位时间内完成有效任务且满足约束的数量，不能用总生成速度替代。

## 一个明确的定义

设窗口 T 内请求 i 的成功、质量、TTFT 和后续延迟条件共同为 Cᵢ，则请求 goodput＝Σ1(Cᵢ)/T。token goodput 可以按满足条件的请求输出量加权，但这是另一个目标；长输出的权重更高。定义必须与业务完成单位一致。

“TTFT 合格率 95%，ITL 合格率 95%”不保证同一批 95% 请求同时合格。联合事件至少需要看二者交集，在最坏相关关系下可低到 90%。因此应直接统计联合 SLO，而非拼接边缘百分位。

## Admission control 的角色

到达超过能力时，系统可排队、拒绝或降级。拒绝减少已接收请求的延迟，但 offered workload 的完成比例下降。报告需同时保存 offered/admitted/completed/failed，才能解释 goodput 的分母与成功标准。静默丢弃或缩短输出会改变任务，不能当作纯容量提升。

## 容量是一条曲线

随着到达率增加，batch 复用提高可能先改善 throughput，随后 KV、计算或通信饱和导致尾延迟陡增。满足目标的最大稳定到达率构成一个运行点；改变 prompt/output 分布、SLO 或质量要求，容量前沿也改变。

prefill/decode 分离、分块 prefill、量化和投机都可能移动这条前沿，但方向由成本结构决定。量化扩大驻留请求数可能提高吞吐，也可能增加排队使单请求更慢；投机减少某些请求步数，却占更多验证预算。

## 选择方案

满足模式寻找在给定预算内达到联合约束的方案；优化模式在可行区域比较目标，不应把不合格但快的版本替换掉合格版本。可行性与最优性分开保存，使 Agent 在探索中仍保有可交付系统。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MLPERF-LLM：MLPerf Inference — Llama 2 与 LoadGen（固定历史版本）](../sources/S-MLPERF-LLM.md)：MLPerf 的 Llama 2 服务基准将特定任务、到达模型、质量和延迟条件组织成评测协议，服务能力依赖这些共同条件。
  [完整原文](../../references/S-MLPERF-LLM/reader.html) · [全文文本](../../references/S-MLPERF-LLM/fulltext.txt)

- [S-DISTSERVE：DistServe](../sources/S-DISTSERVE.md)：DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。
  [完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)

- [S-GENAI-PERF：NVIDIA GenAI-Perf](../sources/S-GENAI-PERF.md)：GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。
  [完整原文](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [服务指标：事件边界、Token 间隔与用户时间](../workloads/metric-events.md)。
- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
- [容量设计：驻留、服务率与阶段配比](../workloads/capacity-planning.md)。
