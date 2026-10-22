# K-WORK-004：性能实验：稳定态、对照与不确定性

可比较的性能结果来自相同任务函数与负载条件，以及能够解释波动的测量窗口。

## 冷路径与稳定路径

加载权重、编译、autotune、图捕获、首次内存触碰和缓存填充都可产生冷成本。稳定态测试可以在其完成后测量，但冷启动目标要单独包含。warmup 应覆盖实际形状与执行路径；只预热一个小 batch 不能消除其他桶的首次成本。

前缀缓存冷热还属于 workload 状态。两版本一个热缓存一个冷缓存，得到的差异不能直接归因于内核。缓存控制应与真实部署目标一致，不必把所有实验都人为冷启动，但必须说明所比较的状态。

## 构造可解释的对照

固定模型/权重、质量要求、输入序列与到达计划、输出设置、硬件分配和软件版本。单因素变化有助于识别机制；多因素组合则评估最终方案。随机化或交替实验顺序可减少温度、频率和邻居负载随时间变化的偏差。

重复测量给出分布，不应只挑最好一次。平均吞吐、尾延迟与失败率有不同估计不确定性，p99 需要足够请求覆盖尾部。若 100 个请求就报告精确到小数点后多位的 p99，其统计稳定性很有限。

## 采集本身改变系统

详细 profiler、堆栈记录、同步计时和计数器 replay 会改变执行。可先用低开销服务计时确认最终表现，再用短窗口深度采集解释原因。两者共享模型与负载条件，却不应把深度采集的耗时当成无扰动成绩。

优化报告最有解释力的部分是：哪段关键路径缩短、哪些资源变化、在哪类请求收益或退化。总数只有与这些机制连起来，才能成为下一次设计依据。

## 把比较写成一个可否定的问题

在运行前说明改变的机制、预期影响的资源、对应的用户指标与最小有意义的效果。例如“减少权重读取应降低小 batch decode 时间，但不一定改善大 prefill”；然后选择能够使这两种解释分开的形状与观测。仅比较两个最终数字，不能判断实际收益是否来自预期机制。

成对比较使用同一组固定输入/到达轨迹，但每次运行的状态应按已声明条件重置或复用。若共享缓存无法隔离，交替顺序仍可能让 B 继承 A 的预热；需要独立实例或显式说明共享状态。协议事件、质量和实际输出长度也必须核对。

## 不确定性属于正确的重复层级

独立构建、实例启动、时间区组和实例内请求是不同层级。继续增加同一实例的 token 采样，不能消除实例放置或环境变化引起的不确定性。配对与区组设计、比值区间和层级重采样的条件见 [实验统计](../tools/statistical-experiments.md)；这里需要把实际执行方式与统计单位对应起来。

尾部指标单独判断样本充分性。请求之间共享排队和热状态，不能默认独立；删去超时或测量结束时未完成请求，会使尾部看起来更好。报告原始计数、窗口、重复次数和不确定范围，比增加小数位更有价值。

## 搜索预算与确认实验

探索期间记录候选、失败和淘汰理由。选定实现后冻结代码、配置与验收规则，用没有参与选择的轨迹和独立运行确认。若同时尝试多种线程数、量化或桶配置，最好一次包含选择噪声，不能直接作为最终效应估计。停止规则应围绕决策阈值和有效重复，而不是不断尝试直到出现显著优势。

深度 profiler 只用来解释受控窗口内的机制；正式成绩来自相应的低扰动测量。必要时做采集开/关对照，判断 instrumentation 是否改变批次、时序或缓存。这样可以把“观测到某事件”与“它在无采集服务中是主要瓶颈”区分开。

<!-- infraagents:provenance -->

## 来源与推导

- [S-GENAI-PERF：NVIDIA GenAI-Perf](../sources/S-GENAI-PERF.md)：GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。
  [完整原文](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt)

- [S-TORCH-PROFILER：PyTorch — torch.profiler](../sources/S-TORCH-PROFILER.md)：框架 profiler 可以采集算子、设备工作、shape、stack 和内存相关信息。更详细的采集会增加运行开销，部分选项也会延长张量引用。
  [完整原文](../../references/S-TORCH-PROFILER/reader.html) · [全文文本](../../references/S-TORCH-PROFILER/fulltext.txt)

- [S-NCU：Nsight Compute — Profiling Guide](../sources/S-NCU.md)：Nsight Compute 提供 kernel 级硬件计数器和派生指标，某些采集需要 replay。计数器能力、采集模式及恢复条件影响所观察的执行。
  [完整原文](../../references/S-NCU/reader.html) · [全文文本](../../references/S-NCU/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-STATS：Quantifying Performance Changes with Effect Size Confidence Intervals](../sources/S-STATS.md)：§6–7 连接多层随机影响、实验重复、效应量和置信区间，并明确方法假设。
  [完整原文](../../references/S-STATS/reader.html) · [全文文本](../../references/S-STATS/fulltext.txt)

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致。
  [完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

## 延伸阅读

- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
