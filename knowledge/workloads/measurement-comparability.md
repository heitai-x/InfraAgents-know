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

<!-- infraagents:provenance -->

## 来源与推导

- S-GENAI-PERF：[完整原文](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt)
- S-TORCH-PROFILER：[完整原文](../../references/S-TORCH-PROFILER/reader.html) · [全文文本](../../references/S-TORCH-PROFILER/fulltext.txt)
- S-NCU：[完整原文](../../references/S-NCU/reader.html) · [全文文本](../../references/S-NCU/fulltext.txt)

- [S-GENAI-PERF：NVIDIA GenAI-Perf](../sources/S-GENAI-PERF.md)：GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。

- [S-TORCH-PROFILER：PyTorch — torch.profiler](../sources/S-TORCH-PROFILER.md)：框架 profiler 可以采集算子、设备工作、shape、stack 和内存相关信息。更详细的采集会增加运行开销，部分选项也会延长张量引用。

- [S-NCU：Nsight Compute — Profiling Guide](../sources/S-NCU.md)：Nsight Compute 提供 kernel 级硬件计数器和派生指标，某些采集需要 replay。计数器能力、采集模式及恢复条件影响所观察的执行。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
