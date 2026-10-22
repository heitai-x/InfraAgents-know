# K-ENGINE-001：vLLM V1：从 Token 进度到调度输出

vLLM V1 的调度可沿请求 token 进度理解，再追踪预算、缓存分配和执行结果怎样改变下一轮状态。

## 固定版本的阅读坐标

本页核对 v0.24.0 的 vllm/v1/core/sched/scheduler.py：Scheduler 在第 68 行，schedule 在第 388 行，_update_after_schedule 在第 1130 行，_try_schedule_encoder_inputs 在第 1280 行。文件 SHA 与符号见 [源码定位证据](../../docs/evidence/expert-source-locations-20260920.json)。这些位置绑定此标签，不能作为其他 checkout 的固定行号。

## 核心状态关系

schedule 用已计算 token 与请求所需 token 的差额表达待推进工作，并把投机 token 和异步输出占位纳入计算。这样 prefill、decode 和 chunked prefill 可以在共同进度模型下选择本轮分配，而不必完全依赖两个独立阶段的硬切换。

token_budget 限制本轮计算量，encoder 还有独立预算，KV 分配决定状态是否装得下。当前运行请求和等待请求的处理、抢占和恢复共同形成 SchedulerOutput；后者交给 runner 转成实际张量。

## 以具体问题穿过源码

若长 prefill 影响 decode，追踪待算数、长 prefill threshold、token 预算与本轮选中顺序，再对照时间线。若请求未进入执行，区分调度预算、encoder 输入、KV 资源和外部传输可用性。若缓存命中但仍重算，沿 KVCacheManager 返回的已计算长度和 logits 需求解释。

## 可以吸收的设计

将长期语义状态与短期调度视图分开，允许多个优化作用在相同 token 进度模型上。InfraAgents 自己构建目标引擎时，可以据此解释哪些字段必要，但不复制整套调度器或默认策略。我们需要的证据是任务条件下的资源关系，源码提供一种具体实现参照。

<!-- infraagents:provenance -->

## 来源与推导

- S-VLLM-SCHED：[完整原文](../../references/S-VLLM-SCHED/reader.html) · [全文文本](../../references/S-VLLM-SCHED/fulltext.txt)
- S-VLLM-CODE：[完整原文](../../references/S-VLLM-CODE/reader.html) · [全文文本](../../references/S-VLLM-CODE/fulltext.txt)
- S-ORCA：[完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

- [S-VLLM-SCHED：vLLM — V1 Scheduler source reference](../sources/S-VLLM-SCHED.md)：vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。

- [S-VLLM-CODE：vLLM V1 固定标签源码](../sources/S-VLLM-CODE.md)：实际读取 v0.24.0 的 Scheduler、KVCacheManager 和 GPUModelRunner 三文件，核对 token 进度、分配参数、执行/采样状态和延迟释放相关入口。

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [vLLM KV 管理：命中、分配和延迟释放](../engine-index/vllm-kv-manager.md)。
- [vLLM ModelRunner：调度视图到设备状态](../engine-index/vllm-model-runner.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [分块 Prefill：服务时间切片与历史可见性](../algorithms/chunked-prefill.md)。
