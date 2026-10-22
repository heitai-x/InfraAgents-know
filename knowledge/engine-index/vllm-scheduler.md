# K-ENGINE-001：vLLM V1：从 Token 进度到调度输出

vLLM V1 的调度可沿请求 token 进度理解，再追踪预算、缓存分配和执行结果怎样改变下一轮状态。

## 固定版本的阅读坐标

本页核对固定提交 9c61864bf8a911a8369f35d79d538c7f11cf3dc2，提交日期为 2026-04-30。以下坐标由完整归档源码解析取得，只定位这一版本。

| 文件 | 符号 | 行号 |
|---|---|---|
| vllm/v1/core/sched/scheduler.py | Scheduler | 67 |
| vllm/v1/core/sched/scheduler.py | schedule | 352 |
| vllm/v1/core/sched/scheduler.py | _update_after_schedule | 987 |
| vllm/v1/core/sched/scheduler.py | _try_schedule_encoder_inputs | 1116 |

[完整原文与本地阅读](../../references/S-VLLM-SCHED/reader.html)；[源码定位证据](../../docs/evidence/astra-source-locations.json)。

## 核心状态关系

schedule 用已计算 token 与请求所需 token 的差额表达待推进工作，并把投机 token 和异步输出占位纳入计算。这样 prefill、decode 和 chunked prefill 可以在共同进度模型下选择本轮分配，而不必完全依赖两个独立阶段的硬切换。

token_budget 限制本轮计算量，encoder 还有独立预算，KV 分配决定状态是否装得下。当前运行请求和等待请求的处理、抢占和恢复共同形成 SchedulerOutput；后者交给 runner 转成实际张量。

## 以具体问题穿过源码

若长 prefill 影响 decode，追踪待算数、长 prefill threshold、token 预算与本轮选中顺序，再对照时间线。若请求未进入执行，区分调度预算、encoder 输入、KV 资源和外部传输可用性。若缓存命中但仍重算，沿 KVCacheManager 返回的已计算长度和 logits 需求解释。

## 可以吸收的设计

将长期语义状态与短期调度视图分开，允许多个优化作用在相同 token 进度模型上。InfraAgents 自己构建目标引擎时，可以据此解释哪些字段必要，但不复制整套调度器或默认策略。我们需要的证据是任务条件下的资源关系，源码提供一种具体实现参照。

## 把实现坐标还原成独立调度契约

阅读固定版本时，先记录四项输出：哪些请求获得推进权、各自的逻辑范围、为本轮保留的资源、哪些结果仍未确认。然后沿执行返回核对这些承诺如何成为实际进度。这样的数据关系可以用于独立设计，而无需沿用既有类划分或队列名称。

比较自己的候选时，可以构造相同资源预算下的长 prefill、短 decode、取消和恢复请求，检查推进量是否守恒、输出是否正确归属，以及等待是否符合任务公平性。若改变为 deadline 或租户配额策略，应保留状态与容量契约，重新验证选择顺序和饥饿行为；源实现的默认次序没有提供目标 workload 的最优性证明。

进一步的 [运行中路由迁移](../algorithms/routing-and-migration.md) 与 [服务主机执行域](../system-semantics/host-runtime.md) 会把准入和完成分散到多个队列。源码中存在外部传输字段仅证明一种交接位置，不能代替全局路由、存储与失败恢复的独立设计。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VLLM-SCHED：vLLM — V1 Scheduler source reference](../sources/S-VLLM-SCHED.md)：vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。
  [完整原文](../../references/S-VLLM-SCHED/reader.html) · [全文文本](../../references/S-VLLM-SCHED/fulltext.txt)

- [S-VLLM-CODE：vLLM V1 固定标签源码](../sources/S-VLLM-CODE.md)：固定历史提交的 Scheduler、KVCacheManager、GPUModelRunner、coordinator 和 block pool 展示 token 进度、分配、执行与采样、层组和回收之间的关系。接口含义以所归档的提交为准。
  [完整原文](../../references/S-VLLM-CODE/reader.html) · [全文文本](../../references/S-VLLM-CODE/fulltext.txt)

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。
  [完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [vLLM KV 管理：命中、分配和延迟释放](../engine-index/vllm-kv-manager.md)。
- [vLLM ModelRunner：调度视图到设备状态](../engine-index/vllm-model-runner.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [分块 Prefill：服务时间切片与历史可见性](../algorithms/chunked-prefill.md)。
