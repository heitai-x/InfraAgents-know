# K-ENGINE-001：调度决策：进度、就绪与资源承诺

动态服务的难点是请求的进度、输入就绪和资源可用性不在同一个时刻变化。调度输出需要表达本轮允许执行的逻辑范围及其资源承诺；按 token、请求、时间片或其他单位选择都是设计点。

## 问题、不变量与设计轴

对请求 i，区分已完成且可恢复的计算状态边界 cᵢ、已安排但结果尚未确认的边界 sᵢ、当前需要推进到的边界 wᵢ，以及已分配容量 aᵢ。在单一连续前缀的简化模型中 cᵢ≤sᵢ≤aᵢ；投机拒绝、失败或状态恢复可能使部分预安排进度失效。待安排工作不能简单用总长度减已确认长度，否则可能重复提交在途工作。

任何策略都要保持输入依赖就绪、逻辑范围不重复、状态容量足够、返回结果属于正确请求和代次。优先级、先来先服务、短任务偏好、deadline、抢占粒度和预留量属于选择；它们影响等待、公平性与吞吐，不能由一个默认队列顺序推出最优性。

## 按剩余工作组织一次调度

```pseudocode
输入：请求状态、在途工作、就绪信号和本轮计算/状态预算
按所选策略形成候选顺序
对候选请求：
    排除尚未就绪或已经在途的逻辑范围
    在计算预算、模型范围与状态资源内选择本轮推进量
    为选中范围建立预留和请求/代次映射；预留失败则不承诺执行
提交本轮执行视图，记录已安排范围
结果到达后依据成功、投机接受或失败修正已确认状态与预留
```

这是独立设计的语义表达，不声称既有引擎采用同一字段或原子事务。[ORCA](../sources/S-ORCA.md) 展示迭代级批处理；[已核对的 vLLM 实例](../sources/S-VLLM-SCHED.md) 用共同 token 进度覆盖 prefill、decode 与投机，并在调度后提前推进内部进度、在执行结果返回时校正。因而一个名为“已计算”的字段不能直接当作设备完成证据。另一种合法组织是等待结果再推进；代价是可能增加 host 等待或降低流水重叠。

## 成本、反转与区分实验

调度耗时包含扫描/排序、metadata 更新、分配检查和提交；选中工作的成本则与 query 数、可见历史、模型阶段和物理状态相关。相同 token 数不代表相同 Attention 读取量。长片有利于摊销准备，却可能延长其他请求等待；短片增加切换与准备频率。缓存友好排序减少重算，也可能延后低复用请求。

用同一到达轨迹改变排序或时间片，分别记录选中逻辑范围、预留、设备完成、可见输出和等待分布。预算耗尽、输入未就绪与 KV 不足应能由原件区分。让结果延迟、取消或拒绝投机，可检验提前推进是否被误当成已经提交的语义。吞吐改善若伴随 offered 请求减少或尾延迟变差，应分别报告，不归为同一服务目标下的无条件收益。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VLLM-SCHED：vLLM — V1 Scheduler source reference](../sources/S-VLLM-SCHED.md)：vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。
  [算法与机制说明](../sources/S-VLLM-SCHED.md)

- [S-VLLM-CODE：vLLM V1 固定标签源码](../sources/S-VLLM-CODE.md)：固定历史提交的 Scheduler、KVCacheManager、GPUModelRunner、coordinator 和 block pool 展示 token 进度、分配、执行与采样、层组和回收之间的关系。接口含义以所归档的提交为准。
  [算法与机制说明](../sources/S-VLLM-CODE.md)

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。
  [完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [vLLM KV 管理：命中、分配和延迟释放](../engine-index/vllm-kv-manager.md)。
- [vLLM ModelRunner：调度视图到设备状态](../engine-index/vllm-model-runner.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [分块 Prefill：服务时间切片与历史可见性](../algorithms/chunked-prefill.md)。
