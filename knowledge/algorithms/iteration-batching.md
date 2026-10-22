# K-ALG-001：连续批处理：从自回归依赖到迭代调度

连续批处理将资源分配单位从整条请求改成一次可执行的 token 推进；批次是请求状态的临时视图。

## 计算依赖决定调度单位

对请求 r，下一 token 的分布依赖已经确认的前缀。不同请求之间没有这种因果边，因此同一迭代可以合并多个请求各自已就绪的计算。一次普通 decode 使每个入选请求前进一个位置；prefill 则能在因果 mask 下同时求出一段已知输入的位置。两类工作共享层权重，但 query 数、历史长度和产生输出的位置不同。

静态批处理等最慢请求结束才更换整个批次。若同批输出长度是 2、3、8，单步成本暂按常数估计，8 轮中只有 13 个有效 token，而固定三槽执行了 24 个槽次。连续调度在请求完成后的边界填入新请求，回收的是这类空槽机会，并不消除每条请求自身的顺序依赖。

## 一轮包含什么

| 对象 | 语义 | 为什么不能混为一个长度 |
|---|---|---|
| 已确认长度 | 模型语义上可读取的前缀 | 投机写入尚未全部确认 |
| 已计算长度 | 当前 KV/状态已生成的位置 | 前缀缓存命中可直接贡献此值 |
| 本轮 token 数 | 分配给请求的计算量 | prefill 可能多个，decode 通常一个 |
| 分配容量 | 已保留的物理槽位 | 包含页尾空槽和 lookahead |

调度结果需要同时描述请求身份、读取的状态版本、写入区间以及结果归属。执行后只有已完成且语义有效的部分进入下一轮；取消、EOS、抢占和投机拒绝都可能改变下一轮集合。调度器维护长期状态，runner 将本轮视图压紧成张量，采样器再将行结果还原到请求。

## 从资源约束推导策略

一轮不能只限制请求数。设请求 r 被分配 qᵣ 个 query token、当前历史长 hᵣ，普通稠密 causal attention 的有效 QK 对数量随 qᵣhᵣ＋qᵣ(qᵣ＋1)/2 变化；线性层工作近似随总 qᵣ 增长。token 预算限制线性层规模，页预算限制驻留与增量分配，最长历史又影响 attention 时间。因此“同样 256 tokens”并非相同成本。

吞吐导向可以用较多就绪 token 摊薄权重读取；交互导向必须限制一轮耗时，使已有 decode 不被长 prefill 拖延。老请求优先可稳定 ITL，却可能让等待中的长请求饥饿；公平机制应按等待时长或累计服务量补偿，而非仅按当前 batch index 排序。

## 缓存和抢占如何进入调度

前缀命中减少待算输入，但命中页仍占物理内存并需要引用保护。资源不足时，暂停并保留页、换出状态、丢弃状态再算，是三种不同成本的抢占方式。重算代价取决于前缀长度和模型，换出代价取决于状态大小与链路，保留代价则是阻止其他请求进入。选择应基于下一次服务机会的估计，而不是把暂停等同于释放。

新请求进入、旧请求离开、批次压紧与页重用的交界是连续批处理真正的实现核心。单请求逐步输出只是一个退化情形；能够导出混合批每一行的逻辑位置、页表和输出请求，才说明调度模型已经落实。更深的主机/设备重叠由异步调度章节推导。

<!-- infraagents:provenance -->

## 来源与推导

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。
  [完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。
  [完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

- [S-VLLM-SCHED：vLLM — V1 Scheduler source reference](../sources/S-VLLM-SCHED.md)：vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。
  [完整原文](../../references/S-VLLM-SCHED/reader.html) · [全文文本](../../references/S-VLLM-SCHED/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [分块 Prefill：服务时间切片与历史可见性](../algorithms/chunked-prefill.md)。
- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
