# K-ALG-023：请求路由与迁移：命中收益、排队和追赶条件

路由选择请求去哪里执行，迁移允许在实际状态增长后重新选择。两者应最小化与任务有关的完整代价，并保持输入、状态和输出提交的一致性。

## 命中最多的节点未必最早完成

候选节点的估计成本可以包括排队、未命中输入计算、远端读取/转换、执行阶段和接管开销。缓存身份先满足语义，再比较性能；过期目录与滞后的负载报告不能直接作为执行许可。

设定 A 有本地前缀，排队 100 ms、剩余计算 10 ms；B 无缓存但无排队，重算 80 ms；C 排队 10 ms，远端取得正确状态需 40 ms、剩余计算 10 ms。对应估计为 110、80、60 ms，C 才是这一设定下较好的选择。若 C 的目标容量或状态身份检查失败，它必须退出合法候选集合，不能仅凭最低分数继续接收。

最短请求队列、最少 token、最少内存占用与最短完成时间也不等价。长历史会增加 KV 和 attention，长输出决定未来驻留，prefill 与 decode 的资源需求不同。未知输出可以使用有范围的估计，但实际准入仍需检查当前资源和近期增量。

## 全局观察与本地决定

[Mooncake](../sources/S-MOONCAKE.md) §5–6 将缓存局部性、池负载与 SLO 联合调度；[Llumnix](../sources/S-LLUMNIX.md) §4 将全局实例配对与本地请求选择分开，并用虚拟使用量表达多种调度目标。这些是减少全局跟踪成本的候选组织，不保证一个内存标量就能预测所有延迟。

多个路由器同时看到同一个空闲节点，可能把下一批请求全部送过去。预留、局部再检查和适度分散可以减少这种竞争，但需要说明何时释放预留与如何拒绝过时决定。负载报告越稀疏，控制开销越低，决策误差可能越大；频繁迁移又会放大网络和状态成本。

## 活跃迁移的追赶条件

对可冻结旧前缀且仅追加新状态的请求，可以一边复制旧状态一边生成新 token，随后复制增量，最后短暂停止源端推进并移交尾部。对可变窗口、递归覆盖或投机回滚，必须另行定义每一阶段的快照边界。

设初始状态 D、持续新增速度 g、有效复制速度 B，忽略其他开销，首阶段耗时 D/B，产生增量 D·g/B；下一阶段的增量继续乘 g/B。只有 B>g 才有这种几何收敛机会。实际链路竞争、不同块开销与生成速度波动都可能打破估计，因此应有停止追赶、暂停复制或中止迁移的条件。

设定一个请求组有 1000 MiB 待迁移，新增 10 MiB/s，复制 100 MiB/s。前三阶段分别约 10、1、0.1 秒，剩余增量为 1 MiB，最终停顿复制约 10 ms，另加提交开销。总迁移时间约 11.11 秒，不能用 10 ms 停顿代表整个迁移免费。如果可用带宽降到 8 MiB/s，小于新增速度，继续原策略不会追上。

## 目标容量、输出与网络的共同预算

目标在每阶段接收前预留必要空间；源在迁移中完成、取消或被抢占时，目的端清理未提交预留。移交时明确最后确认 token、已计算状态与已经发布输出，防止新旧端重复发送或遗漏。数据传输和请求接管的责任见 [恢复协议](../system-semantics/serving-recovery.md)。

迁移可以释放某实例空间、减轻干扰、支持优先级或排空，但同时会占用目的容量和网络。搬走一个大请求可能比重新路由未来请求更昂贵；搬走多个小请求也可能增加控制开销。比较预计避免的等待/抢占成本与迁移全过程成本，并保留足够收益余量，避免围绕预测噪声来回迁移。

## 热点副本与优先级

复制热门前缀把后续请求分散到多个实例，可以同时改善命中与排队；也可能将有限缓存装满同一热点，挤出其他高价值对象。高优先级请求可通过迁走同驻请求获得资源，但被迁请求的等待和 SLO 仍需要记录。总体 goodput 上升不代表每个租户都改善。

验证使用热点逐渐变化、长短请求混合、负载报告延迟和带宽下降的轨迹；同时观察排队、重算、迁移字节、停顿、拒绝、内存碎片与各类 SLO。若节省的计算小于新增排队，或迁移长期追不上，路由应允许回到简单分散或本地重算方案。

<!-- infraagents:provenance -->

## 来源与推导

- [S-LLUMNIX：Llumnix](../sources/S-LLUMNIX.md)：§4 通过分阶段 append-only KV 复制和预留握手支持活跃迁移，采用全局实例级调度与本地请求选择、虚拟使用量。
  [完整原文](../../references/S-LLUMNIX/reader.html) · [全文文本](../../references/S-LLUMNIX/fulltext.txt)

- [S-MOONCAKE：Mooncake: Kimi’s KVCache-centric Architecture for LLM Serving](../sources/S-MOONCAKE.md)：§3–6 描述分离式 KV 池、块身份、层间传输、缓存感知调度、热点副本和过载提前拒绝。
  [完整原文](../../references/S-MOONCAKE/reader.html) · [全文文本](../../references/S-MOONCAKE/fulltext.txt)

- [S-SGLANG：SGLang: Efficient Execution of Structured Language Model Programs](../sources/S-SGLANG.md)：SGLang 的 RadixAttention 利用 token 前缀树组织复用状态，结构化执行场景可以复用重复前缀计算。
  [完整原文](../../references/S-SGLANG/reader.html) · [全文文本](../../references/S-SGLANG/fulltext.txt)

- [S-DISTSERVE：DistServe](../sources/S-DISTSERVE.md)：DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。
  [完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [分布式 KV 存储：目录、有效副本与可恢复状态](../system-semantics/distributed-kv-store.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
- [通信传输：注册、完成、可见与回收](../hardware/transport-and-rdma.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
