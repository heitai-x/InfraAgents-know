# K-WORK-007：多模型放置：共同容量、混合执行与租户公平

多模型服务同时选择哪些版本驻留、各放在哪些设备、请求到达后去哪里，以及有限容量由哪些请求占有。模型名的数量不能描述负载：完整权重不同的模型、共享基座的 adapters，以及同一模型的副本具有不同的共享条件。

## 先确定可共享的对象

完全独立的模型通常需要各自的权重与执行布局。LoRA 可共享相同基座的计算和只读参数，增量矩阵、有效 rank、缩放与接入位置仍按请求确定。tokenizer 或语义不同的模型不能仅因形状相同就合并状态；不同 adapter 也不能默认共享同文本产生的 KV。[多 LoRA 执行](../algorithms/lora-serving.md) 给出线性计算与缓存身份的具体关系。

对每个设备分别计算权重、活跃 adapter、请求状态、图/编译缓存、workspace、加载 staging 和预留增量。共享基座在物理容量中计一次，在各租户计费中需要明确分摊方法；逻辑份额相加不能反过来制造另一份物理基座。CPU 中保存的冷 adapter 也要占主存，并与权重加载、KV 换出和分词工作竞争。

## 驻留、并行与路由的候选

独占副本容易归因和限制干扰，但低频模型可能长期闲置。完整模型共同驻留便于设备时间复用，却减少每个模型的 KV 余量。用模型并行将多个模型分布到同一设备组，可以让某个模型的突发使用更多设备，同时引入通信、阶段不均衡和共同故障影响。[AlpaServe v1](../sources/S-ALPASERVE.md) 系统研究了这种统计复用；其主要评估是一次前向的非自回归模型，不能直接把固定执行时间和 batching 结论套到逐 token 生成。

另一条选择是只驻留热模型，冷模型按需加载。它用启动等待换容量；等待时间包括前一个版本排空、存储争用与预热。对 LoRA，可以保留基座并只交换增量，或为极热且长期固定的 adapter 保留合并版本。路由应把局部命中、排队、可用 KV 和加载成本共同比较，而不只追求最多模型名或最高平均利用率。

聚批和公平性是同一个容量问题的两个面。优先同 adapter 请求能减少活动增量和小算子数量，却可能让低频租户长期等不到组批机会。可用有上限的等待窗口、按预算轮转或独立保留额度限制饥饿；额度单位应对应瓶颈，如 token 预算、KV 字节、设备时间或在途请求，而不是盲目按请求数平均。

## 场景：共享基座的多租户 LoRA 服务

以下全部为可复算的设计设定。单 GPU 可用 24 GiB，基座权重 12 GiB，workspace/执行缓存 3 GiB，额外保留 1 GiB，因此 adapter 与 KV 共同预算为 8 GiB。共有 128 个 adapter，32 层、每层两个 4096×4096 线性层使用 rank 16、FP16 的 LoRA；各请求的 KV 为 128 KiB/token。服务目标同时约束首 token、ITL 与每租户拒绝率，不能仅最大化总 token/s。

每个目标矩阵的增量参数为 16×(4096+4096)×2 字节＝256 KiB。64 个目标矩阵使一个 adapter 为 16 MiB，128 个 adapter 共 2 GiB，适合放在预留充足的主存中。若一轮 16 个请求各用不同 adapter，活跃增量为 256 MiB＝0.25 GiB。每条 2048 token 历史需要 256 MiB KV，16 条共 4 GiB，两者合计 4.25 GiB；若每条还允许生成 512 token，需要另增 1 GiB。页尾、元数据和在途版本仍在这些主项之外。

比较三种候选：A 为当前 adapter 使用合并后的完整权重，按 adapter 串行切换；B 为共享基座加混合 adapter 执行；C 在 B 上按 adapter/rank 限制每轮组合，并给低频请求设置最长等待。A 在单一热 adapter、很少切换时减少低秩路径；本例两个完整 12 GiB 合并副本已经吃满设备容量，无法再容纳执行区和 KV。B 的共同容量更可行；C 可能减少增量驻留和 padding，但需用尾延迟检验等待成本。

不能把 8 GiB 预算全用来增加请求。若 30 条历史均为 2048 token、adapter 各异，则 KV 为 7.5 GiB、增量为 0.46875 GiB，仅剩 32 MiB。第 31 条同类请求至少增加 272 MiB，当前就不应准入；已有请求继续生成也会很快触及预算。准入必须保留近期增长，并允许排队、释放非活跃状态或采用已经验证的抢占路线。

再设有效主机到设备带宽为 16 GiB/s，单个 16 MiB adapter 的纯传输下界为 0.9766 ms，16 个同时未命中为 15.625 ms。若本轮其他工作用时设为 24 ms、ITL 上限为 40 ms，串行加载几乎用尽剩余时间；预取只有在目的空间已预留、身份正确、传输确实重叠且未拖慢计算时才有意义。存储命中和 CPU 解码额外成本不得漏掉。

混合 rank 还改变算术量。8 个 rank 16 与 8 个 rank 64 请求，实际低秩维度和为 640；全补齐到 64 则为 1024，低秩部分有 1.6 倍的维度工作量。这个倍率不适用于基座或整个 forward。分组内核是否优于 padding，要同时测分组、索引、launch 与短组利用率。

## 贯穿状态、质量和公平性的验证

每个 adapter 用独立的小批参考检查 Wx+sB(Ax)，覆盖零增量、异 rank、不同接入层和量化基座；共享执行的结果与对应独立执行比较。构造相同文本、不同 adapter 的前缀，验证缓存不串用。adapter 从 v1 更新到 v2 时，同时保留运行中的 v1 请求、v2 新请求和一条取消请求，检查引用与迟到完成。

负载轨迹包括均匀请求、单一热租户、同时突发、长短混合与冷 adapter 风暴。固定 offered workload 和质量约束，分别报告租户等待/拒绝/完成率、首 token 与 ITL、增量换入字节、各资源峰值及无效工作；总均值不能掩盖冷门租户饥饿。租户级状态配额应能阻止一个长请求消耗所有共享 KV。

当热 adapter 几乎不变时，A 可能重新占优；当全部模型同时饱和时，统计复用空间收缩；严格故障隔离、不同基座或不相容执行图可能要求拆分设备组。这里给出可验证的设计链，没有声称这些设定已经在某个引擎上达到服务目标。

<!-- infraagents:provenance -->

## 来源与推导

- [S-ALPASERVE：AlpaServe](../sources/S-ALPASERVE.md)：§3 比较复制与模型并行的统计复用、通信和阶段不均衡；§4 将模型分桶、设备组与模拟驱动放置联系，§6 检验流量变化与资源约束。
  [完整原文](../../references/S-ALPASERVE/reader.html) · [全文文本](../../references/S-ALPASERVE/fulltext.txt)

- [S-SLORA：S-LoRA](../sources/S-SLORA.md)：§4 共享基座与异 adapter 低秩分支、聚批/准入；§5 adapter/KV 共同分页、预取和非连续 gather 内核；§6 增量的张量并行；附录给聚类与早停策略条件。
  [完整原文](../../references/S-SLORA/reader.html) · [全文文本](../../references/S-SLORA/fulltext.txt)

- [S-LORA：LoRA: Low-Rank Adaptation of Large Language Models](../sources/S-LORA.md)：LoRA 用低秩增量 ΔW＝sBA 表达权重适配；固定 adapter 可在条件允许时合并，多 adapter 可以共享基座并分别计算增量分支。
  [完整原文](../../references/S-LORA/reader.html) · [全文文本](../../references/S-LORA/fulltext.txt)

- [S-VLLM-PREFIX：vLLM — Automatic Prefix Caching](../sources/S-VLLM-PREFIX.md)：所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。
  [完整原文](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [多 LoRA 服务：低秩计算、驻留与缓存身份](../algorithms/lora-serving.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [请求路由与迁移：命中收益、排队和追赶条件](../algorithms/routing-and-migration.md)。
- [弹性服务：启动延迟、可用容量与排空](../workloads/elastic-serving.md)。
- [租户隔离：资源份额、缓存命名空间与故障边界](../system-semantics/tenant-isolation.md)。
- [集群放置：资源向量、拓扑、执行组与故障域](../workloads/cluster-resource-placement.md)。
