# K-SYS-012：内存分配：地址、容量与在途所有权

分配器管理的不只是空闲字节，还包括地址稳定性、布局约束、共享引用和异步消费者的生命期。请求完成时，逻辑状态可以终止，但它占用的物理资源不一定立即可重用。

## 从对象生命期划分空间

权重通常随模型版本存在；KV 和 recurrent state 随请求或可复用前缀存在；工作区随执行阶段存在；图缓冲和通信注册区可能跨多个迭代存在。按这些生命期建池可以减少互相造成的碎片，但池之间的容量划分过硬也可能让一个池有空闲、另一个池无法推进。

设对象 i 的申请量为 rᵢ，按分配粒度舍入后的占用为 aᵢ。内部碎片为 Σ(aᵢ−rᵢ)。外部碎片体现为总空闲足够、却没有满足连续性或映射条件的区间。allocator 保留而未给活跃对象使用的内存又是另一项；它可以加快下一次分配，但仍占物理容量。虚拟地址预留则不能直接加到物理驻留量上。

完整峰值取 maxₜ[权重+活跃状态+可复用缓存+在途缓冲+工作区+保留池]，并避免把同一物理页在共享引用中重复计数。分别取每项最大值再相加是保守上界；只计算权重加稳态 KV 又可能漏掉最危险的过渡峰值。

## 连续缓冲、块表和虚拟映射

连续预分配适合已知上界、地址稳定且容量足够的对象，索引简单，但最大长度预留会浪费尚未使用的空间。应用层固定块池通过块表把逻辑序列映射到离散缓冲，便于共享与写时复制，要求消费者理解块布局。虚拟内存方案预留连续地址，逐步提供物理 backing，保留连续张量接口，但新增映射、页粒度与权限管理成本。

[vAttention v1](../sources/S-VATTENTION.md) §5–6 展示了第三种方案。该版本获得更小页粒度还使用了驱动扩展，不能据此宣称所有 CUDA 环境都直接具备相同能力。设备支持、映射粒度和同步条件必须由目标版本确认。虚拟地址连续也不自动提供跨请求前缀共享、写时复制、跨机迁移或 SSD 换入换出。

两种分页有不同尾部账。设定每请求全层 KV 为 128 KiB/token，应用块长 16 token，31 token 请求占两个 2 MiB 块，尾部浪费 128 KiB。另一设定是 32 层、每层 K 和 V 各 2 KiB/token，物理映射粒度 64 KiB：同一请求每层的 K/V 各需 62 KiB，也总计浪费 128 KiB。若映射粒度变为 2 MiB，每个层状态的尾部开销就大幅增加。这个算例用于显示粒度如何乘上层数与状态类型，不代表任意平台实际支持这些页大小。

## 分配、提交和回收

| 转换 | 必须成立的关系 |
|---|---|
| 空闲 → 预留 | 申请者独占容量预算，其他申请不能重复消费 |
| 预留 → 可读写 | 地址/块表、权限和必要初始化都已完成 |
| 可读写 → 在途 | 明确哪些 CPU、设备或网络消费者仍持有访问权 |
| 在途 → 待回收 | 不再接收新访问，但已有访问仍受保护 |
| 待回收 → 空闲 | 共享引用为零，所有相关异步访问已完成 |

多步分配必须允许部分失败后释放已取得资源。若先更新请求页表再发现其他层无法分配，该请求不能带着不完整状态进入执行。可先预留全部必要增量，再发布新的可执行视图；失败时撤销本次预留，保留旧视图的一致性。

同一个地址被下一代请求重用时，迟到完成事件必须仍能区分旧对象。引用计数负责“谁还可能使用”，完成事件负责“已提交的使用是否结束”；两者不可互相替代。跨 stream 的释放依赖见 [streams 与 allocator](../frameworks/streams-allocator.md)，请求取消的时序见 [资源完成](cancellation-resource-completion.md)。

## 工作区复用与容量调度

顺序执行的两个算子可以共享同一工作区；让它们并发会使生命期重叠，原先的内存规划便失效。融合、图重放、双缓冲、投机树和 P/D 传输都会改变同时存活的对象集合。地址稳定的图缓冲还可能限制压缩或搬移，必须把这种约束回传给容量规划。

准入要为本轮新增 KV、共享尾块的写时复制、采样/归约工作区和不可回收在途页预留空间。缓存驱逐能提供容量，但若被引用或仍在 DMA 中，就不能把“已选择驱逐”计作“现在可用”。达到容量边界时比较等待、分块、驱逐、重算或降低并发，不能依赖偶然的分配成功维持正确性。

## 验证边界与方案反转

用可追踪的设定负载制造同长请求、交错长短请求、共享前缀分叉、异步取消和部分分配失败。检查物理使用守恒、共享页唯一计数、写时复制独立性与回收后迟到事件；同时比较分配耗时分布、碎片、映射开销和可接受并发。

短请求且容量宽裕时，简单连续池可能更好；长短混合与高复用使块池更有吸引力；已有高性能连续 attention kernel 而块表改造昂贵时，可比较虚拟映射。选择依赖页粒度、内核效率、共享需求和调度成本，不存在仅由“分页”两个字决定的最优实现。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VATTENTION：vAttention](../sources/S-VATTENTION.md)：§5 将虚拟地址预留与物理页映射分开；§6 讨论映射成本、页粒度和驱动扩展。
  [完整原文](../../references/S-VATTENTION/reader.html) · [全文文本](../../references/S-VATTENTION/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。
  [完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

正文中的设定算例、资源账和机制组合是基于上述资料的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；本次编辑日期不改变来源时间。

## 延伸阅读

- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
