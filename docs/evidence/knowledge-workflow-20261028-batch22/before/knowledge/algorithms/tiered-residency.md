# K-ALG-017：多级驻留：有效副本、预取、淘汰与重算

HBM、DRAM 与 SSD 提供不同的容量和访问路径。分层管理的目标是让下一次合法计算按时获得正确版本的数据，同时控制同时存活的副本和在途缓冲；对象能放入所有层的总容量，并不保证服务延迟可接受。

## 按对象的更新方式定义副本

权重在一个模型版本内通常只读；已封闭的 KV 前缀可以作为不可变对象复用；正在追加的尾块、recurrent state 和投机状态则会变化。对象身份应包含模型/表示版本、逻辑范围和必要的恢复边界，位置只是该对象当前有哪些副本。

同一对象可以在多个层有有效只读副本，但不能在两处无协议地独立修改并继续声称同一版本。可变状态换出时，可以在安全边界冻结一个版本，或为后续更新使用新的对象；复制进行中仍允许原地写入，会使目的副本混合多个时刻的状态。

| 状态变化 | 必要条件 | 失败时保留的责任 |
|---|---|---|
| 未驻留 → 目标空间已预留 | 容量及位置映射已独占预留 | 释放尚未发布的预留 |
| 已预留 → 传输中 | 源版本稳定，缓冲有明确拥有者 | 记录部分完成，不暴露未完成对象 |
| 传输中 → 有效副本 | 所需范围完整，表示和身份匹配，完成可见 | 不能以“已提交”代替数据就绪 |
| 有效副本 → 已发布 | 消费者取得同一版本和可用位置 | 发布失败不撤销仍在用的源 |
| 源副本 → 可回收 | 下游满足保留/恢复要求，引用与在途访问结束 | 迟到结果仍按旧代次清理 |

这套关系与 [取消和资源完成](../system-semantics/cancellation-resource-completion.md) 相连。SSD 上的缓存若允许丢弃后重建，恢复协议可以不同于必须持久保存的状态，但必须明确是哪一种。

## 容量、流量与驻留收益

每层容量计入有效副本、预留目的空间、在途 staging 和暂时不能释放的旧版本。一次迁移可能短暂同时保留源与目的；将“准备淘汰”立即扣减容量会造成过度准入。压缩表示还需要记录恢复工作区与元数据。

比较保留、搬运与重算。设对象 B 字节，下次使用概率 p；保留的机会成本是占用容量而排挤其他工作，重新取得的成本包括排队、传输、转换和同步，重算成本由模型、前缀与当前 batch 决定。p·避免的关键路径时间/B 可以作为价值线索，但它不包含公平、deadline 和多个对象的依赖，不能作为统一最优淘汰规则。

LRU 以最近访问近似未来重用，适合某些时间局部性；高成本前缀或热门专家可以加入重建成本与大小；单遍扫描的数据则可能污染缓存。策略评估应保留真实重用距离与交错负载，而非只测连续重复的同一对象。

## 预取由消费时间决定

设传输和转换需要 Tₘ，距离消费还有 Tₐ，只有 Tₘ≤Tₐ 且资源可用时才可能完全隐藏；提前 h 个层或迭代并不保证满足条件，因为各段计算时间不同。预取还会与当前 kernel、CPU 计算及其他 I/O 竞争资源，必须计入共享带宽。

连续大权重可以采用分块流水，动态路由的专家则存在错误预取。错误预取消耗带宽与容量，但未改变模型函数；为了避免传输而跳过实际需要的数据则属于另一种近似。各层队列应有界，消费期限较近的请求不能被远期预取永久阻塞。

## 场景：显存不足时的三级存储设计

以下是推导设定。GPU 可用 16 GiB，主存 64 GiB；固定权重与布局合计 20 GiB，设备工作区 1 GiB，双缓冲合计 1 GiB。状态设为 32 层、8 个 KV head、128 维、2 字节，即每 token 全层 KV 为 128 KiB。离线任务有 128 个请求，每个 prompt 128 token、生成至多 32 token，按最多 160 个已计算位置保守计，KV 上界为 2.5 GiB。

若常驻 GPU 的权重为 10 GiB，另 10 GiB 保留主存、按层传输，则所列设备占用为 10+2.5+1+1=14.5 GiB，余量还要覆盖页尾和其他对象。主存另计权重、激活和锁页缓冲。假设 H2D 有效带宽 16 GiB/s，SSD 顺序读 2 GiB/s，并且每个 decode 迭代仍要传输这 10 GiB 权重，则传输下界为 625 ms/迭代；从冷 SSD 完整读取 20 GiB 的下界为 10 秒。

可以比较三条路线。第一条按小批完整推进模型，简单但每批重复搬权重。第二条按层让多个微批共享已载入权重，扩大有效批量，把每轮搬运摊到更多输出；这与 [FlexGen 固定版本](../sources/S-FLEXGEN.md) §4 的调度动机一致。若一轮推进 128 个请求，仅该传输项给出的输出吞吐上界为 128/0.625=204.8 token/s，但每个请求仍等待整轮，不能将吞吐倒数当作 ITL。第三条使用经过质量验证的更小表示使权重尽量常驻；若任务要求保持原数值关系且不允许这种近似，该路线就不合法。

对于 50 ms ITL 的在线目标，625 ms 下界已经排除了上述每轮搬运方案，增加 SSD 容量无助；应比较减少传输的计算放置、其他合法表示或更多设备。对于允许较长完成时间的离线任务，层间权重复用可能有价值，但还需验证实际计算和全任务排空时间。历史长度增加后 KV 可迅速成为新的主项，不能继续沿用短 prompt 的容量账。

## 验证与策略反转

先对固定模型输入比较全驻留与迁移后的状态/输出，再注入部分读取、取消、源回收延迟和模型版本切换。检查物理容量守恒、有效副本身份和单一提交边界。性能同时报告冷启动、稳态传输字节、预取命中、重算、在途峰值与联合 SLO。

更高重用、更长可隐藏计算或更快链路有利于预取；更短 deadline、较小 batch、变化的热点和共享带宽拥塞可能使它退化。分层存储的决策需要随 workload 重算，不能把容量扩展本身当作性能优化已经成立。

<!-- infraagents:provenance -->

## 来源与推导

- [S-FLEXGEN：High-throughput Generative Inference of Large Language Models with a Single GPU](../sources/S-FLEXGEN.md)：§3–4 将生成、状态容量、层/批遍历与 GPU/CPU/disk 放置一起建模；§4.2 讨论把部分主存 KV 对应计算交给 CPU。
  [完整原文](../../references/S-FLEXGEN/reader.html) · [全文文本](../../references/S-FLEXGEN/fulltext.txt)

- [S-SERVERLESSLLM：ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](../sources/S-SERVERLESSLLM.md)：§4 的加载格式、块池和多级流水；§5 的 token 重算迁移；§6 的局部性与启动时间模型。
  [完整原文](../../references/S-SERVERLESSLLM/reader.html) · [全文文本](../../references/S-SERVERLESSLLM/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
- [存储 I/O：请求粒度、队列和加载流水](../hardware/storage-io.md)。
- [KV 卸载与预取：读集合、恢复点与迁移预算](../algorithms/kv-offload-prefetch.md)。
- [权重与专家卸载：重用、路由与在途版本](../algorithms/weight-expert-offload.md)。
- [CPU–GPU 计算放置：传数据、传结果与共同提交](../algorithms/heterogeneous-execution.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [容量设计：驻留、服务率与阶段配比](../workloads/capacity-planning.md)。
