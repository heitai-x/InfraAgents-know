# K-ENGINE-002：状态分配决策：命中、预留、发布与回收

缓存命中、存储已经分配、数据已经可用与请求已经完成是不同事实。分配器必须把这些事实接到执行与恢复边界，单个“长度”或“空闲显存”无法完整描述可推进性。

## 状态与容量的共同约束

设每页容纳 B 个 token；对不共享、全历史驻留的单个状态族，已分配范围 a 需要 ⌈a/B⌉ 页。若从 c 扩展到 c+q+r，其中 q 是本次工作、r 是额外预留，则新增页数为 max(0,⌈(c+q+r)/B⌉−已持有页数)。多个层组、窗口、共享页与压缩残余必须分别计账，不能直接把逻辑页数相加当作唯一物理占用。

例如 B=16、当前 c=31 且持有 2 页，本轮 q=2、r=0 时需再分配 1 页；若再预留 r=16，则需要再分配 2 页。预留增加未来推进余量，也挤占其他请求的容量。完整序列准入与按步分配是不同轴：前者保守地限制过度准入，后者灵活但可能在未来遇到不足。

## 命中与执行的交接

```pseudocode
匹配与当前模型、位置和表示一致的可恢复前缀
确认所需状态族都可恢复；外部尚未到达的数据保持不可消费状态
按本轮工作和预留量计算实际新增容量，保护命中页并建立新页归属
执行或传输完成后发布有效状态；失败时撤销本次未兑现的承诺
请求结束后撤销逻辑持有，等最后消费者结束再决定保留为缓存或复用存储
```

[已核对的块管理实例](../sources/S-VLLM-KV.md) 只复用完整缓存块，并为取得 logits 保留重新计算末尾的机会。若长度 32 的输入全部 KV 已缓存、页长 16，而命中范围受“最多 31 个 token 且完整块”限制，最多复用 16 个，重算可能是 16 个 token，不能固定计为 1。其他设计可以保存 logits 或允许不同恢复粒度，但会改变驻留与身份管理。

## 比较连续、分页和分层驻留

连续地址简化读取，但扩展与碎片可能增加搬运或预留；分页以索引换细粒度容量管理；虚拟映射可以保持地址连续而改变物理绑定；卸载还需目标预留、传输和可见性确认。这些表示都要解释引用、生存期和失败恢复，见[内存分配器](../system-semantics/memory-allocators.md)与[多级驻留](../algorithms/tiered-residency.md)。

区分容量问题与可用性问题，可保持逻辑输入相同，分别延迟外部传输、增加预留、造成跨页扩展或保留在途引用，记录每一状态族的已占、已预留、可消费与可回收量。调用释放函数不等于已经允许覆盖仍被设备或传输读取的存储。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VLLM-KV：vLLM — KVCacheManager source reference](../sources/S-VLLM-KV.md)：KVCacheManager 的相关接口区分已命中状态、新 token 分配、lookahead、外部状态和释放。层组与实际可恢复状态参与分配判断。
  [算法与机制说明](../sources/S-VLLM-KV.md)

- [S-VLLM-CODE：vLLM V1 固定标签源码](../sources/S-VLLM-CODE.md)：固定历史提交的 Scheduler、KVCacheManager、GPUModelRunner、coordinator 和 block pool 展示 token 进度、分配、执行与采样、层组和回收之间的关系。接口含义以所归档的提交为准。
  [算法与机制说明](../sources/S-VLLM-CODE.md)

- [S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager](../sources/S-VLLM-HYBRID.md)：固定历史设计文档区分 KV hidden bytes、token block size 和物理 page size，讨论异构层分组、统一分配以及各组可恢复前缀的交集。具体功能限制只适用于该文档版本。
  [完整原文](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [vLLM V1：从 Token 进度到调度输出](../engine-index/vllm-scheduler.md)。
