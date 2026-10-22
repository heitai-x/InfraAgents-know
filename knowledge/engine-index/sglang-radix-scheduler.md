# K-ENGINE-004：SGLang：Radix 复用与调度状态的连接

SGLang 将共享前缀组织成 Radix 结构，调度器围绕命中状态和可执行批次推进请求。

## 已核对的源码锚点

本页核对固定提交 cf346bb15de90c447ed498ad5e99aae407e04ce2，提交日期为 2026-04-30。以下坐标由完整归档源码解析取得，只定位这一版本。

| 文件 | 符号 | 行号 |
|---|---|---|
| python/sglang/srt/managers/scheduler.py | Scheduler | 322 |
| python/sglang/srt/managers/scheduler.py | get_next_batch_to_run | 2388 |
| python/sglang/srt/managers/scheduler.py | run_batch | 2864 |
| python/sglang/srt/mem_cache/radix_cache.py | RadixCache | 321 |
| python/sglang/srt/mem_cache/radix_cache.py | match_prefix | 398 |
| python/sglang/srt/mem_cache/radix_cache.py | insert | 468 |
| python/sglang/srt/mem_cache/radix_cache.py | evict | 608 |

[完整原文与本地阅读](../../references/S-SGLANG-CODE/reader.html)；[源码定位证据](../../docs/evidence/astra-source-locations.json)。

## 一次请求如何穿过结构

规范 token 前缀进入匹配，得到可复用路径和关联的状态；未命中后缀进入计算。共享节点持有的状态在请求活跃期间需要引用/锁保护，完成后可转为可淘汰内容。新结果插入时可能与既有路径共享或在边内分叉。

Radix 的边表示 token 片段，物理状态的块和 allocator 对象有自己的粒度。匹配得到的 token 范围、当前请求已经算过的范围和可用于下一次执行的页/状态，需要在调度交接处对应。树命中本身不替代全部恢复条件。

## 以问题选择阅读路径

共享前缀不命中时，沿 token identity、路径匹配和实际可用状态追踪；命中后输出错误时，沿分叉、位置、请求归属和物理页保护追踪；缓存占满时，沿引用保护、evictable 集合和请求完成路径追踪。系统性能则将节省的 prefill 与路由/调度成本一起解释。

## 借鉴范围

RadixAttention 提供了前缀结构复用的具体方案，不要求所有目标系统都使用树。固定块链哈希、检查点索引或特征缓存可能更适合不同状态。InfraAgents 应将算法选择建立在模型依赖、共享分布和物理资源之上，源码用于理解一个方案如何承担这些责任。

## 前缀索引与存储系统分别承担责任

Radix 路径定位共享 token 依赖，物理层还要回答状态存在于何处、表示能否被当前执行器消费、哪些读者仍持有它。把索引扩展到远端或压缩存储时，目录命中、字节可读和完整恢复点必须分开判断；仅出现 RadixCache 类名不能证明这些后续能力已经具备。

对独立设计，可以在同一组前缀重用轨迹上比较树路径、固定块链和直接重算。树的匹配/分叉成本、块粒度浪费、对象元数据和远端等待都应入账。若只有短而低频的重复前缀，维护复杂索引可能无法抵偿其成本；长且反复使用的公共前缀则可能使保存与路由更有价值。

将跨页分叉、热前缀淘汰、目录旧记录、传输失败和会话继续放在联合验证中，才能把 [前缀身份](../algorithms/prefix-cache-identity.md) 接到 [分布式副本](../system-semantics/distributed-kv-store.md)。这里列出的是设计与验证关系，不声称这些变化都由本页定位的固定类直接实现。

<!-- infraagents:provenance -->

## 来源与推导

- [S-SGLANG：SGLang: Efficient Execution of Structured Language Model Programs](../sources/S-SGLANG.md)：SGLang 的 RadixAttention 利用 token 前缀树组织复用状态，结构化执行场景可以复用重复前缀计算。
  [完整原文](../../references/S-SGLANG/reader.html) · [全文文本](../../references/S-SGLANG/fulltext.txt)

- [S-SGLANG-CODE：SGLang — scheduler and radix cache source locations](../sources/S-SGLANG-CODE.md)：固定历史提交中的 scheduler.py 与 radix_cache.py 展示 Scheduler、批次选择和执行，以及 RadixCache 的前缀匹配、插入与淘汰。精确符号和行号随本版本源码单独记录。
  [完整原文](../../references/S-SGLANG-CODE/reader.html) · [全文文本](../../references/S-SGLANG-CODE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
