# K-ENGINE-004：前缀复用决策：树、块链与状态保护

前缀复用的动机是相同依赖产生的中间状态可以避免重复计算。索引回答在哪里找到可能相同的前缀，恢复与所有权机制回答它是否能被当前请求安全使用；两者不能合并成一个“命中”布尔值。

## 等价、粒度与可恢复性

相同 token 前缀只有在模型、adapter、位置与其他影响计算的条件一致时才可能复用。树按公共 token 片段共享路径；固定块链用前驱身份和当前块标识前缀；直接重算不保留索引状态。三种方案都合法，取舍取决于重用长度与频率、查找/分叉成本、驻留和回收粒度。

[已核对的 Radix 实例](../sources/S-SGLANG-CODE.md) 在 token 之外使用命名空间区分不能共享的条目，匹配可能切分边以显式表示边界，活跃路径通过引用保护退出可淘汰集合。树边的逻辑 token 粒度与物理页粒度可以不同；不能用最长匹配长度直接代替所有状态族的共同恢复点。

## 一次复用与回收

```pseudocode
以模型相关身份和 token 前缀定位候选缓存
核对可恢复范围、表示与数据可用性，保护本次将消费的路径和状态
计算未命中部分，在合法完成点发布可共享的新状态
请求退出时撤销其持有；仍有消费者的节点继续受保护
需要容量时，按所选策略淘汰没有活动依赖的内容，并同步撤销索引身份
```

这个语义流程不要求树或哈希表。树可从未保护叶子逐步回收，共享祖先仍受保护；块池也需在复用存储前使旧缓存身份失效。远端目录命中、字节到达和执行器可消费是不同阶段，见[分布式 KV](../system-semantics/distributed-kv-store.md)。

## 收益与反例

复用收益来自省下的计算与读取，代价包括查找、身份检查、分叉/插入、保留容量与可能的传输。如果公共前缀短、重用很少或保存它会挤掉更有价值的状态，直接重算可能更合适；长且反复使用的前缀则可能抵偿索引和驻留成本。高命中率本身不说明省了多少时间。

在相同重用轨迹上比较树、块链和不缓存，记录实际跳过的工作与完整时间。构造 token 相同但 adapter 或命名空间不同的输入，应禁止错误共享；一个请求结束、另一个仍读取公共前缀时，公共状态应继续受保护。将传输延迟与目录旧记录分别注入，可以区分索引命中和恢复失败。

<!-- infraagents:provenance -->

## 来源与推导

- [S-SGLANG：SGLang: Efficient Execution of Structured Language Model Programs](../sources/S-SGLANG.md)：SGLang 的 RadixAttention 利用 token 前缀树组织复用状态，结构化执行场景可以复用重复前缀计算。
  [完整原文](../../references/S-SGLANG/reader.html) · [全文文本](../../references/S-SGLANG/fulltext.txt)

- [S-SGLANG-CODE：SGLang — scheduler and radix cache source locations](../sources/S-SGLANG-CODE.md)：固定历史提交中的 scheduler.py 与 radix_cache.py 展示 Scheduler、批次选择和执行，以及 RadixCache 的前缀匹配、插入与淘汰。精确符号和行号随本版本源码单独记录。
  [算法与机制说明](../sources/S-SGLANG-CODE.md)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
