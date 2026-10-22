# K-MODEL-004：SSM 与混合模型：递归状态、扫描与恢复

递归状态是前缀的压缩摘要；它支持低增长成本的 decode，却改变了前缀复用、回滚和批次管理方式。

## 从递推看存储复杂度

简化状态空间关系写为 hₜ＝Aₜhₜ₋₁＋Bₜxₜ，yₜ＝Cₜhₜ＋Dₜxₜ。selective SSM 中部分系数依赖输入。decode 只需前一个递归状态以及模型要求的局部卷积历史，状态大小通常不随完整序列长度线性增长。prefill 则可利用专门的并行 scan/分块算法，不必逐 token 用同一小步 kernel 串行执行。

压缩状态并不意味着可以随机读取任意旧位置。全 KV 保存许多历史表示，递归状态只保留到某个前缀为止的摘要。要从较早位置分叉，需要保存该位置的状态快照或重放其后的输入。

## 状态包的内容

一条请求的状态可能包含每层 SSM 张量、卷积环形缓冲、全 attention KV、窗口 KV、逻辑位置与版本。各层状态大小和更新规则不同。batch 行重排时，这些对象必须使用同一请求映射；只重排主递归张量而忘记卷积状态，会导致似乎随机的跨请求污染。

在原地更新 h 的实现中，第 t＋1 步覆盖第 t 步。异步执行必须保证上一使用者完成；投机更需要保留接受边界的状态。将“把缓存长度改小”作为回滚方法对这种覆盖式状态不起作用。

## 前缀缓存从连续页变成检查点

一个状态快照可以代表长前缀，但只有它对应的准确位置可以直接恢复。假设每 256 token 保存一次递归状态，命中长度 700 时可以从 512 快照重算后续 188 个，而不能凭最终 768 状态倒推 700。是否能逆转取决于递推可逆性及数值稳定性，常规服务不应假定成立。

混合层模型的公共恢复点由所有层需要的状态联合决定。可以让全 KV、窗口历史和递归快照以同一恢复位置形成状态包；这增加写检查点成本，却简化共享与淘汰语义。检查点间隔因此是在存储、重算和命中粒度之间选择。

## 模型组合的成本

将部分全 attention 层换成递归层，会减缓 KV 容量增长，但全 attention 层仍可能在长上下文主导带宽。prefill scan、decode state update 与 attention kernel 的最佳批量形状不同，需要按层组分析。把所有层套入同一种分页接口可能既浪费容量，也掩盖错误的恢复条件。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MAMBA：Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../sources/S-MAMBA.md)：Mamba 使用输入相关的选择性状态空间机制。递归生成保留有限大小的状态，长序列处理可以采用专门的扫描/分块执行方式。
  [完整原文](../../references/S-MAMBA/reader.html) · [全文文本](../../references/S-MAMBA/fulltext.txt)

- [S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager](../sources/S-VLLM-HYBRID.md)：固定历史设计文档区分 KV hidden bytes、token block size 和物理 page size，讨论异构层分组、统一分配以及各组可恢复前缀的交集。具体功能限制只适用于该文档版本。
  [完整原文](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [投机解码：分布保持、状态提交与收益模型](../algorithms/speculation-budget.md)。
- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
