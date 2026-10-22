# K-MODEL-007：窗口与稀疏 Attention：可见性、位置和状态保留

窗口限制的是每个 query 的直接可见历史；逻辑位置与跨层信息传播仍然持续演进。

## 窗口的精确定义

若窗口宽 W 包含当前 token，位置 t 可直接读取 max(0,t−W＋1)…t；有些接口把 window_left 定义为左侧 token 数，此时数值可能相差 1。局部层的直接 receptive field 有限，但叠层后信息可以通过中间表示传播到更远处。因此丢弃旧 KV 与宣称模型只利用最近 W 个原始 token 不是同一个结论。

全 attention 层与窗口层交替时，前者仍需全历史。sink/global tokens 又可能在窗口之外长期保留。缓存管理应记录每层的可见集合，不能因为模型出现一个 window 参数就对全部层滚动覆盖。

## 环形存储与绝对位置

物理槽 t mod W 可以重复使用，逻辑位置 t 仍继续增加。RoPE、相对位置偏置和 attention mask 应使用模型定义的位置，而非物理槽号。窗口跨过环形末端时，逻辑顺序分成两个物理区间，kernel 可以理解环形映射，或通过页表恢复顺序。

旧槽只有在所有读取它的在途 query 完成后才能覆盖。prefill 一次处理多 token 时，窗口随每行 query 移动，不能用最后一个 query 的窗口裁掉前面行还需读取的历史。

## 稀疏规则如何变成性能

块稀疏 attention 需要把合法 Q/K 块关系编码为可高效访问的索引。有效工作约随可见对数增长，索引构造、非连续读取和负载不均衡则是额外成本。密集 kernel 加一个稀疏 mask 可以验证语义，却未必减少计算。

窗口小到使 KV 不再主导时，权重读取、线性层和调度会成为下一瓶颈。保留更长历史供未来可能切换窗口只有在模型/业务确实需要时才有价值；否则可将释放的容量用于更多请求。前缀共享则需要同时保有恢复点所需窗口内容和其他层状态。

<!-- infraagents:provenance -->

## 来源与推导

- S-MISTRAL：[完整原文](../../references/S-MISTRAL/reader.html) · [全文文本](../../references/S-MISTRAL/fulltext.txt)
- S-HF-CACHE：[完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)
- S-VLLM-HYBRID：[完整原文](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt)

- [S-MISTRAL：Mistral 7B](../sources/S-MISTRAL.md)：Mistral 7B 提供滑动窗口 attention 的具体模型例子：局部可见历史与滚动状态可以降低相关层的直接历史成本。

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。

- [S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager](../sources/S-VLLM-HYBRID.md)：已读 0.18 设计文档注明其基于提交 458e74，区分 KV hidden bytes、token block size 和物理 page size，讨论层组统一池以及各组前缀命中的交集。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
