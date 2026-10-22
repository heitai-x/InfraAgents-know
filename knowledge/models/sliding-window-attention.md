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

## 原生可见性与运行时近似分开验证

模型原生窗口、global/sink 位置或其他规定的稀疏集合决定合法 attention 函数。按同一集合做密集 mask 与稀疏内核应数值对应；给一个原本全注意力模型强行缩小窗口，或者按重要性删去当前仍可见 token，则改变函数，需要独立质量评估。[KV 压缩](../algorithms/kv-compression.md) 将量化、可恢复副本淘汰与不可恢复可见内容裁剪分别处理。

量化保留可见位置但改变数值；裁剪减少位置；原生窗口本来就不要求直接读取更旧位置；把状态卸到 DRAM/SSD 则可以保留内容但增加读取时间。四者可能组合，却不能用同一个“缓存减少比例”说明质量和性能。原生窗口层仍可通过前层传播更旧信息，其质量也不能简单等同于把原输入截成最近 W 个 token。

## 索引构造也进入关键路径

规则滑窗可以从 query 位置和窗口宽直接推导块区间；不规则稀疏需要构造、排序或压紧合法块表。设一个 query block 对应 B 个潜在 key blocks，最终只访问 k 个，则索引生成仍可能扫描 B 个候选，且各 query block 的 k 不同。若每轮 host 构造并传表，节省的设备计算可能被 host 与同步抵消。

以 W=256、key block=64 为设定，单行 query 的窗口可能因起点不对齐跨 5 个物理块，而不是固定 4 个。对 Q 个相邻 query 的合并可见范围又扩大到最多 W+Q−1 个逻辑 token，块内还要 mask 每行不合法位置。只用末行窗口建立整个 prefill tile，会错删前面 query 所需历史。

规则块表可缓存几何或增量更新，但输入追加、跨环形边界、global 位置改变和 batch 重排都会影响有效映射。图模式下保留表地址仍需正确更新内容和有效长度。按每个 query block 的实际工作分配任务，通常比仅按请求数更能反映慢块；具体效果取决于稀疏形态与实现。

## 质量与压力的联合边界

用密集合法 mask 参考覆盖窗口前后各一位置、跨页和环形覆盖、混合全局层、短于窗口的请求和多 token chunk。对近似裁剪，增加长距离检索、位置分布变化、干扰上下文与持续生成；只测一个短问答不能支持长上下文质量结论。

性能同时测索引时间、有效与填充成对数、KV/元数据字节、重排流量和总层时间。窗口变小后若权重 GEMM 已成为主导，再降低 KV 容量可能主要提升可接入数量；不应继续按 attention 节约比例线性预测端到端收益。高并发、规则窗口与固定块表更容易摊薄索引成本；低并发或高度不规则集合可能使较简单的密集/小窗口实现更合算。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MISTRAL：Mistral 7B](../sources/S-MISTRAL.md)：Mistral 7B 提供滑动窗口 attention 的具体模型例子：局部可见历史与滚动状态可以降低相关层的直接历史成本。
  [完整原文](../../references/S-MISTRAL/reader.html) · [全文文本](../../references/S-MISTRAL/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：固定文档说明逐层复用历史 K/V、计算并合并新增 K/V，以及增量循环的 attention mask 应覆盖历史与新增 token；并区分 Dynamic、Static、StaticSlidingWindow 等缓存存储形态。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager](../sources/S-VLLM-HYBRID.md)：固定历史设计文档区分 KV hidden bytes、token block size 和物理 page size，讨论异构层分组、统一分配以及各组可恢复前缀的交集。具体功能限制只适用于该文档版本。
  [完整原文](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-KIVI：KIVI](../sources/S-KIVI.md)：§3 分析 K/V 的不同分组轴，使用分组量化与高精度残余区处理流式追加；§4 给出模型和任务相关的评估条件。
  [完整原文](../../references/S-KIVI/reader.html) · [全文文本](../../references/S-KIVI/fulltext.txt)

- [S-H2O：H2O](../sources/S-H2O.md)：§2–4 定义有限 KV 保留集合，研究累计 attention 统计与近期 token，讨论带条件的动态子模分析。
  [完整原文](../../references/S-H2O/reader.html) · [全文文本](../../references/S-H2O/fulltext.txt)

## 延伸阅读

- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
