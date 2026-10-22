# K-ALG-018：KV 卸载与预取：读集合、恢复点与迁移预算

KV 卸载需要同时回答存储在哪里、下一步 attention 会读哪些状态，以及何时能够恢复推进。释放显存、减少重复 prefill 和降低 decode 流量是不同目标，采用的机制也不同。

## 先区分三类使用

暂停请求的 KV 可以换出后在恢复时读回；跨请求前缀缓存可以保存在更低层，命中后恢复；活跃 decode 的历史 KV 则可能每一步都被读取。前两种主要支付切换或命中成本，第三种可能持续支付整段历史的传输。不能把低频前缀缓存的收益外推为频繁活跃卸载的收益。

可见集合由模型决定：全 attention、滑动窗口、MLA 和 hybrid state 具有不同状态表示与读取范围。把一部分 KV 放到 SSD 并不会让模型不再需要它；主动裁剪或近似选择历史属于另一个质量契约。混合模型恢复还要使其他状态与 KV 对齐到共同的逻辑前缀。

## 布局与目的缓冲

按层读取便于与层计算流水，按 token 块管理便于共享与局部替换，按 head/并行分片存放则可能匹配执行器。存储格式与设备布局不同会引入 scatter/gather、解压或转置；代价和临时副本需要计入。共同前缀的物理块不能按每个引用重复换出，也不能因取消一个请求就删除其他请求仍需要的副本。

已封闭前缀与正在追加的尾块分开管理。导出尾块时固定一个有效范围与代次；恢复后尚未写入的空槽不属于有效 KV。量化表示还必须迁移对应 scale/zero point 和分组布局。仅有一个文件长度或哈希不能替代模型版本、位置与布局的语义检查。

## 搬运和重算的比较

令恢复状态为 D 字节，可用带宽 B，恢复时延至少包含排队+D/B+转换+同步；重算需要从持有的 token/特征及必要恢复点重新执行，时间为 Tᵣ。只有它们恢复同一语义边界时才可比较。token 本身不足以恢复任意模型状态、随机分支或已经提交的外部协议，所需额外状态参见 [分布式状态迁移](../system-semantics/distributed-state-transfer.md)。

设定一个待恢复前缀占 512 MiB，主机到设备有效带宽 16 GiB/s，则纯传输下界为 31.25 ms。若同一目标形态的重算成本假设为 20 ms，恢复不能仅凭缓存命中被判为更快；若重算为 200 ms，传输路线才有潜在优势，仍需计入排队和转换。从 2 GiB/s 的 SSD 读取同样数据，仅读取下界就为 250 ms，结论会再次变化。这里的重算数值是比较设定，实际策略必须校准。

## 逐层预取与持续带宽

层 l 的 KV 被消费前，目的空间、数据传输和布局转换都要完成。可以在前面若干层计算时预取，但下一层提前量由计算区间决定，不能只按层号相差固定个数保证。双缓冲减少等待的前提是前一个消费者已结束且所需数据已就绪；更深缓冲会挤占当前活跃 KV 的容量。

对 L 层、Bᵣ 个请求、历史 N、Hkv 个 head、维度 d、s 字节的普通全 attention，完整历史主项为 D=2LBᵣNHkvds。如果这些状态每步都经过同一 H2D 链路，D/B链路 是每步下界。设定 L=32、Bᵣ=8、N=4096、Hkv=8、d=128、s=2，D=4 GiB；16 GiB/s 链路的下界为 250 ms。分页可以组织这笔流量，却不会自动消除它。可比较把部分 attention 计算留在 CPU，见 [异构执行](heterogeneous-execution.md)。

## 发布、取消与部分失败

恢复请求先保留目的块和对应模型状态版本，逐范围接收完成结果；只有所需集合齐备才发布新的可执行视图。部分命中可以复用正确前缀并计算后缀，但不能拼接不同输入或模型版本的状态。若其中一层读取失败，不能让其他层的新映射独自进入下一次 forward。

取消后停止新的预取，丢弃业务输出，但等待已发起的 I/O/DMA/设备读结束再释放缓冲。迟到完成按原请求、对象和代次结算，不能覆盖新分配的页。源副本的删除与请求终态分开判断，保证失败或取消后至少保留策略要求的恢复能力。

## 区分实验和选择反转

用冷缓存、DRAM 命中、SSD 命中、部分前缀和长历史活跃请求分别对照全驻留/卸载/重算。测真正避免的计算、实际读取字节、首次恢复延迟、稳态 ITL、预取浪费和容量峰值。正确性覆盖非整页前缀、共享分叉、位置偏移、量化元数据和页槽复用。

长时间暂停和昂贵前缀有利于保存状态；短前缀、低带宽、繁忙存储或频繁读取可能更适合重算或改变计算放置。只有完整状态契约允许的方案才能参与性能比较。

<!-- infraagents:provenance -->

## 来源与推导

- [S-FLEXGEN：High-throughput Generative Inference of Large Language Models with a Single GPU](../sources/S-FLEXGEN.md)：§3–4 将生成、状态容量、层/批遍历与 GPU/CPU/disk 放置一起建模；§4.2 讨论把部分主存 KV 对应计算交给 CPU。
  [完整原文](../../references/S-FLEXGEN/reader.html) · [全文文本](../../references/S-FLEXGEN/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [多级驻留：有效副本、预取、淘汰与重算](../algorithms/tiered-residency.md)。
- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [分布式状态迁移：身份、传输与提交点](../system-semantics/distributed-state-transfer.md)。
- [CPU–GPU 计算放置：传数据、传结果与共同提交](../algorithms/heterogeneous-execution.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
