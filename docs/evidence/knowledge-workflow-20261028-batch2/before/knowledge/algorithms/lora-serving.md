# K-ALG-014：多 LoRA 服务：低秩计算、驻留与缓存身份

LoRA 为共享基座增加低秩映射；多适配器服务将模型复用问题转换成按请求选择增量权重的问题。

## 从低秩式推导执行路径

对原线性层 W，adapter 给出 ΔW＝sBA，A 为 r×dᵢ，B 为 dₒ×r。输出为 Wx＋sB(Ax)。当 r 远小于输入输出维度，adapter 参数量为 r(dᵢ＋dₒ)，增量计算也远少于完整 dᵢdₒ 的矩阵乘法。

单一固定 adapter 可以在允许的数值条件下把 ΔW 合并到 W，推理不再有额外低秩算子。多 adapter 同时服务时，维护每个合并后的完整 W 会重复基座容量；共享 W 并按请求执行低秩分支通常更符合内存目标。基座量化时，先反量化合并再重量化会增加误差，不能假设与运行时高精度增量逐位等价。

## 混合 batch 的计算组织

共享基座 GEMM 可处理全部 token，adapter 分支按 adapter ID 分组或用分段 kernel 执行。若每个请求 adapter 不同且 batch 很小，会产生很多细小矩阵运算；按 adapter 聚批能提高局部效率，却延长低频 adapter 的等待。因此 adapter 驻留策略、分组方式和调度公平性相互影响。

rank 不同意味着 A/B 形状不同。统一最大 rank 的图桶便于稳定地址，但需要有效 rank 和 padding 语义；按 rank 分桶减少无效算术，却增加图与编译变体。adapter 权重加载是一次独立生命周期：在途请求引用旧版本时，不能原地覆盖同一地址为新版本。

## 为什么缓存 key 必须认识 adapter

只要 adapter 作用在影响缓存状态的路径上，同 token 前缀在不同 adapter 下就可能生成不同 KV。即使 LoRA 只修改部分层，上游隐藏表示的变化也可能传播到之后各层。可以证明不影响某类缓存的特殊接入例外需要按计算依赖分析，默认把有效 adapter 身份纳入状态 key 更清晰。

一个 adapter 从主机换入需要带宽与同步；高频 adapter 常驻、低频按需加载的价值由复用间隔和大小决定。容量预算还包括临时低秩激活、图中固定槽与版本并存，而不仅是文件大小。多 LoRA 的系统收益应在相同租户流量和质量要求下比较。

## adapter 与 KV 争用同一预算

[S-LoRA v1](../sources/S-SLORA.md) 将动态 adapter 与 KV 放入共同分页池，并为非连续增量权重设计异 rank 的 gather 计算。可迁移的关系是“分配、布局和 kernel 必须共同设计”，不是所有模型都具有相同 H 维或天然能使用同一页几何。GQA、不同投影维度、dtype 和量化元数据改变分配粒度；分别计算每类对象的有效字节、对齐浪费和页表成本。

分离固定池便于控制各自上限，但一个池空闲时另一池仍可能失败。共同池能在两类对象之间借用容量，同时使增加一个冷 adapter 可能挤掉更多 KV。调度选择应把“这组请求需要多少增量权重”和“各自状态增长多少”一起估计。已经在执行或传输的 adapter 页不能为了增加 batch 立即撤销。

设定一个 adapter 16 MiB，每 token KV 为 128 KiB；在相同字节预算下多保留一个 adapter 的直接容量机会成本为 128 token KV。它是否值得，取决于后续请求复用、换入代价以及被挤走状态的恢复成本。固定预留 adapter 槽可以稳定 graph 地址，但也需要算入空槽、最大 rank padding 和更新时新旧两份并存。

## 分组、预取与公平等待

共享基座处理混合 token 后，低秩分支按 adapter、rank 和 token 段寻找各自 A/B。分组需要保持输入行与结果行的逆映射；不同 adapter 恰好有相同 rank，不代表可以共用参数。prefill 的多 token 矩阵计算与 decode 的单 token 向量路径，应分别比较补齐、分段和 gather 的代价。

例如 8 个 rank 16 和 8 个 rank 64 请求，实际 rank 总和为 640，统一补齐到 64 需处理 1024 个 rank 位置。节省这部分算术可能被小组调度和索引成本抵消，不能直接推导端到端 1.6 倍收益。低频 adapter 为等待更大组付出的时间，应计入自身首 token 和拒绝率，不能用热租户平均值代替。

预取只针对可预测的下一批，并明确目标预留、传输完成与实际引用。等待队列改变后，预取可能无用；无用流量、被占容量和取消后的回收都是策略成本。CPU 中冷 adapter 太多也可能增加文件读取或页缓存压力，不能把“支持很多 adapter”解释为没有主存和 I/O 上限。

## 热切换与独立数值参考

更新 adapter 时登记新的基座兼容关系、目标层、rank、缩放和版本，写入新存储并验证后发布。旧版本仍被请求或图引用时继续有效；路由切换只改变后续绑定。把 A、B 分别更新会产生混合增量，必须以相容的完整对象发布。对多个 TP rank，所有参与者需共同确认同一版本与分片规则。

验证以逐 adapter 的 Wx+sB(Ax) 独立参考为起点，覆盖零 A/B、非最大 rank、混合 batch 重排、非连续页、旧版本运行中新版本加载，以及加载失败与取消。再检查同文本不同 adapter 的 KV 身份和按租户占用。S-LoRA 的固定 v1 使用特定 Llama 配置，其中 70B 实验并未采用官方 GQA；论文调度附录的简化条件也不构成任意服务负载的最优性证明。

[多模型完整场景](../workloads/multi-model-placement.md) 将共同池、冷 adapter、长度增长、候选布局和租户验证放入同一资源账。只有单一 adapter 长期稳定时，合并权重的简洁路径才可能重新成为更好的选择；多 adapter 并发、量化基座或频繁更新会改变这个判断。

<!-- infraagents:provenance -->

## 来源与推导

- [S-LORA：LoRA: Low-Rank Adaptation of Large Language Models](../sources/S-LORA.md)：LoRA 用低秩增量 ΔW＝sBA 表达权重适配；固定 adapter 可在条件允许时合并，多 adapter 可以共享基座并分别计算增量分支。
  [完整原文](../../references/S-LORA/reader.html) · [全文文本](../../references/S-LORA/fulltext.txt)

- [S-VLLM-PREFIX：vLLM — Automatic Prefix Caching](../sources/S-VLLM-PREFIX.md)：所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。
  [完整原文](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-SLORA：S-LoRA](../sources/S-SLORA.md)：§4 共享基座与异 adapter 低秩分支、聚批/准入；§5 adapter/KV 共同分页、预取和非连续 gather 内核；§6 增量的张量并行；附录给聚类与早停策略条件。
  [完整原文](../../references/S-SLORA/reader.html) · [全文文本](../../references/S-SLORA/fulltext.txt)

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
