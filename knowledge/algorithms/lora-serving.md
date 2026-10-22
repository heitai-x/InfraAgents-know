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

<!-- infraagents:provenance -->

## 来源与推导

- S-LORA：[完整原文](../../references/S-LORA/reader.html) · [全文文本](../../references/S-LORA/fulltext.txt)
- S-VLLM-PREFIX：[完整原文](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt)
- S-TORCH-NUMERIC：[完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-LORA：LoRA: Low-Rank Adaptation of Large Language Models](../sources/S-LORA.md)：LoRA 用低秩增量 ΔW＝sBA 表达权重适配；固定 adapter 可在条件允许时合并，多 adapter 可以共享基座并分别计算增量分支。

- [S-VLLM-PREFIX：vLLM — Automatic Prefix Caching](../sources/S-VLLM-PREFIX.md)：所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
