# S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention

## 原文

[原始 PDF](../../references/S-PAGED/original.pdf) · [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

固定版本：2309.06180v1。来源版本日期：2023-09-12；整理核对日期：2026-10-27。可直接阅读全文；下面是本库核对后抽出的机制与结论。

## 问题

当时的服务系统为每条请求按最大长度预留连续 KV 空间。论文测得只有 20.4%–38.2% 的 KV 内存存放了真实 token 状态，其余是为未来 token 保留的槽、超额预留造成的内部碎片，以及分配器的外部碎片。这限制了批大小，进而限制吞吐。

## 机制

块化 attention（§4.1）。把每条序列的 KV 切成固定 token 数 B 的块，Kⱼ = (k₍ⱼ₋₁₎B₊₁ … k_{jB})，Vⱼ 同理。attention 按块计算：Aᵢⱼ = exp(qᵢᵀKⱼ/√d) / Σ_t exp(qᵢᵀK_t 𝟙)，oᵢ = Σⱼ VⱼAᵢⱼᵀ；kernel 逐块取址，块在物理上可以不连续。各层各 head 的 K/V 可以放在同一块里，也可以各用独立块与独立块表；论文称两种设计性能无差别。

块表（§4.2）。每条请求有逻辑块到物理块的映射，每项记录物理块号与已填充位置数。逻辑块从左到右填满后才分配下一个物理块，因此每条请求的浪费被限制在最后一块以内。GPU 上由一个 block engine 预分配一大段显存再切成物理块；CPU 内存也按同样方式管理以支持换出。

共享与写时复制（§4.4）。同一物理块可被多个逻辑块映射，每个物理块维护引用计数；某个序列要写入引用计数大于 1 的块时，先分配新块、复制内容、把原块计数减一，再写新块。并行采样、beam search 与共享系统提示都靠这一机制共享前缀 KV。beam search 中被淘汰候选的逻辑块释放后，物理块计数归零即回收。

抢占的两条路（§4.5）。换出：把被抢占序列的块复制到 CPU 内存，恢复时再搬回；CPU 侧换出空间不超过 GPU 物理块总数。重算：丢弃状态，恢复时把已生成 token 与原 prompt 拼接一次性重新 prefill。

## 结论

- 块大小的权衡（§7.2，Fig. 18b）：块越大，kernel 并行处理的位置越多、硬件利用率越高，但内部碎片越大。ShareGPT 轨迹上 16–128 都表现最好；Alpaca 轨迹（序列更短）上 16 和 32 好，更大的块明显变差。论文选 16。
- 块化 kernel 的额外开销（§7.1，Fig. 18a）：访问块表、额外分支与变长处理使 attention kernel 延迟比高度优化的连续实现高 20–26%；论文认为它只影响 attention 算子，可以接受。
- 端到端：相同延迟下吞吐比当时的系统高 2–4 倍，序列越长、模型越大、采样越复杂提升越明显。

## 本库据此推导

- "逻辑位置顺序不变、物理放置任意"是分页的不变量；块大小是粒度轴上的一个点，论文的 16 来自两条特定轨迹，不是常数，见 [分页 KV](../algorithms/paged-kv-capacity.md)。
- 引用计数管理逻辑所有者、完成事件管理物理使用者，二者都成立才能重用；论文只写了前者，后者由异步执行引入，见 [内存分配](../system-semantics/memory-allocators.md)。
- 20–26% 的 kernel 开销是"间接寻址换容量"这条交换的实测价格之一，见 [Attention 执行接口](../frameworks/attention-backends.md)。
- 换出与重算的比较由状态大小、链路带宽与重算长度决定，见 [连续批处理](../algorithms/iteration-batching.md) 与 [KV 卸载](../algorithms/kv-offload-prefetch.md)。

## 适用边界

论文的数字针对 2023 年的 A100 与 OPT/LLaMA 模型。块化 attention 的开销比例随 kernel 实现变化，后来的实现可以降低它。版本日期不证明模型训练实际包含该资料。
