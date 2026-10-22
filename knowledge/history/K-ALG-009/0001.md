# K-ALG-009：异构状态缓存：层组、恢复位置与联合分配

混合模型的缓存是多类状态的联合系统，统一的是逻辑恢复位置，物理块形状可以不同。

## 将缓存描述成状态族

全注意力保存越来越长的 KV；窗口注意力只需直接可见的最近历史；MLA 保存压缩 latent 与位置分量；SSM 保存递归状态和局部卷积历史。encoder context 又是固定只读状态。用统一的“每层同样 KV 页”表示会掩盖长度、形状和更新方式的差异。

可以按状态类型、dtype、每 token 字节、块长、层数和保留规则分组。对组 g，token block 长 Pᵍ、每 token 全组字节 cᵍ，则块容量为 Pᵍcᵍ；递归组的检查点则按一次完整状态的字节计。总容量是各池驻留块之和，加上共享元数据和临时分叉。

## 物理池的三种组织

| 组织 | 好处 | 代价 |
|---|---|---|
| 每状态族独立池 | 布局直接，生命周期清楚 | 某池闲置而另一池耗尽 |
| 同字节尺寸统一块池 | 容量可跨族调剂 | token 块长不同，映射复杂 |
| 大块内子分配 | 兼容不同尺寸 | 碎片、回收和对齐账务增加 |

若为了统一字节块取各组单位尺寸的最小公倍数，必须计算真实 LCM。384 与 512 的 gcd 是 128，LCM 为 1536，而非乘积 196608。这还仅是两个单位大小的共同倍数，不说明它就是最佳页：层数、对齐要求、窗口长及内核支持可能让统一策略不合算。用更大块减少元数据时，要同时计算尾部浪费和回收粒度。

## 恢复状态是集合问题

对每一状态族定义可恢复位置集合 Rᵍ。模型可恢复位置属于所有 Rᵍ 的交集。全注意力页可能覆盖连续前缀，递归状态只保存在稀疏检查点，窗口状态还需包含恢复点之前的完整可见窗口。一个 token 前缀哈希命中并不构成全模型恢复资格。

例如全注意力已有位置 0–1023，递归检查点仅有 512 与 768，窗口池只保留 640–1023 且窗口 128。768 可恢复需要相应窗口历史存在；512 即使有递归检查点，也缺少窗口内容。最长可继续的公共位置由这些条件共同得出。

## 联合分配与淘汰

一次请求推进需要多个池都获得资源。若先占住 A 池再等待 B 池，而另一个请求反向占用，会造成容量僵局或饥饿。设计可以先计算完整资源向量并原子预留，或让失败预留可完全回滚。这里的原子性是调度事务语义，不要求一个硬件原子指令。

淘汰时也要维护可恢复组合。删除一个共享前缀的递归检查点，可能让大量保留 KV 失去继续执行价值；只按单池 LRU 看不出这种损失。对状态包按可恢复位置计价，比简单追求各池局部命中率更接近真实重算收益。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager](../sources/S-VLLM-HYBRID.md)：固定历史设计文档区分 KV hidden bytes、token block size 和物理 page size，讨论异构层分组、统一分配以及各组可恢复前缀的交集。具体功能限制只适用于该文档版本。
  [完整原文](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt)

- [S-MAMBA：Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../sources/S-MAMBA.md)：Mamba 使用输入相关的选择性状态空间机制。递归生成保留有限大小的状态，长序列处理可以采用专门的扫描/分块执行方式。
  [完整原文](../../references/S-MAMBA/reader.html) · [全文文本](../../references/S-MAMBA/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [SSM 与混合模型：递归状态、扫描与恢复](../models/recurrent-hybrid-state.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](../models/mla.md)。
- [窗口与稀疏 Attention：可见性、位置和状态保留](../models/sliding-window-attention.md)。
- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
