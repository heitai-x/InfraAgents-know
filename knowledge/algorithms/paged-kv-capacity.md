# K-ALG-002：分页 KV：逻辑位置、物理布局与共享所有权

分页把序列长度的动态增长转换为块映射变化；attention 语义仍由逻辑位置与有效长度决定。

## 从稠密缓存到页表

一层普通注意力缓存的逻辑轴是请求、token 位置、KV head、head 内通道，以及独立的 K/V。若每请求预留最大长度，短请求的大部分容量闲置；若每次增长都重新连接大张量，则反复搬移旧状态。固定 token 数 P 的物理页将这两项成本换成页表访问和页尾碎片。

逻辑位置 t 的页号是 ⌊t/P⌋，页内偏移是 t mod P。请求页表把逻辑页号映射到物理页 ID。物理页可以不连续，但所有逻辑 token 的顺序必须保持。存储布局 NHD 与 HND 分别让页内 token 或 head 位于不同的中间轴；它们描述物理张量次序，并不改变逻辑 head 的含义。转置一个描述符而未转换数据不会改变实际布局。

## 一个可算清的例子

设 32 层、8 个 KV heads、每 head 128 元素、K/V 都为 2 字节，则每 token 的全层 KV 是 2×32×8×128×2＝131072 字节，即 128 KiB。P＝16 时，一组覆盖全部层的逻辑块需要 2 MiB；这不要求实现把所有层放进同一个张量。

长度 18 的请求需要两块，分配 4 MiB，有效数据 2.25 MiB。多个请求长度与 P 的余数近似均匀且都独占尾页时，平均浪费约 (P−1)/2 个 token 的容量。更大 P 减少页表长度，却提高尾部浪费和共享分叉粒度。小 P 降低浪费，却增加元数据、非连续访存与调度开销。

变长批可用一段平坦页 ID 列表加每请求的起止偏移描述：请求 A 占前两项，B 占下一项。末页有效 token 数单独保存。空请求不能伪装成拥有一页且长度 P；页尾长度为 0 的约定必须与内核一致。query 的分段偏移与 KV 的页偏移各表达不同轴。

## 共享后必须引入所有权

| 状态 | 允许的操作 | 转换依据 |
|---|---|---|
| 空闲 | 分配给新逻辑页 | 不存在未完成读写 |
| 私有写入 | 当前所有者追加 | 分配成功且映射已发布 |
| 共享只读 | 多请求读取 | 相同前缀的有效状态已完成 |
| 私有分叉 | 复制有效前缀后继续写 | 多所有者中的一个需要修改 |
| 等待回收 | 等待设备和传输完成 | 逻辑引用归零但执行仍可能使用 |

引用计数管理逻辑所有者，完成事件管理物理使用者。两者条件都成立后才可重用。若共享的是已填满的整页，后续 token 可写新页；若共享未满页，追加需要私有尾页或 copy-on-write。页表本身也可能被已排队的 graph 读取，因此修改页表和回收数据都要按执行代次隔离。

## 分页与计算优化的组合

PagedAttention 解决动态存储映射；FlashAttention 解决片上分块和中间矩阵 IO。二者可组合，前提是 kernel 直接理解页表，或显式付出 gather 成稠密布局的成本。分页不自动降低全部 KV 读取量：长上下文 decode 仍需读取可见历史。前缀共享节省容量和 prefill 重算，也不意味着不同 query 可以完全共享后续 attention 计算。

容量规划还需扣除权重、激活峰值、图池、通信区、量化 scales、分配器碎片和暂存副本。只用理论 KV/token 除剩余显存得到的是理想上界；实际可同时驻留多少请求由这些项和长度分布共同决定。

<!-- infraagents:provenance -->

## 来源与推导

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：截止日前的 FlashInfer 固定源码区分 query 分段、KV 页表及末页长度。相关 wrapper 的 plan 安排计算与工作区，run 执行 attention；plan 的捕获限制与图模式缓冲须按该版本接口核对。
  [完整原文](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt)

- [S-VLLM-KV：vLLM — KVCacheManager source reference](../sources/S-VLLM-KV.md)：KVCacheManager 的相关接口区分已命中状态、新 token 分配、lookahead、外部状态和释放。层组与实际可恢复状态参与分配判断。
  [完整原文](../../references/S-VLLM-KV/reader.html) · [全文文本](../../references/S-VLLM-KV/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [MHA、GQA 与 MQA：头映射和带宽复用](../models/gqa-kv-heads.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
