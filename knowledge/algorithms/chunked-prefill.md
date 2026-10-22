# K-ALG-007：分块 Prefill：服务时间切片与历史可见性

分块 prefill 将长输入的执行切成可调度时间片；每块仍读取其之前完整且有效的上下文。

## 为什么要切片

假设单独处理一个长 prompt 需要 200 ms，而正在生成的会话希望每 30 ms 收到 token。即便长 prompt 的总计算效率很好，它独占下一轮也会造成一次很长的输出停顿。将其分为若干块，让 decode 在块边界获得服务，改变的是可抢占粒度。总 prefill 计算量并不会因此按块数降低。

已有历史长 h，本块 q 个 token 时，第 i 个本块 query 应看见 h＋i＋1 个 token。前块 KV 与本块新 KV 都参与 attention。把每块当成新序列会切断跨块语义；每块重复完整历史计算又会丢失缓存收益。分块只是执行分区，逻辑位置连续。

## token 预算和时间预算

线性层近似随 q 增长，attention 随 qh＋q(q＋1)/2 增长。因此同样 q 的后续块可能更慢。固定 token 块简单且便于图桶，时间预算模型则可以结合 h、batch 和平台选择 q。若目标是限制 decode 间隔，可以先为已就绪 decode 保留服务量，再将剩余时间估计分给 prefill，而不是只看剩余 token 数。

块太小会重复 launch、调度和权重读取，降低 GEMM 利用率；块太大提高局部效率，却拉长等待。一个 4096 token prompt 采用 512 的块需要 8 次参与调度，其总 TTFT 还包含这 8 次间隙中的等待。稳定 ITL 与降低新请求 TTFT 是相互作用的目标。

## 混合批的数据形态

若本轮请求 A 有 128 个 prefill query，B、C 各 1 个 decode query，则线性层可面对 130 行压紧 token；attention 分段长度仍是 128、1、1，历史长度各自独立。只有 A 完成最后一块时，才从相应末行提取下一 token logits；中间块不应向用户产生未请求的采样输出。

chunk size 可以对齐页粒度以简化分配，但语义上不要求页长和 chunk 长相等。部分前缀命中、窗口层和多模态分段可能造成非整页边界。encoder-decoder 的 encoder 是否能分块，是其自身可见性决定的：双向 encoder 的任意切片通常改变表示，不能直接套用 causal decoder 的这一推导。

<!-- infraagents:provenance -->

## 来源与推导

- S-SARATHI：[完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)
- S-ORCA：[完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)
- S-HF-CACHE：[完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
