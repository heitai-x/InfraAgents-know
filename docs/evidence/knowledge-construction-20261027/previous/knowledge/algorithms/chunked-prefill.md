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

## 从剩余延迟预算反求块长

若最紧迫 decode 请求距离上次输出已过去 w，其目标间隔为 D，主机提交与响应预计耗时 tₕ，则本轮设备预算最多为 D−w−tₕ。这只是局部预算，网络与上游排队若也属于 SLO，必须使用相同的事件定义。负预算说明当前请求已经面临违约，不能通过选择负块长解决。

一个可校准的初始模型可以令本轮时间为 a+bΣqᵣ+cΣ[qᵣhᵣ+qᵣ(qᵣ+1)/2]，再按实际 kernel、batch、dtype 和平台分段。它把线性层与 attention 的形态区分开，不意味着真实执行一定线性或三项可以完全相加。先为已就绪 decode 保留预算，再选择满足时间和容量条件的最大合法 prefill 工作集合。

设定设备剩余预算为 10 ms，某一固定形态下模型为 2+0.01q+0.000001[qh+q(q+1)/2] ms。求解并取满足预算的最大整数，h=1000 时 q≤704；h=7000 时 q≤464。若可用图桶只有 256 和 512，后一种形态只能先考虑 256。该算例仅展示历史长度和桶约束如何改变选择，系数必须由目标运行校准，不能照抄为默认配置。

## 校准误差与组合约束

用同类形态最近的预测残差判断预算是否可靠，保留保守余量或缩小不确定形态的块。突然编译、页映射、共享页写时复制或专家偏斜可能使平均模型失效；它们应作为可解释状态或独立路径处理，而不是无限增大一个全局安全系数。

块长同时影响页预留、kernel 选择和调度机会，优先级也影响哪些请求进入。若多个长 prompt 竞争，单纯让每个请求取最大块可能增加其他请求等待；轮转、按剩余工作和按 deadline 分配都可比较。语音/视觉前处理造成的就绪时间与 causal token 流不同，应按阶段依赖进入队列，不能仅凭已经收到原始输入就视为 prefill 全部可执行。

对完整 prompt，以不同块序列和相同绝对位置计算，比较每个公共边界的状态与 logits；再在固定外部到达下测 TTFT、ITL、总 prefill 时间和抢占次数。若块减小后 ITL 好转而 TTFT 或总服务成本明显恶化，应重新选择目标运行点；低负载且没有其他请求等待时，较大块甚至整段 prefill 可能更合适。

<!-- infraagents:provenance -->

## 来源与推导

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。
  [完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。
  [完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：固定文档说明逐层复用历史 K/V、计算并合并新增 K/V，以及增量循环的 attention mask 应覆盖历史与新增 token；并区分 Dynamic、Static、StaticSlidingWindow 等缓存存储形态。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
