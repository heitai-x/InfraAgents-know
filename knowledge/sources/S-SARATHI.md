# S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

## 本地完整原文

[完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

固定版本：2308.16369v1。来源版本日期：2023-08-31；整理核对日期：2026-10-28。

## 来源支持的机制与定义

固定 v1 的 §3–4 从两类成本出发：prefill 有较多已知 query 行，可利用矩阵计算；decode 每请求通常只有一行，读取权重的代价难以由单行摊薄。混合批的组成又会改变 pipeline 各阶段时长，使不同微批的时间不均衡。

§4 的 chunked-prefills 把一个已知 prompt 切成多个块，后块继续读此前块的 KV。Figure 6 用逐步增长的 causal 可见矩阵展示这一点；不是把每块作为新请求独立计算。每块内部仍可同时计算多行，因为这些输入 token 已知，而 mask 排除其逻辑未来。

decode-maximal batching 选一个 prefill 块，将其余批槽填入 decode；论文希望 prefill 提供计算密集的主体，让 decode 共同摊薄相应线性层成本，并使微批工作更均匀。在固定 chunk 大小下，长 prefill 可以提供多次混合机会。这个组合依赖有足够已就绪的 decode、合适的 chunk 与设备形态；prefill 不足时仍会出现 decode 尾部，额外 chunk 边界也会重复准备和执行成本。

论文中的计算均匀性目标不等于每个固定 token 块严格等时。历史 h 增大后，q 行 Attention 的合法对数为 qh+q(q+1)/2；同一 q 可以在不同运行点支付不同读取、计算和 kernel 费用。不能把“chunk size”无条件当成时间片长度，或者把论文中的混合收益视为长历史 decode 的免费计算。

## InfraAgents 如何据此推导

[分块 prefill](../algorithms/chunked-prefill.md)和[连续批处理](../algorithms/iteration-batching.md)据此推导时间预算、页增量、混合批行映射与首尾成本。后者给出的最大可行 q、页预算反转、单请求一轮在途、轮转公平前提与诊断数据均为本库独立构造；并不声称固定 SARATHI v1 采用同样的时间模型或取消协议。

对一段相同 prompt，把各块的 qh+q(q+1)/2 相加，恰好等于整段的 causal 对数。切片降低单次不可抢占区间，通常会重复固定成本；只有加入并发请求和服务目标，才能判断这种交换是否值得。实验要同时比较已有 decode 的 ITL、新请求 TTFT、总服务时间、实际提供负载和拒绝，不能只引用混合 GEMM 的每 token 成本。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
