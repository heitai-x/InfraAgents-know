# K-SYS-010：系统设计推导：把模型、缓存、执行与目标组合起来

专家指导的产物是一条从任务条件到系统结构的推导链：状态模型、资源模型、候选设计及其可区分的结果。

## 统一问题表述

输入条件包括模型函数与状态、硬件资源、到达与长度分布、质量和时延目标。设计变量包括状态布局、调度粒度、执行后端、并行策略和阶段划分。输出应解释为什么这些变量组合在当前条件下有价值，以及改变哪个条件会促使设计改变。

可以用四类关系串联：模型决定可复用状态；状态和 workload 决定容量与流量；硬件和 backend 决定执行成本；调度与队列决定用户看到的延迟。只优化其中一环可能把瓶颈推向另一环，设计需要顺着完整链重新求解。

## 场景一：长系统提示、高前缀复用的 GQA 聊天服务

先从 Hkv 计算 KV/token，再用长度分布得到驻留需求。大量相同系统前缀满足计算等价条件，适合共享前缀状态。分页让共享完整块与各请求私有尾部共存。前缀命中减少 prefill 工作，剩余新输入仍可分块，避免阻塞活跃 decode。

当命中后 query 很短，主机调度、采样和 launch 的比例可能上升，因此固定缓冲的图执行与异步准备值得评估。图前缀页表在每轮内容变化，稳定地址与代次隔离一起设计。若加入多 LoRA，缓存身份与图形状还要包含 adapter 依赖；共享收益可能下降，应重新估算容量和路由。

本设计的预测是 prefill 计算量按命中 token 减少，KV 物理驻留按实际共享页减少，单 token decode attention 仍读取其可见历史。若观测只有 hit rate 上升而队列更长，可能是亲和路由集中负载，应比较排队增加和计算节省。

## 场景二：语音 Encoder-decoder 多请求服务

源音频先经过明确的 processor 与 encoder，固定源可缓存各层 cross KV。decoder self KV 随目标长度增长，cross KV 由源长度决定，两个容量预算分开。encoder 批次按音频长度聚合，decoder 按迭代连续推进，阶段间传递带身份的源状态。

若要流式输入，先判断 encoder 是否保证已提交表示稳定。双向 encoder 不具备该性质时，任意追加源并重用旧 cross KV 会改变函数。可选按模型支持的块协议接入，或将独立窗口识别作为另一种明确任务。首个稳定文本时延需包括输入窗口等待，不能只测 decode。

## 场景三：MLA＋MoE 的多卡长上下文服务

MLA 减少持久 attention 状态，但是否避免历史展开决定实际带宽收益；MoE 全部专家权重仍需驻留或按策略移动。容量账必须分别计 latent、专家、临时分组激活与通信区。EP 改变专家归属，TP 改变 attention/线性层分片，两者通信频率和消息分布不同。

节点内外链路不同，先给高频同步选择强互联组，再计算远端 dispatch 的路径和偏斜。若 prefill/decode 分离，还要计 latent 状态在不同分片之间的传输与转换。较小 KV 不等于网络完全无代价；也不意味着专家热点被解决。

## 场景四：静态编译平台上的混合长度任务

模型函数固定，服务形状变化。按 prompt、query、context 和 batch 的联合分布选择有限桶，设备保留 KV，控制面只推进真实长度。短高频请求应有合适的小桶；稀少长请求可以用较粗桶，避免过多 executable 与 workspace 驻留。

前缀命中会改变待算 query 分布，连续批会改变 batch 分布，投机会改变验证长度分布。桶选择因此应在这些机制组合后的 workload 上推导。若先按无缓存独立请求选桶，再加入其他机制，原配置可能失去意义。

## 从推导到 RSI 增量

规划说明当前链条中哪个关系最不确定，执行实现对应机制，诊断对其预测和实际状态进行核对。进化把新认识写回受影响的机制章节：补足成本项、修正可恢复条件或新增组合分支。成功不必产生新条目；更精确地理解一个现有机制，往往比增加一个孤立提示更有价值。

<!-- infraagents:provenance -->

## 来源与推导

- S-PAGED：[完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)
- S-SARATHI：[完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)
- S-DISTSERVE：[完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)
- S-MLA：[完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)
- S-NXDI：[完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。

- [S-DISTSERVE：DistServe](../sources/S-DISTSERVE.md)：DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。

- [S-NXDI：NxD Inference Features Configuration Guide](../sources/S-NXDI.md)：NxD Inference 说明静态形状可执行程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可配置不同桶；持久状态与模型执行共同受配置约束。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [容量设计：驻留、服务率与阶段配比](../workloads/capacity-planning.md)。
