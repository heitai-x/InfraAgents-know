# 算法

本分类条目的适用条件、来源与未验证范围在各正文中说明。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-ALG-001：连续批处理：从自回归依赖到迭代调度](iteration-batching.md) | v1 · seed | 连续批处理将资源分配单位从整条请求改成一次可执行的 token 推进；批次是请求状态的临时视图。 |
| [K-ALG-002：分页 KV：逻辑位置、物理布局与共享所有权](paged-kv-capacity.md) | v1 · seed | 分页把序列长度的动态增长转换为块映射变化；attention 语义仍由逻辑位置与有效长度决定。 |
| [K-ALG-003：前缀缓存：计算等价类、Radix 与淘汰收益](prefix-cache-identity.md) | v1 · seed | 可复用前缀是能产生等价中间状态的一组输入和执行条件；字符串相同只是其中一个投影。 |
| [K-ALG-004：Attention 的 IO、在线 softmax 与分块归约](attention-io.md) | v1 · seed | 精确 attention 可以通过改变数据流避免存储完整分数矩阵；算法复杂度、HBM 流量与数值归约顺序分别决定成本。 |
| [K-ALG-005：量化：误差传播、分组尺度与执行格式](quantization-contract.md) | v1 · seed | 量化把数值表示、数据布局和执行内核一起改变；收益由减少的字节与新增的变换成本共同决定。 |
| [K-ALG-006：投机解码：分布保持、状态提交与收益模型](speculation-budget.md) | v1 · seed | 投机用便宜候选换取目标模型的并行验证；真正推进系统的是被确认的前缀，而非写入过缓存的全部 token。 |
| [K-ALG-007：分块 Prefill：服务时间切片与历史可见性](chunked-prefill.md) | v1 · seed | 分块 prefill 将长输入的执行切成可调度时间片；每块仍读取其之前完整且有效的上下文。 |
| [K-ALG-008：异步调度：依赖图、代次与双缓冲](async-scheduling.md) | v1 · seed | 异步调度只重叠彼此独立的工作；自回归 token、缓存映射和资源回收仍由明确依赖连接。 |
| [K-ALG-009：异构状态缓存：层组、恢复位置与联合分配](heterogeneous-state-cache.md) | v1 · seed | 混合模型的缓存是多类状态的联合系统，统一的是逻辑恢复位置，物理块形状可以不同。 |
| [K-ALG-010：交叉注意力缓存：固定源、独立几何与可见性](cross-attention-cache.md) | v1 · seed | 当 encoder 输出和投影权重固定时，cross-attention 的 K/V 与 decoder 步数无关，可以一次生成、多次读取。 |
| [K-ALG-011：批量采样：概率变换、请求随机性与行映射](batched-sampling.md) | v1 · seed | 采样器消费每个请求自己的条件分布；向量化改变执行形态，不能改变参数、随机状态或输出归属。 |
| [K-ALG-012：结构化生成：语言状态、token 边界与约束采样](structured-output.md) | v1 · seed | 结构化生成将语法状态与语言模型状态同步推进，合法 token 是从当前语法状态仍可完成目标语言的延伸。 |
| [K-ALG-013：Prefill/Decode 分离：状态传输与队列耦合](disaggregated-serving.md) | v1 · seed | 分离式服务把两个阶段的资源策略解耦，同时引入 KV 传输、布局转换和跨节点状态提交。 |
| [K-ALG-014：多 LoRA 服务：低秩计算、驻留与缓存身份](lora-serving.md) | v1 · seed | LoRA 为共享基座增加低秩映射；多适配器服务将模型复用问题转换成按请求选择增量权重的问题。 |
| [K-ALG-015：推理并行：张量分解、流水、专家与上下文](parallelism.md) | v1 · seed | 并行策略改变张量归属和同步边界；选择依据是每阶段的计算、状态与通信分解。 |
