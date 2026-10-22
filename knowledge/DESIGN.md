# 从任务条件推导系统设计

先用 [组合设计](system-semantics/design-synthesis.md) 建立完整问题，再选与当前决策有关的机制。索引摘要用于导航，具体设计依据来自正文中的关系与推导。

角色只从当前运行的时间 profile 读取正文和原始来源。`checked_at`、`updated_at` 和本地归档时间不是事实发生时间；需要先通过 [时间边界规则](../docs/23-time-bounded-knowledge.md) 的来源版本检查，再把专题用于设计。

## 确定模型函数与状态

从 [模型计算图](models/prefill-decode-state.md) 出发，识别 [GQA](models/gqa-kv-heads.md)、[MLA](models/mla.md)、[MoE](models/moe-routing.md)、[SSM/混合](models/recurrent-hybrid-state.md) 或 [窗口层](models/sliding-window-attention.md)。对于 [encoder-decoder](models/encoder-decoder-streaming.md)、[VLM](models/vision-language.md)、[扩散](models/image-diffusion.md)、[视频](models/video-generation.md)、[语音生成](models/speech-generation.md) 和 [全模态](models/omni-multimodal.md)，还需还原阶段与模态之间的依赖。

这一步的产物是输入输出、逻辑位置、可见集合和持久状态的明确描述。[Tokenizer 与 I/O](system-semantics/tokenization-io.md) 将用户输入连接到这份描述。

再通过 [权重加载](frameworks/checkpoint-loading.md) 还原制品、张量别名与执行布局，以 [基础算子](algorithms/operator-primitives.md) 明确线性、归一化、位置旋转和索引的函数契约。加载完成与实例就绪分开判断，冷启动和预热的峰值都进入容量账。

## 推导状态表示与调度

由 [分页 KV](algorithms/paged-kv-capacity.md) 计算字节和所有权，以 [前缀等价](algorithms/prefix-cache-identity.md) 推导共享，混合层使用 [异构状态](algorithms/heterogeneous-state-cache.md)，固定源使用 [cross KV](algorithms/cross-attention-cache.md)。

[连续批](algorithms/iteration-batching.md) 决定推进单位，[分块 prefill](algorithms/chunked-prefill.md) 决定服务时间片，[异步调度](algorithms/async-scheduling.md) 决定可以重叠的区间。[请求身份](system-semantics/request-batch-identity.md) 和 [资源完成](system-semantics/cancellation-resource-completion.md) 贯穿这些组合。

[内存分配](system-semantics/memory-allocators.md) 把权重、KV、工作区和在途资源放在同一份容量账中，并比较连续池、块表与虚拟地址映射；状态表示和调度必须满足同一组分配与回收条件。

## 映射到平台和执行路径

先建立 [容量/roofline](hardware/memory-and-roofline.md) 与 [通信拓扑](hardware/topology-parallelism.md)，再阅读目标 [CUDA](hardware/cuda-execution.md)、[ROCm](hardware/rocm-execution.md)、[Metal](hardware/metal-unified-memory.md)、[Trainium](hardware/trainium-dataflow.md) 或 [CPU](hardware/cpu-serving.md)。

[Attention IO](algorithms/attention-io.md) 解释 kernel 数据流，[backend 契约](frameworks/attention-backends.md) 连接语义与布局，[编译分桶](frameworks/compile-dynamic-shapes.md)、[CUDA Graph](frameworks/cuda-graph-buffers.md)、[MLX](frameworks/mlx-serving.md)、[Neuron](frameworks/neuron-serving.md) 与 [NKI](frameworks/nki-tiling.md) 承接具体执行模型。

CPU 进一步进入 [GEMM/GEMV](hardware/cpu-linear-kernels.md) 的寄存器、cache/TLB 与线程划分，以及 [低位执行](frameworks/cpu-quantized-kernels.md) 的解包、零点补偿和累加路径。[主机运行时](system-semantics/host-runtime.md) 将算子线程与分词、队列、IPC 和响应处理的资源预算连接起来。

## 组合额外能力

[量化](algorithms/quantization-contract.md) 改变表示和误差，[投机](algorithms/speculation-budget.md) 改变确认 token 的计算过程，[LoRA](algorithms/lora-serving.md) 改变权重依赖，[结构化生成](algorithms/structured-output.md) 改变合法输出分布。加入它们时，重新推导缓存身份、状态回滚、形状和成本。

[并行](algorithms/parallelism.md) 与 [P/D 分离](algorithms/disaggregated-serving.md) 改变资源归属；[分布式状态](system-semantics/distributed-state-transfer.md) 和 [集合顺序](system-semantics/collective-ordering.md) 说明组合协议。

## 回到服务目标

[协议与背压](system-semantics/service-protocol.md) 定义用户能看到的行为。[负载分布](workloads/input-distribution.md)、[到达过程](workloads/arrival-and-concurrency.md)、[指标事件](workloads/metric-events.md) 和 [goodput](workloads/slo-goodput.md) 定义优化问题，[容量规划](workloads/capacity-planning.md) 将需求映射到资源。

[正确性 oracle](tools/correctness-oracles.md) 与 [数值参考](frameworks/numerical-reference.md) 验证函数关系，[时间线](tools/timeline-attribution.md) 和 [计数器](tools/kernel-counter-replay.md) 检验成本解释。研究具体实现时再进入 [引擎索引](engine-index/README.md)。

[多轮与分支负载](workloads/session-workloads.md) 说明请求依赖、思考时间和状态驻留；[实验统计](tools/statistical-experiments.md) 将重复层级、效应量、尾部证据与独立保留验证连接起来。完整场景需要贯穿这些关系，不能仅由单章的局部机制证明系统目标已达成。
