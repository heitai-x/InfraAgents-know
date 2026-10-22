# 从任务条件推导系统设计

先用 [组合设计](system-semantics/design-synthesis.md) 建立完整问题，再选与当前决策有关的机制。索引摘要用于导航，具体设计依据来自正文中的关系与推导。

角色只从当前运行的时间 profile 读取正文和原始来源。`checked_at`、`updated_at` 和本地归档时间不是事实发生时间；需要先通过 [时间边界规则](../docs/23-time-bounded-knowledge.md) 的来源版本检查，再把专题用于设计。

本入口按完整推理系统组织阅读顺序；[推理范围与组合验证](../docs/inference-coverage.md) 列出 24 个能力域和集群、存储、部署三项平台依赖。运行实际可读范围还由知识条件、来源资格及原始资料访问限制决定。

## 确定模型函数与状态

从 [模型计算图](models/prefill-decode-state.md) 出发，识别 [GQA](models/gqa-kv-heads.md)、[MLA](models/mla.md)、[MoE](models/moe-routing.md)、[SSM/混合](models/recurrent-hybrid-state.md) 或 [窗口层](models/sliding-window-attention.md)。对于 [encoder-decoder](models/encoder-decoder-streaming.md)、[VLM](models/vision-language.md)、[扩散](models/image-diffusion.md)、[视频](models/video-generation.md)、[语音生成](models/speech-generation.md) 和 [全模态](models/omni-multimodal.md)，还需还原阶段与模态之间的依赖。

这一步的产物是输入输出、逻辑位置、可见集合和持久状态的明确描述。[Tokenizer 与 I/O](system-semantics/tokenization-io.md) 将用户输入连接到这份描述。

非生成负载分别进入 [Embedding 与重排](models/embedding-reranking.md)、[视觉识别与检测](models/vision-inference.md)、[推荐](models/recommendation-inference.md) 和 [图模型](models/graph-inference.md)。它们各有 pooling/评分、图像几何、稀疏特征或邻居依赖；[检索与 RAG](system-semantics/retrieval-pipeline.md) 再将编码、索引、材料版本和下游生成连接起来。

再通过 [权重加载](frameworks/checkpoint-loading.md) 还原制品、张量别名与执行布局，以 [基础算子](algorithms/operator-primitives.md) 明确线性、归一化、位置旋转和索引的函数契约。加载完成与实例就绪分开判断，冷启动和预热的峰值都进入容量账。

[模型生命周期](system-semantics/model-lifecycle.md) 将加载、编译/预热、准入、排空和版本切换连成状态链，明确旧请求何时结束、新模型何时可以接收请求，以及对应资源何时回收。

## 推导状态表示与调度

由 [分页 KV](algorithms/paged-kv-capacity.md) 计算字节和所有权，以 [前缀等价](algorithms/prefix-cache-identity.md) 推导共享，混合层使用 [异构状态](algorithms/heterogeneous-state-cache.md)，固定源使用 [cross KV](algorithms/cross-attention-cache.md)。

[连续批](algorithms/iteration-batching.md) 决定推进单位，[分块 prefill](algorithms/chunked-prefill.md) 决定服务时间片，[异步调度](algorithms/async-scheduling.md) 决定可以重叠的区间。[请求身份](system-semantics/request-batch-identity.md) 和 [资源完成](system-semantics/cancellation-resource-completion.md) 贯穿这些组合。

[内存分配](system-semantics/memory-allocators.md) 把权重、KV、工作区和在途资源放在同一份容量账中，并比较连续池、块表与虚拟地址映射；状态表示和调度必须满足同一组分配与回收条件。

## 映射到平台和执行路径

先建立 [容量/roofline](hardware/memory-and-roofline.md) 与 [通信拓扑](hardware/topology-parallelism.md)，再阅读目标 [CUDA](hardware/cuda-execution.md)、[ROCm](hardware/rocm-execution.md)、[Metal](hardware/metal-unified-memory.md)、[Trainium](hardware/trainium-dataflow.md) 或 [CPU](hardware/cpu-serving.md)。

[Attention IO](algorithms/attention-io.md) 解释 kernel 数据流，[backend 契约](frameworks/attention-backends.md) 连接语义与布局，[编译分桶](frameworks/compile-dynamic-shapes.md)、[CUDA Graph](frameworks/cuda-graph-buffers.md)、[MLX](frameworks/mlx-serving.md)、[Neuron](frameworks/neuron-serving.md) 与 [NKI](frameworks/nki-tiling.md) 承接具体执行模型。

CPU 进一步进入 [GEMM/GEMV](hardware/cpu-linear-kernels.md) 的寄存器、cache/TLB 与线程划分，以及 [低位执行](frameworks/cpu-quantized-kernels.md) 的解包、零点补偿和累加路径。[主机运行时](system-semantics/host-runtime.md) 将算子线程与分词、队列、IPC 和响应处理的资源预算连接起来。

GPU 则通过 [矩阵与归约流水](hardware/gpu-kernel-pipelines.md) 把片上布局、矩阵指令、归约和缓冲完成关系连接起来；[CUDA 平台](hardware/cuda-execution.md) 中的交互服务算例将这些选择接回容量与 ITL。

## 在容量与传输约束下组织驻留

[HBM–DRAM–SSD 驻留](algorithms/tiered-residency.md) 先定义对象版本、有效副本和迁移提交，再由 [存储 I/O](hardware/storage-io.md) 还原真实读取路径。[KV 卸载](algorithms/kv-offload-prefetch.md) 比较恢复、持续读取与重算，[权重和专家卸载](algorithms/weight-expert-offload.md) 则比较层间重用、路由并集与热点变化。

在 [CPU–GPU 计算放置](algorithms/heterogeneous-execution.md) 中比较搬数据到 GPU 与把计算留在 CPU 的完整路径；[共享内存](hardware/shared-memory-paths.md) 进一步区分统一地址、可迁移驻留和物理共享。三级存储与异构计算各有可复算场景，容量可行、质量合格和 SLO 可达分别判断。

[KV 压缩](algorithms/kv-compression.md) 区分计算表示、历史裁剪与网络字节流。[分布式 KV 存储](system-semantics/distributed-kv-store.md) 建立目录、副本与恢复点，[传输与 RDMA](hardware/transport-and-rdma.md) 将注册、完成和 GPU 可见性连接到应用提交。[路由与迁移](algorithms/routing-and-migration.md) 比较命中与排队，[服务恢复](system-semantics/serving-recovery.md) 处理失败、重试与唯一提交权。

材料出现在不同上文或顺序中时，进入 [非前缀上下文复用](algorithms/context-reuse.md)，分别判断位置变换、完整前缀等价和选择性重算的质量条件；它与文档 embedding 缓存、普通材料缓存具有不同依赖。

## 组合额外能力

[量化](algorithms/quantization-contract.md) 改变表示和误差，[投机](algorithms/speculation-budget.md) 改变确认 token 的计算过程，[LoRA](algorithms/lora-serving.md) 改变权重依赖，[结构化生成](algorithms/structured-output.md) 改变合法输出分布。加入它们时，重新推导缓存身份、状态回滚、形状和成本。

[多样本与搜索分支](algorithms/branch-generation.md) 统一梳理父子状态、树形可见性和共同提交，同时保持 beam、多样本采样与投机的不同目标。完整长前缀与 P/D 场景分别见 [前缀缓存](algorithms/prefix-cache-identity.md) 和 [P/D 分离](algorithms/disaggregated-serving.md)。

[并行](algorithms/parallelism.md) 与 [P/D 分离](algorithms/disaggregated-serving.md) 改变资源归属；[分布式状态](system-semantics/distributed-state-transfer.md) 和 [集合顺序](system-semantics/collective-ordering.md) 说明组合协议。

## 组织多模型与平台依赖

[多模型放置](workloads/multi-model-placement.md) 先确认可共享的基座、adapter、执行缓冲和缓存，再比较驻留、分组与副本。[租户隔离](system-semantics/tenant-isolation.md) 将身份、缓存命名空间、配额、资源干扰和故障边界分别约束。[弹性服务](workloads/elastic-serving.md) 根据真实就绪容量与冷启动/排空时间响应突发，不能把已申请实例数直接当作服务能力。

[集群放置](workloads/cluster-resource-placement.md) 检查每个 rank 的资源向量、通信拓扑、共同启动和故障域；[推理存储服务](system-semantics/storage-services.md) 区分权重、KV、索引与提交记录的可见性和持久性；[部署环境](system-semantics/deployment-environment.md) 核对镜像、宿主驱动、设备身份、quota 和 NUMA。这三项决定推理方案是否真正可部署，并与加载、迁移和恢复共用容量及带宽预算。

## 回到服务目标

[协议与背压](system-semantics/service-protocol.md) 定义用户能看到的行为。[负载分布](workloads/input-distribution.md)、[到达过程](workloads/arrival-and-concurrency.md)、[指标事件](workloads/metric-events.md) 和 [goodput](workloads/slo-goodput.md) 定义优化问题，[容量规划](workloads/capacity-planning.md) 将需求映射到资源。

[正确性 oracle](tools/correctness-oracles.md) 与 [数值参考](frameworks/numerical-reference.md) 验证函数关系，[时间线](tools/timeline-attribution.md) 和 [计数器](tools/kernel-counter-replay.md) 检验成本解释。研究具体实现时再进入 [引擎索引](engine-index/README.md)。

[多轮与分支负载](workloads/session-workloads.md) 说明请求依赖、思考时间和状态驻留；[实验统计](tools/statistical-experiments.md) 将重复层级、效应量、尾部证据与独立保留验证连接起来。完整场景需要贯穿这些关系，不能仅由单章的局部机制证明系统目标已达成。

[成本与能耗](workloads/cost-and-energy.md) 以同一质量与 SLO 下的合格结果为分母，计入闲置、冷启动、失败和资源计量边界。方案缩小模型、减少候选或改变生成长度时，先确认任务定义与质量是否仍可比较。

## 沿完整场景检查组合

以下场景各自连接任务条件、状态、资源账、多个候选、验证与选择反转。正文中的数值为明确假设下的设定算例；它们用于形成可检验的设计，不表示已经在目标任务上取得相应性能。

| 场景 | 主要阅读入口 | 需要串起的关系 |
|---|---|---|
| C01：CPU 多请求生成 | [完整组合设计](system-semantics/design-synthesis.md) | CPU 线性层、KV、NUMA/线程、批量与联合 SLO |
| C02：单 GPU 交互 | [CUDA 执行](hardware/cuda-execution.md) | kernel、图池、状态容量、迭代时间与交互尾延迟 |
| C03：长上下文高复用 | [前缀缓存](algorithms/prefix-cache-identity.md) | 等价前缀、共享容量、排队、远端命中与重算 |
| C04：显存不足与三级存储 | [分层驻留](algorithms/tiered-residency.md) | 有效副本、迁移提交、加载峰值与链路下界 |
| C05：CPU–GPU 混合执行 | [异构执行](algorithms/heterogeneous-execution.md) | 计算放置、返回合并、共享资源竞争与同步 |
| C06：多卡 MLA/MoE | [并行组合](algorithms/parallelism.md) | TP/EP/PP 候选、latent/专家状态、逐层通信和慢端 |
| C07：P/D 与远端 KV | [P/D 分离](algorithms/disaggregated-serving.md) | 首 token 时间轴、状态接管、池配比与失败恢复 |
| C08：多模型/LoRA | [共同放置](workloads/multi-model-placement.md)、[低秩执行](algorithms/lora-serving.md) | 基座共享、异 rank、adapter 冷加载、缓存身份与公平等待 |
| C09：流式语音 | [转写与源编码](models/encoder-decoder-streaming.md)、[语音生成](models/speech-generation.md) | 源可见性、稳定文字提交、首包、播放供给与打断 |
| C10：VLM/全模态 | [VLM 三阶段](models/vision-language.md)、[全模态交互](models/omni-multimodal.md) | 视觉/语言缓存、多时钟、领先距离和背压 |
| C11：图像/视频扩散 | [图像生成](models/image-diffusion.md)、[视频生成](models/video-generation.md) | scheduler、CFG、时空切分、VAE 状态和完整交付质量 |
| C12：非生成推理 | [Embedding/重排](models/embedding-reranking.md)、[检索/RAG](system-semantics/retrieval-pipeline.md)、[CNN/检测](models/vision-inference.md)、[推荐](models/recommendation-inference.md)、[GNN](models/graph-inference.md) | 五类独立输入/输出和质量契约，各自计量批量、状态、搬运及结果；不合并为一个通用 token 指标 |

出现跨阶段的异常时，从 [诊断入口](DIAGNOSIS.md) 找到对应关系，再回到本表检查组合假设。章节和场景的编辑状态以 [构建记录](../docs/BUILD-STATUS.md) 与 [逐篇计划](../docs/expansion-plan.json) 为准。
