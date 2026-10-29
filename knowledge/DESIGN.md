# 从任务条件推导系统设计

本页按技术主题提供可选入口，没有固定阅读或研发顺序。场景与示例不穷尽候选；Task 的要求和所选机制的语义决定实际检查，原型、整体编译、联合设计和其他未列方案均可按条件探索。

可按当前问题直接选择专题；[组合设计](system-semantics/design-synthesis.md) 提供连接多个机制的条件化例子。索引摘要用于导航，具体设计依据来自正文中的关系与推导。

角色只从当前运行的时间 profile 读取正文和原始来源。`checked_at`、`updated_at` 和本地归档时间不是事实发生时间；需要先通过 [时间边界规则](../docs/23-time-bounded-knowledge.md) 的来源版本检查，再把专题用于设计。

本入口按完整推理系统的技术主题组织导航；[推理范围与组合验证](../docs/inference-coverage.md) 列出 24 个能力域和集群、存储、部署三项平台依赖。运行实际可读范围还由知识条件、来源资格及原始资料访问限制决定。

## 确定模型函数与状态

从 [模型计算图](models/prefill-decode-state.md) 出发，识别 [GQA](models/gqa-kv-heads.md)、[MLA](models/mla.md)、[MoE](models/moe-routing.md)、[SSM/混合](models/recurrent-hybrid-state.md) 或 [窗口层](models/sliding-window-attention.md)。对于 [encoder-decoder](models/encoder-decoder-streaming.md)、[VLM](models/vision-language.md)、[扩散](models/image-diffusion.md)、[视频](models/video-generation.md)、[语音生成](models/speech-generation.md) 和 [全模态](models/omni-multimodal.md)，还需还原阶段与模态之间的依赖。

这些专题解释输入输出、逻辑位置、可见集合和持久状态之间的关系。[Tokenizer 与 I/O](system-semantics/tokenization-io.md) 将用户输入连接到这份描述。

非生成负载分别进入 [Embedding 与重排](models/embedding-reranking.md)、[视觉识别与检测](models/vision-inference.md)、[推荐](models/recommendation-inference.md) 和 [图模型](models/graph-inference.md)。它们各有 pooling/评分、图像几何、稀疏特征或邻居依赖；[检索与 RAG](system-semantics/retrieval-pipeline.md) 再将编码、索引、材料版本和下游生成连接起来。

通过 [权重加载](frameworks/checkpoint-loading.md) 还原制品、张量别名与执行布局，以 [基础算子](algorithms/operator-primitives.md) 明确线性、归一化、位置旋转和索引的函数契约。加载完成与实例就绪分开判断，冷启动和预热的峰值都进入容量账。

[模型生命周期](system-semantics/model-lifecycle.md) 将加载、编译/预热、准入、排空和版本切换连成状态链，明确旧请求何时结束、新模型何时可以接收请求，以及对应资源何时回收。

## 推导状态表示与调度

由 [分页 KV](algorithms/paged-kv-capacity.md) 计算字节和所有权，以 [前缀等价](algorithms/prefix-cache-identity.md) 推导共享，混合层使用 [异构状态](algorithms/heterogeneous-state-cache.md)，固定源使用 [cross KV](algorithms/cross-attention-cache.md)。

[连续批](algorithms/iteration-batching.md) 决定推进单位，[分块 prefill](algorithms/chunked-prefill.md) 决定服务时间片，[异步调度](algorithms/async-scheduling.md) 决定可以重叠的区间。[请求身份](system-semantics/request-batch-identity.md) 和 [资源完成](system-semantics/cancellation-resource-completion.md) 贯穿这些组合。

[内存分配](system-semantics/memory-allocators.md) 把权重、KV、工作区和在途资源放在同一份容量账中，并比较连续池、块表与虚拟地址映射；状态表示和调度必须满足同一组分配与回收条件。

## 映射到平台和执行路径

[容量/roofline](hardware/memory-and-roofline.md) 与 [通信拓扑](hardware/topology-parallelism.md) 可与目标平台交叉阅读： [CUDA](hardware/cuda-execution.md)、[ROCm](hardware/rocm-execution.md)、[Metal](hardware/metal-unified-memory.md)、[Trainium](hardware/trainium-dataflow.md) 或 [CPU](hardware/cpu-serving.md)。

[Attention IO](algorithms/attention-io.md) 解释 kernel 数据流，[backend 契约](frameworks/attention-backends.md) 连接语义与布局，[编译分桶](frameworks/compile-dynamic-shapes.md)、[CUDA Graph](frameworks/cuda-graph-buffers.md)、[MLX](frameworks/mlx-serving.md)、[Neuron](frameworks/neuron-serving.md) 与 [NKI](frameworks/nki-tiling.md) 承接具体执行模型。

CPU 进一步进入 [GEMM/GEMV](hardware/cpu-linear-kernels.md) 的寄存器、cache/TLB 与线程划分，以及 [低位执行](frameworks/cpu-quantized-kernels.md) 的解包、零点补偿和累加路径。[主机运行时](system-semantics/host-runtime.md) 将算子线程与分词、队列、IPC 和响应处理的资源预算连接起来。

GPU 则通过 [矩阵与归约流水](hardware/gpu-kernel-pipelines.md) 把片上布局、矩阵指令、归约和缓冲完成关系连接起来；[CUDA 平台](hardware/cuda-execution.md) 中的交互服务算例将这些选择接回容量与 ITL。

## 在容量与传输约束下组织驻留

[HBM–DRAM–SSD 驻留](algorithms/tiered-residency.md) 解释对象版本、有效副本和迁移提交；[存储 I/O](hardware/storage-io.md) 解释真实读取路径。[KV 卸载](algorithms/kv-offload-prefetch.md) 比较恢复、持续读取与重算，[权重和专家卸载](algorithms/weight-expert-offload.md) 则比较层间重用、路由并集与热点变化。

在 [CPU–GPU 计算放置](algorithms/heterogeneous-execution.md) 中比较搬数据到 GPU 与把计算留在 CPU 的完整路径；[共享内存](hardware/shared-memory-paths.md) 进一步区分统一地址、可迁移驻留和物理共享。三级存储与异构计算各有可复算场景，容量可行、质量合格和 SLO 可达分别判断。

[KV 压缩](algorithms/kv-compression.md) 区分计算表示、历史裁剪与网络字节流。[分布式 KV 存储](system-semantics/distributed-kv-store.md) 建立目录、副本与恢复点，[传输与 RDMA](hardware/transport-and-rdma.md) 将注册、完成和 GPU 可见性连接到应用提交。[路由与迁移](algorithms/routing-and-migration.md) 比较命中与排队，[服务恢复](system-semantics/serving-recovery.md) 处理失败、重试与唯一提交权。

材料出现在不同上文或顺序中时，进入 [非前缀上下文复用](algorithms/context-reuse.md)，分别判断位置变换、完整前缀等价和选择性重算的质量条件；它与文档 embedding 缓存、普通材料缓存具有不同依赖。

## 组合额外能力

[量化](algorithms/quantization-contract.md) 改变表示和误差，[投机](algorithms/speculation-budget.md) 改变确认 token 的计算过程，[LoRA](algorithms/lora-serving.md) 改变权重依赖，[结构化生成](algorithms/structured-output.md) 改变合法输出分布。加入它们时，重新推导缓存身份、状态回滚、形状和成本。

[多样本与搜索分支](algorithms/branch-generation.md) 统一梳理父子状态、树形可见性和共同提交，同时保持 beam、多样本采样与投机的不同目标。完整长前缀与 P/D 场景分别见 [前缀缓存](algorithms/prefix-cache-identity.md) 和 [P/D 分离](algorithms/disaggregated-serving.md)。

[并行](algorithms/parallelism.md) 与 [P/D 分离](algorithms/disaggregated-serving.md) 改变资源归属；[分布式状态](system-semantics/distributed-state-transfer.md) 和 [集合顺序](system-semantics/collective-ordering.md) 说明组合协议。

## 组织多模型与平台依赖

[多模型放置](workloads/multi-model-placement.md) 将基座、adapter、执行缓冲与缓存的共享条件连接到驻留、分组和副本选择。[租户隔离](system-semantics/tenant-isolation.md) 将身份、缓存命名空间、配额、资源干扰和故障边界分别约束。[弹性服务](workloads/elastic-serving.md) 根据真实就绪容量与冷启动/排空时间响应突发，不能把已申请实例数直接当作服务能力。

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

## 其他相关机制的直接入口

以下是尚未在上文直接列出的专题，按既有类别补齐导航，不构成必读列表。完整问题仍可通过 [主题索引](INDEX.md) 检索。

| 类别 | 相关专题 |
|---|---|
| algorithms | [批量采样：概率变换、请求随机性与行映射](algorithms/batched-sampling.md) |
| engine-index | [vLLM V1：从 Token 进度到调度输出](engine-index/vllm-scheduler.md)、[vLLM KV 管理：命中、分配和延迟释放](engine-index/vllm-kv-manager.md)、[vLLM ModelRunner：调度视图到设备状态](engine-index/vllm-model-runner.md)、[SGLang：Radix 复用与调度状态的连接](engine-index/sglang-radix-scheduler.md)、[TensorRT-LLM：执行器、Overlap 与传输责任](engine-index/tensorrt-llm.md) |
| frameworks | [PyTorch 推理：模块模式、Autograd 与状态突变](frameworks/inference-eval.md)、[Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md)、[Triton Kernel：块程序、布局与融合边界](frameworks/triton-kernels.md) |
| hardware | [主机准备、DMA 与流水缓冲](hardware/host-device-transfer.md)、[数值格式、打包布局与矩阵指令](hardware/precision-and-kernel-support.md) |
| system-semantics | [主机与设备时间：偏序、区间并集和关键路径](system-semantics/host-device-timing.md)、[位置、Mask 与缓存：同一可见性关系的三种表示](system-semantics/position-mask-cache.md)、[工程认识如何成为可迁移的专家知识](system-semantics/evidence-and-independent-checks.md) |
| tools | [可观测性：执行事件与采集事件的映射](tools/trace-observability.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md)、[失败分析：最早偏离、最小复现与机制归因](tools/failure-evidence.md) |
| workloads | [性能实验：稳定态、对照与不确定性](workloads/measurement-comparability.md) |

[使用问题与探索边界](../docs/knowledge-use-cases.md) 说明场景索引、知识事实和任务要求的不同作用。

## 补充问题入口

- [K-ALG-001：连续批处理：从自回归依赖到迭代调度](algorithms/iteration-batching.md)：连续批处理将资源分配单位从整条请求改成一次可执行的 token 推进；批次是请求状态的临时视图。
- [K-ALG-002：分页 KV：逻辑位置、物理布局与共享所有权](algorithms/paged-kv-capacity.md)：分页把序列长度的动态增长转换为块映射变化；attention 语义仍由逻辑位置与有效长度决定。
- [K-ALG-003：前缀缓存：计算等价类、Radix 与淘汰收益](algorithms/prefix-cache-identity.md)：可复用前缀是能产生等价中间状态的一组输入和执行条件；字符串相同只是其中一个投影。
- [K-ALG-004：Attention 的 IO、在线 softmax 与分块归约](algorithms/attention-io.md)：精确 attention 可以通过改变数据流避免存储完整分数矩阵；算法复杂度、HBM 流量与数值归约顺序分别决定成本。
- [K-ALG-005：量化：误差传播、分组尺度与执行格式](algorithms/quantization-contract.md)：权重/激活/KV 的校准、缩放与执行格式共同决定量化收益；分别验证制品解释、真实 kernel 和任务质量。
- [K-ALG-006：投机解码：分布保持、状态提交与收益模型](algorithms/speculation-budget.md)：投机用便宜候选换取目标模型的并行验证；真正推进系统的是被确认的前缀，而非写入过缓存的全部 token。
- [K-ALG-007：分块 Prefill：服务时间切片与历史可见性](algorithms/chunked-prefill.md)：分块 prefill 将长输入的执行切成可调度时间片；每块仍读取其之前完整且有效的上下文。
- [K-ALG-008：异步调度：依赖图、代次与双缓冲](algorithms/async-scheduling.md)：异步调度只重叠彼此独立的工作；自回归 token、缓存映射和资源回收仍由明确依赖连接。
- [K-ALG-009：异构状态缓存：层组、恢复位置与联合分配](algorithms/heterogeneous-state-cache.md)：混合模型的缓存是多类状态的联合系统，统一的是逻辑恢复位置，物理块形状可以不同。
- [K-ALG-010：交叉注意力缓存：固定源、独立几何与可见性](algorithms/cross-attention-cache.md)：当 encoder 输出和投影权重固定时，cross-attention 的 K/V 与 decoder 步数无关，可以一次生成、多次读取。
- [K-ALG-011：批量采样：概率变换、请求随机性与行映射](algorithms/batched-sampling.md)：采样器消费每个请求自己的条件分布；向量化改变执行形态，不能改变参数、随机状态或输出归属。
- [K-ALG-012：结构化生成：语言状态、token 边界与约束采样](algorithms/structured-output.md)：结构化生成将语法状态与语言模型状态同步推进，合法 token 是从当前语法状态仍可完成目标语言的延伸。
- [K-ALG-013：Prefill/Decode 分离：状态传输与队列耦合](algorithms/disaggregated-serving.md)：分离式服务把两个阶段的资源策略解耦，同时引入 KV 传输、布局转换和跨节点状态提交。
- [K-ALG-014：多 LoRA 服务：低秩计算、驻留与缓存身份](algorithms/lora-serving.md)：LoRA 为共享基座增加低秩映射；多适配器服务将模型复用问题转换成按请求选择增量权重的问题。
- [K-ALG-015：推理并行：张量分解、流水、专家与上下文](algorithms/parallelism.md)：并行策略改变张量归属和同步边界；选择依据是每阶段的计算、状态与通信分解。
- [K-ENGINE-001：vLLM V1：从 Token 进度到调度输出](engine-index/vllm-scheduler.md)：vLLM V1 的调度可沿请求 token 进度理解，再追踪预算、缓存分配和执行结果怎样改变下一轮状态。
- [K-ENGINE-002：vLLM KV 管理：命中、分配和延迟释放](engine-index/vllm-kv-manager.md)：KVCacheManager 将逻辑 token 进度转换成层组块分配，命中长度、保留容量和可回收块各有独立含义。
- [K-ENGINE-003：vLLM ModelRunner：调度视图到设备状态](engine-index/vllm-model-runner.md)：ModelRunner 把逻辑调度结果转成设备输入与持久批状态，并将前向和采样结果按请求归还。
- [K-ENGINE-004：SGLang：Radix 复用与调度状态的连接](engine-index/sglang-radix-scheduler.md)：SGLang 将共享前缀组织成 Radix 结构，调度器围绕命中状态和可执行批次推进请求。
- [K-ENGINE-005：TensorRT-LLM：执行器、Overlap 与传输责任](engine-index/tensorrt-llm.md)：TensorRT-LLM 的执行器将请求调度、模型执行、采样和资源管理组合成后台迭代，适合研究异步服务的责任划分。
- [K-FW-001：PyTorch 推理：模块模式、Autograd 与状态突变](frameworks/inference-eval.md)：推理行为由模型模式、梯度记录、dtype/device、模块状态与执行路径共同决定。
- [K-FW-002：数值参考：误差尺度、条件数与首次分歧](frameworks/numerical-reference.md)：一个有解释力的数值参考保持任务函数和输入一致，再分层衡量实现重排与近似产生的差异。
- [K-FW-003：Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md)：stream 解决执行顺序，内存生存期解决地址何时可重用；正确的异步程序需要同时满足两种关系。
- [K-FW-004：编译与动态形状：Guard、分桶和变体成本](frameworks/compile-dynamic-shapes.md)：图获取、IR/lowering、布局和 guard 将函数映射到目标程序；编译缓存及在线变体必须保持副作用、容量与生命周期契约。
- [K-FW-005：CUDA Graph：捕获边界、计划状态与变长服务](frameworks/cuda-graph-buffers.md)：CUDA Graph 的固定地址、动态计划和在途内容所有权需共同验证；卸载、adapter 切换和图池竞争改变合法性与收益。
- [K-FW-006：Attention 执行接口：可见性、几何、状态布局与执行模式的交集](frameworks/attention-backends.md)：Attention 的合法性来自可见性、几何和状态契约；分块、分区、布局与准备复用改变并行度、驻留和成本频率。
- [K-FW-007：Triton Kernel：块程序、布局与融合边界](frameworks/triton-kernels.md)：块程序、布局与融合边界连接固定Triton实现；解释descriptor、log2统计、索引范围与调优成本的适用条件。
- [K-FW-008：MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md)：MLX 的数组表达式可以延迟执行，服务循环必须明确计算何时提交、何时完成和何时释放状态。
- [K-FW-009：Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md)：Neuron 服务通过形状桶、设备驻留状态和分布式执行组，把可变请求映射到编译后的程序族。
- [K-FW-010：NKI：Partition 轴、片上生存期与 Attention 数据流](frameworks/nki-tiling.md)：NKI 的 tile 是带硬件轴语义的片上对象；布局与生存期共同决定 DMA、矩阵和归约引擎的供给。
- [K-HW-001：容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md)：容量决定系统能容纳哪些状态，带宽与计算能力决定搬运和处理这些状态的下界，关键路径决定哪些下界实际显现。
- [K-HW-002：通信拓扑：从 Rank 组到瓶颈链路](hardware/topology-parallelism.md)：拓扑、方向与共享链路决定通信成本；将理想ring启动项和发送流量放到具体带宽假设中解释。
- [K-HW-003：主机准备、DMA 与流水缓冲](hardware/host-device-transfer.md)：主机准备、DMA、计算与消费按对象依赖连接；量化有限批、各池槽位与共享带宽，允许不同放置与传输方式。
- [K-HW-004：数值格式、打包布局与矩阵指令](hardware/precision-and-kernel-support.md)：固定架构/指令能力、量化参数布局与实际 dispatch 分别核验，防止把存储格式、指令支持和目标内核性能混为一谈。
- [K-HW-005：CUDA 平台：SM 数据流、占用率与执行粒度](hardware/cuda-execution.md)：CUDA 优化应把算法的数据复用映射到 SM 的寄存器、共享存储和 warp 执行，再判断主机提交是否成为瓶颈。
- [K-HW-006：ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md)：ROCm 移植应保持模型语义，同时按 AMD 的执行资源、矩阵指令和库 dispatch 重新推导数据布局与粒度。
- [K-HW-007：Apple Silicon：统一内存、带宽与交互式推理](hardware/metal-unified-memory.md)：统一内存减少显式设备间搬运，但 CPU/GPU 仍共享容量、带宽并通过依赖控制可见性。
- [K-HW-008：Trainium：HBM、SBUF、PSUM 与编译执行](hardware/trainium-dataflow.md)：Trainium 的高效执行来自把张量分块映射到片上存储和专用引擎，同时让静态可执行程序承接动态服务状态。
- [K-HW-009：CPU 推理：线程、缓存层级与 NUMA 分区](hardware/cpu-serving.md)：CPU 服务优化需要同时管理向量算术、内存带宽、线程调度与请求级并发，更多线程不等于更多有效工作。
- [K-MODEL-001：从模型计算图推导 Prefill、Decode 与持久状态](models/prefill-decode-state.md)：模型计算图和状态语义解释 token、位置、请求与执行阶段的依赖；参考边界帮助验证实现，不规定构建顺序。
- [K-MODEL-002：MHA、GQA 与 MQA：头映射和带宽复用](models/gqa-kv-heads.md)：GQA 让多个 query heads 读取同一组 K/V；逻辑共享可以减少状态，但是否减少实际访存取决于 kernel。
- [K-MODEL-003：MoE：路由、Dispatch/Combine 与专家负载](models/moe-routing.md)：MoE 的条件计算节省激活 FLOPs，但全部专家的驻留和路由后的不均衡决定真实服务成本。
- [K-MODEL-004：SSM 与混合模型：递归状态、扫描与恢复](models/recurrent-hybrid-state.md)：递归状态是前缀的压缩摘要；它支持低增长成本的 decode，却改变了前缀复用、回滚和批次管理方式。
- [K-MODEL-005：Encoder-decoder：源表示、两套 Attention 与流式边界](models/encoder-decoder-streaming.md)：Encoder-decoder 将源编码和目标生成分开；固定源复用、流式追加和目标自回归具有不同的依赖结构。
- [K-MODEL-006：MLA：压缩状态、矩阵吸收与解耦 RoPE](models/mla.md)：MLA 通过缓存低维 latent 改变 KV 的表示；真正节省 decode 流量需要执行内核沿压缩表示计算。
- [K-MODEL-007：窗口与稀疏 Attention：可见性、位置和状态保留](models/sliding-window-attention.md)：窗口限制的是每个 query 的直接可见历史；逻辑位置与跨层信息传播仍然持续演进。
- [K-MODEL-008：视觉语言模型：Patch、模态桥接与联合位置](models/vision-language.md)：VLM 的真实输入规模由处理器生成的视觉 token 决定，文本长度只描述其中一部分。
- [K-MODEL-009：图像扩散：迭代求解、条件复用与 CFG](models/image-diffusion.md)：扩散服务围绕不断变化的 latent 做多步求解；可复用对象由每步依赖决定，不能沿用自回归 KV 的全部规则。
- [K-MODEL-010：视频生成：时空 Token、全局耦合与分块解码](models/video-generation.md)：视频生成把时间加入 latent 几何，时间压缩、时空 attention 和输出解码共同决定可服务规模。
- [K-MODEL-011：语音生成：Codec Token、声学块与播放时钟](models/speech-generation.md)：语音服务需要把模型生成时钟转换为音频播放时钟，首包和稳定供给由多个阶段共同决定。
- [K-MODEL-012：全模态系统：多时钟、分支依赖与背压](models/omni-multimodal.md)：全模态推理是多个有依赖的计算分支组成的系统；文本、视频和音频不能压成同一个 token 长度。
- [K-SYS-001：主机与设备时间：偏序、区间并集和关键路径](system-semantics/host-device-timing.md)：服务延迟来自生产、提交、执行与消费之间的依赖路径；主机 API 区间与设备工作区间可以重叠。
- [K-SYS-002：请求身份：逻辑序列、批次视图与结果归属](system-semantics/request-batch-identity.md)：请求是长期语义对象，batch 行与物理槽是临时执行位置；正确性依赖它们之间可追溯的映射。
- [K-SYS-003：位置、Mask 与缓存：同一可见性关系的三种表示](system-semantics/position-mask-cache.md)：缓存中的内容必须对应模型定义的逻辑位置；mask 决定读取集合，物理布局只决定这些内容放在哪里。
- [K-SYS-004：取消与完成：业务终止、设备排空和资源回收](system-semantics/cancellation-resource-completion.md)：取消撤销未来服务意图，完成证明既有工作已结束；物理资源重用必须等待最后一个使用者。
- [K-SYS-005：集合通信：参与集合、操作顺序与失败传播](system-semantics/collective-ordering.md)：集合通信连接参与者、张量解释、count和完成关系；固定NCCL组提交语义及可辨识反例支持所选边界的验证。
- [K-SYS-006：工程认识如何成为可迁移的专家知识](system-semantics/evidence-and-independent-checks.md)：知识的增量是对系统机制与设计关系的新增解释；运行结果为这些解释提供事实、区分和修正。
- [K-SYS-007：输入输出语义：Tokenizer、Chat Template 与流式文本](system-semantics/tokenization-io.md)：模型处理的是规范化后的 token/特征序列，服务协议必须精确保存从用户输入到模型输入再到输出文本的映射。
- [K-SYS-008：服务协议：接收、流式背压与终止语义](system-semantics/service-protocol.md)：对外协议、发送缓冲、背压与终止语义依Task和具体接口解释；结果交付与资源完成分别核对。
- [K-SYS-009：分布式状态迁移：身份、传输与提交点](system-semantics/distributed-state-transfer.md)：状态迁移是一项所有权和可见性协议；字节到达只有在布局和版本被共同解释后才成为可用模型状态。
- [K-SYS-010：系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md)：在声明条件下推导模型、状态、资源与执行关系；场景不穷尽候选，不规定组件划分、研发顺序或任务外功能。
- [K-TOOLS-001：可观测性：执行事件与采集事件的映射](tools/trace-observability.md)：trace 是由采集范围和工具机制生成的观测，不是系统全部执行事实的无损副本。
- [K-TOOLS-002：时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)：时间线分析先建立关联和依赖，再用区间与关键路径解释延迟；热点累计时长只是入口。
- [K-TOOLS-003：Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)：硬件计数器帮助识别单 kernel 的资源限制，但它测到的执行可能与正常并发服务具有不同环境。
- [K-TOOLS-004：框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md)：框架 profiler 解释模型操作怎样落到设备工作，并通过 shape、调用栈与内存事件连接两层视图。
- [K-TOOLS-005：失败分析：最早偏离、最小复现与机制归因](tools/failure-evidence.md)：失败记录的价值在于还原系统从哪一个状态开始偏离预期，并区分输入、实现、资源和外部环境。
- [K-TOOLS-006：正确性 Oracle：函数参考、状态变形与服务契约](tools/correctness-oracles.md)：独立数值、身份与协议参考及负对照按Task和所选机制检查；缺失、重复或空集不能冒充完整结果。
- [K-WORK-001：服务指标：事件边界、Token 间隔与用户时间](workloads/metric-events.md)：一个延迟指标是两个明确事件的时间差；指标名字相同而事件定义不同，会导致不同结论。
- [K-WORK-002：到达过程与并发：排队、反馈和饱和点](workloads/arrival-and-concurrency.md)：负载生成器定义了系统受到的压力；闭环并发和独立到达率回答不同容量问题。
- [K-WORK-003：工作负载分布：长度、复用、模态与相关性](workloads/input-distribution.md)：系统成本由输入输出的联合分布决定；平均长度会丢失影响缓存、批次和尾部行为的结构。
- [K-WORK-004：性能实验：稳定态、对照与不确定性](workloads/measurement-comparability.md)：可比较的性能结果来自相同任务函数与负载条件，以及能够解释波动的测量窗口。
- [K-WORK-005：Goodput：联合 SLO、拒绝和容量前沿](workloads/slo-goodput.md)：服务能力是单位时间内完成有效任务且满足约束的数量，不能用总生成速度替代。
- [K-WORK-006：容量设计：驻留、服务率与阶段配比](workloads/capacity-planning.md)：容量设计把每请求资源需求和阶段服务时间连接到到达过程，得出实例数、并发和缓存预算。
- [K-FW-011：权重加载：从制品身份到可执行布局](frameworks/checkpoint-loading.md)：加载必须同时建立张量语义、执行布局、完整参数与就绪边界；流式流水的价值由启动关键路径和峰值副本决定。
- [K-ALG-016：基础算子：从张量契约到合法融合](algorithms/operator-primitives.md)：从函数、张量地址与别名推导合法执行；展开QKV、归一化、RoPE和融合成本，按所选机制检查数值与真实消费者。
- [K-HW-010：CPU GEMM/GEMV：寄存器、缓存与线程划分](hardware/cpu-linear-kernels.md)：CPU 线性层需把 shape 的重用机会落实到寄存器、cache/TLB、打包和线程分区；同一配置不适用于全部 GEMV/GEMM。
- [K-FW-012：CPU 低位执行：解包、补偿与累加路径](frameworks/cpu-quantized-kernels.md)：低位存储、解包和实际乘加是不同执行路径；补偿、组尺度、累加宽度和舍入共同决定正确性与收益。
- [K-SYS-012：内存分配：地址、容量与在途所有权](system-semantics/memory-allocators.md)：分配器同时管理地址、物理容量与异步所有权；比较连续池、块表和虚拟映射时必须计入碎片、工作区和在途页。
- [K-SYS-015：服务主机运行时：执行域、队列与缓冲所有权](system-semantics/host-runtime.md)：执行域、队列与缓冲的成本和完成关系；资源有界要求可由准入、额度或队列等不同机制实现。
- [K-WORK-010：多轮与分支负载：依赖、状态驻留和到达过程](workloads/session-workloads.md)：会话依赖、思考时间和分支决定真实到达与状态驻留；前缀复用收益需要和会话成功、容量及重用距离一起衡量。
- [K-TOOLS-007：性能实验：独立重复、效应量与尾部证据](tools/statistical-experiments.md)：重复层级、效应量和结论范围；可选知识条件实验区分交付失败、成本与删失，不向Agent规定构建流程。
- [K-HW-011：GPU 矩阵与归约流水：布局、资源和完成依赖](hardware/gpu-kernel-pipelines.md)：GPU tile、片上布局和异步流水共同受寄存器、共享存储及完成依赖约束；GEMM 与 attention 的最优组织随形状和架构改变。
- [K-ALG-017：多级驻留：有效副本、预取、淘汰与重算](algorithms/tiered-residency.md)：多级驻留维护有效版本与在途副本，并联合比较保留、预取、淘汰和重算；三级容量可行不等于在线延迟可行。
- [K-HW-012：存储 I/O：请求粒度、队列和加载流水](hardware/storage-io.md)：实际存储供给由布局、pagecache/直接 I/O、请求粒度、队列和 staging 共同决定；提交、可读、持久化和可回收分开建模。
- [K-ALG-018：KV 卸载与预取：读集合、恢复点与迁移预算](algorithms/kv-offload-prefetch.md)：KV 卸载按实际可见集合与共同恢复点组织；暂停恢复、前缀命中和活跃 decode 的持续传输需要不同预算。
- [K-ALG-019：权重与专家卸载：重用、路由与在途版本](algorithms/weight-expert-offload.md)：权重与专家卸载同时依赖批次重用、专家并集、热点和版本；按 token 稀疏不能直接推断 batch 读取稀疏。
- [K-ALG-020：CPU–GPU 计算放置：传数据、传结果与共同提交](algorithms/heterogeneous-execution.md)：CPU–GPU 分工比较冷权重搬运、CPU 计算与分片合并的完整关键路径；激活稀疏预测须有独立质量契约。
- [K-HW-013：共享内存路径：统一地址、物理驻留与显式迁移](hardware/shared-memory-paths.md)：统一地址、可迁移驻留和物理共享内存是不同访问关系；重用、页放大、可见性和带宽竞争共同决定复制选择。
- [K-ALG-021：KV 压缩：表示、近似保留与传输格式](algorithms/kv-compression.md)：KV 量化、历史裁剪与传输编码改变不同成本与语义；分组轴、残余区、元数据和质量必须分别计入。
- [K-SYS-013：分布式 KV 存储：目录、有效副本与可恢复状态](system-semantics/distributed-kv-store.md)：分布式 KV 目录只定位候选状态；有效副本、共同恢复点、目标可见性与排队/传输成本决定真实可复用性。
- [K-HW-014：通信传输：注册、完成、可见与回收](hardware/transport-and-rdma.md)：网络提交、CQE、消费者可见与应用接管是不同事件；注册、代次和完成协议共同决定缓冲何时可回收。
- [K-SYS-014：推理服务恢复：提交边界、执行组与重试身份](system-semantics/serving-recovery.md)：恢复需要明确计算、确认与外部提交边界，以及唯一提交权；可重算 KV 不等于可恢复整个请求或恰好一次交付。
- [K-ALG-023：请求路由与迁移：命中收益、排队和追赶条件](algorithms/routing-and-migration.md)：缓存亲和与动态迁移需要同时比较排队、状态增长、网络和提交责任；局部命中与低迁移停顿不保证整体收益。
- [K-ALG-024：多样本与搜索分支：父子状态、评分与共同提交](algorithms/branch-generation.md)：beam、多样本与投机树共享存储机制但优化目标不同；父子路径、logits 行、随机/语法状态与提交边界必须一致。
- [K-SYS-011：模型生命周期：加载、就绪、排空与版本切换](system-semantics/model-lifecycle.md)：模型就绪需要完整制品、执行路径和容量；请求绑定版本，新旧并存、排空和卸载由共同生命周期约束。
- [K-WORK-007：多模型放置：共同容量、混合执行与租户公平](workloads/multi-model-placement.md)：多模型与 LoRA 的共享条件决定共同容量、混合执行和公平等待；完整场景联合比较合并、共享及分组路径。
- [K-WORK-008：弹性服务：启动延迟、可用容量与排空](workloads/elastic-serving.md)：扩缩容要以实际就绪容量和启动/排空延迟解释队列；冷启动流量、故障余量与状态所有权决定策略是否可行。
- [K-WORK-009：服务成本与能耗：按合格结果计算资源代价](workloads/cost-and-energy.md)：成本与能耗以相同质量/SLO 的合格结果为分母，计入闲置、加载和失败，并区分计量域、积分与价格假设。
- [K-SYS-016：租户隔离：资源份额、缓存命名空间与故障边界](system-semantics/tenant-isolation.md)：租户身份、缓存权限、资源限额和故障域需要分别约束；固定文档区分 cgroup、time-slicing、MPS 与 MIG 资源契约。
- [K-WORK-011：集群放置：资源向量、拓扑、执行组与故障域](workloads/cluster-resource-placement.md)：集群放置先检查逐 rank 资源和拓扑，再比较装箱、局部性、共同启动与故障后容量；总空闲量不保证可部署。
- [K-SYS-018：推理存储服务：制品、缓存、索引与提交记录](system-semantics/storage-services.md)：权重、KV、索引和提交记录具有不同持久性；版本清单、对象可见性、分发与恢复竞争共同决定推理存储契约。
- [K-SYS-019：部署环境：版本契约、可见设备与真实资源边界](system-semantics/deployment-environment.md)：镜像、宿主驱动、设备身份与有效 quota 共同定义部署环境；就绪、资源峰值和代表路径验证才能支持服务可用性。
- [K-ALG-022：非前缀上下文复用：依赖、位置与选择性重算](algorithms/context-reuse.md)：非前缀文本重复不保证逐层状态相同；按上文、位置与可见性判断精确复用，并对选择性重算独立计量质量和加载成本。
- [K-MODEL-013：Embedding 与重排：编码函数、批量与评分契约](models/embedding-reranking.md)：编码器、pooling、normalize 与句对评分共同定义 embedding/重排函数；长度聚批、padding 与输入版本决定成本及结果。
- [K-SYS-017：向量检索与 RAG：索引、版本与端到端质量](system-semantics/retrieval-pipeline.md)：检索的距离契约、候选过滤和索引版本共同决定合法结果；Flat/IVF/图搜索与压缩需连接材料相关性和完整 RAG 时间预算。
- [K-MODEL-014：视觉识别与卷积：空间布局、预处理与任务输出](models/vision-inference.md)：视觉识别把图像变换、卷积布局、残差生命周期与分类/检测输出连接；按分辨率、工作区和几何契约验证完整请求。
- [K-MODEL-015：推荐推理：稀疏表、特征交互与请求批量](models/recommendation-inference.md)：推荐推理分别计量稀疏表 lookup、bag 聚合、特征交互与 dense 计算，并以特征/参数版本约束分片、缓存和更新。
- [K-MODEL-016：图模型推理：邻居依赖、采样与不规则执行](models/graph-inference.md)：图推理先展开邻居依赖再逐层聚合；采样、图/特征版本、去重语义和不规则搬运共同决定质量与资源上界。
- [K-TOOLS-008：GPU 从零表征：可持续带宽、矩阵吞吐与测量条件](tools/gpu-characterization.md)：用带条件的微基准建立 HBM/L2、矩阵、launch/图、PCIe、时钟功耗与多消费者参数包，并对接资源模型；峰值与异形态结果不能直接代入。
- [K-SYS-024：专用化杠杆：固定模型、负载与硬件时多出的自由度](system-semantics/specialization-levers.md)：模型、workload 或硬件冻结时，预转换、分桶、离线定参与单一路径等杠杆的收益来源、前提、失效条件与代价；非施工清单。
- [K-SYS-025：已知张力：单实例交互式服务中的机制冲突](system-semantics/known-tensions.md)：decode 带宽与 ITL、prefill/decode 干扰、投机与批、KV 量化悬崖、图/workspace 挤 KV、小批主机占比：机制、来源、可观测与反转。
