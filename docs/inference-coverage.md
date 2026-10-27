# 推理知识库的范围与组合验证

当前修订保持既有108篇；v2.2-r1深化11篇原专题，本轮再修正4篇的流程化表达，继续沿用模型、算法、硬件、框架、引擎索引、系统语义、工作负载和工具八类。这里按初始审计的 24 个能力域说明当前阅读入口、需要连接的机制和验证边界；集群放置、推理存储与部署环境作为三项跨域平台依赖单独展开。

知识库面向一个完整推理任务：从固定模型与输入，经过制品加载、算子/执行器、状态管理、调度和平台资源，直到对外输出、质量判断、故障恢复与成本评估。范围包含生成式和非生成式推理，以及支持这些推理服务的集群、存储和部署契约。训练数据构建、优化器/反向传播、完整训练平台生命周期不在本轮范围；硬件章节提供固定来源支持的机制与接口边界，不宣称穷尽全部设备、指令和产品版本。

资料版本上限为 **2026-04-30（含当日）**。章节编辑日期、归档日期和本轮审阅日期不改变来源资格，也不证明模型训练实际包含某项资料。具体运行还要遵循选定的知识条件、时间 profile、冻结版本及原始资料访问限制。本页是仓库全局范围说明，不能代替运行期选择策略。

正文、来源和场景承担不同职责：正文展开机制与设计推导；来源提供固定版本及其适用边界；场景用声明的假设连接多个专题并构造验证。seed 表示来源种子，不表示目标任务已有实测支持。篇数、来源数量、链接有效或算例可复算，也不能单独证明某个推理引擎已经达到性能目标。实际建设和交付状态见 [构建记录](BUILD-STATUS.md)，逐篇工作与场景登记见 [扩充计划](expansion-plan.json)。

## 本轮补充的归属

CPU–GPU知识此前已经由异构计算、主机设备流水、异步调度、stream/分配器与共享路径覆盖；本轮把有限批、缓冲槽位和共享带宽量化放回 [主机设备流水](../knowledge/hardware/host-device-transfer.md)。通信count/标识反例归 [collective](../knowledge/system-semantics/collective-ordering.md)，启动与带宽成本归 [拓扑](../knowledge/hardware/topology-parallelism.md)。算子地址和具体API增量分别归原算子、Attention、服务与主机专题。

使用场景是可选的问题入口，见 [使用问题与探索边界](knowledge-use-cases.md)。能力域和部署例子不穷尽合法方案，不产生固定组件、施工顺序或任务以外的验收要求。当前修订见 [变更清单](actionable-knowledge/change-manifest.json)，六篇历史撤回稿的归属见 [撤回登记](actionable-knowledge/withdrawn-chapters.json)。

## 24 个能力域的阅读与验证

表中的“组合验证”列出各机制的可检查关系；哪些关系适用取决于实际任务、所选机制和当前问题。它们不构成统一检查清单，也未在本页被声明为已完成实验。[完整索引](../knowledge/INDEX.md) 提供所有章节，[设计入口](../knowledge/DESIGN.md) 按主题导航，[诊断入口](../knowledge/DIAGNOSIS.md) 从实际行为反推机制。

| 能力域 | 主要入口 | 组合验证应回答的问题 |
|---|---|---|
| D01：任务、负载、SLO 与资源目标 | [指标事件](../knowledge/workloads/metric-events.md)、[到达与并发](../knowledge/workloads/arrival-and-concurrency.md)、[输入分布](../knowledge/workloads/input-distribution.md)、[Goodput](../knowledge/workloads/slo-goodput.md)、[成本与能耗](../knowledge/workloads/cost-and-energy.md) | 到达、长度、复用、取消和质量是否固定；吞吐、拒绝、尾延迟与合格结果成本是否使用一致的观察边界；多轮负载是否包含用户反馈与思考时间。 |
| D02：模型结构与输入输出语义 | [模型计算图](../knowledge/models/prefill-decode-state.md)、[Tokenizer/I/O](../knowledge/system-semantics/tokenization-io.md)、[位置与 mask](../knowledge/system-semantics/position-mask-cache.md)、[Embedding/重排](../knowledge/models/embedding-reranking.md) | 固定配置、权重、前处理与任务头能否还原完整函数；padding、位置、可见性及持久状态是否一致；生成和非生成输出是否各有参考。 |
| D03：模型制品、权重加载与冷启动 | [制品加载](../knowledge/frameworks/checkpoint-loading.md)、[模型生命周期](../knowledge/system-semantics/model-lifecycle.md)、[推理存储](../knowledge/system-semantics/storage-services.md) | 分片、别名、转换格式、编译/预热与就绪是否属于同一版本；源/目标副本和新旧实例并存峰值是否可容纳；缺片、取消或回退时谁回收资源。 |
| D04：基础算子与数值计算 | [算子原语](../knowledge/algorithms/operator-primitives.md)、[Attention IO](../knowledge/algorithms/attention-io.md)、[数值参考](../knowledge/frameworks/numerical-reference.md) | GEMM/GEMV、embedding、norm、RoPE、softmax 与索引是否遵守真实形状/布局；尾块、归约和融合能否对照独立函数定位首次分歧。 |
| D05：GPU 架构与内核实现 | [GPU 流水](../knowledge/hardware/gpu-kernel-pipelines.md)、[CUDA](../knowledge/hardware/cuda-execution.md)、[ROCm](../knowledge/hardware/rocm-execution.md)、[块程序与调优](../knowledge/frameworks/triton-kernels.md) | 所选 tile/流水的寄存器、shared/LDS、spill、并行度和搬运成本是什么；局部改善是否保留真实 batch 与服务路径收益。 |
| D06：CPU 架构、内核与线程运行时 | [CPU 服务](../knowledge/hardware/cpu-serving.md)、[CPU 线性层](../knowledge/hardware/cpu-linear-kernels.md)、[CPU 低位执行](../knowledge/frameworks/cpu-quantized-kernels.md)、[主机运行时](../knowledge/system-semantics/host-runtime.md) | 小 batch decode 与 prefill 的重用差异能否落实到打包、cache/TLB、NUMA 和线程；算子线程与分词/协议处理是否争用同一预算。 |
| D07：量化、压缩与近似计算 | [量化契约](../knowledge/algorithms/quantization-contract.md)、[数值格式与 kernel](../knowledge/hardware/precision-and-kernel-support.md)、[KV 压缩](../knowledge/algorithms/kv-compression.md) | 校准集、scale/zero point、实际 dispatch 与误差是否分别验证；节省的码字节、元数据、副本和转换是否一起计账；组合近似是否仍满足任务质量。 |
| D08：KV 与其他状态的分配、共享和回收 | [分页 KV](../knowledge/algorithms/paged-kv-capacity.md)、[异构状态](../knowledge/algorithms/heterogeneous-state-cache.md)、[分配器](../knowledge/system-semantics/memory-allocators.md)、[取消与完成](../knowledge/system-semantics/cancellation-resource-completion.md) | 跨页、共享分叉、递归状态、抢占和取消的恢复点是否共同合法；逻辑所有权、物理槽与在途引用是否守恒，何时允许复用。 |
| D09：HBM、DRAM、SSD 分层存储 | [分层驻留](../knowledge/algorithms/tiered-residency.md)、[KV 卸载](../knowledge/algorithms/kv-offload-prefetch.md)、[权重/专家卸载](../knowledge/algorithms/weight-expert-offload.md)、[存储 I/O](../knowledge/hardware/storage-io.md) | 保留、搬运和重算的完整成本如何比较；加载完成、消费者可见和状态发布如何区分；失败/部分完成及流水缓冲是否计入峰值。 |
| D10：CPU 与 GPU 协同计算 | [异构执行](../knowledge/algorithms/heterogeneous-execution.md)、[共享内存路径](../knowledge/hardware/shared-memory-paths.md)、[宿主/设备传输](../knowledge/hardware/host-device-transfer.md) | 计算按层、算子或专家放置时，输出如何合并；共享带宽、远端读取、缺页与同步是否抵消并行；激活稀疏等前提是否实际成立。 |
| D11：执行器、编译与异步运行时 | [编译与 lowering](../knowledge/frameworks/compile-dynamic-shapes.md)、[CUDA Graph](../knowledge/frameworks/cuda-graph-buffers.md)、[Stream/allocator](../knowledge/frameworks/streams-allocator.md)、[Attention backend](../knowledge/frameworks/attention-backends.md) | IR 改写、布局和 guard 保留了什么语义；编译、计划、图和缓存各自依赖什么；动态长度、adapter/权重切换、多 stream 与旧缓冲生命周期是否一致。 |
| D12：单实例调度与准入 | [连续批](../knowledge/algorithms/iteration-batching.md)、[分块 prefill](../knowledge/algorithms/chunked-prefill.md)、[容量设计](../knowledge/workloads/capacity-planning.md)、[到达过程](../knowledge/workloads/arrival-and-concurrency.md) | 在相同 offered workload 下，时间片、未知输出长度、抢占与优先级如何影响等待和容量；公平性与联合 SLO 是否随吞吐一起验证。 |
| D13：采样、投机与分支生成 | [批量采样](../knowledge/algorithms/batched-sampling.md)、[投机](../knowledge/algorithms/speculation-budget.md)、[结构化输出](../knowledge/algorithms/structured-output.md)、[分支生成](../knowledge/algorithms/branch-generation.md) | 贪心、分布保持和允许近似的目标是否明确；父子可见性、logits 行、随机/grammar 状态与 KV 的提交/回滚是否一致；草稿成本是否进入收益账。 |
| D14：MoE 执行与专家管理 | [MoE 路由](../knowledge/models/moe-routing.md)、[MLA](../knowledge/models/mla.md)、[专家卸载](../knowledge/algorithms/weight-expert-offload.md)、[并行](../knowledge/algorithms/parallelism.md) | token→专家→返回位置是否可追踪；batch 专家并集、路由偏斜、复制和分组 kernel 是否保留模型语义；慢专家与通信重叠如何影响尾部。 |
| D15：多设备并行与拓扑映射 | [并行组合](../knowledge/algorithms/parallelism.md)、[拓扑](../knowledge/hardware/topology-parallelism.md)、[GQA 头映射](../knowledge/models/gqa-kv-heads.md)、[集群放置](../knowledge/workloads/cluster-resource-placement.md) | TP/PP/DP/EP/CP 的逐 rank 张量与状态是否容纳；非整除划分、复制、通信和气泡是否计入；相同结果参考下何时改变并行组合。 |
| D16：网络传输与通信实现 | [传输/RDMA](../knowledge/hardware/transport-and-rdma.md)、[集合顺序](../knowledge/system-semantics/collective-ordering.md)、[分布式状态传输](../knowledge/system-semantics/distributed-state-transfer.md) | 注册、发送、CQE、GPU 可见与应用接管分别由谁确认；NIC 亲和、拥塞和错误完成如何影响缓冲回收及整个执行组。 |
| D17：P/D 分离、全局缓存与路由 | [P/D 分离](../knowledge/algorithms/disaggregated-serving.md)、[分布式 KV](../knowledge/system-semantics/distributed-kv-store.md)、[路由迁移](../knowledge/algorithms/routing-and-migration.md)、[非前缀复用](../knowledge/algorithms/context-reuse.md) | 本地/远端命中、排队、传输和重算是否用同一时间轴比较；热副本、迁移责任与接管失败如何处理；精确前缀与近似片段融合是否区分。 |
| D18：服务宿主、操作系统与 I/O | [主机运行时](../knowledge/system-semantics/host-runtime.md)、[服务协议](../knowledge/system-semantics/service-protocol.md)、[存储 I/O](../knowledge/hardware/storage-io.md)、[部署环境](../knowledge/system-semantics/deployment-environment.md) | 入口、tokenizer、IPC、模型提交与响应的队列/缓冲是否连通；quota、线程与异步 I/O 的限制是否可观察；背压是否传播到准入。 |
| D19：多模型、多租户和弹性部署 | [多模型放置](../knowledge/workloads/multi-model-placement.md)、[LoRA](../knowledge/algorithms/lora-serving.md)、[弹性](../knowledge/workloads/elastic-serving.md)、[租户隔离](../knowledge/system-semantics/tenant-isolation.md) | 哪些对象可共享、哪些状态必须隔离；异 rank 与冷 adapter 是否影响公平；扩缩容和滚动更新是否以真正就绪/排空为依据。 |
| D20：故障恢复、隔离与长期可靠性 | [服务恢复](../knowledge/system-semantics/serving-recovery.md)、[取消/完成](../knowledge/system-semantics/cancellation-resource-completion.md)、[租户隔离](../knowledge/system-semantics/tenant-isolation.md)、[失败证据](../knowledge/tools/failure-evidence.md) | rank、进程、网络或 I/O 失败后，哪些输出已提交、哪些状态可重建；重试与迟到完成是否保持唯一提交权；长期资源是否回到基线。 |
| D21：多模态、语音和扩散服务 | [流式 encoder-decoder](../knowledge/models/encoder-decoder-streaming.md)、[语音](../knowledge/models/speech-generation.md)、[VLM](../knowledge/models/vision-language.md)、[全模态](../knowledge/models/omni-multimodal.md)、[图像](../knowledge/models/image-diffusion.md)、[视频](../knowledge/models/video-generation.md) | 输入处理、专用模型阶段和交付时钟如何连接；chunk、lookahead、CFG、scheduler 与 VAE 状态是否正确；首包与完整输出质量是否联合约束。 |
| D22：Embedding、检索和其他推理负载 | [Embedding/重排](../knowledge/models/embedding-reranking.md)、[检索/RAG](../knowledge/system-semantics/retrieval-pipeline.md)、[CNN/检测](../knowledge/models/vision-inference.md)、[推荐](../knowledge/models/recommendation-inference.md)、[GNN](../knowledge/models/graph-inference.md) | 分别验证编码/排序、合法近邻与材料质量、图像几何、bag/交互和采样邻域；每类使用自己的计算链、状态、资源与质量指标。 |
| D23：观测、验证与实验统计 | [可观测性](../knowledge/tools/trace-observability.md)、[时间线](../knowledge/tools/timeline-attribution.md)、[计数器](../knowledge/tools/kernel-counter-replay.md)、[独立 oracle](../knowledge/tools/correctness-oracles.md)、[统计实验](../knowledge/tools/statistical-experiments.md) | 材料能否区分竞争解释；区组/配对、重复层级、自相关、尾部样本和搜索选择偏差是否处理；局部正确、机制解释、端到端收益与迁移性是否分开。 |
| D24：完整系统推导与设计探索 | [组合设计](../knowledge/system-semantics/design-synthesis.md)、[独立检查](../knowledge/system-semantics/evidence-and-independent-checks.md)、[容量规划](../knowledge/workloads/capacity-planning.md)、[场景入口](../knowledge/DESIGN.md) | 能否依据 workload/约束解释方案的合法性、成本预测和失败条件，并根据区分实验保留、修改或更换方案；证据是否只支持实际检查过的范围。 |

## 三项推理平台依赖

集群与存储平台需要进入知识库，因为它们决定设备和数据何时就绪、模型能否共同启动、状态能否迁移，以及失败后服务是否恢复。三项平台专题横跨前述能力域，不另建一套与推理脱节的产品分类。

| 平台专题 | 推理需要掌握的内容 | 需要连接与验证的边界 |
|---|---|---|
| [N33 / K-WORK-011：集群资源放置](../knowledge/workloads/cluster-resource-placement.md) | 逐 rank 资源向量、设备/NIC/NUMA 拓扑、共同启动、碎片、故障域与故障后容量 | 接 D15 并行、D19 多模型弹性和 D20 恢复；验证“总量足够但拓扑不可放置”、部分 rank 就绪、热点共置与单故障后的 SLO。不扩展为通用集群调度器的完整实现或所有调度产品手册。 |
| [N34 / K-SYS-018：推理存储服务](../knowledge/system-semantics/storage-services.md) | 权重制品、可重建 KV、检索索引与提交记录各自的持久性；对象/文件/块接口、完整版本发布、分发和恢复竞争 | 接 D03 加载、D09 分层 I/O、D17 缓存迁移、D22 检索；验证缺片、部分可见、旧引用、重复读取、冷启动风暴与恢复带宽。固定 Ceph 等资料用于机制依据，不替代任意存储系统的全部一致性保证。 |
| [N35 / K-SYS-019：部署环境](../knowledge/system-semantics/deployment-environment.md) | 镜像与宿主驱动相容、设备身份、有效 quota、亲和性、健康/就绪和更新 | 接 D06/D11 的执行路径、D18 主机与 D19 生命周期；验证同镜像异节点、设备可见但不可用、CPU 限流和新旧版本共存。范围是推理可执行环境的契约，不覆盖完整运维平台生命周期。 |

CPU、CUDA、ROCm、Metal 和 Trainium 章节保留各自执行模型；MLX、Neuron/NKI、Triton 与 attention backend 章节说明如何承接这些条件。使用新平台或来源截止日后的 API 时，需要建立新的版本依据和验证，不能把某个平台的内存可见性、指令形状或图行为自动套到其他平台。

## 从能力域到十二类完整场景

[设计入口的场景表](../knowledge/DESIGN.md) 连接 C01–C12：CPU、单 GPU、长前缀、三级存储、CPU–GPU 协同、多卡 MLA/MoE、P/D、LoRA、多模态/语音、扩散以及非生成推理。C12 内进一步区分 embedding/重排、检索/RAG、CNN/检测、推荐与 GNN 五份独立契约。场景覆盖是对机制能否组合的检查，不能由一个代表模型代替全部负载。

场景中的任务函数、状态、成本与反例展示了机制之间的联系。读者可以选取当前问题需要的关系，直接实现、构造局部实验或比较有实际意义的方案；候选数量、记录形式和研发顺序由任务与实际不确定性决定。已有设定算例用于检查量纲、数量和逻辑关系，实际设备的性能参数仍需测量。

实验后继续按现有知识构建逻辑处理：观察保留真实任务、版本与时间，运行期 learned 与预置 seed 分层；Evolution 提案、独立复核和发布规则负责决定哪些认识可写回。外部论文、官方文档与源码索引提供可追溯的依据，独立正文围绕推理问题组织；不要求使用知识库时先运行参考项目。

## 与原覆盖映射的关系

[原 COVERAGE.md](../knowledge/COVERAGE.md) 保留原 73 篇与固定参考集的映射事实。该表说明迁移和组织关系，本页说明本轮推理能力范围，两者不互相替代。当前章节清单以 [catalog](../knowledge/catalog.json) 和 [索引](../knowledge/INDEX.md) 为准，来源资格与原始材料见 [来源登记](../knowledge/SOURCES.md)；整库发布和目标任务收益应分别依据相应交付记录与实验，不能由本范围说明推定。
