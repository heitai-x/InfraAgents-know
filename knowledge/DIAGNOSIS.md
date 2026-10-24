# 从行为回到机制

本页按技术主题提供可选入口，没有固定阅读或研发顺序。场景与示例不穷尽候选；Task 的要求和所选机制的语义决定实际检查，原型、整体编译、联合设计和其他未列方案均可按条件探索。

诊断是对系统关系的反向推理。先把现象定位到输入、状态、执行、调度或输出阶段，再选择能区分解释的观测。

本表覆盖模型与 kernel，也包含多模型、检索及推理平台依赖。[范围说明](../docs/inference-coverage.md) 保留能力域与组合验证的对应；具体运行仍按选定知识条件、来源时间和访问权限读取。

| 实际行为 | 应还原的关系 | 专家正文 |
|---|---|---|
| 权重加载结束但服务仍不能接收请求 | 编译/预热、容量预留、健康与就绪条件 | [模型生命周期](system-semantics/model-lifecycle.md)、[部署环境](system-semantics/deployment-environment.md) |
| 切换模型后旧请求的结果或缓存混入新版本 | 请求绑定、旧版排空、执行缓冲与制品身份 | [模型生命周期](system-semantics/model-lifecycle.md)、[请求身份](system-semantics/request-batch-identity.md) |
| 多 adapter 合批后结果错误或小租户长期等待 | rank/scale、分组映射、共同容量和公平性 | [多模型放置](workloads/multi-model-placement.md)、[LoRA](algorithms/lora-serving.md) |
| 扩容数量增加而队列继续变长 | 实际就绪时间、制品分发竞争与可用服务率 | [弹性服务](workloads/elastic-serving.md)、[推理存储](system-semantics/storage-services.md) |
| 配额相同但租户互相干扰或状态越界 | 共享机制的资源边界、缓存命名空间和故障域 | [租户隔离](system-semantics/tenant-isolation.md) |
| 集群总空闲资源充足但模型执行组无法启动 | 逐 rank 容量、拓扑、共同启动与碎片 | [集群放置](workloads/cluster-resource-placement.md) |
| 文件存在却读到不完整或不一致的模型/索引 | 分片完整性、版本发布点、对象可见性和读取快照 | [推理存储](system-semantics/storage-services.md)、[权重加载](frameworks/checkpoint-loading.md) |
| 同一镜像在另一节点性能或可用性不同 | 驱动/用户态相容、设备映射、NUMA 与实际 quota | [部署环境](system-semantics/deployment-environment.md) |
| 每请求成本或能耗下降但合格结果也减少 | 计量域、闲置/失败成本与质量/SLO 分母 | [成本与能耗](workloads/cost-and-energy.md)、[Goodput](workloads/slo-goodput.md) |
| 材料文本相同，换顺序复用 KV 后答案改变 | 上文依赖、位置变换和选择性重算质量 | [非前缀复用](algorithms/context-reuse.md) |
| 单条 embedding 正确，合批后向量或分数变化 | padding/packing 可见性、pooling、截断与候选身份 | [Embedding 与重排](models/embedding-reranking.md) |
| 检索 recall 尚可但过滤后为空或 RAG 答案变差 | 合法候选集合、向量/材料版本、重排与 prompt 组装 | [检索与 RAG](system-semantics/retrieval-pipeline.md) |
| 分类正常但检测框偏移，或换布局后质量下降 | resize/crop 坐标、stride、残差与后处理顺序 | [视觉识别与检测](models/vision-inference.md) |
| 推荐 lookup 看似正确，最终评分异常 | 特征映射、重复 ID、bag 归约、交互列顺序与参数版本 | [推荐推理](models/recommendation-inference.md) |
| GNN 同样多目标节点，耗时与输出波动很大 | 实际边/唯一节点、采样子图、特征 miss 和图版本 | [图模型](models/graph-inference.md) |
| KV 文件明显变小但显存或速度未改善 | 元数据、残余区、完整解码副本与实际消费路径 | [KV 压缩](algorithms/kv-compression.md) |
| 远端缓存命中却比重算慢 | 目录、源队列、传输、转换和目标容量 | [分布式 KV](system-semantics/distributed-kv-store.md)、[路由](algorithms/routing-and-migration.md) |
| NIC 完成后 GPU 偶发读到旧数据 | 注册代次、GPU 可见性与消费者提交顺序 | [传输与 RDMA](hardware/transport-and-rdma.md) |
| 断线重试后重复或遗漏输出 | 计算、确认、外部提交边界与尝试身份 | [恢复](system-semantics/serving-recovery.md) |
| 树验证结果不同于独立路径 | 祖先可见性、logits 行、RNG/grammar 与回滚 | [分支生成](algorithms/branch-generation.md) |
| GPU 增加流水级数反而变慢 | 寄存器/共享存储、spill、同步与实际重叠 | [GPU 流水](hardware/gpu-kernel-pipelines.md) |
| 显存释放成功但 decode 间隔很长 | 每步实际读取的 KV/权重及链路下界 | [多级驻留](algorithms/tiered-residency.md)、[KV 卸载](algorithms/kv-offload-prefetch.md) |
| SSD 标称很快但模型供给不足 | 请求大小、IOPS、排队、冷热页缓存与 staging | [存储 I/O](hardware/storage-io.md) |
| MoE 每 token 稀疏但仍频繁加载 | batch 的专家并集、热点漂移与无效预取 | [专家卸载](algorithms/weight-expert-offload.md) |
| CPU–GPU 都在忙但总延迟没有降低 | 分片合并、共享带宽、同步与冷部分工作量 | [异构执行](algorithms/heterogeneous-execution.md) |
| 没有显式复制仍出现传输或等待 | 统一地址与物理位置、缺页迁移、远端读取和求值完成 | [共享内存路径](hardware/shared-memory-paths.md) |
| 权重大小能容纳但加载时 OOM | 源/目标副本、转换区、预热与新旧版本共存的峰值 | [加载](frameworks/checkpoint-loading.md)、[分配](system-semantics/memory-allocators.md) |
| CPU 小 batch 快而 prefill 退化 | 形状重用、打包摊销、cache/TLB 与线程划分 | [CPU 线性层](hardware/cpu-linear-kernels.md)、[低位执行](frameworks/cpu-quantized-kernels.md) |
| 低位内核结果有系统偏差 | 组尺度、零点补偿、窄累加和负数舍入 | [低位执行](frameworks/cpu-quantized-kernels.md)、[算子](algorithms/operator-primitives.md) |
| 模型空闲但请求仍堆积 | 输入处理、单执行域饱和、队列与背压 | [主机运行时](system-semantics/host-runtime.md) |
| 多轮压测越慢到达越少 | 会话依赖和闭环用户反馈 | [会话负载](workloads/session-workloads.md)、[到达](workloads/arrival-and-concurrency.md) |
| 多次试验挑出提升但复测消失 | 独立重复层级、环境漂移和选择偏差 | [实验统计](tools/statistical-experiments.md) |
| 单请求正确，混合批错误 | batch 视图与持久请求状态的逆映射 | [身份](system-semantics/request-batch-identity.md)、[采样](algorithms/batched-sampling.md) |
| 前缀命中后结果不同 | 状态的计算等价类、共同恢复位置 | [前缀](algorithms/prefix-cache-identity.md)、[异构状态](algorithms/heterogeneous-state-cache.md) |
| 分块后 logits 分歧 | query 的绝对位置与历史可见集合 | [位置/mask](system-semantics/position-mask-cache.md)、[分块](algorithms/chunked-prefill.md) |
| 图重放或取消后串请求 | 地址稳定性、内容代次和最后使用者 | [图执行](frameworks/cuda-graph-buffers.md)、[生命周期](system-semantics/cancellation-resource-completion.md) |
| decode 随源长异常增长 | encoder、cross 投影与读源 attention 的成本分解 | [Cross KV](algorithms/cross-attention-cache.md) |
| MLA 容量小但带宽未降 | 是否仍把全部历史 latent 展开 | [MLA](models/mla.md) |
| MoE 某些 batch 特别慢 | 专家负载分布、dispatch 流量和慢端 | [MoE](models/moe-routing.md)、[拓扑](hardware/topology-parallelism.md) |
| 首 token 快但输出经常停顿 | 迭代时间片、排队与客户端缓冲 | [连续批](algorithms/iteration-batching.md)、[指标](workloads/metric-events.md) |
| CPU 忙而 GPU 有空洞 | 准备、提交、隐式同步和可重叠区间 | [异步调度](algorithms/async-scheduling.md)、[数据流水](hardware/host-device-transfer.md) |
| 某些长度第一次特别慢 | 编译 guard、图桶、autotune 与冷缓存 | [编译](frameworks/compile-dynamic-shapes.md)、[实验](workloads/measurement-comparability.md) |
| 吞吐提高但用户失败增加 | offered/admitted/completed 与联合 SLO | [Goodput](workloads/slo-goodput.md)、[到达](workloads/arrival-and-concurrency.md) |
| 音频平均足够快仍卡顿 | 生成时钟、解码 lookahead 和播放缓冲 | [语音](models/speech-generation.md)、[全模态](models/omni-multimodal.md) |
| 流式转写重复撤回文字或边界识别下降 | 源 lookahead、chunk 可见性和稳定输出提交 | [Encoder-decoder 流式](models/encoder-decoder-streaming.md) |
| VLM 复用后对应错图，或音视频输出错位 | 媒体处理/视觉特征版本、模态块身份与多时钟依赖 | [VLM](models/vision-language.md)、[全模态](models/omni-multimodal.md) |
| 去噪 kernel 更快，完整图像生成却未改善 | scheduler 步数、CFG、跨步复用、VAE 与交付边界 | [图像扩散](models/image-diffusion.md) |
| 视频分块后出现时间接缝或显存仍持续增长 | 全局时空耦合、重叠区、VAE 状态及在途输出 | [视频生成](models/video-generation.md) |
| 分布式卡住，最后只见超时 | rank 共同视图、操作顺序和首个失败 | [集合通信](system-semantics/collective-ordering.md)、[失败分析](tools/failure-evidence.md) |

[可观测性](tools/trace-observability.md) 说明当前材料能回答什么，[时间关系](system-semantics/host-device-timing.md) 说明累计时间为何不等于延迟。新的解释最终回到原机制章节，保持知识按系统关系组织。

## 用完整场景复查解释

局部解释需要放回整个请求路径。场景包含设定算例和验证设计，不表示目标任务已经实测通过；完整目录见 [设计入口](DESIGN.md)。

- C01 [CPU 多请求](system-semantics/design-synthesis.md) 与 C02 [单 GPU 交互](hardware/cuda-execution.md)，检查算子变快后是否把瓶颈转移给线程、图池、调度或输出。
- C03 [长前缀复用](algorithms/prefix-cache-identity.md)、C04 [三级存储](algorithms/tiered-residency.md) 与 C05 [CPU–GPU 协同](algorithms/heterogeneous-execution.md)，检查命中、搬运和重算是否计入同一份时间及容量账。
- C06 [多卡 MLA/MoE](algorithms/parallelism.md)、C07 [P/D 接管](algorithms/disaggregated-serving.md) 与 C08 [多模型/LoRA](workloads/multi-model-placement.md)，检查张量归属、版本、慢端、资源共享和公平等待。
- C09 [流式语音](models/speech-generation.md)、C10 [VLM/全模态](models/omni-multimodal.md) 与 C11 [图像/视频扩散](models/video-generation.md)，检查阶段依赖、实时提交与最终完整输出的质量。
- C12 分别复查 [Embedding/重排](models/embedding-reranking.md)、[检索/RAG](system-semantics/retrieval-pipeline.md)、[CNN/检测](models/vision-inference.md)、[推荐](models/recommendation-inference.md) 和 [GNN](models/graph-inference.md)；每类都有自己的参考函数与质量指标。

## 其他相关机制的直接入口

以下是尚未在上文直接列出的专题，按既有类别补齐导航，不构成必读列表。完整问题仍可通过 [主题索引](INDEX.md) 检索。

| 类别 | 相关专题 |
|---|---|
| algorithms | [分页 KV：逻辑位置、物理布局与共享所有权](algorithms/paged-kv-capacity.md)、[Attention 的 IO、在线 softmax 与分块归约](algorithms/attention-io.md)、[量化：误差传播、分组尺度与执行格式](algorithms/quantization-contract.md)、[投机解码：分布保持、状态提交与收益模型](algorithms/speculation-budget.md)、[结构化生成：语言状态、token 边界与约束采样](algorithms/structured-output.md) |
| engine-index | [vLLM V1：从 Token 进度到调度输出](engine-index/vllm-scheduler.md)、[vLLM KV 管理：命中、分配和延迟释放](engine-index/vllm-kv-manager.md)、[vLLM ModelRunner：调度视图到设备状态](engine-index/vllm-model-runner.md)、[SGLang：Radix 复用与调度状态的连接](engine-index/sglang-radix-scheduler.md)、[TensorRT-LLM：执行器、Overlap 与传输责任](engine-index/tensorrt-llm.md) |
| frameworks | [PyTorch 推理：模块模式、Autograd 与状态突变](frameworks/inference-eval.md)、[数值参考：误差尺度、条件数与首次分歧](frameworks/numerical-reference.md)、[Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md)、[Attention Backend：语义、布局与执行契约的交集](frameworks/attention-backends.md)、[Triton Kernel：块程序、布局与融合边界](frameworks/triton-kernels.md)、[MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md)、[Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md)、[NKI：Partition 轴、片上生存期与 Attention 数据流](frameworks/nki-tiling.md) |
| hardware | [容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md)、[数值格式、打包布局与矩阵指令](hardware/precision-and-kernel-support.md)、[ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md)、[Apple Silicon：统一内存、带宽与交互式推理](hardware/metal-unified-memory.md)、[Trainium：HBM、SBUF、PSUM 与编译执行](hardware/trainium-dataflow.md)、[CPU 推理：线程、缓存层级与 NUMA 分区](hardware/cpu-serving.md) |
| models | [从模型计算图推导 Prefill、Decode 与持久状态](models/prefill-decode-state.md)、[MHA、GQA 与 MQA：头映射和带宽复用](models/gqa-kv-heads.md)、[SSM 与混合模型：递归状态、扫描与恢复](models/recurrent-hybrid-state.md)、[窗口与稀疏 Attention：可见性、位置和状态保留](models/sliding-window-attention.md) |
| system-semantics | [工程认识如何成为可迁移的专家知识](system-semantics/evidence-and-independent-checks.md)、[输入输出语义：Tokenizer、Chat Template 与流式文本](system-semantics/tokenization-io.md)、[服务协议：接收、流式背压与终止语义](system-semantics/service-protocol.md)、[分布式状态迁移：身份、传输与提交点](system-semantics/distributed-state-transfer.md) |
| tools | [时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)、[Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md)、[正确性 Oracle：函数参考、状态变形与服务契约](tools/correctness-oracles.md) |
| workloads | [工作负载分布：长度、复用、模态与相关性](workloads/input-distribution.md)、[容量设计：驻留、服务率与阶段配比](workloads/capacity-planning.md) |

[使用问题与探索边界](../docs/knowledge-use-cases.md) 说明场景索引、知识事实和任务要求的不同作用。
