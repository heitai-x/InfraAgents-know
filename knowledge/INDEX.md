# 知识索引

按任务条件检索；seed 是一手资料种子，supported 是条件化运行支持。候选、冲突和弃用内容须显式区分。

[设计入口](DESIGN.md) · [诊断入口](DIAGNOSIS.md) · [覆盖矩阵](COVERAGE.md) · [来源登记](SOURCES.md)

## 算法

[分类入口](algorithms/README.md)，共 16 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-ALG-001：连续批处理：从自回归依赖到迭代调度](algorithms/iteration-batching.md) | v1 · seed | 连续批处理将资源分配单位从整条请求改成一次可执行的 token 推进；批次是请求状态的临时视图。 |
| [K-ALG-002：分页 KV：逻辑位置、物理布局与共享所有权](algorithms/paged-kv-capacity.md) | v1 · seed | 分页把序列长度的动态增长转换为块映射变化；attention 语义仍由逻辑位置与有效长度决定。 |
| [K-ALG-003：前缀缓存：计算等价类、Radix 与淘汰收益](algorithms/prefix-cache-identity.md) | v1 · seed | 可复用前缀是能产生等价中间状态的一组输入和执行条件；字符串相同只是其中一个投影。 |
| [K-ALG-004：Attention 的 IO、在线 softmax 与分块归约](algorithms/attention-io.md) | v1 · seed | 精确 attention 可以通过改变数据流避免存储完整分数矩阵；算法复杂度、HBM 流量与数值归约顺序分别决定成本。 |
| [K-ALG-005：量化：误差传播、分组尺度与执行格式](algorithms/quantization-contract.md) | v1 · seed | 量化把数值表示、数据布局和执行内核一起改变；收益由减少的字节与新增的变换成本共同决定。 |
| [K-ALG-006：投机解码：分布保持、状态提交与收益模型](algorithms/speculation-budget.md) | v1 · seed | 投机用便宜候选换取目标模型的并行验证；真正推进系统的是被确认的前缀，而非写入过缓存的全部 token。 |
| [K-ALG-007：分块 Prefill：服务时间切片与历史可见性](algorithms/chunked-prefill.md) | v1 · seed | 分块 prefill 将长输入的执行切成可调度时间片；每块仍读取其之前完整且有效的上下文。 |
| [K-ALG-008：异步调度：依赖图、代次与双缓冲](algorithms/async-scheduling.md) | v1 · seed | 异步调度只重叠彼此独立的工作；自回归 token、缓存映射和资源回收仍由明确依赖连接。 |
| [K-ALG-009：异构状态缓存：层组、恢复位置与联合分配](algorithms/heterogeneous-state-cache.md) | v1 · seed | 混合模型的缓存是多类状态的联合系统，统一的是逻辑恢复位置，物理块形状可以不同。 |
| [K-ALG-010：交叉注意力缓存：固定源、独立几何与可见性](algorithms/cross-attention-cache.md) | v1 · seed | 当 encoder 输出和投影权重固定时，cross-attention 的 K/V 与 decoder 步数无关，可以一次生成、多次读取。 |
| [K-ALG-011：批量采样：概率变换、请求随机性与行映射](algorithms/batched-sampling.md) | v1 · seed | 采样器消费每个请求自己的条件分布；向量化改变执行形态，不能改变参数、随机状态或输出归属。 |
| [K-ALG-012：结构化生成：语言状态、token 边界与约束采样](algorithms/structured-output.md) | v1 · seed | 结构化生成将语法状态与语言模型状态同步推进，合法 token 是从当前语法状态仍可完成目标语言的延伸。 |
| [K-ALG-013：Prefill/Decode 分离：状态传输与队列耦合](algorithms/disaggregated-serving.md) | v1 · seed | 分离式服务把两个阶段的资源策略解耦，同时引入 KV 传输、布局转换和跨节点状态提交。 |
| [K-ALG-014：多 LoRA 服务：低秩计算、驻留与缓存身份](algorithms/lora-serving.md) | v1 · seed | LoRA 为共享基座增加低秩映射；多适配器服务将模型复用问题转换成按请求选择增量权重的问题。 |
| [K-ALG-015：推理并行：张量分解、流水、专家与上下文](algorithms/parallelism.md) | v1 · seed | 并行策略改变张量归属和同步边界；选择依据是每阶段的计算、状态与通信分解。 |
| [K-ALG-016：基础算子：从张量契约到合法融合](algorithms/operator-primitives.md) | v1 · seed | 算子契约包含轴、布局、数值和别名；合法融合需要保持函数与状态关系，再比较字节、启动和资源占用。 |

## 模型

[分类入口](models/README.md)，共 12 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-MODEL-001：从模型计算图推导 Prefill、Decode 与持久状态](models/prefill-decode-state.md) | v1 · seed | 模型接入首先是一项计算图和状态语义工作：识别哪些量随 token、位置、请求和阶段变化，再选择执行机制。 |
| [K-MODEL-002：MHA、GQA 与 MQA：头映射和带宽复用](models/gqa-kv-heads.md) | v1 · seed | GQA 让多个 query heads 读取同一组 K/V；逻辑共享可以减少状态，但是否减少实际访存取决于 kernel。 |
| [K-MODEL-003：MoE：路由、Dispatch/Combine 与专家负载](models/moe-routing.md) | v1 · seed | MoE 的条件计算节省激活 FLOPs，但全部专家的驻留和路由后的不均衡决定真实服务成本。 |
| [K-MODEL-004：SSM 与混合模型：递归状态、扫描与恢复](models/recurrent-hybrid-state.md) | v1 · seed | 递归状态是前缀的压缩摘要；它支持低增长成本的 decode，却改变了前缀复用、回滚和批次管理方式。 |
| [K-MODEL-005：Encoder-decoder：源表示、两套 Attention 与流式边界](models/encoder-decoder-streaming.md) | v1 · seed | Encoder-decoder 将源编码和目标生成分开；固定源复用、流式追加和目标自回归具有不同的依赖结构。 |
| [K-MODEL-006：MLA：压缩状态、矩阵吸收与解耦 RoPE](models/mla.md) | v1 · seed | MLA 通过缓存低维 latent 改变 KV 的表示；真正节省 decode 流量需要执行内核沿压缩表示计算。 |
| [K-MODEL-007：窗口与稀疏 Attention：可见性、位置和状态保留](models/sliding-window-attention.md) | v1 · seed | 窗口限制的是每个 query 的直接可见历史；逻辑位置与跨层信息传播仍然持续演进。 |
| [K-MODEL-008：视觉语言模型：Patch、模态桥接与联合位置](models/vision-language.md) | v1 · seed | VLM 的真实输入规模由处理器生成的视觉 token 决定，文本长度只描述其中一部分。 |
| [K-MODEL-009：图像扩散：迭代求解、条件复用与 CFG](models/image-diffusion.md) | v1 · seed | 扩散服务围绕不断变化的 latent 做多步求解；可复用对象由每步依赖决定，不能沿用自回归 KV 的全部规则。 |
| [K-MODEL-010：视频生成：时空 Token、全局耦合与分块解码](models/video-generation.md) | v1 · seed | 视频生成把时间加入 latent 几何，时间压缩、时空 attention 和输出解码共同决定可服务规模。 |
| [K-MODEL-011：语音生成：Codec Token、声学块与播放时钟](models/speech-generation.md) | v1 · seed | 语音服务需要把模型生成时钟转换为音频播放时钟，首包和稳定供给由多个阶段共同决定。 |
| [K-MODEL-012：全模态系统：多时钟、分支依赖与背压](models/omni-multimodal.md) | v1 · seed | 全模态推理是多个有依赖的计算分支组成的系统；文本、视频和音频不能压成同一个 token 长度。 |

## 硬件平台

[分类入口](hardware/README.md)，共 10 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-HW-001：容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md) | v1 · seed | 容量决定系统能容纳哪些状态，带宽与计算能力决定搬运和处理这些状态的下界，关键路径决定哪些下界实际显现。 |
| [K-HW-002：通信拓扑：从 Rank 组到瓶颈链路](hardware/topology-parallelism.md) | v1 · seed | 多卡性能由数据经过的实际路径决定；rank 数只是拓扑模型的一部分。 |
| [K-HW-003：主机准备、DMA 与流水缓冲](hardware/host-device-transfer.md) | v1 · seed | 输入流水的吞吐取决于准备、传输和计算三个阶段的平衡；数据可见性和缓冲所有权决定能否正确重叠。 |
| [K-HW-004：数值格式、打包布局与矩阵指令](hardware/precision-and-kernel-support.md) | v1 · seed | 硬件支持某种 dtype、库能接受该 dtype、目标形状有高效 kernel，是三个不同层次的能力。 |
| [K-HW-005：CUDA 平台：SM 数据流、占用率与执行粒度](hardware/cuda-execution.md) | v1 · seed | CUDA 优化应把算法的数据复用映射到 SM 的寄存器、共享存储和 warp 执行，再判断主机提交是否成为瓶颈。 |
| [K-HW-006：ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md) | v1 · seed | ROCm 移植应保持模型语义，同时按 AMD 的执行资源、矩阵指令和库 dispatch 重新推导数据布局与粒度。 |
| [K-HW-007：Apple Silicon：统一内存、带宽与交互式推理](hardware/metal-unified-memory.md) | v1 · seed | 统一内存减少显式设备间搬运，但 CPU/GPU 仍共享容量、带宽并通过依赖控制可见性。 |
| [K-HW-008：Trainium：HBM、SBUF、PSUM 与编译执行](hardware/trainium-dataflow.md) | v1 · seed | Trainium 的高效执行来自把张量分块映射到片上存储和专用引擎，同时让静态可执行程序承接动态服务状态。 |
| [K-HW-009：CPU 推理：线程、缓存层级与 NUMA 分区](hardware/cpu-serving.md) | v1 · seed | CPU 服务优化需要同时管理向量算术、内存带宽、线程调度与请求级并发，更多线程不等于更多有效工作。 |
| [K-HW-010：CPU GEMM/GEMV：寄存器、缓存与线程划分](hardware/cpu-linear-kernels.md) | v1 · seed | CPU 线性层需把 shape 的重用机会落实到寄存器、cache/TLB、打包和线程分区；同一配置不适用于全部 GEMV/GEMM。 |

## 框架

[分类入口](frameworks/README.md)，共 12 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-FW-001：PyTorch 推理：模块模式、Autograd 与状态突变](frameworks/inference-eval.md) | v1 · seed | 推理行为由模型模式、梯度记录、dtype/device、模块状态与执行路径共同决定。 |
| [K-FW-002：数值参考：误差尺度、条件数与首次分歧](frameworks/numerical-reference.md) | v1 · seed | 一个有解释力的数值参考保持任务函数和输入一致，再分层衡量实现重排与近似产生的差异。 |
| [K-FW-003：Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md) | v1 · seed | stream 解决执行顺序，内存生存期解决地址何时可重用；正确的异步程序需要同时满足两种关系。 |
| [K-FW-004：编译与动态形状：Guard、分桶和变体成本](frameworks/compile-dynamic-shapes.md) | v1 · seed | 编译把一组可证明成立的执行假设变成专门程序；动态服务需要管理这些假设覆盖的形状与状态空间。 |
| [K-FW-005：CUDA Graph：捕获边界、计划状态与变长服务](frameworks/cuda-graph-buffers.md) | v1 · seed | CUDA Graph 重放一组已捕获的设备工作；服务动态性通过稳定缓冲中的值、分桶和捕获外控制面承接。 |
| [K-FW-006：Attention Backend：语义、布局与执行契约的交集](frameworks/attention-backends.md) | v1 · seed | 选择 attention backend 是匹配可见性、张量几何、状态布局和执行模式，性能排名只对确定的运行点成立。 |
| [K-FW-007：Triton Kernel：块程序、布局与融合边界](frameworks/triton-kernels.md) | v1 · seed | Triton 用块级程序描述张量工作，性能来自把复用、归约和线程布局映射到目标 GPU 资源。 |
| [K-FW-008：MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md) | v1 · seed | MLX 的数组表达式可以延迟执行，服务循环必须明确计算何时提交、何时完成和何时释放状态。 |
| [K-FW-009：Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md) | v1 · seed | Neuron 服务通过形状桶、设备驻留状态和分布式执行组，把可变请求映射到编译后的程序族。 |
| [K-FW-010：NKI：Partition 轴、片上生存期与 Attention 数据流](frameworks/nki-tiling.md) | v1 · seed | NKI 的 tile 是带硬件轴语义的片上对象；布局与生存期共同决定 DMA、矩阵和归约引擎的供给。 |
| [K-FW-011：权重加载：从制品身份到可执行布局](frameworks/checkpoint-loading.md) | v1 · seed | 加载必须同时建立张量语义、执行布局、完整参数与就绪边界；流式流水的价值由启动关键路径和峰值副本决定。 |
| [K-FW-012：CPU 低位执行：解包、补偿与累加路径](frameworks/cpu-quantized-kernels.md) | v1 · seed | 低位存储、解包和实际乘加是不同执行路径；补偿、组尺度、累加宽度和舍入共同决定正确性与收益。 |

## 引擎源码索引

[分类入口](engine-index/README.md)，共 5 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-ENGINE-001：vLLM V1：从 Token 进度到调度输出](engine-index/vllm-scheduler.md) | v1 · seed | vLLM V1 的调度可沿请求 token 进度理解，再追踪预算、缓存分配和执行结果怎样改变下一轮状态。 |
| [K-ENGINE-002：vLLM KV 管理：命中、分配和延迟释放](engine-index/vllm-kv-manager.md) | v1 · seed | KVCacheManager 将逻辑 token 进度转换成层组块分配，命中长度、保留容量和可回收块各有独立含义。 |
| [K-ENGINE-003：vLLM ModelRunner：调度视图到设备状态](engine-index/vllm-model-runner.md) | v1 · seed | ModelRunner 把逻辑调度结果转成设备输入与持久批状态，并将前向和采样结果按请求归还。 |
| [K-ENGINE-004：SGLang：Radix 复用与调度状态的连接](engine-index/sglang-radix-scheduler.md) | v1 · seed | SGLang 将共享前缀组织成 Radix 结构，调度器围绕命中状态和可执行批次推进请求。 |
| [K-ENGINE-005：TensorRT-LLM：执行器、Overlap 与传输责任](engine-index/tensorrt-llm.md) | v1 · seed | TensorRT-LLM 的执行器将请求调度、模型执行、采样和资源管理组合成后台迭代，适合研究异步服务的责任划分。 |

## 工具

[分类入口](tools/README.md)，共 7 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-TOOLS-001：可观测性：执行事件与采集事件的映射](tools/trace-observability.md) | v1 · seed | trace 是由采集范围和工具机制生成的观测，不是系统全部执行事实的无损副本。 |
| [K-TOOLS-002：时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md) | v1 · seed | 时间线分析先建立关联和依赖，再用区间与关键路径解释延迟；热点累计时长只是入口。 |
| [K-TOOLS-003：Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md) | v1 · seed | 硬件计数器帮助识别单 kernel 的资源限制，但它测到的执行可能与正常并发服务具有不同环境。 |
| [K-TOOLS-004：框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | v1 · seed | 框架 profiler 解释模型操作怎样落到设备工作，并通过 shape、调用栈与内存事件连接两层视图。 |
| [K-TOOLS-005：失败分析：最早偏离、最小复现与机制归因](tools/failure-evidence.md) | v1 · seed | 失败记录的价值在于还原系统从哪一个状态开始偏离预期，并区分输入、实现、资源和外部环境。 |
| [K-TOOLS-006：正确性 Oracle：函数参考、状态变形与服务契约](tools/correctness-oracles.md) | v1 · seed | 正确性验证需要一组互补的 oracle：数值函数、状态转换、采样分布和外部协议分别有可检查的关系。 |
| [K-TOOLS-007：性能实验：独立重复、效应量与尾部证据](tools/statistical-experiments.md) | v1 · seed | 性能结论需要与独立重复层级相匹配的效应量和不确定性；尾部失败、搜索偏差和保留验证不能由样本数量替代。 |

## 工作负载

[分类入口](workloads/README.md)，共 7 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-WORK-001：服务指标：事件边界、Token 间隔与用户时间](workloads/metric-events.md) | v1 · seed | 一个延迟指标是两个明确事件的时间差；指标名字相同而事件定义不同，会导致不同结论。 |
| [K-WORK-002：到达过程与并发：排队、反馈和饱和点](workloads/arrival-and-concurrency.md) | v1 · seed | 负载生成器定义了系统受到的压力；闭环并发和独立到达率回答不同容量问题。 |
| [K-WORK-003：工作负载分布：长度、复用、模态与相关性](workloads/input-distribution.md) | v1 · seed | 系统成本由输入输出的联合分布决定；平均长度会丢失影响缓存、批次和尾部行为的结构。 |
| [K-WORK-004：性能实验：稳定态、对照与不确定性](workloads/measurement-comparability.md) | v1 · seed | 可比较的性能结果来自相同任务函数与负载条件，以及能够解释波动的测量窗口。 |
| [K-WORK-005：Goodput：联合 SLO、拒绝和容量前沿](workloads/slo-goodput.md) | v1 · seed | 服务能力是单位时间内完成有效任务且满足约束的数量，不能用总生成速度替代。 |
| [K-WORK-006：容量设计：驻留、服务率与阶段配比](workloads/capacity-planning.md) | v1 · seed | 容量设计把每请求资源需求和阶段服务时间连接到到达过程，得出实例数、并发和缓存预算。 |
| [K-WORK-010：多轮与分支负载：依赖、状态驻留和到达过程](workloads/session-workloads.md) | v1 · seed | 会话依赖、思考时间和分支决定真实到达与状态驻留；前缀复用收益需要和会话成功、容量及重用距离一起衡量。 |

## 系统语义与一致性

[分类入口](system-semantics/README.md)，共 12 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-SYS-001：主机与设备时间：偏序、区间并集和关键路径](system-semantics/host-device-timing.md) | v1 · seed | 服务延迟来自生产、提交、执行与消费之间的依赖路径；主机 API 区间与设备工作区间可以重叠。 |
| [K-SYS-002：请求身份：逻辑序列、批次视图与结果归属](system-semantics/request-batch-identity.md) | v1 · seed | 请求是长期语义对象，batch 行与物理槽是临时执行位置；正确性依赖它们之间可追溯的映射。 |
| [K-SYS-003：位置、Mask 与缓存：同一可见性关系的三种表示](system-semantics/position-mask-cache.md) | v1 · seed | 缓存中的内容必须对应模型定义的逻辑位置；mask 决定读取集合，物理布局只决定这些内容放在哪里。 |
| [K-SYS-004：取消与完成：业务终止、设备排空和资源回收](system-semantics/cancellation-resource-completion.md) | v1 · seed | 取消撤销未来服务意图，完成证明既有工作已结束；物理资源重用必须等待最后一个使用者。 |
| [K-SYS-005：集合通信：参与集合、操作顺序与失败传播](system-semantics/collective-ordering.md) | v1 · seed | 分布式前向是一组共同执行协议；每个 rank 的局部控制流必须形成匹配的通信序列。 |
| [K-SYS-006：工程认识如何成为可迁移的专家知识](system-semantics/evidence-and-independent-checks.md) | v1 · seed | 知识的增量是对系统机制与设计关系的新增解释；运行结果为这些解释提供事实、区分和修正。 |
| [K-SYS-007：输入输出语义：Tokenizer、Chat Template 与流式文本](system-semantics/tokenization-io.md) | v1 · seed | 模型处理的是规范化后的 token/特征序列，服务协议必须精确保存从用户输入到模型输入再到输出文本的映射。 |
| [K-SYS-008：服务协议：接收、流式背压与终止语义](system-semantics/service-protocol.md) | v1 · seed | 推理服务将内部生成状态映射为可观察的网络事件；接收、执行完成和客户端收到结果是不同的完成点。 |
| [K-SYS-009：分布式状态迁移：身份、传输与提交点](system-semantics/distributed-state-transfer.md) | v1 · seed | 状态迁移是一项所有权和可见性协议；字节到达只有在布局和版本被共同解释后才成为可用模型状态。 |
| [K-SYS-010：系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md) | v1 · seed | 专家指导的产物是一条从任务条件到系统结构的推导链：状态模型、资源模型、候选设计及其可区分的结果。 |
| [K-SYS-012：内存分配：地址、容量与在途所有权](system-semantics/memory-allocators.md) | v1 · seed | 分配器同时管理地址、物理容量与异步所有权；比较连续池、块表和虚拟映射时必须计入碎片、工作区和在途页。 |
| [K-SYS-015：服务主机运行时：执行域、队列与缓冲所有权](system-semantics/host-runtime.md) | v1 · seed | 主机执行域、队列和缓冲所有权共同限制推理服务；背压与取消必须连接模型调度和在途资源完成。 |

## 运行与演进

[案例](cases/README.md) · [提案](proposals/README.md)。历史正文与结构化事件保留在 history/ 和 events/，原始代码与 trace 仍位于运行产物中。
