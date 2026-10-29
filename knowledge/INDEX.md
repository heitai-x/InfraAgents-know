# 知识索引

按任务条件检索；seed 是一手资料种子，supported 是条件化运行支持。候选、冲突和弃用内容须显式区分。

[设计入口](DESIGN.md) · [诊断入口](DIAGNOSIS.md) · [覆盖矩阵](COVERAGE.md) · [来源登记](SOURCES.md)

## 算法

[分类入口](algorithms/README.md)，共 24 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-ALG-001：连续批处理：从自回归依赖到迭代调度](algorithms/iteration-batching.md) | v5 · seed | 从已知token与连续KV边界推导就绪性，完整预算/执行视图/采样提交算法连接混合调度、取消、资源与SLO成本。 |
| [K-ALG-002：分页 KV：逻辑位置、物理布局与共享所有权](algorithms/paged-kv-capacity.md) | v5 · seed | 从逻辑位置到NHD/HND字节地址，完整追加/COW/发布/失败回滚/在途回收算法连接共享页、容量和页长取舍。 |
| [K-ALG-003：前缀缓存：计算等价类、Radix 与淘汰收益](algorithms/prefix-cache-identity.md) | v5 · seed | 可复用前缀是能产生等价中间状态的一组输入和执行条件；字符串相同只是其中一个投影。 |
| [K-ALG-004：Attention 的 IO、在线 softmax 与分块归约](algorithms/attention-io.md) | v5 · seed | 精确 attention 可以通过改变数据流避免存储完整分数矩阵；算法复杂度、HBM 流量与数值归约顺序分别决定成本。 |
| [K-ALG-005：量化：误差传播、分组尺度与执行格式](algorithms/quantization-contract.md) | v4 · seed | 权重/激活/KV 的校准、缩放与执行格式共同决定量化收益；分别验证制品解释、真实 kernel 和任务质量。 |
| [K-ALG-006：投机解码：分布保持、状态提交与收益模型](algorithms/speculation-budget.md) | v5 · seed | 投机用便宜候选换取目标模型的并行验证；真正推进系统的是被确认的前缀，而非写入过缓存的全部 token。 |
| [K-ALG-007：分块 Prefill：服务时间切片与历史可见性](algorithms/chunked-prefill.md) | v5 · seed | 分块 prefill 将长输入的执行切成可调度时间片；每块仍读取其之前完整且有效的上下文。 |
| [K-ALG-008：异步调度：依赖图、代次与双缓冲](algorithms/async-scheduling.md) | v4 · seed | 异步调度只重叠彼此独立的工作；自回归 token、缓存映射和资源回收仍由明确依赖连接。 |
| [K-ALG-009：异构状态缓存：层组、恢复位置与联合分配](algorithms/heterogeneous-state-cache.md) | v4 · seed | 混合模型的缓存是多类状态的联合系统，统一的是逻辑恢复位置，物理块形状可以不同。 |
| [K-ALG-010：交叉注意力缓存：固定源、独立几何与可见性](algorithms/cross-attention-cache.md) | v5 · seed | 当 encoder 输出和投影权重固定时，cross-attention 的 K/V 与 decoder 步数无关，可以一次生成、多次读取。 |
| [K-ALG-011：批量采样：概率变换、请求随机性与行映射](algorithms/batched-sampling.md) | v4 · seed | 采样器消费每个请求自己的条件分布；向量化改变执行形态，不能改变参数、随机状态或输出归属。 |
| [K-ALG-012：结构化生成：语言状态、token 边界与约束采样](algorithms/structured-output.md) | v4 · seed | 结构化生成将语法状态与语言模型状态同步推进，合法 token 是从当前语法状态仍可完成目标语言的延伸。 |
| [K-ALG-013：Prefill/Decode 分离：状态传输与队列耦合](algorithms/disaggregated-serving.md) | v4 · seed | 分离式服务把两个阶段的资源策略解耦，同时引入 KV 传输、布局转换和跨节点状态提交。 |
| [K-ALG-014：多 LoRA 服务：低秩计算、驻留与缓存身份](algorithms/lora-serving.md) | v5 · seed | LoRA 为共享基座增加低秩映射；多适配器服务将模型复用问题转换成按请求选择增量权重的问题。 |
| [K-ALG-015：推理并行：张量分解、流水、专家与上下文](algorithms/parallelism.md) | v5 · seed | 并行策略改变张量归属和同步边界；选择依据是每阶段的计算、状态与通信分解。 |
| [K-ALG-016：基础算子：从张量契约到合法融合](algorithms/operator-primitives.md) | v7 · seed | 从函数、张量地址与别名推导合法执行；展开QKV、归一化、RoPE和融合成本，按所选机制检查数值与真实消费者。 |
| [K-ALG-017：多级驻留：有效副本、预取、淘汰与重算](algorithms/tiered-residency.md) | v3 · seed | 多级驻留维护有效版本与在途副本，并联合比较保留、预取、淘汰和重算；三级容量可行不等于在线延迟可行。 |
| [K-ALG-018：KV 卸载与预取：读集合、恢复点与迁移预算](algorithms/kv-offload-prefetch.md) | v4 · seed | KV 卸载按实际可见集合与共同恢复点组织；暂停恢复、前缀命中和活跃 decode 的持续传输需要不同预算。 |
| [K-ALG-019：权重与专家卸载：重用、路由与在途版本](algorithms/weight-expert-offload.md) | v4 · seed | 权重与专家卸载同时依赖批次重用、专家并集、热点和版本；按 token 稀疏不能直接推断 batch 读取稀疏。 |
| [K-ALG-020：CPU–GPU 计算放置：传数据、传结果与共同提交](algorithms/heterogeneous-execution.md) | v5 · seed | CPU–GPU 分工比较冷权重搬运、CPU 计算与分片合并的完整关键路径；激活稀疏预测须有独立质量契约。 |
| [K-ALG-021：KV 压缩：表示、近似保留与传输格式](algorithms/kv-compression.md) | v5 · seed | KV 量化、历史裁剪与传输编码改变不同成本与语义；分组轴、残余区、元数据和质量必须分别计入。 |
| [K-ALG-022：非前缀上下文复用：依赖、位置与选择性重算](algorithms/context-reuse.md) | v4 · seed | 非前缀文本重复不保证逐层状态相同；按上文、位置与可见性判断精确复用，并对选择性重算独立计量质量和加载成本。 |
| [K-ALG-023：请求路由与迁移：命中收益、排队和追赶条件](algorithms/routing-and-migration.md) | v3 · seed | 缓存亲和与动态迁移需要同时比较排队、状态增长、网络和提交责任；局部命中与低迁移停顿不保证整体收益。 |
| [K-ALG-024：多样本与搜索分支：父子状态、评分与共同提交](algorithms/branch-generation.md) | v3 · seed | beam、多样本与投机树共享存储机制但优化目标不同；父子路径、logits 行、随机/语法状态与提交边界必须一致。 |

## 模型

[分类入口](models/README.md)，共 16 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-MODEL-001：从模型计算图推导 Prefill、Decode 与持久状态](models/prefill-decode-state.md) | v7 · seed | 模型计算图和状态语义解释 token、位置、请求与执行阶段的依赖；参考边界帮助验证实现，不规定构建顺序。 |
| [K-MODEL-002：MHA、GQA 与 MQA：头映射和带宽复用](models/gqa-kv-heads.md) | v5 · seed | GQA 让多个 query heads 读取同一组 K/V；逻辑共享可以减少状态，但是否减少实际访存取决于 kernel。 |
| [K-MODEL-003：MoE：路由、Dispatch/Combine 与专家负载](models/moe-routing.md) | v5 · seed | MoE 的条件计算节省激活 FLOPs，但全部专家的驻留和路由后的不均衡决定真实服务成本。 |
| [K-MODEL-004：SSM 与混合模型：递归状态、扫描与恢复](models/recurrent-hybrid-state.md) | v5 · seed | 递归状态是前缀的压缩摘要；它支持低增长成本的 decode，却改变了前缀复用、回滚和批次管理方式。 |
| [K-MODEL-005：Encoder-decoder：源表示、两套 Attention 与流式边界](models/encoder-decoder-streaming.md) | v4 · seed | Encoder-decoder 将源编码和目标生成分开；固定源复用、流式追加和目标自回归具有不同的依赖结构。 |
| [K-MODEL-006：MLA：压缩状态、矩阵吸收与解耦 RoPE](models/mla.md) | v5 · seed | MLA 通过缓存低维 latent 改变 KV 的表示；真正节省 decode 流量需要执行内核沿压缩表示计算。 |
| [K-MODEL-007：窗口与稀疏 Attention：可见性、位置和状态保留](models/sliding-window-attention.md) | v5 · seed | 窗口限制的是每个 query 的直接可见历史；逻辑位置与跨层信息传播仍然持续演进。 |
| [K-MODEL-008：视觉语言模型：Patch、模态桥接与联合位置](models/vision-language.md) | v5 · seed | VLM 的真实输入规模由处理器生成的视觉 token 决定，文本长度只描述其中一部分。 |
| [K-MODEL-009：图像扩散：迭代求解、条件复用与 CFG](models/image-diffusion.md) | v4 · seed | 扩散服务围绕不断变化的 latent 做多步求解；可复用对象由每步依赖决定，不能沿用自回归 KV 的全部规则。 |
| [K-MODEL-010：视频生成：时空 Token、全局耦合与分块解码](models/video-generation.md) | v5 · seed | 视频生成把时间加入 latent 几何，时间压缩、时空 attention 和输出解码共同决定可服务规模。 |
| [K-MODEL-011：语音生成：Codec Token、声学块与播放时钟](models/speech-generation.md) | v4 · seed | 语音服务需要把模型生成时钟转换为音频播放时钟，首包和稳定供给由多个阶段共同决定。 |
| [K-MODEL-012：全模态系统：多时钟、分支依赖与背压](models/omni-multimodal.md) | v4 · seed | 全模态推理是多个有依赖的计算分支组成的系统；文本、视频和音频不能压成同一个 token 长度。 |
| [K-MODEL-013：Embedding 与重排：编码函数、批量与评分契约](models/embedding-reranking.md) | v3 · seed | 编码器、pooling、normalize 与句对评分共同定义 embedding/重排函数；长度聚批、padding 与输入版本决定成本及结果。 |
| [K-MODEL-014：视觉识别与卷积：空间布局、预处理与任务输出](models/vision-inference.md) | v3 · seed | 视觉识别把图像变换、卷积布局、残差生命周期与分类/检测输出连接；按分辨率、工作区和几何契约验证完整请求。 |
| [K-MODEL-015：推荐推理：稀疏表、特征交互与请求批量](models/recommendation-inference.md) | v3 · seed | 推荐推理分别计量稀疏表 lookup、bag 聚合、特征交互与 dense 计算，并以特征/参数版本约束分片、缓存和更新。 |
| [K-MODEL-016：图模型推理：邻居依赖、采样与不规则执行](models/graph-inference.md) | v3 · seed | 图推理先展开邻居依赖再逐层聚合；采样、图/特征版本、去重语义和不规则搬运共同决定质量与资源上界。 |

## 硬件平台

[分类入口](hardware/README.md)，共 14 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-HW-001：容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md) | v5 · seed | 容量决定系统能容纳哪些状态，带宽与计算能力决定搬运和处理这些状态的下界，关键路径决定哪些下界实际显现。 |
| [K-HW-002：通信拓扑：从 Rank 组到瓶颈链路](hardware/topology-parallelism.md) | v6 · seed | 拓扑、方向与共享链路决定通信成本；将理想ring启动项和发送流量放到具体带宽假设中解释。 |
| [K-HW-003：主机准备、DMA 与流水缓冲](hardware/host-device-transfer.md) | v6 · seed | 主机准备、DMA、计算与消费按对象依赖连接；量化有限批、各池槽位与共享带宽，允许不同放置与传输方式。 |
| [K-HW-004：数值格式、打包布局与矩阵指令](hardware/precision-and-kernel-support.md) | v4 · seed | 固定架构/指令能力、量化参数布局与实际 dispatch 分别核验，防止把存储格式、指令支持和目标内核性能混为一谈。 |
| [K-HW-005：CUDA 平台：SM 数据流、占用率与执行粒度](hardware/cuda-execution.md) | v5 · seed | CUDA 优化应把算法的数据复用映射到 SM 的寄存器、共享存储和 warp 执行，再判断主机提交是否成为瓶颈。 |
| [K-HW-006：ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md) | v5 · seed | ROCm 移植应保持模型语义，同时按 AMD 的执行资源、矩阵指令和库 dispatch 重新推导数据布局与粒度。 |
| [K-HW-007：Apple Silicon：统一内存、带宽与交互式推理](hardware/metal-unified-memory.md) | v5 · seed | 统一内存减少显式设备间搬运，但 CPU/GPU 仍共享容量、带宽并通过依赖控制可见性。 |
| [K-HW-008：Trainium：HBM、SBUF、PSUM 与编译执行](hardware/trainium-dataflow.md) | v5 · seed | Trainium 的高效执行来自把张量分块映射到片上存储和专用引擎，同时让静态可执行程序承接动态服务状态。 |
| [K-HW-009：CPU 推理：线程、缓存层级与 NUMA 分区](hardware/cpu-serving.md) | v5 · seed | CPU 服务优化需要同时管理向量算术、内存带宽、线程调度与请求级并发，更多线程不等于更多有效工作。 |
| [K-HW-010：CPU GEMM/GEMV：寄存器、缓存与线程划分](hardware/cpu-linear-kernels.md) | v5 · seed | CPU 线性层需把 shape 的重用机会落实到寄存器、cache/TLB、打包和线程分区；同一配置不适用于全部 GEMV/GEMM。 |
| [K-HW-011：GPU 矩阵与归约流水：布局、资源和完成依赖](hardware/gpu-kernel-pipelines.md) | v5 · seed | 完整SIMT GEMM映射和带代次多读者缓冲算法，联立有限流水时序、资源驻留与网格成本，区分同步合法性与实际重叠。 |
| [K-HW-012：存储 I/O：请求粒度、队列和加载流水](hardware/storage-io.md) | v4 · seed | 实际存储供给由布局、pagecache/直接 I/O、请求粒度、队列和 staging 共同决定；提交、可读、持久化和可回收分开建模。 |
| [K-HW-013：共享内存路径：统一地址、物理驻留与显式迁移](hardware/shared-memory-paths.md) | v3 · seed | 统一地址、可迁移驻留和物理共享内存是不同访问关系；重用、页放大、可见性和带宽竞争共同决定复制选择。 |
| [K-HW-014：通信传输：注册、完成、可见与回收](hardware/transport-and-rdma.md) | v4 · seed | 网络提交、CQE、消费者可见与应用接管是不同事件；注册、代次和完成协议共同决定缓冲何时可回收。 |

## 框架

[分类入口](frameworks/README.md)，共 12 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-FW-001：PyTorch 推理：模块模式、Autograd 与状态突变](frameworks/inference-eval.md) | v5 · seed | 推理行为由模型模式、梯度记录、dtype/device、模块状态与执行路径共同决定。 |
| [K-FW-002：数值参考：误差尺度、条件数与首次分歧](frameworks/numerical-reference.md) | v5 · seed | 一个有解释力的数值参考保持任务函数和输入一致，再分层衡量实现重排与近似产生的差异。 |
| [K-FW-003：Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md) | v5 · seed | stream 解决执行顺序，内存生存期解决地址何时可重用；正确的异步程序需要同时满足两种关系。 |
| [K-FW-004：编译与动态形状：Guard、分桶和变体成本](frameworks/compile-dynamic-shapes.md) | v5 · seed | 图获取、IR/lowering、布局和 guard 将函数映射到目标程序；编译缓存及在线变体必须保持副作用、容量与生命周期契约。 |
| [K-FW-005：CUDA Graph：捕获边界、计划状态与变长服务](frameworks/cuda-graph-buffers.md) | v5 · seed | CUDA Graph 的固定地址、动态计划和在途内容所有权需共同验证；卸载、adapter 切换和图池竞争改变合法性与收益。 |
| [K-FW-006：Attention 执行：数学语义、分块归约、低精度与硬件数据流](frameworks/attention-backends.md) | v9 · seed | 从数学不变量到FA2/FA3、低精度与成本；完整warp映射及共享前缀/量化尾块/图快照案例连接算法、地址与状态生命周期。 |
| [K-FW-007：Triton Kernel：块程序、布局与融合边界](frameworks/triton-kernels.md) | v6 · seed | 块程序、布局与融合边界连接固定Triton实现；解释descriptor、log2统计、索引范围与调优成本的适用条件。 |
| [K-FW-008：MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md) | v4 · seed | MLX 的数组表达式可以延迟执行，服务循环必须明确计算何时提交、何时完成和何时释放状态。 |
| [K-FW-009：Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md) | v4 · seed | Neuron 服务通过形状桶、设备驻留状态和分布式执行组，把可变请求映射到编译后的程序族。 |
| [K-FW-010：NKI：Partition 轴、片上生存期与 Attention 数据流](frameworks/nki-tiling.md) | v5 · seed | NKI 的 tile 是带硬件轴语义的片上对象；布局与生存期共同决定 DMA、矩阵和归约引擎的供给。 |
| [K-FW-011：权重加载：从制品身份到可执行布局](frameworks/checkpoint-loading.md) | v4 · seed | 加载必须同时建立张量语义、执行布局、完整参数与就绪边界；流式流水的价值由启动关键路径和峰值副本决定。 |
| [K-FW-012：CPU 低位执行：解包、补偿与累加路径](frameworks/cpu-quantized-kernels.md) | v4 · seed | 低位存储、解包和实际乘加是不同执行路径；补偿、组尺度、累加宽度和舍入共同决定正确性与收益。 |

## 引擎源码索引

[分类入口](engine-index/README.md)，共 5 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-ENGINE-001：调度决策：进度、就绪与资源承诺](engine-index/vllm-scheduler.md) | v4 · seed | 动态服务的难点是请求的进度、输入就绪和资源可用性不在同一个时刻变化。调度输出需要表达本轮允许执行的逻辑范围及其资源承诺；按 token、请求、时间片或其他单位选择都是设计点。 |
| [K-ENGINE-002：状态分配决策：命中、预留、发布与回收](engine-index/vllm-kv-manager.md) | v4 · seed | 缓存命中、存储已经分配、数据已经可用与请求已经完成是不同事实。分配器必须把这些事实接到执行与恢复边界，单个“长度”或“空闲显存”无法完整描述可推进性。 |
| [K-ENGINE-003：执行视图决策：压紧、固定槽与结果归属](engine-index/vllm-model-runner.md) | v4 · seed | 动态请求集合需要转成设备能执行的张量，但逻辑身份不应随物理行号变化。执行器的核心问题是保持请求、query 位置、缓存写入与输出行之间的可逆对应，并管理这些表示的生存期。 |
| [K-ENGINE-004：前缀复用决策：树、块链与状态保护](engine-index/sglang-radix-scheduler.md) | v4 · seed | 前缀复用的动机是相同依赖产生的中间状态可以避免重复计算。索引回答在哪里找到可能相同的前缀，恢复与所有权机制回答它是否能被当前请求安全使用；两者不能合并成一个“命中”布尔值。 |
| [K-ENGINE-005：异步执行决策：结果确认、在途状态与回收](engine-index/tensorrt-llm.md) | v4 · seed | 主机准备、设备计算、采样与外部状态传输可以部分重叠，但自回归依赖与存储生存期不会因异步提交而消失。设计需要区分业务完成、输出提交、设备完成和各个外部消费者结束。 |

## 工具

[分类入口](tools/README.md)，共 8 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-TOOLS-001：可观测性：执行事件与采集事件的映射](tools/trace-observability.md) | v6 · seed | trace 是由采集范围和工具机制生成的观测，不是系统全部执行事实的无损副本。 |
| [K-TOOLS-002：时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md) | v5 · seed | 时间线分析先建立关联和依赖，再用区间与关键路径解释延迟；热点累计时长只是入口。 |
| [K-TOOLS-003：Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md) | v5 · seed | 硬件计数器帮助识别单 kernel 的资源限制，但它测到的执行可能与正常并发服务具有不同环境。 |
| [K-TOOLS-004：框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | v4 · seed | 框架 profiler 解释模型操作怎样落到设备工作，并通过 shape、调用栈与内存事件连接两层视图。 |
| [K-TOOLS-005：失败分析：最早偏离、最小复现与机制归因](tools/failure-evidence.md) | v5 · seed | 失败记录的价值在于还原系统从哪一个状态开始偏离预期，并区分输入、实现、资源和外部环境。 |
| [K-TOOLS-006：正确性 Oracle：函数参考、状态变形与服务契约](tools/correctness-oracles.md) | v8 · seed | 独立数值、身份与协议参考及负对照按Task和所选机制检查；缺失、重复或空集不能冒充完整结果。 |
| [K-TOOLS-007：性能实验：独立重复、效应量与尾部证据](tools/statistical-experiments.md) | v6 · seed | 重复层级、效应量和结论范围；可选知识条件实验区分交付失败、成本与删失，不向Agent规定构建流程。 |
| [K-TOOLS-008：GPU 从零表征：可持续带宽、矩阵吞吐与测量条件](tools/gpu-characterization.md) | v2 · seed | 用带条件的微基准建立 HBM/L2、矩阵、launch/图、PCIe、时钟功耗与多消费者参数包，并对接资源模型；峰值与异形态结果不能直接代入。 |

## 工作负载

[分类入口](workloads/README.md)，共 11 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-WORK-001：服务指标：事件边界、Token 间隔与用户时间](workloads/metric-events.md) | v5 · seed | 一个延迟指标是两个明确事件的时间差；指标名字相同而事件定义不同，会导致不同结论。 |
| [K-WORK-002：到达过程与并发：排队、反馈和饱和点](workloads/arrival-and-concurrency.md) | v5 · seed | 负载生成器定义了系统受到的压力；闭环并发和独立到达率回答不同容量问题。 |
| [K-WORK-003：工作负载分布：长度、复用、模态与相关性](workloads/input-distribution.md) | v4 · seed | 系统成本由输入输出的联合分布决定；平均长度会丢失影响缓存、批次和尾部行为的结构。 |
| [K-WORK-004：性能实验：稳定态、对照与不确定性](workloads/measurement-comparability.md) | v6 · seed | 可比较的性能结果来自相同任务函数与负载条件，以及能够解释波动的测量窗口。 |
| [K-WORK-005：Goodput：联合 SLO、拒绝和容量前沿](workloads/slo-goodput.md) | v6 · seed | 服务能力是单位时间内完成有效任务且满足约束的数量，不能用总生成速度替代。 |
| [K-WORK-006：容量设计：驻留、服务率与阶段配比](workloads/capacity-planning.md) | v6 · seed | 容量设计把每请求资源需求和阶段服务时间连接到到达过程，得出实例数、并发和缓存预算。 |
| [K-WORK-007：多模型放置：共同容量、混合执行与租户公平](workloads/multi-model-placement.md) | v3 · seed | 多模型与 LoRA 的共享条件决定共同容量、混合执行和公平等待；完整场景联合比较合并、共享及分组路径。 |
| [K-WORK-008：弹性服务：启动延迟、可用容量与排空](workloads/elastic-serving.md) | v4 · seed | 扩缩容要以实际就绪容量和启动/排空延迟解释队列；冷启动流量、故障余量与状态所有权决定策略是否可行。 |
| [K-WORK-009：服务成本与能耗：按合格结果计算资源代价](workloads/cost-and-energy.md) | v3 · seed | 成本与能耗以相同质量/SLO 的合格结果为分母，计入闲置、加载和失败，并区分计量域、积分与价格假设。 |
| [K-WORK-010：多轮与分支负载：依赖、状态驻留和到达过程](workloads/session-workloads.md) | v4 · seed | 会话依赖、思考时间和分支决定真实到达与状态驻留；前缀复用收益需要和会话成功、容量及重用距离一起衡量。 |
| [K-WORK-011：集群放置：资源向量、拓扑、执行组与故障域](workloads/cluster-resource-placement.md) | v3 · seed | 集群放置先检查逐 rank 资源和拓扑，再比较装箱、局部性、共同启动与故障后容量；总空闲量不保证可部署。 |

## 系统语义与一致性

[分类入口](system-semantics/README.md)，共 21 条。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-SYS-001：主机与设备时间：偏序、区间并集和关键路径](system-semantics/host-device-timing.md) | v5 · seed | 服务延迟来自生产、提交、执行与消费之间的依赖路径；主机 API 区间与设备工作区间可以重叠。 |
| [K-SYS-002：请求身份：逻辑序列、批次视图与结果归属](system-semantics/request-batch-identity.md) | v4 · seed | 请求是长期语义对象，batch 行与物理槽是临时执行位置；正确性依赖它们之间可追溯的映射。 |
| [K-SYS-003：位置、Mask 与缓存：同一可见性关系的三种表示](system-semantics/position-mask-cache.md) | v6 · seed | 缓存中的内容必须对应模型定义的逻辑位置；mask 决定读取集合，物理布局只决定这些内容放在哪里。 |
| [K-SYS-004：取消与完成：业务终止、设备排空和资源回收](system-semantics/cancellation-resource-completion.md) | v5 · seed | 取消撤销未来服务意图，完成证明既有工作已结束；物理资源重用必须等待最后一个使用者。 |
| [K-SYS-005：集合通信：参与集合、操作顺序与失败传播](system-semantics/collective-ordering.md) | v6 · seed | 集合通信连接参与者、张量解释、count和完成关系；固定NCCL组提交语义及可辨识反例支持所选边界的验证。 |
| [K-SYS-006：工程认识如何成为可迁移的专家知识](system-semantics/evidence-and-independent-checks.md) | v4 · seed | 知识的增量是对系统机制与设计关系的新增解释；运行结果为这些解释提供事实、区分和修正。 |
| [K-SYS-007：输入输出语义：Tokenizer、Chat Template 与流式文本](system-semantics/tokenization-io.md) | v5 · seed | 模型处理的是规范化后的 token/特征序列，服务协议必须精确保存从用户输入到模型输入再到输出文本的映射。 |
| [K-SYS-008：服务协议：接收、流式背压与终止语义](system-semantics/service-protocol.md) | v6 · seed | 对外协议、发送缓冲、背压与终止语义依Task和具体接口解释；结果交付与资源完成分别核对。 |
| [K-SYS-009：分布式状态迁移：身份、传输与提交点](system-semantics/distributed-state-transfer.md) | v4 · seed | 状态迁移是一项所有权和可见性协议；字节到达只有在布局和版本被共同解释后才成为可用模型状态。 |
| [K-SYS-010：系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md) | v7 · seed | 在声明条件下推导模型、状态、资源与执行关系；场景不穷尽候选，不规定组件划分、研发顺序或任务外功能。 |
| [K-SYS-011：模型生命周期：加载、就绪、排空与版本切换](system-semantics/model-lifecycle.md) | v4 · seed | 模型就绪需要完整制品、执行路径和容量；请求绑定版本，新旧并存、排空和卸载由共同生命周期约束。 |
| [K-SYS-012：内存分配：地址、容量与在途所有权](system-semantics/memory-allocators.md) | v3 · seed | 分配器同时管理地址、物理容量与异步所有权；比较连续池、块表和虚拟映射时必须计入碎片、工作区和在途页。 |
| [K-SYS-013：分布式 KV 存储：目录、有效副本与可恢复状态](system-semantics/distributed-kv-store.md) | v4 · seed | 分布式 KV 目录只定位候选状态；有效副本、共同恢复点、目标可见性与排队/传输成本决定真实可复用性。 |
| [K-SYS-014：推理服务恢复：提交边界、执行组与重试身份](system-semantics/serving-recovery.md) | v3 · seed | 恢复需要明确计算、确认与外部提交边界，以及唯一提交权；可重算 KV 不等于可恢复整个请求或恰好一次交付。 |
| [K-SYS-015：服务主机运行时：执行域、队列与缓冲所有权](system-semantics/host-runtime.md) | v5 · seed | 执行域、队列与缓冲的成本和完成关系；资源有界要求可由准入、额度或队列等不同机制实现。 |
| [K-SYS-016：租户隔离：资源份额、缓存命名空间与故障边界](system-semantics/tenant-isolation.md) | v3 · seed | 租户身份、缓存权限、资源限额和故障域需要分别约束；固定文档区分 cgroup、time-slicing、MPS 与 MIG 资源契约。 |
| [K-SYS-017：向量检索与 RAG：索引、版本与端到端质量](system-semantics/retrieval-pipeline.md) | v3 · seed | 检索的距离契约、候选过滤和索引版本共同决定合法结果；Flat/IVF/图搜索与压缩需连接材料相关性和完整 RAG 时间预算。 |
| [K-SYS-018：推理存储服务：制品、缓存、索引与提交记录](system-semantics/storage-services.md) | v3 · seed | 权重、KV、索引和提交记录具有不同持久性；版本清单、对象可见性、分发与恢复竞争共同决定推理存储契约。 |
| [K-SYS-019：部署环境：版本契约、可见设备与真实资源边界](system-semantics/deployment-environment.md) | v3 · seed | 镜像、宿主驱动、设备身份与有效 quota 共同定义部署环境；就绪、资源峰值和代表路径验证才能支持服务可用性。 |
| [K-SYS-024：专用化杠杆：固定模型、负载与硬件时多出的自由度](system-semantics/specialization-levers.md) | v2 · seed | 模型、workload 或硬件冻结时，预转换、分桶、离线定参与单一路径等杠杆的收益来源、前提、失效条件与代价；非施工清单。 |
| [K-SYS-025：已知张力：单实例交互式服务中的机制冲突](system-semantics/known-tensions.md) | v2 · seed | decode 带宽与 ITL、prefill/decode 干扰、投机与批、KV 量化悬崖、图/workspace 挤 KV、小批主机占比：机制、来源、可观测与反转。 |

## 运行与演进

[案例](cases/README.md) · [提案](proposals/README.md)。历史正文与结构化事件保留在 history/ 和 events/，原始代码与 trace 仍位于运行产物中。
