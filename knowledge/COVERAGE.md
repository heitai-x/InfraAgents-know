# 专家知识覆盖与 VibeSys 对照

本版本保留原知识库的主题组织和覆盖映射。该映射是编辑结构，不以参考项目的日期证明知识的时间资格；事实依据只来自逐条登记的截止日前原始资料。

## 阅读层次

[模型](models/README.md) → [算法与状态](algorithms/README.md) → [硬件](hardware/README.md) / [框架](frameworks/README.md) → [系统组合](system-semantics/design-synthesis.md) → [工作负载与容量](workloads/README.md)。[引擎](engine-index/README.md) 提供固定版本的实现映射，[工具](tools/README.md) 解释如何观测和验证机制。

## 逐项主题映射

| VibeSys 参考文件 | InfraAgents 专题 | 内容关系 |
|---|---|---|
| algorithms/async-scheduling.md | [异步调度：依赖图、代次与双缓冲](algorithms/async-scheduling.md)、[Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/batched-sampling.md | [批量采样：概率变换、请求随机性与行映射](algorithms/batched-sampling.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/chunked-prefill.md | [分块 Prefill：服务时间切片与历史可见性](algorithms/chunked-prefill.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/continuous-batching.md | [连续批处理：从自回归依赖到迭代调度](algorithms/iteration-batching.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/cross-attention-kv-cache.md | [交叉注意力缓存：固定源、独立几何与可见性](algorithms/cross-attention-cache.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/disaggregated-serving.md | [Prefill/Decode 分离：状态传输与队列耦合](algorithms/disaggregated-serving.md)、[分布式状态迁移：身份、传输与提交点](system-semantics/distributed-state-transfer.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/heterogeneous-kv-cache.md | [异构状态缓存：层组、恢复位置与联合分配](algorithms/heterogeneous-state-cache.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/moe-routing-dispatch.md | [MoE：路由、Dispatch/Combine 与专家负载](models/moe-routing.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/paged-attention.md | [分页 KV：逻辑位置、物理布局与共享所有权](algorithms/paged-kv-capacity.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/parallelism.md | [推理并行：张量分解、流水、专家与上下文](algorithms/parallelism.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/quantization-schemes.md | [量化：误差传播、分组尺度与执行格式](algorithms/quantization-contract.md)、[数值格式、打包布局与矩阵指令](hardware/precision-and-kernel-support.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/radix-prefix-caching.md | [前缀缓存：计算等价类、Radix 与淘汰收益](algorithms/prefix-cache-identity.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/speculative-decoding.md | [投机解码：分布保持、状态提交与收益模型](algorithms/speculation-budget.md) | 以计算关系、状态表示和成本推导展开。 |
| algorithms/structured-output.md | [结构化生成：语言状态、token 边界与约束采样](algorithms/structured-output.md) | 以计算关系、状态表示和成本推导展开。 |
| engines/sglang.md | [SGLang：Radix 复用与调度状态的连接](engine-index/sglang-radix-scheduler.md) | 已核对固定源码的具体入口与机制映射。 |
| engines/trtllm.md | [TensorRT-LLM：执行器、Overlap 与传输责任](engine-index/tensorrt-llm.md) | 已核对固定源码的具体入口与机制映射。 |
| engines/vllm.md | [vLLM V1：从 Token 进度到调度输出](engine-index/vllm-scheduler.md)、[vLLM KV 管理：命中、分配和延迟释放](engine-index/vllm-kv-manager.md)、[vLLM ModelRunner：调度视图到设备状态](engine-index/vllm-model-runner.md) | 已核对固定源码的具体入口与机制映射。 |
| frameworks/pytorch.md | [PyTorch 推理：模块模式、Autograd 与状态突变](frameworks/inference-eval.md)、[数值参考：误差尺度、条件数与首次分歧](frameworks/numerical-reference.md)、[Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md)、[编译与动态形状：Guard、分桶和变体成本](frameworks/compile-dynamic-shapes.md) | 拆成执行模式、数值、异步内存和编译专题。 |
| frameworks/triton.md | [Triton Kernel：块程序、布局与融合边界](frameworks/triton-kernels.md) | 块程序、布局、融合与调优。 |
| models/attention-variants.md | [MHA、GQA 与 MQA：头映射和带宽复用](models/gqa-kv-heads.md)、[MLA：压缩状态、矩阵吸收与解耦 RoPE](models/mla.md)、[窗口与稀疏 Attention：可见性、位置和状态保留](models/sliding-window-attention.md)、[交叉注意力缓存：固定源、独立几何与可见性](algorithms/cross-attention-cache.md) | 从模型状态与依赖推导服务结构。 |
| models/image-generation.md | [图像扩散：迭代求解、条件复用与 CFG](models/image-diffusion.md) | 从模型状态与依赖推导服务结构。 |
| models/omni-multimodal.md | [全模态系统：多时钟、分支依赖与背压](models/omni-multimodal.md) | 从模型状态与依赖推导服务结构。 |
| models/speech-generation.md | [语音生成：Codec Token、声学块与播放时钟](models/speech-generation.md) | 从模型状态与依赖推导服务结构。 |
| models/speech-language.md | [Encoder-decoder：源表示、两套 Attention 与流式边界](models/encoder-decoder-streaming.md) | 从模型状态与依赖推导服务结构。 |
| models/ssm-hybrid.md | [SSM 与混合模型：递归状态、扫描与恢复](models/recurrent-hybrid-state.md) | 从模型状态与依赖推导服务结构。 |
| models/text-dense.md | [从模型计算图推导 Prefill、Decode 与持久状态](models/prefill-decode-state.md) | 从模型状态与依赖推导服务结构。 |
| models/text-moe.md | [MoE：路由、Dispatch/Combine 与专家负载](models/moe-routing.md) | 从模型状态与依赖推导服务结构。 |
| models/video-generation.md | [视频生成：时空 Token、全局耦合与分块解码](models/video-generation.md) | 从模型状态与依赖推导服务结构。 |
| models/vision-language.md | [视觉语言模型：Patch、模态桥接与联合位置](models/vision-language.md) | 从模型状态与依赖推导服务结构。 |
| platforms/cpu/floor.md | [CPU 推理：线程、缓存层级与 NUMA 分区](hardware/cpu-serving.md)、[系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md) | 平台选择由资源与任务推导，不设置无条件功能底线。 |
| platforms/cpu/hardware.md | [CPU 推理：线程、缓存层级与 NUMA 分区](hardware/cpu-serving.md)、[容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md) | 执行资源、数据层级与容量模型。 |
| platforms/cpu/profiler.md | [CPU 推理：线程、缓存层级与 NUMA 分区](hardware/cpu-serving.md)、[时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)、[Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | 对应平台执行语义与可观测模型；本文不复制安装/命令手册。 |
| platforms/cuda/async-scheduling.md | [异步调度：依赖图、代次与双缓冲](algorithms/async-scheduling.md)、[Stream 与分配器：依赖顺序和对象生存期](frameworks/streams-allocator.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/attention-backend-comparison.md | [Attention Backend：语义、布局与执行契约的交集](frameworks/attention-backends.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/continuous-batching.md | [连续批处理：从自回归依赖到迭代调度](algorithms/iteration-batching.md)、[CUDA Graph：捕获边界、计划状态与变长服务](frameworks/cuda-graph-buffers.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/cuda-graph.md | [CUDA Graph：捕获边界、计划状态与变长服务](frameworks/cuda-graph-buffers.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/flashattention.md | [Attention Backend：语义、布局与执行契约的交集](frameworks/attention-backends.md)、[Attention 的 IO、在线 softmax 与分块归约](algorithms/attention-io.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/flashinfer.md | [Attention Backend：语义、布局与执行契约的交集](frameworks/attention-backends.md)、[CUDA Graph：捕获边界、计划状态与变长服务](frameworks/cuda-graph-buffers.md)、[分页 KV：逻辑位置、物理布局与共享所有权](algorithms/paged-kv-capacity.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/floor.md | [CUDA 平台：SM 数据流、占用率与执行粒度](hardware/cuda-execution.md)、[系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md) | 平台选择由资源与任务推导，不设置无条件功能底线。 |
| platforms/cuda/hardware.md | [CUDA 平台：SM 数据流、占用率与执行粒度](hardware/cuda-execution.md)、[容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md) | 执行资源、数据层级与容量模型。 |
| platforms/cuda/profiler.md | [CUDA 平台：SM 数据流、占用率与执行粒度](hardware/cuda-execution.md)、[时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)、[Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | 对应平台执行语义与可观测模型；本文不复制安装/命令手册。 |
| platforms/cuda/sdpa.md | [Attention Backend：语义、布局与执行契约的交集](frameworks/attention-backends.md)、[PyTorch 推理：模块模式、Autograd 与状态突变](frameworks/inference-eval.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/speculative-decoding.md | [投机解码：分布保持、状态提交与收益模型](algorithms/speculation-budget.md)、[CUDA Graph：捕获边界、计划状态与变长服务](frameworks/cuda-graph-buffers.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/cuda/triton-kernels.md | [Triton Kernel：块程序、布局与融合边界](frameworks/triton-kernels.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/metal/async-scheduling.md | [MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md)、[异步调度：依赖图、代次与双缓冲](algorithms/async-scheduling.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/metal/continuous-batching.md | [MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md)、[连续批处理：从自回归依赖到迭代调度](algorithms/iteration-batching.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/metal/floor.md | [Apple Silicon：统一内存、带宽与交互式推理](hardware/metal-unified-memory.md)、[系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md) | 平台选择由资源与任务推导，不设置无条件功能底线。 |
| platforms/metal/hardware.md | [Apple Silicon：统一内存、带宽与交互式推理](hardware/metal-unified-memory.md)、[容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md) | 执行资源、数据层级与容量模型。 |
| platforms/metal/mlx-serving.md | [MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md)、[服务协议：接收、流式背压与终止语义](system-semantics/service-protocol.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/metal/mlx.md | [MLX 服务：惰性求值、编译与请求状态](frameworks/mlx-serving.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/metal/profiler.md | [Apple Silicon：统一内存、带宽与交互式推理](hardware/metal-unified-memory.md)、[时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)、[Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | 对应平台执行语义与可观测模型；本文不复制安装/命令手册。 |
| platforms/rocm/aiter.md | [ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md)、[Attention Backend：语义、布局与执行契约的交集](frameworks/attention-backends.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/rocm/floor.md | [ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md)、[系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md) | 平台选择由资源与任务推导，不设置无条件功能底线。 |
| platforms/rocm/hardware.md | [ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md)、[容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md) | 执行资源、数据层级与容量模型。 |
| platforms/rocm/profiler.md | [ROCm 平台：CU、Wave 与 AMD 算子路径](hardware/rocm-execution.md)、[时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)、[Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | 对应平台执行语义与可观测模型；本文不复制安装/命令手册。 |
| platforms/trainium/async-scheduling.md | [Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md)、[异步调度：依赖图、代次与双缓冲](algorithms/async-scheduling.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/trainium/continuous-batching.md | [Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md)、[连续批处理：从自回归依赖到迭代调度](algorithms/iteration-batching.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/trainium/floor.md | [Trainium：HBM、SBUF、PSUM 与编译执行](hardware/trainium-dataflow.md)、[系统设计推导：把模型、缓存、执行与目标组合起来](system-semantics/design-synthesis.md) | 平台选择由资源与任务推导，不设置无条件功能底线。 |
| platforms/trainium/hardware.md | [Trainium：HBM、SBUF、PSUM 与编译执行](hardware/trainium-dataflow.md)、[容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md) | 执行资源、数据层级与容量模型。 |
| platforms/trainium/neuron-flash-attention.md | [NKI：Partition 轴、片上生存期与 Attention 数据流](frameworks/nki-tiling.md)、[Attention 的 IO、在线 softmax 与分块归约](algorithms/attention-io.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/trainium/neuron-pytorch.md | [Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/trainium/nxd-inference.md | [Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/trainium/nxd-kv-cache.md | [Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md)、[异构状态缓存：层组、恢复位置与联合分配](algorithms/heterogeneous-state-cache.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| platforms/trainium/profiler.md | [Trainium：HBM、SBUF、PSUM 与编译执行](hardware/trainium-dataflow.md)、[时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)、[Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | 对应平台执行语义与可观测模型；本文不复制安装/命令手册。 |
| platforms/trainium/speculative-decoding.md | [Neuron/NxD：静态程序承接动态生成](frameworks/neuron-serving.md)、[投机解码：分布保持、状态提交与收益模型](algorithms/speculation-budget.md) | 机制正文与平台条件交叉阅读，具体接口绑定所列资料版本。 |
| tooling/accuracy-checker.md | [正确性 Oracle：函数参考、状态变形与服务契约](tools/correctness-oracles.md)、[数值参考：误差尺度、条件数与首次分歧](frameworks/numerical-reference.md) | 保留方法与协议语义，以推导组织独立正文。 |
| tooling/fastapi-serving.md | [服务协议：接收、流式背压与终止语义](system-semantics/service-protocol.md) | 保留方法与协议语义，以推导组织独立正文。 |
| tooling/io-handling.md | [输入输出语义：Tokenizer、Chat Template 与流式文本](system-semantics/tokenization-io.md) | 保留方法与协议语义，以推导组织独立正文。 |
| tooling/lora-serving.md | [多 LoRA 服务：低秩计算、驻留与缓存身份](algorithms/lora-serving.md) | 保留方法与协议语义，以推导组织独立正文。 |
| tooling/openai-api.md | [服务协议：接收、流式背压与终止语义](system-semantics/service-protocol.md) | 保留方法与协议语义，以推导组织独立正文。 |
| tooling/performance-modeling.md | [容量、算术强度与服务 Roofline](hardware/memory-and-roofline.md)、[容量设计：驻留、服务率与阶段配比](workloads/capacity-planning.md) | 保留方法与协议语义，以推导组织独立正文。 |
| tooling/profiler.md | [可观测性：执行事件与采集事件的映射](tools/trace-observability.md)、[时间线分析：从请求路径定位执行瓶颈](tools/timeline-attribution.md)、[Kernel 计数器：吞吐、访存与 Replay 语境](tools/kernel-counter-replay.md)、[框架 Profiler：算子归属、内存与图边界](tools/framework-profiler-scope.md) | 保留方法与协议语义，以推导组织独立正文。 |
| tooling/serving-benchmark.md | [服务指标：事件边界、Token 间隔与用户时间](workloads/metric-events.md)、[到达过程与并发：排队、反馈和饱和点](workloads/arrival-and-concurrency.md)、[工作负载分布：长度、复用、模态与相关性](workloads/input-distribution.md)、[性能实验：稳定态、对照与不确定性](workloads/measurement-comparability.md)、[Goodput：联合 SLO、拒绝和容量前沿](workloads/slo-goodput.md) | 保留方法与协议语义，以推导组织独立正文。 |

## 进一步展开的核心关系

- 统一从依赖等价解释前缀、cross KV、递归检查点和多模态特征复用。
- 用可恢复位置集合解释混合缓存，给出联合分配、淘汰和 LCM 的可复算例。
- 从概率修正和接受边界解释投机，将 KV、SSM、grammar 与随机状态放在同一提交关系中。
- 分离逻辑身份、物理槽和执行代次，连接连续批、graph、取消与跨设备传输。
- 把模型结构、硬件成本、队列、协议与 SLO 串成完整设计，最后将新认识写回对应章节。

来源和实际核对范围见 [登记](SOURCES.md)，四个完整设计实例见 [组合推导](system-semantics/design-synthesis.md)。
