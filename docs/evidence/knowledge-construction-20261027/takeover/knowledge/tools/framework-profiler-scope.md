# K-TOOLS-004：框架 Profiler：算子归属、内存与图边界

框架 profiler 解释模型操作怎样落到设备工作，并通过 shape、调用栈与内存事件连接两层视图。

## 算子与 kernel 并非一一对应

一个高层算子可能触发多个 kernel、临时复制和同步；多个算子可能被编译融合成一个 kernel。host operator duration 与设备持续时间也不同。eager 与 compiled trace 的命名粒度变化不表示工作凭空消失，需要沿输入输出和生成图解释。

shape 记录可以发现意外的全历史重算、padding、KV 展开和非连续复制；内存事件可以区分持久 KV、临时激活和分配器保留。reserved memory 与真实活跃 tensor 字节不等同，释放引用也未必让缓存分配器立即归还驱动。

## 观测开销的来源

记录 stack、shape、内存和大量事件增加主机工作，shape 追踪还可能延长张量引用。对 launch 受限系统，这些额外成本尤其显著。分段采集与短代表窗口有助于减少扰动，但必须覆盖问题发生的状态转换。

惰性框架需要明确求值边界；编译平台需要把 host 阶段、executable 和设备时间关联。只看到函数调用顺序而看不到实际求值，不足以判断设备忙闲。

## 五类平台的观测落点

| 平台 | 系统尺度 | 设备或内核尺度 | 与框架视图的连接 |
|---|---|---|---|
| CUDA | Nsight Systems 的 API、stream、copy 与主机时间 | Nsight Compute 的 kernel 计数器 | 用范围和 correlation 关联具体算子实例 |
| ROCm | rocprofv3 的 HIP/HSA、dispatch 与传输记录 | GPU 对应硬件计数器与派生指标 | 保留进程、agent、queue、dispatch 身份 |
| Metal | MLX 求值边界与 Metal 对象标签 | 固定版本支持的 GPU capture 与 Xcode 操作依赖视图 | 把 MLX 数组求值连接到实际 Metal 工作；未被此来源覆盖的计数器能力另行核实 |
| Trainium | Neuron Explorer 的 runtime、API、加载与主机事件 | NeuronCore 指令 trace 与依赖 | 设备 trace 与匹配的编译 NEFF 对应，解释实际 executable |
| CPU | 线程调度、进程 CPU 时间与 framework spans | perf 的 cycles、instructions、cache 等可用事件 | 区分算子计算、线程竞争与主机预处理 |

这些工具回答不同尺度的问题。ROCm 的 counter 可用集合依设备而变，多个指标可能需要多遍采集；Trainium 系统 trace 说明调用路径，设备 trace 才解释片上执行；Metal 的统一内存仍需观察 CPU/GPU 并发和容量压力。CPU IPC 是 instructions/cycles，较低值可能来自访存、依赖或调度，不是某个原因的唯一签名。

跨平台可以共用“关键路径、字节流量、有效工作”的问题表述，但事件名称和采集机制应按平台分别解释。将不同厂商同名 utilization 百分比直接比较，通常缺少共同分母。

## 从框架视图提出可检验解释

若每 token 都出现大张量 concatenate，可能在复制历史；若首次每种 shape 都有长编译段，可能 guard 变体过多；若某个 scalar extraction 后出现空洞，可能产生同步。每个解释都应回到具体张量的大小、位置和生存期，再用更窄的观测或干预验证。

## 从待裁决问题选择采集层

工具选择可以从证据需求反推。若问题是输出错误，首先需要模型边界和独立数值参考；若是 batch 变化后多出复制，需要 shape、布局与内存生存期；若是设备空洞，需要主机准备、提交和同步；若是片上资源限制，再进入专用计数器。直接对整段服务启用最细采集，常会增加开销而仍缺少业务关联。

对每个候选解释写出能让它与其他解释分开的信号，并确定现有工具是否提供。若低精度路径疑似回退，框架算子名只能定位候选位置，还需要对应 shape、实际生成执行路径和数值结果；若图缓存反复失效，需要编译/捕获次数与变体身份，而不只是累计编译时间。

## 内存事件要接到状态生命周期

权重、持久 KV、临时激活、编译工作区、固定图缓冲和分配器保留属于不同用途。框架看到的 tensor 分配不一定覆盖外部分配器、通信注册或驱动内部空间。比较峰值时应注明来自框架活跃字节、分配器 reserved、设备总占用还是主机 RSS，不能将不同口径互相替代。

设框架记录活跃 tensor 10 GiB、分配器保留 14 GiB、设备整体占用 16 GiB。4 GiB 差额可能包括可复用缓存与碎片，另 2 GiB 则可能属于框架观测范围外；这些只是待核查分类，不能凭减法证明全部可回收。需要在明确的同步完成点比较对象生存期，并结合实际容量失败位置。

惰性求值与图执行还会延后或合并工作。记录定义时刻、实际求值和输出消费三个边界，才能解释一个 host 操作到底创建了描述、提交了执行，还是等待既有执行完成。

## 工具不可用时仍能保留哪些结论

权限不足、硬件不支持或固定版本缺少某种事件时，保留可执行的最小观测：客户端完成时间、主机经过时间、有效输入输出规模和明确同步后的结果。它们可以验证候选是否满足目标，也可以用容量/流量下界排除明显不可能的解释，但不足以证明某个 GPU stall 类别或 CPU 缓存层是原因。

例如无法取得设备计数器时，可以比较同一数值路径在不同 batch、上下文长度和合法布局下的曲线，并结合已知字节模型形成假设。只有当额外证据足以区分计算、带宽和调度时，才收敛到具体归因。不同平台的工具能力允许不对称，结论的精度也应随证据调整。

## 用分层测量控制观测扰动

先做无 profiler 的重复服务测量，再捕获代表窗口的框架/系统事件，最后针对少量实例采集细粒度计数。每层保留模型版本、shape、负载阶段、执行模式与采集选项，确保局部解释能对应到实际请求。

代表窗口至少覆盖冷/热、典型/尾部或其他对当前假设有影响的运行点。对启用 stack、shape 或 memory 记录后的行为改变，应记录扰动，必要时缩小窗口或拆开采集；不能用被 profiler 改变的排队分布直接宣称生产尾延迟改善。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TORCH-PROFILER：PyTorch — torch.profiler](../sources/S-TORCH-PROFILER.md)：框架 profiler 可以采集算子、设备工作、shape、stack 和内存相关信息。更详细的采集会增加运行开销，部分选项也会延长张量引用。
  [完整原文](../../references/S-TORCH-PROFILER/reader.html) · [全文文本](../../references/S-TORCH-PROFILER/fulltext.txt)

- [S-MLX：MLX：Lazy Evaluation](../sources/S-MLX.md)：MLX 的惰性求值先建立计算图；显式求值或主机访问结果等操作触发执行。过密求值和过大的未求值图会产生不同开销。
  [完整原文](../../references/S-MLX/reader.html) · [全文文本](../../references/S-MLX/fulltext.txt)

- [S-NKI：Trainium2 Architecture Guide for NKI](../sources/S-NKI.md)：已读 Trainium2 NKI 资料区分 HBM、SBUF、PSUM、DMA 与计算引擎；片上张量的 partition/free 轴和转置路径具有具体硬件意义。
  [完整原文](../../references/S-NKI/reader.html) · [全文文本](../../references/S-NKI/fulltext.txt)

- [S-ROCPROF：ROCprofiler-SDK：rocprofv3](../sources/S-ROCPROF.md)：ROCprofiler-SDK 的 rocprofv3 区分 API/activity tracing 与 kernel counter collection，记录进程、agent、queue、dispatch 等身份。计数器可用集合和并行采集能力依 GPU 而定。
  [完整原文](../../references/S-ROCPROF/reader.html) · [全文文本](../../references/S-ROCPROF/fulltext.txt)

- [S-METAL-TOOLS：MLX — Metal Debugger（固定历史文档）](../sources/S-METAL-TOOLS.md)：MLX 的固定 Metal Debugger 文档说明 GPU capture、Metal 对象标签，以及在 Xcode 中打开与重放 trace、查看操作依赖。它不单独证明完整 Instruments 时间线或所有硬件计数器的覆盖。
  [完整原文](../../references/S-METAL-TOOLS/reader.html) · [全文文本](../../references/S-METAL-TOOLS/fulltext.txt)

- [S-NEURON-EXPLORER：Neuron Explorer：Get started](../sources/S-NEURON-EXPLORER.md)：Neuron Explorer 区分系统事件与设备指令 trace；系统视图覆盖 runtime、API、加载与主机，设备视图需要匹配的编译产物与 trace。
  [完整原文](../../references/S-NEURON-EXPLORER/reader.html) · [全文文本](../../references/S-NEURON-EXPLORER/fulltext.txt)

- [S-PERF：Linux perf-stat manual](../sources/S-PERF.md)：perf stat 统计指定执行范围内的 CPU 性能事件，事件类型、计数对象和运行区间决定结果含义。
  [完整原文](../../references/S-PERF/reader.html) · [全文文本](../../references/S-PERF/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [PyTorch 推理：模块模式、Autograd 与状态突变](../frameworks/inference-eval.md)。
