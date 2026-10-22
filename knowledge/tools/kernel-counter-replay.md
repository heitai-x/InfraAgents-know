# K-TOOLS-003：Kernel 计数器：吞吐、访存与 Replay 语境

硬件计数器帮助识别单 kernel 的资源限制，但它测到的执行可能与正常并发服务具有不同环境。

## 计数器连接哪种模型

计算吞吐、内存事务、缓存命中、占用率和 stall 等指标分别对应不同资源。带宽接近可持续上限且算术强度低，支持流量受限；低带宽也低算力可能来自并行度不足、依赖长或访存不规则，不能直接判定“没有瓶颈”。

occupancy 高表示更多执行资源驻留，不能证明有效工作多。stall 分类说明调度器观察到等待的原因，但常存在因果链：寄存器压力降低驻留，导致访存延迟无法隐藏，最终表现为内存等待。应连接 tile、布局和资源使用解释。

## Replay 改变了什么

有限硬件计数器可能需要多次执行同一 kernel 采集不同组。replay 可改变 cache 热度、输入状态、并发和时序；某些捕获方式还要保存恢复内存。对会修改持久状态的 kernel，工具恢复能力与采集模式尤其相关。

孤立 kernel 的有效带宽不能直接加总成服务带宽，共享 HBM、L2 和其他执行单元会改变并发行为。将单 kernel 优化接回完整服务后，还需要检查它是否仍位于关键路径。

## 有解释力的使用方式

先由时间线确定代表 kernel 与实际 shape，再用少量指标回答明确问题：是否重复读同一数据、是否 spill、tile 是否过大、矩阵指令是否有效使用。随后用相同输入验证数值，并在正常执行环境确认整体效果。CUDA、ROCm 和 Trainium 使用各自工具，但这种从问题到计数器的关系可迁移。

## CPU 计数必须固定运行范围与分母

[perf 的固定版本文档](../sources/S-PERF.md) 区分线程、进程、CPU 集合和系统范围，并说明计数缩放、事件分组和复用。CPU IPC 为同一范围和时段内 instructions/cycles；若两个量在不同阶段采集，非平稳的 prefill/decode 混合会使比值失去对应含义。

硬件计数器不足时，多个事件轮流占用计数器。缩放将观测外推到启用区间，只有实际计数区间具有代表性时才有良好解释。记录实际运行比例并优先选择回答当前问题的一小组事件；长时间复用平均数不能自动还原短 burst。跨核心迁移、SMT 共用资源及系统范围事件还需要结合调度和目标身份分析。

设定 2×10⁹ 条指令、1×10⁹ cycles，IPC 为 2；另一次是 1×10⁹ 条指令、0.75×10⁹ cycles，IPC 约 1.33。第二次 IPC 更低，却使用更少 cycles。若频率和有效工作相同，它可能更快。因此 SIMD 或融合减少指令后，不能只以 IPC 升降决定优劣。实际耗时还受频率、等待和并发影响。

## 从计算与访存结构选择计数

| 机制假设 | 需要相互印证的证据 | 不能直接推出的结论 |
|---|---|---|
| CPU 权重流量受限 | 每步有效权重字节、内存带宽、线程扩展曲线 | cache miss 多就一定是 DRAM 饱和 |
| 地址或 TLB 压力 | 可用的地址翻译事件、页布局与工作集对照 | 任一 cache miss 都是 TLB miss |
| 低位解包占主导 | unpack/scale 阶段时间、实际 dispatch、计算与流量变化 | 权重位数下降就会同比加速 |
| GPU 寄存器压力 | 编译资源、local-memory 流量、驻留和 tile 对照 | occupancy 较低就必须缩小 tile |
| 异步流水等待 | 数据就绪时序、阶段等待、共享存储与资源使用 | 存在异步指令就已经隐藏延迟 |
| 网络或存储阻塞 | 同一范围字节、完成速率、排队和消费间隙 | CPU/GPU 低利用率就没有资源瓶颈 |

具体事件名称、含义和可用性依 PMU、GPU 代际及工具版本确定。指令能力与实际执行路径应分别验证；源码声明支持某种 dtype 不代表本次形状走到了对应矩阵或向量指令。

## 流量账与计数器需要同一层次

理论张量字节通常是有效负载；事务包含对齐、未合并访问、缓存行和重复读取。相同张量的 HBM、L2、设备间和主机传输计数不能直接相加后称为一种带宽。先说明正在考察哪条链路、读写是否都计入，再用有效字节与事务字节之比分析放大。

例如设有效读取为 512 MiB，目标链路记录 768 MiB，则放大为 1.5 倍；它说明存在额外流量，但不能仅由比值分辨尾块、重复读取和其他请求。需要限定采集范围或改变布局/工作集形成不同预测。共享控制器的计数尤其不能自动归属于单个进程。

## 保持 replay 与正常服务的对应

[Nsight Compute 固定版本](../sources/S-NCU.md) 的 kernel、application 和 range replay 具有不同的保存、重放及并发条件。选择模式时应核实 kernel 是否与主机或其他 kernel 协作，以及工具能否保持所需状态。多次 application replay 还要求能够正确匹配同一个逻辑执行实例。

内核计数用于解释资源限制，正常服务测量用于确认效果。若单 kernel 更快却造成更多临时内存、同步或调度等待，最终方案仍可能更慢；这类反例应回到完整资源账，而非继续增加指标来回避服务结果。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NCU：Nsight Compute — Profiling Guide](../sources/S-NCU.md)：Nsight Compute 提供 kernel 级硬件计数器和派生指标，某些采集需要 replay。计数器能力、采集模式及恢复条件影响所观察的执行。
  [完整原文](../../references/S-NCU/reader.html) · [全文文本](../../references/S-NCU/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-ROCM：HIP：Hardware implementation](../sources/S-ROCM.md)：HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。
  [完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-PERF：Linux perf-stat manual](../sources/S-PERF.md)：perf stat 统计指定执行范围内的 CPU 性能事件，事件类型、计数对象和运行区间决定结果含义。
  [完整原文](../../references/S-PERF/reader.html) · [全文文本](../../references/S-PERF/fulltext.txt)

- [S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation](../sources/S-LINUX-IO.md)：Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。
  [完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

## 延伸阅读

- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [Triton Kernel：块程序、布局与融合边界](../frameworks/triton-kernels.md)。
- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
