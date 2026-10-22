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

<!-- infraagents:provenance -->

## 来源与推导

- [S-NCU：Nsight Compute — Profiling Guide](../sources/S-NCU.md)：Nsight Compute 提供 kernel 级硬件计数器和派生指标，某些采集需要 replay。计数器能力、采集模式及恢复条件影响所观察的执行。
  [完整原文](../../references/S-NCU/reader.html) · [全文文本](../../references/S-NCU/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-ROCM：HIP：Hardware implementation](../sources/S-ROCM.md)：HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。
  [完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [Triton Kernel：块程序、布局与融合边界](../frameworks/triton-kernels.md)。
- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
