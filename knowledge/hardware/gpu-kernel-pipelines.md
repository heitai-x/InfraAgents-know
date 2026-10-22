# K-HW-011：GPU 矩阵与归约流水：布局、资源和完成依赖

一个 GPU kernel 的设计同时决定数据怎样进入计算单元、哪个线程持有什么片段、临时状态占多少资源，以及何时允许覆盖。tile 更大、流水更深或矩阵指令更多，都只有在这些关系成立时才可能改善性能。

## GEMM 的数据依赖与资源账

对 C=A·B，令输出块为 Mₜ×Nₜ、沿归约轴每次处理 Kₜ。一个阶段读取 A 的 Mₜ×Kₜ 片段和 B 的 Kₜ×Nₜ 片段，输出 accumulator 跨多个 K 阶段保留。数据依赖是：全局读取 → 片上数据就绪 → 矩阵乘加 → accumulator 可消费 → epilogue → 输出可见。下一个 K 阶段的读取可以提前，但不能覆盖仍被矩阵指令读取的片上缓冲。

若输入每元素 s 字节、流水深度 n，单 block 的输入缓冲主项为 n·s·(MₜKₜ+KₜNₜ)，输出累加主项为 a·MₜNₜ，a 为累加元素字节。还要计入对齐、barrier、转置区、地址与 epilogue 临时量。共享存储、寄存器、线程数和 block 数约束共同限制驻留；仅检查共享存储够用不足以证明没有 spill。

设定 Mₜ=Nₜ=128、Kₜ=64、输入 2 字节、累加 4 字节。每级输入缓冲为 32 KiB，三级为 96 KiB；输出 accumulator 为 64 KiB。若由 128 个线程均摊，仅输出就平均需要每线程 128 个 32 bit 寄存器，尚未计其他变量。改成两级减少 32 KiB 片上缓冲，但能否增加驻留 block，还取决于寄存器与目标架构限制。这个算例描述资源占用，不指定任意 GPU 的最佳 tile。

## 先适配指令布局，再安排线程

矩阵指令要求的 fragment 布局不等于用户矩阵的行列布局。全局合并访问、共享存储 bank、矩阵操作数布局与输出写回应作为一条路径比较。局部转置或 swizzle 增加地址/搬运工作，却可能减少非合并事务和 bank 冲突；也可能让后续消费者再支付一次转换。

固定来源中，[FlashAttention-3 v1](../sources/S-FLASHATTENTION3.md) §2–3 以 Hopper 的 TMA、异步 WGMMA 和 warpgroup 组织流水，warpgroup 由 4 个 warp 组成。这是具体架构上的实现入口。[ROCm 固定文档](../sources/S-ROCM.md) 则描述 CDNA 的 MFMA、VGPR/AGPR 与 LDS。两条路径的线程片段、寄存器资源和异步能力需要分别核实，不能将 CUDA 指令、32 线程约定或某一 bank 数直接替换成 AMD 名称。

相同 dtype 也可能有不同操作数布局限制。FA3 对 FP8 的讨论说明，前一 GEMM 的 accumulator 要交给后一 GEMM 时，转换与排列是实际成本；FP16 路径可用不意味着只更改 dtype 就能得到合法 FP8 kernel。低精度还要重新验证 scale、累加、异常值和任务质量。

## 生产者、消费者与流水深度

将部分线程负责加载、其他线程负责计算，可以让指令发射与资源分配更清晰。环形缓冲的每一代至少需要两个条件：生产者确认旧消费者已结束，消费者确认新数据已就绪。只等待复制完成不能证明矩阵指令已不再读取该缓冲；同一个槽反复使用还必须区分代次。

设一个 tile 的加载、计算分别为 L、C。足够缓冲与独立资源下，稳态周期下界接近 max(L,C)，第一块仍要付出加载与计算的启动路径。流水更深可覆盖更长延迟，但增加驻留状态、barrier 与寄存器需求，可能降低并行度或引入 spill。多个 stream 也不意味着多个专属计算或复制引擎，重叠效果需要从实际时间线确认。

## Attention 中归约与矩阵流水怎样连接

Q tile 与多块 K/V 组合时，QK 产生分数，在线 softmax 更新每行最大值 m 和归一化质量 ℓ，随后把加权 V 合入未归一化输出 U。若最大值从 mₒ 改为 mₙ，旧质量与旧 U 都要乘 exp(mₒ−mₙ)，最后才用 ℓ 归一化；完整推导见 [attention IO](../algorithms/attention-io.md)。

核对缩放方向的设定反例：两块分数分别为 0、2，对应 V 分别为 1、0。输出应为 1/(1+exp(2))≈0.1192，落在 V 的凸包内。若把旧累加量向相反方向放大，会立即破坏该关系。应从归一化恒等式检查伪代码，不能仅因材料是固定论文就照抄所有公式排版。

QK、softmax、PV 在同一块内有顺序依赖；跨块或跨 warpgroup 可以重叠部分矩阵与标量工作，但要同时保留所需分数、概率和输出代次。额外 Bq×Bk 分数片段会增加寄存器压力。增大 Bq 提高 K/V 重用，增大 Bk 改变加载和归约粒度；长历史 decode 的少量 query 又可能需要拆分 KV 以获得并行度，再支付部分统计合并成本。

## 如何验证选择及其反转

在代表 shape 上分别比较单/多级缓冲、普通/专门分工、不同 tile 和融合程度。检查实际生成的指令路径、共享存储与寄存器、spill、内存事务及时间线，而不把源代码中的异步调用当成已经重叠的证据。正确性覆盖非整除尾部、全 mask 行、不同历史长度、GQA 映射和归约次序。

短输入或小矩阵可能由启动与调度支配，深流水的启动/收尾反而较贵；长 prefill 更容易摊薄这些成本；长历史小 batch decode 可能受 KV 带宽与并行度约束。最终对照还要回到服务 SLO，避免局部 kernel 的工作区或驻留资源挤占可服务请求。

## 流水的首尾与短输入必须单独推导

固定 [FA3 v1](../sources/S-FLASHATTENTION3.md) 的多级伪代码展示了重叠意图，但打印的循环边界和部分状态交接不足以直接作为实现。例如两阶段示意在只有两块 K/V 时，需要明确计算第二块分数，并依次使用两块各自的概率与 V；不能让循环为空后仍引用未生成的最后概率。独立时序表应保证每个必要 K/V 块恰好计算一次、输出使用同一最大值尺度、收尾按最终归一化分母处理，并覆盖 1、2、3 块、非整除尾部和取消前的最后访问。

其三阶段实验也说明，更多理论重叠可能因额外概率片段和缩放状态挤占寄存器，迫使 tile 变小；实际编译结果又未得到预期的全部重叠。因此候选选择应同时比较算法依赖、生成指令和资源占用，不能只比较源代码里的阶段数。

<!-- infraagents:provenance -->

## 来源与推导

- [S-FLASHATTENTION3：FlashAttention-3](../sources/S-FLASHATTENTION3.md)：§2–3 以 Hopper 的 TMA、异步 WGMMA、warp specialization 和多级缓冲组织 attention；讨论寄存器、布局和 FP8 数值限制。
  [完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

- [S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../sources/S-FLASH.md)：FlashAttention 通过片上分块和在线 softmax，避免将完整 attention 分数与概率矩阵写入设备主存；目标函数仍是精确 attention 的数学形式。
  [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-ROCM：HIP：Hardware implementation](../sources/S-ROCM.md)：HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。
  [完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [CUDA 平台：SM 数据流、占用率与执行粒度](../hardware/cuda-execution.md)。
- [ROCm 平台：CU、Wave 与 AMD 算子路径](../hardware/rocm-execution.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [Triton Kernel：块程序、布局与融合边界](../frameworks/triton-kernels.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
