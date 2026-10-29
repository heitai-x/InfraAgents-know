# K-HW-005：CUDA 平台：SM 数据流、占用率与执行粒度

CUDA 优化应把算法的数据复用映射到 SM 的寄存器、共享存储和 warp 执行，再判断主机提交是否成为瓶颈。

## 执行层级与数据层级

grid 由 thread blocks 组成，block 内线程按 warp 调度，block 的寄存器和共享存储需求决定能有多少 blocks 同时驻留。增加 occupancy 可以隐藏访存延迟，但过多线程若降低每线程复用或增加竞争，并不必然更快。矩阵计算、标量归约和数据搬运常有不同瓶颈。

连续线程访问连续地址通常有利于合并内存事务；不规则页表访问需要把索引开销和数据读取分开看。L2 复用可能让实际 HBM 字节低于张量总读取字节，反之非合并加载会搬入未使用数据。布局设计应从线程实际读取哪些元素推导。

## Attention 的 tile 选择

较大的 Q/K tile 可以复用数据和降低中间写入，但消耗寄存器、共享内存和归约空间。长历史 decode 的 query 太少时，沿 KV 切分可增加并行度，再合并 softmax 统计。不同 SM 代际具有不同资源和指令能力，因此一个 tile 配置不应作为所有 GPU 的通用常量。

## 从 Eager 到图执行

大量很短 kernel 使 CPU dispatch 与 launch 间隙显现，CUDA Graph 可以摊薄重复提交成本。若主成本是一个长 GEMM 或 HBM 读取，图不会直接改变这部分工作。融合减少 launch 和中间流量，却可能增加寄存器压力；图减少提交但保留内部 kernel。两种机制可以组合，收益来源不同。

GPU、CPU 和 copy engine 能否重叠受依赖与硬件资源约束。event 建立时序，stream 不表示专用硬件通道。多进程并发、频率变化和共享设备又会影响可用资源；采用实测可持续能力代入 roofline 比套用某代峰值更有用。

平台基线由场景推导：小请求关注 launch 与状态管理；长 prefill 关注 attention IO 与矩阵利用率；大 batch decode 关注 KV 流量；多卡关注集合通信。不存在一个对所有场景都最优的固定功能清单。

## 场景：单 GPU 交互服务的完整选择

以下全部为设计设定。一个固定 GQA decoder 在 GPU 上驻留权重 14 GiB；设备可用容量 24 GiB，工作区 2 GiB、graph/固定缓冲 1 GiB、预留余量 1 GiB。模型状态设为 32 层、8 个 KV head、128 维、每元素 2 字节；8 个活跃请求各有 2048 个已计算历史位置，目标 ITL 为 30 ms，主机与响应预算暂设 3 ms。有效设备读取带宽假设为 1 TiB/s，具体 kernel 能力仍需测量。

每 token 全层 KV 为 128 KiB，每请求 256 MiB，8 个请求共 2 GiB；所列容量合计 20 GiB。若仅用平均长度推导 24 个请求可占用 6 GiB KV，就会恰好用完剩余空间，没有留下本轮增长或额外工作区，不能直接作为硬准入上限。长 prompt、写时复制和图桶必须重新进入峰值账。

比较 A：全驻留、已验证 backend、eager 连续批；B：加入受时间预算约束的分块 prefill、合法前缀共享及固定缓冲的图执行；C：在任务允许且质量验证通过时使用更低精度 kernel。A 是语义与成本对照，B 改变状态/执行组织，C 还改变数值路径，应先逐项区分，再评估最终组合。

在“每轮读取全部权重和每请求完整 KV”的设定下，8 个 decode 的读取量为 16 GiB，下界为 15.625 ms。30−3−15.625=11.375 ms 是其他开销的剩余预算上限，不是已经可供 prefill 使用的时间；实际矩阵、同步、额外流量与排队仍会消耗它。混合批的可行块长需从对应形状的服务曲面得到。

若 8 个请求都共享同一合法 1024 token 前缀，唯一 KV 可以由 2 GiB 降到 128 MiB 公共部分加 8×128 MiB 私有部分，即 1.125 GiB。decode 的逻辑可见历史仍未减少；实际 HBM 流量是否因 cache 重用下降要另测。适配器、模型或 processor 身份不同，则不能使用这个共享算例。

B 的图缓冲需要稳定地址与代次保护；更多桶和流水缓冲可能挤占 KV。C 的低位路径要核对布局、尺度与质量，FA3 的某个形状实验不能直接填入该服务的性能估计。GPU 已受 HBM 流量限制时，减少主机提交未必改变主项；短请求和大量小 kernel 的情况则可能相反。

验证从固定前缀的独立 forward、冷/热前缀、非整除页与 batch 重排开始，再覆盖取消后复用、长 prefill 与 decode 交错、图桶切换和低位异常值。性能固定到达轨迹，记录 TTFT/ITL、有效完成、主机空洞、实际字节与工作区峰值，使用独立重复确认选择。

低负载下 A 的简单路径可能足够；高前缀复用使 B 更有价值；超长历史会把重心移向 attention 和容量；质量门槛收紧可能排除 C。需求或硬件改变后重新计算容量与关键路径，而不是保留一份固定“GPU 优化清单”。

## 按 compute capability 的执行资源与特性事实

固定 CUDA 12.8.0 编程指南按 compute capability 给出每个 SM 的驻留上限、共享存储配置和特性支持。下表只抄录常见代际的行；它们是硬件与驱动的事实，不是本库的设定算例，表外代际须按同一来源逐项核对。

| 项目 | 7.5 | 8.0 | 8.6 | 8.9 | 9.0 |
|---|---|---|---|---|---|
| 每 SM 最大驻留 block 数 | 16 | 32 | 16 | 24 | 32 |
| 每 SM 最大驻留 warp 数 | 32 | 64 | 48 | 48 | 64 |
| 每 SM 最大驻留线程数 | 1024 | 2048 | 1536 | 1536 | 2048 |
| 每 SM 共享存储上限 | 64 KB | 164 KB | 100 KB | 100 KB | 228 KB |
| 每 block 共享存储上限 | 64 KB | 163 KB | 99 KB | 99 KB | 227 KB |
| 每 SM 32 位寄存器数 | 64 K | 64 K | 64 K | 64 K | 64 K |
| 每线程最大寄存器数 | 255 | 255 | 255 | 255 | 255 |

所有列的 warp 大小为 32、每 block 最多 1024 线程、共享存储 bank 数为 32、每线程本地存储上限 512 KB。超过 48 KB 的每 block 共享存储须以动态共享存储显式申请。统一数据缓存在 8.0/8.7 为 192 KB、在 8.6/8.9 为 128 KB、在 9.0 为 256 KB，可划给共享存储的档位随之不同。

特性方面，硬件加速的异步复制（memcpy_async 与 cp.async 指令族）、split arrive/wait barrier 和 L2 驻留管理从 8.x 起可用；线程块簇、分布式共享存储、TMA 单元和 DPX 指令从 9.0 起可用；bf16 算术从 8.x 起可用。8.9 与 9.0 的第四代 Tensor Core 接受 FP8 输入，8.0/8.6/8.7 的第三代 Tensor Core 不含 FP8。128 位整数原子只在 9.0 及以后提供。

这些行对设计的意义是约束而不是配方。8.6/8.9 每 SM 1536 线程、每 block 99 KB，使“深流水大 tile”在片上容量上比 9.0 紧得多：三级 32 KiB 输入缓冲占 96 KiB，已接近单 block 上限且不留 barrier 和转置区；同一 kernel 在 9.0 上可用 227 KB。tile 配置与级数因此不能跨代际照搬。没有 TMA 与簇时，地址生成和跨 block 的部分和交换要由普通线程指令与全局存储承担，流水的组织方式见 [GPU 流水](gpu-kernel-pipelines.md)。特性表缺失某项，只说明该硬件路径不存在，不说明该问题无解。

轴上的点：驻留数与每线程寄存器/共享存储是同一资源的两个方向；缺失的硬件异步能力可由软件流水、更多 block 并行或改变数据划分补偿。轴上未列点同样合法。

## L2 驻留与访问策略窗口

8.0 及以后的设备可以划出一部分 L2 作为持久访问的保留区，并通过访问策略窗口为一段连续全局内存声明持久属性；窗口的命中比例 hitRatio 决定其中多大比例的访问按持久处理，其余按流式处理。固定指南的例子：保留区 16 KB、窗口 32 KB、hitRatio 0.5 时，硬件随机选取约 16 KB 作为持久部分；hitRatio 1.0 则试图缓存整个窗口，保留区小于窗口时按最近使用逐出。多个 stream 的窗口竞争同一保留区时，用小于 1 的比例可以避免互相抖动。

它对推理的意义在于把“反复读取的小对象”与“每轮扫过一次的大对象”分开：每步重读的位置表、小 adapter、分页索引或热块适合持久窗口，整份权重或长历史 KV 远大于 L2 时持久化只会把流式访问挤出去。窗口可以设在 stream 上，也可以作为 CUDA Graph 的 kernel 节点属性，因此图执行不排斥它。

前提与失效条件：MIG 模式下保留区功能被禁用；MPS 下保留区大小只能在服务启动时设定。效果必须用 HBM 事务计数或工作集扫描区分，不能从“设置成功”推出命中。表征方法见 [GPU 从零表征](../tools/gpu-characterization.md) 的 L2 一节。

## 持久 kernel 是 launch 轴上的另一端

每轮 decode 的工作可以是几十到上百个独立 kernel（eager），可以是一次图重放，也可以是一个常驻整步的 kernel 按内部工作队列顺序执行各层。三者位于同一条轴上：提交次数与依赖解析成本从“每 kernel 一次”到“每步一次”再到“每次服务开始一次”递减，代价是调度灵活性、资源占用和实现复杂度递增。

不变量不随位置改变：各层之间的数据依赖必须由某种同步表达（kernel 边界、图边或 kernel 内的 barrier 与标志）；workspace 与 KV 页的所有权仍按代次保护；取消与抢占需要一个可以插入的边界。常驻 kernel 把这些边界移进 kernel 内部，launch 与间隙成本由 block 间同步与负载不均衡取代；它占用的 SM、寄存器和共享存储在整步内不释放，其他工作（采样、复制、另一批）能否并行取决于它留下的资源。

固定来源没有给出这一端的实测收益；选择它的理由只能来自测得的 launch/间隙占比与同步成本，见 [GPU 从零表征](../tools/gpu-characterization.md) 的 launch 一节。轴上未列点同样合法。

## 时钟、功耗与 ECC 改变“可持续”的含义

固定 Nsight Compute 说明，为了减少测量随时钟状态变化，它默认把 GPU 时钟限制到基准值；驱动的热限流不受该设置控制并总是优先；replay 的后几遍可能处在更高时钟状态。因此 profiler 内读到的 kernel 时间与服务稳态并非同一条件，比较时要声明时钟控制方式。

固定 Best Practices 给出两种带宽口径：理论带宽由内存时钟、接口宽度和倍率算出，有效带宽由程序实际读写字节 (B_r+B_w) 除以时间得到。在 GDDR 设备上开启 ECC 会把可用显存减少 6.25%，并因每次事务额外取 ECC 位使有效带宽下降约 20%，具体幅度随访问模式变化；HBM2 有专用 ECC 资源，不付这笔带宽。容量账和流量账都应使用实际配置下的数值。

对单卡交互服务，这些条件意味着 ITL 预算中的带宽与算力项取持续负载下的可持续点，而不是空闲后短时 boost 的峰值；同一配置在另一机箱、另一功耗上限下需要重新测量。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities](../sources/S-CUDA-CAPABILITY.md)：CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。
  [完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-NCU：Nsight Compute — Profiling Guide](../sources/S-NCU.md)：Nsight Compute 提供 kernel 级硬件计数器和派生指标，某些采集需要 replay。计数器能力、采集模式及恢复条件影响所观察的执行。
  [完整原文](../../references/S-NCU/reader.html) · [全文文本](../../references/S-NCU/fulltext.txt)

### 代际事实的定位

能力表与特性行抄录自固定 CUDA 12.8.0 编程指南的 Feature Support 与 Technical Specifications per Compute Capability 两表及 §16.7–16.8 的架构说明；L2 保留区与访问策略窗口来自 §3.2.3；时钟控制来自 Nsight Compute §2.6.2；ECC 与两种带宽口径来自 Best Practices §8.2。表中数值可在归档的 reader.html 中逐项复核。持久 kernel 一节没有来源实测，只是 launch 与依赖成本的推导。

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-FLASHATTENTION3：FlashAttention-3](../sources/S-FLASHATTENTION3.md)：§2–3 以 Hopper 的 TMA、异步 WGMMA、warp specialization 和多级缓冲组织 attention；讨论寄存器、布局和 FP8 数值限制。
  [完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [Triton Kernel：块程序、布局与融合边界](../frameworks/triton-kernels.md)。
- [GPU 矩阵与归约流水：布局、资源和完成依赖](../hardware/gpu-kernel-pipelines.md)。
- [GPU 从零表征：可持续带宽、矩阵吞吐与测量条件](../tools/gpu-characterization.md)。
