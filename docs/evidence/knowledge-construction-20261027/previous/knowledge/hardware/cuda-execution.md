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

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities](../sources/S-CUDA-CAPABILITY.md)：CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。
  [完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-FLASHATTENTION3：FlashAttention-3](../sources/S-FLASHATTENTION3.md)：§2–3 以 Hopper 的 TMA、异步 WGMMA、warp specialization 和多级缓冲组织 attention；讨论寄存器、布局和 FP8 数值限制。
  [完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [Triton Kernel：块程序、布局与融合边界](../frameworks/triton-kernels.md)。
