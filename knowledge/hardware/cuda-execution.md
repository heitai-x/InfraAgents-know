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

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities](../sources/S-CUDA-CAPABILITY.md)：CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。
  [完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [Triton Kernel：块程序、布局与融合边界](../frameworks/triton-kernels.md)。
