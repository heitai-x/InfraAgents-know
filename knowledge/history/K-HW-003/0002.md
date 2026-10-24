# K-HW-003：主机准备、DMA 与流水缓冲

输入流水的吞吐取决于准备、传输和计算三个阶段的平衡；数据可见性和缓冲所有权决定能否正确重叠。

## 准备不是一次复制

文本服务包含解析和 tokenization，多模态还包含媒体解码、采样、resize、特征整理。CPU 输出通常要整理成连续数组，必要时经 pinned staging，再由 DMA 到设备。每一步可能持有一份副本，峰值主存和锁页容量都进入资源预算。

pageable 主存的异步传输可能经过额外 staging；pinned memory 提供 DMA 所需的稳定物理页面，但锁页本身有成本，也占用系统不可自由换出的内存。将所有主存永久锁页会改变其他进程的可用容量。

## 三阶段流水的推导

若准备、传输、计算分别耗时 3、2、8，串行每批 13；在独立资源和足够缓冲下，稳态周期可接近 8，但第一批延迟仍至少 13。为准备 n＋1、传输 n、计算 n−1 同时存在，需要不同缓冲代次，不能仅靠一块可变数组实现三段重叠。

主机发送缓冲在 DMA 读取完成前保持不变；设备输入在 kernel 读完前保持不变；D2H 输出在传输完成后才能由 CPU 消费。返回一个异步句柄与数据已经可读是不同事件，数据访问必须依赖完成关系。

## 何时减少传输次数更有效

小而频繁的 token IDs、lengths 和 sampling metadata 可能受调用与同步成本主导，合并提交比只减少字节更有效。大图像或 KV 状态则更接近带宽问题。将采样留在设备可以减少逐 token D2H 同步，却需要在设备/主机之间明确 EOS 与输出交付时机。

缓存输入特征可减少重复 processor 和传输，但缓存身份必须涵盖处理配置。实际管线的最慢阶段也可能随并发改变：单请求 GPU 忙，多个媒体请求却让 CPU 解码饱和。各阶段队列长度与完成时间能够区分这种迁移。

## 多级输入与共享带宽

输入还可能从 SSD 或远端存储进入主存，再完成解码、转换、锁页 staging 与 H2D。每段需要分别记有效字节和实际传输字节；布局转换可能读写整个对象，多个阶段又可能共享主存控制器或同一 PCIe 上游链路。理想三阶段取最大值的模型只有在资源与依赖允许时成立。

小 metadata 应先分析提交次数和同步，大权重/KV 应先分析可持续带宽。若同时预取权重、读取媒体和执行 CPU 冷部分，原先单独测得的 DMA 带宽不一定仍可用。[存储 I/O](storage-io.md) 与 [计算放置](../algorithms/heterogeneous-execution.md) 需要使用同一份资源预算。

## 缓冲池的槽位与代次

一个槽位依次承担读取目的、转换输入、传输源和待回收对象。真正的回收点是所有消费者结束，而不是最后一个主机函数返回。I/O 可以乱序完成，CPU 取消也不能倒转已提交的 DMA；完成记录必须保留对象、范围、请求与代次。

预留缓冲数可以由阶段时延、吞吐和每批字节估计，但应限制最大在途字节。过深队列可能增加等待与取消成本，甚至挤出本来能驻留的 KV。发生输入读取或转换错误时，不向下一阶段发布部分旧/新混合的数据，并让已在途工作完成资源清理。

## 共享路径的替代方案

物理共享内存、映射主机页和可迁移统一内存有不同的字节与可见性关系，见 [共享内存路径](shared-memory-paths.md)。选择零显式复制路径时，仍需检查实际远端读取、缺页和 CPU/GPU 竞争；不能仅凭应用没有调用复制接口就认定传输成本为零。

验证分别改变输入大小、重复次数、CPU 准备强度和设备消费速度，观察各队列、实际完成时间及缓冲峰值。若更深流水只增加队列而未减少设备空洞，说明限制可能来自共享资源或必需依赖，应缩小在途预算或重分配工作。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。
  [完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation](../sources/S-LINUX-IO.md)：Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。
  [完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [视觉语言模型：Patch、模态桥接与联合位置](../models/vision-language.md)。
