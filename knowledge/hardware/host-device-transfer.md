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

<!-- infraagents:provenance -->

## 来源与推导

- S-CUDA-ASYNC：[完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)
- S-CUDA-BEST：[完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)
- S-CUDA-ALLOC：[完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [视觉语言模型：Patch、模态桥接与联合位置](../models/vision-language.md)。
