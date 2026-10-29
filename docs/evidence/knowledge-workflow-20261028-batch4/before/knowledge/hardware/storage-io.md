# K-HW-012：存储 I/O：请求粒度、队列和加载流水

SSD 带宽是某种访问模式下的能力，推理实际获得的是经过文件布局、操作系统、主存缓冲和设备传输后的供给速度。加载与卸载设计需要逐段记字节、计排队，并明确每个完成事件释放什么资源。

## 文件范围不是张量布局

权重按 shard、tensor 或目标设备分片保存在文件中；执行器可能按层、专家或 tile 访问。一个逻辑读取范围可能跨多个文件或物理 extent，非连续张量还可能引入多个小读取与后续重排。将相邻数据合并可摊薄提交成本，但读取不需要的数据会增加流量与暂存占用。

[ServerlessLLM](../sources/S-SERVERLESSLLM.md) §4 提供按目标布局整理权重和分块流水的实例；[Linux 6.12 的 blk-mq 与 iomap 文档](../sources/S-LINUX-IO.md) 区分文件范围映射、软件队列、硬件派发与完成。操作系统和设备可能重排请求，完成顺序不保证与提交顺序相同；应用应按对象、范围和代次接收结果。

## 页缓存、mmap 与直接 I/O

缓冲读取利用页缓存，重复读取可能不再访问 SSD；mmap 让文件范围进入虚拟地址空间，但首次访问仍可能触发缺页、读取与内存压力。映射成功既不证明内容已驻留，也不证明访问时没有等待。多个进程可能共享只读文件页，而转换后的私有布局仍可能各持一份副本。

直接 I/O 绕过页缓存，便于由应用管理块池和并发，但对齐、文件系统和具体路径有约束，也可能触发其他同步工作。固定 iomap 资料还区分排队完成、部分完成和特定条件下的回退；不能把某个接口接受请求当成所有字节都已交付。直接 I/O 不自动等于异步，也不自动等于 SSD 直接把数据送到 GPU。

选择取决于重用：频繁复用且页缓存足够时，缓存路径可能更便宜；单遍大流量与精确应用缓存管理可能适合直接 I/O。比较时同时测冷/热缓存、读取 CPU 时间、实际存储字节和内存峰值，避免把热缓存读速误记为 SSD 带宽。

## 请求大小、IOPS 与队列深度

设每次请求大小 b、完成率 I 次/s，数据带宽为 Ib。小块可能先达到 IOPS 或 CPU 提交上限，大块则可能受设备/链路带宽约束；顺序与随机访问还会受到布局、设备内部并行和文件系统行为影响，不能一律假设代价相同。

设定吞吐目标 4 GiB/s，平均 I/O 完成时间 1 ms，请求大小 1 MiB。在稳定且这些均值匹配同一人群的简化模型下，至少约需 4096 次/s×0.001 s≈4.096 个在途请求，因此考虑至少 5 个并发槽。它是带宽时延积的起点；增加深度可能继续提高吞吐，也可能只增加排队和尾延迟，必须测实际曲线。

队列预算同时限制请求数与在途字节。预取较远的权重不能饿死即将被消费的 KV；多个模型冷启动也会争夺同一 SSD、文件服务器、主存和 PCIe 上游链路。优先级需要对应消费 deadline、重用收益和公平性，不能单纯让提交最快的线程独占全部槽。

## SSD → DRAM → GPU 的一块数据

一块读取依次经过：预留目的缓冲 → 发起文件范围读取 → 确认已完成范围和错误状态 → 必要的校验/布局转换 → 提交设备传输 → 设备数据就绪 → kernel 消费结束。主机缓冲只有在读取完成且后续 DMA 不再读取后才可复用。部分失败只使对应范围不可用，不能把未写满的旧缓冲发布为新 tensor。

设定 16 MiB 块，SSD 读取 4 GiB/s、主机到设备 16 GiB/s，单块理想时间分别为 3.90625 ms 和 0.9765625 ms。两阶段无资源竞争时，连续 n 块的理想完成时间是两阶段启动之和再加 (n−1) 倍慢阶段时间。转换、CPU 调度、共享主存带宽和真实 I/O 尾部会增加成本；仅为隐藏 1 ms 传输而无限扩大预取深度没有收益保证。

写出还要区分应用缓冲可重用、数据在存储栈中完成、以及满足所需持久性。可重建的 KV 缓存与不可丢失的制品/状态记录有不同要求；声称能够跨进程或机器故障恢复时，必须明确发布与持久化边界，而不能只依据普通写调用返回。

绕过主机缓冲、让存储控制器直接向设备内存 DMA，是这条路径上的另一个设计点：它去掉主存中转的一次写读与对应的主存带宽占用，前提是存储栈、驱动与设备都支持把设备内存作为 DMA 目标（固定 GPUDirect 文档 §2.7 描述了外部 DMA 访问 GPU 内存所需的注册与映射条件）。它不改变存储本身的 IOPS、请求粒度与排队规律，也不替代布局转换——若文件布局不是执行布局，转换仍要在设备上以 kernel 完成。可测的项是：主存带宽占用是否下降、单块延迟是否因少一跳而缩短、以及直通路径不可用时的回退成本。本库没有归档该路径的实测，它只是"中转层数"这一轴上的一端。轴上未列点同样合法。

## 与卸载策略共同验证

构造连续大权重、随机专家块、长 KV 范围和多模型同时加载四种路径，扫描块大小与并发。记录有效字节、实际读取字节、排队、短读/错误、CPU 时间、主存峰值和用户指标。对取消、请求重排、延迟完成和块槽复用，检查旧结果不会覆盖新对象。

存储层改善只在它位于关键路径时转化为推理收益。模型已常驻时，更快 SSD 可能主要改善启动；状态频繁往返 SSD 时，容量可行但在线 ITL 仍可能不可行。应与 [多级驻留](../algorithms/tiered-residency.md) 的保留、预取与重算选择一起计算。

<!-- infraagents:provenance -->

## 来源与推导

- [S-GPUDIRECT：NVIDIA GPUDirect RDMA — CUDA 12.8.0](../sources/S-GPUDIRECT.md)：§2.7 描述外部 DMA 引擎访问 GPU 内存所需的注册、映射与撤销条件；本章只据此说明存储直通路径的前提，不报告其吞吐。
  [完整原文](../../references/S-GPUDIRECT/reader.html) · [全文文本](../../references/S-GPUDIRECT/fulltext.txt)

- [S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation](../sources/S-LINUX-IO.md)：Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。
  [完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

- [S-SERVERLESSLLM：ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](../sources/S-SERVERLESSLLM.md)：§4 的加载格式、块池和多级流水；§5 的 token 重算迁移；§6 的局部性与启动时间模型。
  [完整原文](../../references/S-SERVERLESSLLM/reader.html) · [全文文本](../../references/S-SERVERLESSLLM/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
- [多级驻留：有效副本、预取、淘汰与重算](../algorithms/tiered-residency.md)。
- [KV 卸载与预取：读集合、恢复点与迁移预算](../algorithms/kv-offload-prefetch.md)。
- [主机准备、DMA 与流水缓冲](../hardware/host-device-transfer.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
