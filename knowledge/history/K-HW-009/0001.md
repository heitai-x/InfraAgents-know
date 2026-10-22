# K-HW-009：CPU 推理：线程、缓存层级与 NUMA 分区

CPU 服务优化需要同时管理向量算术、内存带宽、线程调度与请求级并发，更多线程不等于更多有效工作。

## 工作粒度与资源

大 GEMM 可以使用多核和向量/矩阵指令，小 batch decode 可能更接近读取权重的带宽问题。CPU cache 容量、NUMA 内存位置和权重打包影响重复读取；低位权重减少容量，却需要合适的解包与向量内核。

线程有两层：每个算子内部的并行线程，以及多个请求/进程的外部并发。若 8 个 worker 各启动 16 个算子线程，而机器只有 32 个可用核，就可能形成严重超额订阅。上下文切换、cache 驱逐与同步成本会增加尾延迟。

## NUMA 归属的推导

多 socket 机器访问本地内存与远端内存的路径不同。把计算线程固定在一侧，却在另一侧首次触碰/分配全部权重，会让大量读取穿过互联。可按 NUMA 域布置完整服务副本，或显式划分模型和内存；选择取决于模型能否在单域容量内驻留和跨域通信成本。

权重预打包优化某类 GEMM，可能增加额外副本。多个 worker 是否共享只读页，受进程创建方式、后续转换和写时复制影响。模型文件映射大小不能直接等同于物理驻留总量。

## 低成本原型可以验证什么

CPU 小模型适合验证请求状态、缓存等价、采样、取消、协议和 RSI 更新链。这些是算法与生命周期性质。它不能替代目标 GPU 的 kernel、图、通信或吞吐验证。将 CPU 用作明确的语义参考，可以让后续硬件优化有可对照的函数。

完整 CPU 服务仍需把解析、tokenization、矩阵计算和响应编码拆开建模。若 Python 调度受限，增加算子线程无助；若带宽饱和，扩大并发也可能只增加排队。用每阶段服务时间与核/内存占用形成资源模型，再分配线程预算。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CPU-TUNING：PyTorch Performance Tuning Guide](../sources/S-CPU-TUNING.md)：PyTorch CPU 调优资料涉及线程组织、NUMA、向量化与执行配置；线程数和内存归属影响真实可用资源。
  [完整原文](../../references/S-CPU-TUNING/reader.html) · [全文文本](../../references/S-CPU-TUNING/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
- [PyTorch 推理：模块模式、Autograd 与状态突变](../frameworks/inference-eval.md)。
