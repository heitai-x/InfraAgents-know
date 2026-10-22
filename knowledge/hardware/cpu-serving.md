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

## 建立 CPU 执行能力表

对目标 CPU 记录实际可用核、硬件线程、NUMA 域、cache 层级、页配置、有效内存带宽和运行额度。指令集名称只是能力入口：硬件支持、操作系统状态管理、编译产物和实际调度路径同时满足，某种向量或矩阵内核才可使用。BF16/INT8 张量存在并不证明它们用了相应加速指令。

[GEMM/GEMV 专题](cpu-linear-kernels.md) 将向量宽度、寄存器块、cache/TLB 与打包连接起来；[低位执行](../frameworks/cpu-quantized-kernels.md) 区分压缩读取、tile 解包和整数点积。在本章的服务层，要把这些局部选择落实为不同 batch、历史长度和并发下的内核与线程预算，而不是使用统一的“最大线程”设置。

## 绑核、内存放置与副本策略

绑核约束计算在哪里运行，内存策略约束页在哪里建立和迁移，两者需要共同核对。在采用首次触碰策略的环境中，单线程初始化权重可能把页集中到一个 NUMA 域；仅在随后把计算线程分散到两侧，不能自动获得两侧本地带宽。大页可以改变 TLB 压力，但其分配、容量和碎片条件也要测量，不能只根据页大小推断收益。

比较三种结构：每个 NUMA 域完整副本、跨域共享只读权重、按层或输出通道划分权重。完整副本减少远端读取，但增加常驻和预打包容量；共享副本节省容量，可能把读取集中到一个内存控制器；分片可以使用多域容量和带宽，却引入中间结果或归约通信。选择还受模型能否单域驻留、请求并发和服务隔离要求影响。

设定两个各 16 核的域，一个完整服务实例的权重与固定区为 20 GiB，单域可用内存为 32 GiB。双副本在容量上可行，每个副本仍只剩 12 GiB 处理 KV、激活和工作区；将两个域合成一个实例虽然扩大了总容量，却不能假定一个请求就获得双倍带宽。要分别比较吞吐、单请求延迟和跨域流量。

## 线程池、等待和预取

让长期线程池复用工作线程，可以减少每个算子的创建成本；池太大又会占用调度、分词和网络所需 CPU。自旋等待降低部分唤醒延迟但持续耗核，阻塞等待降低空闲消耗但增加唤醒时间。有效预算以部署中真实可运行的 CPU 为准，不能用宿主机总核数替代受配额约束的核数。

预取需要可预测的地址和合适提前量。提前过少不能覆盖延迟，过多可能挤出即将重用的数据；稀疏或动态专家访问还可能浪费带宽。它不增加内存总带宽，不能解决已经饱和的主存通路。连续布局和批次重用往往要先于低层预取调参判断。

## 用信号区分四类退化

带宽上限、cache/TLB 工作集、线程同步和输入流水是不同解释。联合检查每阶段 CPU 时间、有效吞吐、上下文切换、缓存/地址翻译相关事件及 NUMA 流量；计数器名称和含义必须匹配目标 PMU。perf 统计中的复用缩放、采集开销和受权限限制的事件也应显式记录。

实验依次改变线程数、请求并发、绑核/放置、M/K/N 和预打包复用次数，避免一次同时改变全部条件。若 kernel 变快而服务 ITL 退化，检查算子池是否挤占了调度线程；若 CPU 总利用率低而排队增长，检查单一执行域或内存通路。最终验证仍包含跨请求缓存隔离、取消和长时间资源守恒。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CPU-TUNING：PyTorch Performance Tuning Guide](../sources/S-CPU-TUNING.md)：PyTorch CPU 调优资料涉及线程组织、NUMA、向量化与执行配置；线程数和内存归属影响真实可用资源。
  [完整原文](../../references/S-CPU-TUNING/reader.html) · [全文文本](../../references/S-CPU-TUNING/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-GOTO：Anatomy of High-Performance Matrix Multiplication](../sources/S-GOTO.md)：§3–6 从多级存储模型推导 GEMM 的分解、panel 打包、cache/TLB 条件和实现参数。
  [完整原文](../../references/S-GOTO/reader.html) · [全文文本](../../references/S-GOTO/fulltext.txt)

- [S-INT8：Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](../sources/S-INT8.md)：§2 给出仿射量化、整数点积、零点补偿、bias 和输出重新缩放；附录 A–B 讨论组合算子、舍入及 NEON 中间累加条件。
  [完整原文](../../references/S-INT8/reader.html) · [全文文本](../../references/S-INT8/fulltext.txt)

- [S-PERF：Linux perf-stat manual](../sources/S-PERF.md)：perf stat 统计指定执行范围内的 CPU 性能事件，事件类型、计数对象和运行区间决定结果含义。
  [完整原文](../../references/S-PERF/reader.html) · [全文文本](../../references/S-PERF/fulltext.txt)

## 延伸阅读

- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
- [PyTorch 推理：模块模式、Autograd 与状态突变](../frameworks/inference-eval.md)。
