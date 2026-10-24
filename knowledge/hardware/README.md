# 硬件平台

本分类条目的适用条件、来源与未验证范围在各正文中说明。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-HW-001：容量、算术强度与服务 Roofline](memory-and-roofline.md) | v2 · seed | 容量决定系统能容纳哪些状态，带宽与计算能力决定搬运和处理这些状态的下界，关键路径决定哪些下界实际显现。 |
| [K-HW-002：通信拓扑：从 Rank 组到瓶颈链路](topology-parallelism.md) | v3 · seed | 拓扑、方向与共享链路决定通信成本；将理想ring启动项和发送流量放到具体带宽假设中解释。 |
| [K-HW-003：主机准备、DMA 与流水缓冲](host-device-transfer.md) | v4 · seed | 主机准备、DMA、计算与消费按对象依赖连接；量化有限批、各池槽位与共享带宽，允许不同放置与传输方式。 |
| [K-HW-004：数值格式、打包布局与矩阵指令](precision-and-kernel-support.md) | v2 · seed | 固定架构/指令能力、量化参数布局与实际 dispatch 分别核验，防止把存储格式、指令支持和目标内核性能混为一谈。 |
| [K-HW-005：CUDA 平台：SM 数据流、占用率与执行粒度](cuda-execution.md) | v2 · seed | CUDA 优化应把算法的数据复用映射到 SM 的寄存器、共享存储和 warp 执行，再判断主机提交是否成为瓶颈。 |
| [K-HW-006：ROCm 平台：CU、Wave 与 AMD 算子路径](rocm-execution.md) | v2 · seed | ROCm 移植应保持模型语义，同时按 AMD 的执行资源、矩阵指令和库 dispatch 重新推导数据布局与粒度。 |
| [K-HW-007：Apple Silicon：统一内存、带宽与交互式推理](metal-unified-memory.md) | v2 · seed | 统一内存减少显式设备间搬运，但 CPU/GPU 仍共享容量、带宽并通过依赖控制可见性。 |
| [K-HW-008：Trainium：HBM、SBUF、PSUM 与编译执行](trainium-dataflow.md) | v2 · seed | Trainium 的高效执行来自把张量分块映射到片上存储和专用引擎，同时让静态可执行程序承接动态服务状态。 |
| [K-HW-009：CPU 推理：线程、缓存层级与 NUMA 分区](cpu-serving.md) | v2 · seed | CPU 服务优化需要同时管理向量算术、内存带宽、线程调度与请求级并发，更多线程不等于更多有效工作。 |
| [K-HW-010：CPU GEMM/GEMV：寄存器、缓存与线程划分](cpu-linear-kernels.md) | v2 · seed | CPU 线性层需把 shape 的重用机会落实到寄存器、cache/TLB、打包和线程分区；同一配置不适用于全部 GEMV/GEMM。 |
| [K-HW-011：GPU 矩阵与归约流水：布局、资源和完成依赖](gpu-kernel-pipelines.md) | v2 · seed | GPU tile、片上布局和异步流水共同受寄存器、共享存储及完成依赖约束；GEMM 与 attention 的最优组织随形状和架构改变。 |
| [K-HW-012：存储 I/O：请求粒度、队列和加载流水](storage-io.md) | v1 · seed | 实际存储供给由布局、pagecache/直接 I/O、请求粒度、队列和 staging 共同决定；提交、可读、持久化和可回收分开建模。 |
| [K-HW-013：共享内存路径：统一地址、物理驻留与显式迁移](shared-memory-paths.md) | v1 · seed | 统一地址、可迁移驻留和物理共享内存是不同访问关系；重用、页放大、可见性和带宽竞争共同决定复制选择。 |
| [K-HW-014：通信传输：注册、完成、可见与回收](transport-and-rdma.md) | v1 · seed | 网络提交、CQE、消费者可见与应用接管是不同事件；注册、代次和完成协议共同决定缓冲何时可回收。 |
