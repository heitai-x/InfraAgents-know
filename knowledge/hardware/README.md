# 硬件平台

本分类条目的适用条件、来源与未验证范围在各正文中说明。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-HW-001：容量、算术强度与服务 Roofline](memory-and-roofline.md) | v2 · seed | 容量决定系统能容纳哪些状态，带宽与计算能力决定搬运和处理这些状态的下界，关键路径决定哪些下界实际显现。 |
| [K-HW-002：通信拓扑：从 Rank 组到瓶颈链路](topology-parallelism.md) | v1 · seed | 多卡性能由数据经过的实际路径决定；rank 数只是拓扑模型的一部分。 |
| [K-HW-003：主机准备、DMA 与流水缓冲](host-device-transfer.md) | v1 · seed | 输入流水的吞吐取决于准备、传输和计算三个阶段的平衡；数据可见性和缓冲所有权决定能否正确重叠。 |
| [K-HW-004：数值格式、打包布局与矩阵指令](precision-and-kernel-support.md) | v1 · seed | 硬件支持某种 dtype、库能接受该 dtype、目标形状有高效 kernel，是三个不同层次的能力。 |
| [K-HW-005：CUDA 平台：SM 数据流、占用率与执行粒度](cuda-execution.md) | v1 · seed | CUDA 优化应把算法的数据复用映射到 SM 的寄存器、共享存储和 warp 执行，再判断主机提交是否成为瓶颈。 |
| [K-HW-006：ROCm 平台：CU、Wave 与 AMD 算子路径](rocm-execution.md) | v1 · seed | ROCm 移植应保持模型语义，同时按 AMD 的执行资源、矩阵指令和库 dispatch 重新推导数据布局与粒度。 |
| [K-HW-007：Apple Silicon：统一内存、带宽与交互式推理](metal-unified-memory.md) | v1 · seed | 统一内存减少显式设备间搬运，但 CPU/GPU 仍共享容量、带宽并通过依赖控制可见性。 |
| [K-HW-008：Trainium：HBM、SBUF、PSUM 与编译执行](trainium-dataflow.md) | v1 · seed | Trainium 的高效执行来自把张量分块映射到片上存储和专用引擎，同时让静态可执行程序承接动态服务状态。 |
| [K-HW-009：CPU 推理：线程、缓存层级与 NUMA 分区](cpu-serving.md) | v2 · seed | CPU 服务优化需要同时管理向量算术、内存带宽、线程调度与请求级并发，更多线程不等于更多有效工作。 |
| [K-HW-010：CPU GEMM/GEMV：寄存器、缓存与线程划分](cpu-linear-kernels.md) | v1 · seed | CPU 线性层需把 shape 的重用机会落实到寄存器、cache/TLB、打包和线程分区；同一配置不适用于全部 GEMV/GEMM。 |
