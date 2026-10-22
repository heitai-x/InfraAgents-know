# 框架

本分类条目的适用条件、来源与未验证范围在各正文中说明。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-FW-001：PyTorch 推理：模块模式、Autograd 与状态突变](inference-eval.md) | v1 · seed | 推理行为由模型模式、梯度记录、dtype/device、模块状态与执行路径共同决定。 |
| [K-FW-002：数值参考：误差尺度、条件数与首次分歧](numerical-reference.md) | v1 · seed | 一个有解释力的数值参考保持任务函数和输入一致，再分层衡量实现重排与近似产生的差异。 |
| [K-FW-003：Stream 与分配器：依赖顺序和对象生存期](streams-allocator.md) | v1 · seed | stream 解决执行顺序，内存生存期解决地址何时可重用；正确的异步程序需要同时满足两种关系。 |
| [K-FW-004：编译与动态形状：Guard、分桶和变体成本](compile-dynamic-shapes.md) | v1 · seed | 编译把一组可证明成立的执行假设变成专门程序；动态服务需要管理这些假设覆盖的形状与状态空间。 |
| [K-FW-005：CUDA Graph：捕获边界、计划状态与变长服务](cuda-graph-buffers.md) | v1 · seed | CUDA Graph 重放一组已捕获的设备工作；服务动态性通过稳定缓冲中的值、分桶和捕获外控制面承接。 |
| [K-FW-006：Attention Backend：语义、布局与执行契约的交集](attention-backends.md) | v1 · seed | 选择 attention backend 是匹配可见性、张量几何、状态布局和执行模式，性能排名只对确定的运行点成立。 |
| [K-FW-007：Triton Kernel：块程序、布局与融合边界](triton-kernels.md) | v1 · seed | Triton 用块级程序描述张量工作，性能来自把复用、归约和线程布局映射到目标 GPU 资源。 |
| [K-FW-008：MLX 服务：惰性求值、编译与请求状态](mlx-serving.md) | v1 · seed | MLX 的数组表达式可以延迟执行，服务循环必须明确计算何时提交、何时完成和何时释放状态。 |
| [K-FW-009：Neuron/NxD：静态程序承接动态生成](neuron-serving.md) | v1 · seed | Neuron 服务通过形状桶、设备驻留状态和分布式执行组，把可变请求映射到编译后的程序族。 |
| [K-FW-010：NKI：Partition 轴、片上生存期与 Attention 数据流](nki-tiling.md) | v1 · seed | NKI 的 tile 是带硬件轴语义的片上对象；布局与生存期共同决定 DMA、矩阵和归约引擎的供给。 |
| [K-FW-011：权重加载：从制品身份到可执行布局](checkpoint-loading.md) | v1 · seed | 加载必须同时建立张量语义、执行布局、完整参数与就绪边界；流式流水的价值由启动关键路径和峰值副本决定。 |
| [K-FW-012：CPU 低位执行：解包、补偿与累加路径](cpu-quantized-kernels.md) | v1 · seed | 低位存储、解包和实际乘加是不同执行路径；补偿、组尺度、累加宽度和舍入共同决定正确性与收益。 |
