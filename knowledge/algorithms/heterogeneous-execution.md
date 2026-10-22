# K-ALG-020：CPU–GPU 计算放置：传数据、传结果与共同提交

异构执行可以按层、算子、输出分片或专家分配计算。决策比较的是完整依赖路径上的计算、传输、同步和容量，而不是孤立的 CPU 与 GPU 峰值算力。

## 三类候选先保持同一函数

第一类让 CPU 保存部分权重、需要时搬到 GPU 计算；第二类让这些权重对应的计算留在 CPU，只交换输入/输出；第三类让 CPU/GPU 同时处理同一层不同部分，再合并结果。若全权重能合法常驻 GPU，也应保留为对照，不能人为引入搬运再证明异构更快。

按输出通道切分 Y=XW 时，双方的结果可以按原通道位置拼接；按归约轴切分则需要求部分和。norm、softmax 或非线性跨越分片时，不一定允许先局部应用再合并。两边的数据布局、dtype、尺度和累计顺序必须服从相同数值契约。

## 成本比较和共享资源

冷权重大小 W𝒸、输入输出交换量 A、链路带宽 Bₗ、CPU 冷部分时间 T𝒸、GPU 冷部分时间 T𝗀。简单的搬权重路径约为 W𝒸/Bₗ+T𝗀；留在 CPU 路径约为 A/Bₗ+T𝒸+同步/合并。可以重叠的区间再按依赖图处理，不能把原本串行的传输全部视为隐藏。

CPU 计算还会与 tokenization、I/O 和 DMA 竞争核与主存带宽；GPU 冷/热部分也可能竞争同一矩阵或内存资源。把它们放到不同线程或 stream 不会自动产生独立能力。容量上同时计权重副本、输入/结果缓冲与在途代次。

## Attention 留在 CPU 的一种条件

若长历史 KV 已在主存，搬整个历史到 GPU 的字节可能远大于交换 query 和 attention 输出。设相同形状下历史为 N 个位置，KV 流量随 N 增长，而单步 query/结果不随历史线性增长；[FlexGen](../sources/S-FLEXGEN.md) §4.2 据此讨论把部分 decode attention 交给 CPU。

这不是所有长历史都适合 CPU 的结论。CPU 的 KV 读取、点积、softmax、布局和线程开销仍需计算；多个请求会竞争主存。若模型采用窗口、MLA、压缩或状态已经在 HBM，原字节比例又会改变。把计算移过去必须实现同一 mask、位置、归一化与精度关系。

## 热点神经元路线的前提

[PowerInfer v1](../sources/S-POWERINFER.md) 通过离线激活统计安排热点在 GPU、冷部分在 CPU，再用预测器和稀疏算子减少执行工作。其评估包括 OPT 以及特定 ReLU/ReGLU 模型变体；不能据此假设普通稠密模型都具有相同稀疏性。

预测为活跃但实际为零可能增加工作，预测为不活跃而实际有贡献则可能改变输出。若要保持原函数，必须有补算/验证或其他足够条件；若允许近似，则把预测器、校准分布和质量回归纳入模型契约。热点统计只表示给定分布下的访问机会，输入变化或 batch 增大都可能改变冷部分的规模。

## 场景：同一冷权重的三种放置

以下全部时间与带宽为设定值。某层冷权重 512 MiB，设备容量不足以永久容纳它，链路为 16 GiB/s；CPU 冷部分计算设为 8 ms，GPU 冷部分设为 1 ms，GPU 热部分设为 6 ms。CPU 路径全部输入与结果交换合计 2 MiB，额外同步 1 ms，结果合并 0.5 ms。

搬权重需 31.25 ms，即使与热部分充分重叠，仍需再执行冷部分；含合并的理想路径约为 max(31.25,6)+1+0.5=32.75 ms。CPU/GPU 分工若在共同输入就绪后可独立执行，则约为 max(6,8+0.1221+1)+0.5=9.6221 ms。这个计算只证明在这些假设下存在比较优势，真实共享带宽、依赖与计算时间还需测量。

第三个候选是把整层留在 CPU，减少层内同步，却让热部分也失去 GPU 计算机会；其总时间必须独立测量。若增加 batch 后，CPU 冷部分变成 80 ms，而搬运加 GPU 计算只需约 40 ms，选择可能反转。若容量允许全部驻留 GPU，之前 31.25 ms 的搬运项消失，也必须重新比较。

### 从执行依赖到验证

为一次层执行记录输入版本、双方分片、各自完成事件、结果原位置和输出代次。两边成功后才提交共同输出；一边失败不能把另一边的局部结果当成完整层。取消后等待既有访问结束，再回收缓冲；迟到 CPU 结果不能被接到新一轮 GPU 输出上。

语义检查先用固定稠密参考验证分片/拼接或部分和，再检查不同完成顺序、空分片、非整除边界和重复使用缓冲。若加入稀疏预测，另做精确稀疏参考、误预测与任务质量测试，不能用固定 seed 文本偶然相同替代这些关系。

性能记录每段 CPU/GPU 时间、实际交换字节、重叠区间、主存竞争与用户时延。扫描 batch、冷比例、历史长度和其他主机负载；逐层局部收益还需经过完整模型累加与服务调度验证。交互小批、长 KV 在主存或明显热点是可研究条件，不能直接构成通用的异构执行规则。

<!-- infraagents:provenance -->

## 来源与推导

- [S-POWERINFER：PowerInfer](../sources/S-POWERINFER.md)：§3–6 以激活局部性、预测器、神经元放置和 CPU/GPU 结果合并组织混合执行；§8.1 明确评估模型包含特定 ReLU/ReGLU 变体。
  [完整原文](../../references/S-POWERINFER/reader.html) · [全文文本](../../references/S-POWERINFER/fulltext.txt)

- [S-FLEXGEN：High-throughput Generative Inference of Large Language Models with a Single GPU](../sources/S-FLEXGEN.md)：§3–4 将生成、状态容量、层/批遍历与 GPU/CPU/disk 放置一起建模；§4.2 讨论把部分主存 KV 对应计算交给 CPU。
  [完整原文](../../references/S-FLEXGEN/reader.html) · [全文文本](../../references/S-FLEXGEN/fulltext.txt)

- [S-CPU-TUNING：PyTorch Performance Tuning Guide](../sources/S-CPU-TUNING.md)：PyTorch CPU 调优资料涉及线程组织、NUMA、向量化与执行配置；线程数和内存归属影响真实可用资源。
  [完整原文](../../references/S-CPU-TUNING/reader.html) · [全文文本](../../references/S-CPU-TUNING/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [CPU GEMM/GEMV：寄存器、缓存与线程划分](../hardware/cpu-linear-kernels.md)。
- [主机准备、DMA 与流水缓冲](../hardware/host-device-transfer.md)。
- [共享内存路径：统一地址、物理驻留与显式迁移](../hardware/shared-memory-paths.md)。
- [KV 卸载与预取：读集合、恢复点与迁移预算](../algorithms/kv-offload-prefetch.md)。
- [权重与专家卸载：重用、路由与在途版本](../algorithms/weight-expert-offload.md)。
- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
