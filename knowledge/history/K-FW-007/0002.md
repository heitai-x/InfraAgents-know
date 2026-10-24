# K-FW-007：Triton Kernel：块程序、布局与融合边界

Triton 用块级程序描述张量工作，性能来自把复用、归约和线程布局映射到目标 GPU 资源。

## 程序实例与张量 tile

一个 program instance 通常负责输出 tile，内部对输入 tile 进行加载、计算和归约。tile 的逻辑形状不直接等同于硬件线程数，编译器布局决定元素分给哪些线程及寄存器。mask 保证尾部越界元素不参与错误读取或计算，但被 mask 的线程和资源未必完全免费。

对矩阵乘法，增加 M/N tile 提高数据复用，增加 K tile 改变循环与片上驻留。更大 tile 也增加寄存器和 shared memory，使可并发 blocks 变少。decode 的小 M 与 prefill 的大 M 需要不同候选配置。

## 融合从数据流推导

把 bias、激活、归一化或量化与相邻算子融合，可以减少中间 HBM 写读和 launch。若融合引入大归约或让原本矩阵高效路径失效，则收益可能消失。一个合理边界让中间量在寄存器/片上消费，同时保持足够并行度。

attention 使用在线 softmax 归约，数学统计量允许沿 KV 分块。全 mask 行、非整 tile、不同 query/KV 长度和 scale 都要遵循原函数。与普通逐元素 kernel 相比，它对归约次序和累计精度更敏感。

## Autotune 和编译成本

autotune key 应包含真实影响最优配置和合法性的 shape/dtype/布局条件。首见形状触发多配置试跑会增加冷延迟；固定 workload 可以预先形成少量候选，动态服务则应控制变体数量。调优目标应覆盖实际 padding 和 launch 路径，而非孤立最漂亮的 tile。

可移植语言不保证同一配置跨 CUDA/ROCm 最优。不同 warp/wave、矩阵指令、寄存器和 LDS/shared memory 约束需要各自调优。引擎应优先复用已支持的 kernel，在明确缺失的形态上才承担新内核的维护成本。

## 从矩阵形状推到程序与线程工作

以 C[M,N]=A[M,K]B[K,N] 为例，输出 program 坐标决定 M/N tile，归约沿 K 分段。A 的地址由逻辑行、K 索引及两个 stride 共同确定，B 同理；只有原始布局与公式吻合时才可把它简化为连续偏移。调度 program 的遍历顺序可以改变相邻块对 A/B 的 cache 复用，但不改变每个输出元素的归约集合。

设定 M=128、N=256、K=96。方案 A 用 BM=32、BN=64、BK=32，产生 16 个输出块、每块 3 次 K 迭代；方案 B 用 64×128×32，产生 4 个输出块。输入 FP16 时，A 每 stage 的 A/B tile 共 6 KiB，B 为 12 KiB；若双缓冲则分别为 12/24 KiB，尚未含 padding、barrier 和其他状态。FP32 输出累加 tile 分别含 2,048/8,192 个值，即 8/32 KiB 的逻辑累计存储。

若每个 program 使用 4 个 CUDA warp，共 128 个线程，均分累计值时平均分别为每线程 16/64 个 FP32 值。这只是逻辑下界式估计：矩阵指令 fragment 布局、临时输入、地址计算及编译器复制会改变实际寄存器需求，不能把 program 的张量维度直接当作线程维度或最终寄存器数。方案 B 复用更强，但块数更少、资源更多，在小矩阵上可能无法填满设备。

M=1 的 decode 将大量 M tile 行填空时，大 tile 的潜在复用并不存在。可以比较更小 M tile、GEMV、split-K 或跨请求聚批；split-K 会引入部分和、合并及不同归约次序。它适合补足并行度的条件需要与额外流量一起计算，不能只看 K 很大。

## 合并访问、片上流水与 spill

逻辑相邻元素应尽量映射到相邻线程的合并内存访问；shared/LDS 中的布局还要配合矩阵加载和 bank 组织。为了满足 tensor 指令而做的 transpose/swizzle 应与加载、消费及写回一致。移植到不同 warp/wave 或矩阵单元时，要重新核对线程布局，不能只把 num_warps 换一个数字。

流水多一个 stage，增加提前加载机会，也增加片上缓冲和活跃寄存器。资源过大可能减少驻留块，寄存器不够还会 spill 到 local memory，产生额外全局层级流量。“本地变量”并不保证只在片上。应从生成程序的寄存器/shared 使用、spill 和实际 occupancy 解释时延，再判断是否缩 tile、减 stage 或改变消费顺序。

尾块需要同时处理读写和归约身份。矩阵乘缺失的 K 元素可作为零贡献；attention 中越界 key 应从可见性中排除，全 mask query 需定义输出；归约最大值不能把无效元素无条件填零。预取越界地址也可能有问题，不能等到最终 store mask 才保护。非连续 stride、对齐变化、非整除维度是必要的验证点。

## 调优预算需要服务于完整工作负载

TVM 给出计算与 schedule 分离、tensorization 和显式同步的机制；Ansor 把结构选择、参数搜索、成本模型与实际硬件测量分开。它们说明搜索空间不仅是 BM/BN 两个数，还包括融合边界、缓存位置、归约划分和布局。候选首先满足合法性，再进入性能测量。

调优优先级可以由调用频率、当前关键路径占比和近期可获得的改善共同决定。设某 kernel 每请求出现 32 次，每次可节省 2 μs，则理想串行贡献为 64 μs；只出现一次的 kernel 即使快 20 μs，端到端贡献也更小。存在重叠时应看关键路径，不能机械相加。两个候选只差测量噪声时，应增加有意义的重复或保留更简单可靠的方案。

Autotune key 应包含影响合法性与性能的 dtype、shape、stride、mask模式和目标架构，不必加入与选择无关的请求 ID。试跑会写输出/状态的 kernel 需要隔离或恢复相同输入，避免后面的候选得到被前一个候选修改的 KV、随机状态或计数。首次编译、调优开销和缓存驻留应与稳定态分开报告。

最后对调优集之外的合法形状及真实服务混合验证，保留库实现或简单实现作为独立参考。大 tile/深流水在大批规则矩阵上可能占优，小 M、长尾、竞争性并发或寄存器紧张时则可能反转。通用编译语言、手写 CUDA 和厂商库都应按同一完整路径比较，不把历史论文中的某一次胜负变成永久能力排序。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRITON：Triton：Fused Attention](../sources/S-TRITON.md)：Triton 的 fused attention 示例展示块级张量程序、在线 softmax 归约、tile 划分与相应配置选择。逻辑 tile 和实际线程/寄存器布局有映射关系。
  [完整原文](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-ROCM：HIP：Hardware implementation](../sources/S-ROCM.md)：HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。
  [完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-TVM：TVM](../sources/S-TVM.md)：固定 v1 分离图级改写、张量计算、schedule 与硬件 intrinsic，描述融合、布局、协作加载、tensorization、显式同步和 code generation/runtime。
  [完整原文](../../references/S-TVM/reader.html) · [全文文本](../../references/S-TVM/fulltext.txt)

- [S-ANSOR：Ansor](../sources/S-ANSOR.md)：固定 v1 用结构草图、参数采样、进化搜索和学习成本模型优化完整张量程序，并按目标函数分配跨子图调优预算。
  [完整原文](../../references/S-ANSOR/reader.html) · [全文文本](../../references/S-ANSOR/fulltext.txt)

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [CUDA 平台：SM 数据流、占用率与执行粒度](../hardware/cuda-execution.md)。
- [ROCm 平台：CU、Wave 与 AMD 算子路径](../hardware/rocm-execution.md)。
