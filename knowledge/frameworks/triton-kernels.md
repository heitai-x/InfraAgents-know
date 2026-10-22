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

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRITON：Triton：Fused Attention](../sources/S-TRITON.md)：Triton 的 fused attention 示例展示块级张量程序、在线 softmax 归约、tile 划分与相应配置选择。逻辑 tile 和实际线程/寄存器布局有映射关系。
  [完整原文](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-ROCM：HIP：Hardware implementation](../sources/S-ROCM.md)：HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。
  [完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [CUDA 平台：SM 数据流、占用率与执行粒度](../hardware/cuda-execution.md)。
- [ROCm 平台：CU、Wave 与 AMD 算子路径](../hardware/rocm-execution.md)。
