# K-HW-006：ROCm 平台：CU、Wave 与 AMD 算子路径

ROCm 移植应保持模型语义，同时按 AMD 的执行资源、矩阵指令和库 dispatch 重新推导数据布局与粒度。

## 可移植 API 背后的不同执行资源

HIP 的线程、block 和 stream 模型与 CUDA 有相近抽象，但底层 CU、SIMD、wave/warp 大小、寄存器和 LDS 资源依架构不同。AMD CDNA 与 RDNA 也不能视为同一组常量。移植时硬编码 32 线程 warp、shuffle mask 或共享内存布局，可能产生语义或效率问题。

矩阵运算在适当架构上使用 MFMA 等指令。寄存器压力决定同时驻留 wave 数，LDS 用于 workgroup 内合作。CUDA 上有效的 tile 与线程分配不一定匹配这些资源，尤其是非标准 head_dim 和小 batch decode。

## 算子库与引擎的接口

ROCm attention 可能走 PyTorch SDPA 的可用实现、FlashAttention 的 AMD 路径、Triton 或 AITER 等库。真正兼容性是模型语义、dtype、shape、paged layout、mask、图支持和目标架构的交集。AITER 是 AMD 算子实现入口，不意味着所有模型形态在所有 GPU 都有同样支持。

权重转换与量化格式也可能按 kernel 专门预打包。跨 backend 共用模型文件不等于共用设备内布局。集合通信使用对应 ROCm 通信栈时，还应确认 rank、拓扑和 stream 完成语义，而不是只替换库名。

## 性能推导的落点

长上下文的 IO、分块归约、调度和状态所有权仍是同一数学问题；变化的是资源参数与实现。对照时先固定模型输出语义，再测实际 kernel 路径、HBM 流量、CU 利用和主机空洞。某 NVIDIA 插件不可用时可以选择数值等价路径保持接入，但应重新测其成本，不能继承原平台的吞吐结论。

## 将移植拆成语义、布局与资源三层

先以相同固定权重、输入、位置/mask 和 dtype 验证函数；再匹配目标 MFMA/向量路径的 fragment、累加和 LDS 布局；最后选择 block、wave 与流水深度。某个库可以接受 tensor，不代表它选择了目标 GPU 的最佳内核，也不代表与其他 backend 的所有组合均受支持。

在固定 HIP 硬件文档中，CDNA 的矩阵执行、VGPR/AGPR 和 LDS 提供了资源推导入口。具体指令形状、支持精度、bank 行为和异步搬运仍需绑定实际架构与编译目标；不要将一页涵盖多个代际的概述当作每个设备共同具备的能力清单。NVIDIA 的 TMA/WGMMA 调度不能仅替换名称就变成 AMD 流水。

## 预测可区分的性能信号

对相同 shape 分别改变 tile、wave 分配与缓冲级数，检查有效内存事务、LDS 冲突、寄存器压力和实际发射路径。减少 HBM 访问却变慢，可能是额外布局转换、spill、并行度或同步造成；它们需要不同观测，不能统一归因于“ROCm 较慢”。

完整服务还要检查 CPU 提交、集合通信与图支持的实际路径。比较时固定质量和工作量，单独记录原始格式转换与预打包成本；跨平台输出误差与性能差异都应在各自测量范围内解释。

<!-- infraagents:provenance -->

## 来源与推导

- [S-ROCM：HIP：Hardware implementation](../sources/S-ROCM.md)：HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。
  [完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

- [S-AITER：ROCm AITER 官方仓库](../sources/S-AITER.md)：AITER 是 AMD ROCm 的算子实现项目，提供面向相应硬件的 attention、矩阵或相关算子路径。项目覆盖不代表每种形状和设备都有相同实现。
  [完整原文](../../references/S-AITER/reader.html) · [全文文本](../../references/S-AITER/fulltext.txt)

- [S-TRITON：Triton：Fused Attention](../sources/S-TRITON.md)：Triton 的 fused attention 示例展示块级张量程序、在线 softmax 归约、tile 划分与相应配置选择。逻辑 tile 和实际线程/寄存器布局有映射关系。
  [完整原文](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Attention Backend：语义、布局与执行契约的交集](../frameworks/attention-backends.md)。
- [Triton Kernel：块程序、布局与融合边界](../frameworks/triton-kernels.md)。
- [数值格式、打包布局与矩阵指令](../hardware/precision-and-kernel-support.md)。
