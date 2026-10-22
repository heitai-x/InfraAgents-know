# K-HW-004：数值格式、打包布局与矩阵指令

硬件支持某种 dtype、库能接受该 dtype、目标形状有高效 kernel，是三个不同层次的能力。

## 格式定义了可表示的值

浮点格式在指数范围与尾数精度之间分配位数；定点量化依赖外部 scales 和零点。FP16 与 BF16 同为 16 bits，却有不同指数/尾数；FP8 或 FP4 也不是一种唯一格式。输入存储精度、乘法精度、累加精度和输出精度应分别描述。

长归约对累加误差敏感，因此低精度输入配较高精度累加很常见。溢出、下溢和饱和与舍入误差是不同现象。模型有离群通道时，逐组或混合格式可以改变可表示范围，但也改变 scale 加载和布局。

## 指令的形状约束

矩阵指令处理特定 tile、数据类型与布局。理论峰值只有足够对齐、足够大且能连续供数的工作才能接近。小 M decode、大 K 但窄 N、非标准 head_dim 或有大量 padding 的 MoE 分组，都可能落入不同 kernel 路径。

一个库接受 INT4 权重，可能直接使用低位指令，也可能在 tile 内解包到半精度，还可能退回普通算子。三条路径都可能数值正确，性能和临时内存却不同。二进制 ABI、目标架构编译和实际 dispatch 共同决定运行路径。

## 打包格式是执行协议

若相邻低位权重被打包在同一字节，nibble 顺序、分组轴、scale 布局和预转置都要与内核一致。权重文件格式、运行时转换格式和设备内布局可能不同。量化模型加载后再次持有完整高精度副本，会让文件小而驻留并不小。

设计时先从矩阵形状、质量容忍度和瓶颈字节得出候选格式，再核对目标架构和库版本。不能根据“支持 FP8”给全部 attention、MoE、采样和通信推断相同比例加速。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities](../sources/S-CUDA-CAPABILITY.md)：CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。
  [完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](../sources/S-GPTQ.md)：GPTQ 是训练后权重量化方法，使用校准输入所反映的层输入结构，近似考虑量化误差对层输出的影响。它关注的目标不只是孤立权重元素的舍入误差。
  [完整原文](../../references/S-GPTQ/reader.html) · [全文文本](../../references/S-GPTQ/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [量化：误差传播、分组尺度与执行格式](../algorithms/quantization-contract.md)。
- [CUDA 平台：SM 数据流、占用率与执行粒度](../hardware/cuda-execution.md)。
- [ROCm 平台：CU、Wave 与 AMD 算子路径](../hardware/rocm-execution.md)。
