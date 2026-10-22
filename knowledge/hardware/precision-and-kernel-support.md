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

## 固定平台能力表应到具体指令接口

下表采用已归档的 CUDA 12.8 Programming Guide。它描述该版本接口与架构条件，不能替代目标二进制的 dispatch 验证，也不是整个 GPU 产品目录。

| 具体路径 | 输入与累加 | 文档中的形状/架构条件 | 不能由此推出的结论 |
|---|---|---|---|
| WMMA 半精度 | FP16×FP16，FP32 或 FP16 accumulator | 文档列 16×16×16、32×8×16、8×32×16；warp matrix 操作支持条件从 CC 7.0 起 | 所有 FP16 张量乘法均使用该路径 |
| WMMA BF16 | BF16×BF16，FP32 accumulator | alternate floating point 支持从 CC 8.0 起，列出与 half 对应的形状 | BF16 的指数范围等于 FP32 就意味着相同精度 |
| WMMA TF32 | 数值转换后的 float 输入，FP32 accumulator | 16×16×8，CC 8.0 起的相应功能；WMMA 需要规定的 TF32 转换 | 输入存储为 FP32 就完成了完整 FP32 乘法 |
| WMMA 整数 | signed/unsigned 8-bit 输入，int accumulator | 文档列出特定 16×16×16 等组合，仍需核对目标架构和编译结果 | W8A16 或任意 zero-point/group-scale 可直接套入 |
| experimental sub-byte WMMA | s4/u4 输入，int accumulator | 8×8×32，规定输入布局与步长；文档明确相关 s4/u4 变体在 sm_90 移除 | 更新 GPU 一定保留旧 INT4 指令接口 |
| CC 9.0 Tensor Core FP8 | E4M3 或 E5M2 输入能力 | 该架构节明确列出 FP8；具体 kernel/指令、累加与缩放仍须分别确定 | 普通 WMMA 上表就是完整 FP8 API，或整个模型自动用 FP8 |

WMMA fragment 是架构相关的不透明分布，不能把它当作普通数组跨不兼容的编译单元传递。其加载要求 warp 成员一致参与，并规定指针对齐和 leading dimension；这些条件不因上层 tensor 的总元素数正确而自动满足。硬件能力、公开 API 和目标内核是相邻但独立的证明层。

## Scale 布局、尾部与累加范围

W/A 的每组 scale 若位于归约轴内部，要对相应 K 段的部分结果分别应用；单一 epilogue 只能直接承接可在归约外提出的尺度。输入 token、输出通道、组号、packed nibble 和 scale 数组的索引映射应共同给出。预打包改变布局后，scale/zero point 的排列也必须随之变换。

整数矩阵乘仍有溢出边界。对未经 zero-point 修正的 signed INT8 乘积，绝对上界可用 128²=16,384；K=4,096 时和的绝对上界为 67,108,864，小于 INT32 最大值，但更长归约、初始 accumulator、偏置和 zero-point 修正需重新计算。不能从这一例推断所有 INT8 GEMM 都不会溢出。

量化尾块必须填入表示实数零的码和正确元数据，FP8 转换则需处理格式对应的饱和、非有限值及缩放。一个 s4 nibble 0xF 应按约定解释为 −1，而不是无符号 15；一种对称格式的码不能套用另一种非对称 zero-point。独立解包测试应先于整模型质量测试。

## CPU/GPU 的实际 dispatch 核验

先登记执行请求的真实 M/N/K、dtype、stride、group size、对齐、ISA/compute capability、后端版本与编译选项。然后观察运行时实际选中的函数/内核和生成指令，而不是只读硬件型号或库宣称支持列表。CPU 可能由标量、SIMD、整数 dot 或矩阵扩展承担，GPU 可能先转换再调用浮点矩阵内核；未核验的路径应保持为待测候选。

对 CPU，线程池、NUMA、预打包缓存及 shape 专门化会改变同一 dtype 的结果；需确认是否重复打包、是否误用通用 fallback、低位 unpack 是否占主要时间。对 GPU，查看转换 kernel、矩阵 kernel、scale epilogue、工作区与同步的完整时间线，确认没有每轮生成完整高精度副本。这里只定义核验方法，不声称已经在目标机器观察到某条指令。

一个有用的三路比较是：高精度参考；独立解量化后用高精度算子的参考；直接消费压缩表示的目标 kernel。第二路与第三路比较定位格式/实现问题，第一路与第二路比较定位量化误差。再加入非对齐尾块、转置 stride、常数组、极值、长归约及真实小 batch，防止在整齐大矩阵上通过而在实际 decode 或稀疏专家上走不同路径。

硬件格式选择最后仍由瓶颈决定。减少权重码可缓解容量与带宽，却增加转换；更低 bit 的指令若没有合法高效 kernel，可能输给成熟的较高精度路径。反过来，即使累加仍为 FP32，压缩输入与 tile 内转换也可能降低完整成本。应以质量、实际字节、峰值内存和端到端耗时联合选择。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities](../sources/S-CUDA-CAPABILITY.md)：CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。
  [完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](../sources/S-GPTQ.md)：GPTQ 是训练后权重量化方法，使用校准输入所反映的层输入结构，近似考虑量化误差对层输出的影响。它关注的目标不只是孤立权重元素的舍入误差。
  [完整原文](../../references/S-GPTQ/reader.html) · [全文文本](../../references/S-GPTQ/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-SMOOTHQUANT：SmoothQuant](../sources/S-SMOOTHQUANT.md)：固定 v1 用通道等价缩放平衡权重/激活量化困难，区分三种激活量化设置，并通过具体 INT8 GEMM/BMM 路径评估。
  [完整原文](../../references/S-SMOOTHQUANT/reader.html) · [全文文本](../../references/S-SMOOTHQUANT/fulltext.txt)

- [S-AWQ：AWQ](../sources/S-AWQ.md)：固定 v1 以激活感知通道缩放和 clipping 降低低位权重误差，说明 grouped quantization、在线解量化及特定 GPU kernel 路径。
  [完整原文](../../references/S-AWQ/reader.html) · [全文文本](../../references/S-AWQ/fulltext.txt)

- [S-FLASHINFERPAPER：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](../sources/S-FLASHINFERPAPER.md)：固定 v1 描述 BSR/ragged/可组合 KV 表示、attention 状态合并、JIT 变体、动态负载计划与 CUDA Graph workspace，以及 GQA 和 FP8 KV 路径。
  [完整原文](../../references/S-FLASHINFERPAPER/reader.html) · [全文文本](../../references/S-FLASHINFERPAPER/fulltext.txt)

## 延伸阅读

- [量化：误差传播、分组尺度与执行格式](../algorithms/quantization-contract.md)。
- [CUDA 平台：SM 数据流、占用率与执行粒度](../hardware/cuda-execution.md)。
- [ROCm 平台：CU、Wave 与 AMD 算子路径](../hardware/rocm-execution.md)。
