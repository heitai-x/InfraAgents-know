# K-ALG-016：基础算子：从张量契约到合法融合

完整 forward 是张量变换和状态更新的组合。实现一个名字相同的算子并不充分：轴含义、广播、布局、累加精度、特殊值和原地写入关系共同定义它是否可替代参考。

## 线性、索引与布局

令 X 为 M×K，W 为 K×N，Y=XW+b 为 M×N；M 可以是一个 decode token、一批请求的有效 token，或 prefill 的 token 展平。乘加算作两次浮点运算时，主项是 2MKN。忽略重复搬运，至少涉及输入、权重与输出 s(MK+KN+MN) 字节；已有输出参与累加时还要读它。实际流量由分块和 cache 重用决定，不能用这条理想下界直接代替带宽测量。

矩阵转置的逻辑视图可以只改变 stride，但下游内核若只接受连续布局，仍会发生重排。预打包、QKV 合并和输出投影融合应从布局消费关系推导；同一个张量的两类消费者可能要求不同布局，局部减少一次写出却造成另一处完整转换。

Embedding 是按 token ID 从 V×H 表中取行，M 个 token 的有效输出为 M×H，不是把整个词表做一次矩阵乘。重复 ID 可复用数据，但输出仍保留输入顺序；词表边界、padding 行和模型定义的缩放不能省略。gather 需要说明索引轴和越界行为；scatter 需要说明重叠索引是赋值还是归约。并发 scatter 的数学和确定性契约取决于这些定义。

## 归一化与位置旋转

沿 hidden 轴的 LayerNorm 使用均值 μ 与方差 v，输出为 γ(x−μ)/√(v+ε)+β。采用常见带稳定项约定的 RMSNorm 为 γx/√(mean(x²)+ε)，不减均值。[RMSNorm 原文](../sources/S-RMSNORM.md) §3–4 给出两者的定义与区别；ε 的位置、是否有 bias、统计精度和转换顺序以具体模型为准。不能因为两者都叫 norm 就互换。

一次归约需要整条归约域的统计量。单遍保留行、分段归约后合并、两遍读取再应用统计量是不同实现：大行使保留策略耗费寄存器，小行可能由启动与线程协调主导。归约顺序改变舍入，方差很小时 ε 与累加精度尤其敏感。原地融合 residual 与 norm 之前，还要确认旧 residual 是否仍被其他分支使用。

RoPE 对一对分量 (a,b) 在位置 p 作旋转：(a cos(pθ)−b sin(pθ), a sin(pθ)+b cos(pθ))。同一频率下 R(p)ᵀR(q)=R(q−p)，解释了相对位置如何进入 QK 内积。[RoFormer 固定版本](../sources/S-ROPE.md) §3.2 给出这一构造。实际模型还要固定配对方式、旋转维度和频率；改变配对约定必须配套变换权重，不能只更换一个 kernel。

把旋转融合到 Q/K 写出可减少中间流量，但缓存应明确保存旋转前还是旋转后的 K。分块 prefill、前缀命中和窗口移动改变的是本次工作范围，不能随意把逻辑位置重置为零。测试可以先检验旋转保持范数，再检验两个位置旋转后的内积关系；这些代数性质不能替代具体模型参考。

## 激活、softmax 与融合边界

门控 FFN 可写为 U=XWᵤ，G=φ(XW𝗀)，输出为 (G⊙U)W𝒹；门函数、bias 和参数顺序属于模型定义。SiLU、精确 GELU 与 GELU 的近似形式产生不同函数。把门控乘法与激活融合可减少 U/G 的往返写读，但可能提高寄存器压力，或限制可用矩阵内核。

softmax 使用 m=max(z)，pᵢ=exp(zᵢ−m)/Σⱼexp(zⱼ−m)。减最大值限制指数溢出；并不自动定义全 mask 行、NaN、Inf 的行为。padding 元素应根据参考的可见集合处理，不能把错误 token 混入归约后再清零。分段 softmax 必须合并各段最大值与归一化质量，不能把每段独立归一化后直接拼接；详细合并关系见 [attention IO](attention-io.md)。

融合改变执行边界，也可能改变舍入点。线性、bias、激活、量化若在参考中有中间截断，融合必须保留所要求的数值关系。整数 accumulator、浮点 residual 和输出 scale 不能无条件合并；跨设备归约更不能省略必要通信。合法融合先满足数据依赖和别名约束，再比较字节、启动成本与资源占用。

## 可复算比较与定位

设定 X 为 32×4096、W 为 4096×4096，输入权重输出均用 2 字节。线性主项为 1,073,741,824 FLOP，理想流量为 32 MiB 权重加 0.25 MiB 输入和 0.25 MiB 输出，约 31.51 FLOP/byte。M=1 时强度约为 1 FLOP/byte。算式只改变 M 就显著改变重用机会，因此不能按同一 tile 和线程数处理两者；softmax、norm 等归约又有不同的上界。

建立逐算子参考时覆盖非连续 stride、非整除尾维、重复索引、极端数值和别名。完整 forward 逐层定位首次分歧，再区分参数加载、轴排列、位置/mask、有限精度重排和近似误差。若融合后小矩阵变快、大矩阵变慢，应检查内核选择、spill、转置与额外工作区，不能仅凭 kernel 数变少断言端到端改善。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-RMSNORM：Root Mean Square Layer Normalization](../sources/S-RMSNORM.md)：§3–4 比较 LayerNorm 与 RMSNorm，给出不减均值的 RMS 归一化关系及其不变性边界。
  [完整原文](../../references/S-RMSNORM/reader.html) · [全文文本](../../references/S-RMSNORM/fulltext.txt)

- [S-ROPE：RoFormer: Enhanced Transformer with Rotary Position Embedding](../sources/S-ROPE.md)：§3.1–3.3 以二维旋转及分量对推广建立位置编码和相对位置内积关系。
  [完整原文](../../references/S-ROPE/reader.html) · [全文文本](../../references/S-ROPE/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../sources/S-FLASH.md)：FlashAttention 通过片上分块和在线 softmax，避免将完整 attention 分数与概率矩阵写入设备主存；目标函数仍是精确 attention 的数学形式。
  [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

- [S-CPU-TUNING：PyTorch Performance Tuning Guide](../sources/S-CPU-TUNING.md)：PyTorch CPU 调优资料涉及线程组织、NUMA、向量化与执行配置；线程数和内存归属影响真实可用资源。
  [完整原文](../../references/S-CPU-TUNING/reader.html) · [全文文本](../../references/S-CPU-TUNING/fulltext.txt)

正文中的设定算例、资源账和机制组合是基于上述资料的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；本次编辑日期不改变来源时间。

## 延伸阅读

- [从模型计算图推导 Prefill、Decode 与持久状态](../models/prefill-decode-state.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [CPU GEMM/GEMV：寄存器、缓存与线程划分](../hardware/cpu-linear-kernels.md)。
- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
