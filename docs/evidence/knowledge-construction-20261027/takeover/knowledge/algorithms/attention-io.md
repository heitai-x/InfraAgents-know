# K-ALG-004：Attention 的 IO、在线 softmax 与分块归约

精确 attention 可以通过改变数据流避免存储完整分数矩阵；算法复杂度、HBM 流量与数值归约顺序分别决定成本。

## 问题、不变量与设计轴

问题：对每个 query 行计算其合法 key 集合上的 softmax 加权和，key 集合随请求与位置不同，长度可达数万，而片上存储只能容纳一小块。要回答的是"分数矩阵的哪一部分在何时位于哪一层存储，归约按什么顺序完成"。

不变量：输出等于数学定义的 softmax(QKᵀ·scale+mask)V（在约定的浮点容差内）；被 mask 的位置权重为零且不参与分母；分块或分区合并保持全局最大值与归一化质量的一致缩放；全 mask 行有明确定义的输出。

成本项：QK 与 PV 的乘加（随 MN 增长，不因数据流改变而减少）、跨存储边界的字节（随分块方式改变）、归约与合并的额外工作、片上资源占用对并行度的限制、以及数值路径的舍入。

设计轴：切分方向（沿 query / 沿 KV / 二者）；中间量驻留层（寄存器 / 共享存储 / HBM）；归约组织（单程在线 / 两程 / 分区后合并）；可见性表达（稠密 mask / 因果 / 窗口 / 块稀疏索引）；精度策略（分数、概率、累加各自的格式）。FlashAttention 位于"沿 query 切分、中间量驻留片上、单程在线归约"；split-K 增加沿 KV 的切分与一次合并；物化分数矩阵的直接实现在 HBM 驻留端。轴上未列点同样合法。

## 从数学式看主要中间量

对一个 head，Q 为 M×d，K 为 N×d，V 为 N×dᵥ，输出是 softmax(QKᵀ/√d＋mask)V。直接实现先生成 M×N 的分数，再存概率矩阵，HBM 流量和峰值空间都随 MN 增长。QK 与 PV 的主要 FLOPs 分别约 2MNd 和 2MNdᵥ；消除中间矩阵不等于消除这些乘加。

长 prefill 的 M、N 同时较大，按 tile 在片上保留局部 Q/K/V 与归约状态可减少往返。单 token decode 的 M 很小，瓶颈可能转为读取 K/V 历史、launch 和低并行度，因此同一 kernel 的最优切分不会自然覆盖两个阶段。

## 在线 softmax 如何保证等价

已处理一组分数时，保留最大值 m、归一化分母 ℓ＝Σ exp(s−m) 和未归一化加权和 u＝Σ exp(s−m)v。另一块得到 mᵦ、ℓᵦ、uᵦ。合并时选择 m′＝max(m,mᵦ)，则 ℓ′＝exp(m−m′)ℓ＋exp(mᵦ−m′)ℓᵦ，u′ 按相同缩放相加，最终输出 u′/ℓ′。

这是一种带尺度的可合并统计量。它解释了为什么可以分块读 K/V、为什么 split-K 能合并局部 attention、为什么直接平均不同块的归一化输出是错误的。被 mask 的位置贡献零权重；全 mask 行需要明确定义输出，避免 −∞ 与 −∞ 相减形成 NaN。

## 切分带来的真实代价

沿 query 切分使不同输出行独立，减少跨块合并；沿 KV 切分增加并行度，但产生局部输出和统计量，还要一次合并。短 batch 的长历史 decode 可能从 split-K 获益，短历史则可能因合并和调度成本变慢。tile 越大，数据复用越好，但寄存器与共享内存占用也更高，可能降低并发 resident blocks。

mask 是可见性关系而非一个统一布尔开关。因果、窗口、块稀疏、跨模态自定义可见性要求不同有效 tile。若 kernel 遍历全部 tile 后才乘零，并未获得稀疏 FLOPs 收益。稀疏布局同时需要高效索引和足够大的有效块，否则元数据开销可能主导。

## 数值与设计选择

数学上的重排等价不保证浮点逐位一致。分块大小、累加精度、exp 实现、混合精度和分区合并都会改变舍入。一个有用的参考保留同一 mask、位置、scale 和输入值，只改变实现；比较最大/相对误差及误差在长序列上的累积，再判断生成分布是否满足任务目标。

如果算力足够但带宽受限，减少中间矩阵和重复读取有意义；如果有效 QK 对本身巨大，则窗口/稀疏等结构变化可能有效，但那需要模型本来支持或质量验证。IO 优化和模型近似是不同的设计自由度，应在推导中各自明确。

## 归约状态与流水状态分别验证

在线 softmax 的数学状态是行最大值、归一化质量和未归一化加权和；实现流水还管理 K/V 缓冲、分数片段、矩阵指令和写回代次。改变加载顺序或异步深度可以保持同一归约关系，但必须确保消费者看到对应块的完整数据，且旧缓冲没有提前被覆盖。

用两块分数 0、2 和对应 V 值 1、0，可以独立算出输出 1/(1+exp(2))≈0.1192。更新最大值后，旧加权和应随 exp(m_old−m_new) 缩小，再与新块合并；结果应保持在有效 V 的凸包内。固定版本的论文、伪代码和生成实现都应接受这样的数学检查，不能让出处代替等价性验证。

短 query 沿 KV 拆分后，需要把各部分的最大值、质量和加权和重新合并；多部分独立归一化后直接相加没有同样含义。全 mask 的空部分还要显式处理，避免负无穷相减产生未定义中间量。不同归约树可能产生有限精度差异，质量与数值参考仍按原契约检查。

性能上，更多 KV 分块增加并行度，也增加部分结果与合并；更大 tile 减少读取轮次，却增加寄存器与片上占用。若 KV 来自主存或 SSD，还要先比较持续传输下界与 [计算放置](heterogeneous-execution.md)，不能只对设备内 attention kernel 调参。

<!-- infraagents:provenance -->

## 来源与推导

- [S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../sources/S-FLASH.md)：FlashAttention 通过片上分块和在线 softmax，避免将完整 attention 分数与概率矩阵写入设备主存；目标函数仍是精确 attention 的数学形式。
  [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

- [S-TRITON：Triton：Fused Attention](../sources/S-TRITON.md)：Triton 的 fused attention 示例展示块级张量程序、在线 softmax 归约、tile 划分与相应配置选择。逻辑 tile 和实际线程/寄存器布局有映射关系。
  [完整原文](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-FLASHATTENTION3：FlashAttention-3](../sources/S-FLASHATTENTION3.md)：§2–3 以 Hopper 的 TMA、异步 WGMMA、warp specialization 和多级缓冲组织 attention；讨论寄存器、布局和 FP8 数值限制。
  [完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

## 延伸阅读

- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [Attention Backend：语义、布局与执行契约的交集](../frameworks/attention-backends.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](../models/mla.md)。
