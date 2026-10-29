# S-FLASHATTENTION3：FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

## 原文

[原始 PDF](../../references/S-FLASHATTENTION3/original.pdf) · [完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

固定版本：2407.08608v1。来源版本日期：2024-07-11；整理核对日期：2026-10-27。可直接阅读全文；下面是本库核对后抽出的算法与结论。

## 问题

在 Hopper 上，FlashAttention-2 的前向只达到 FP16 矩阵峰值的约 35%。原因有两个：矩阵单元与特殊函数单元的吞吐相差两个量级以上（论文给出 H100 SXM5 为 989 TFLOPS 矩阵算力对 3.9 TFLOPS 特殊函数），softmax 里的指数成为串行瓶颈；以及同步的 kernel 结构不能利用新一代硬件的异步加载与异步矩阵指令。

## 算法

论文 Algorithm 1（CTA 视角，不含消费者内部重叠）：

1. 建立 s 级环形共享存储缓冲及其 barrier。
2. 生产者 warpgroup：释放多余寄存器；发出 Qi 的加载并提交完成信号；对 j = 0..Tc−1，等待第 j mod s 级缓冲被消费完，发出 Kj、Vj 的异步加载到该级，完成后提交信号。
3. 消费者 warpgroup：按消费者 warp 数重新分配寄存器；初始化 Oi = 0、ℓi = 0、mi = −∞；等待 Qi；对每个 j：等待 Kj；Sij = QiKjᵀ（异步矩阵指令，提交并等待）；m_old = mi，mi = max(m_old, rowmax(Sij))；P̃ = exp(Sij − mi)，ℓi = e^{m_old−mi}ℓi + rowsum(P̃)；等待 Vj；Oi = diag(e^{m_old−mi})⁻¹Oi + P̃Vj；释放第 j mod s 级缓冲给生产者。
4. 收尾：Oi = diag(ℓi)⁻¹Oi，Li = mi + log ℓi，写回 HBM。

三项技术：

- 生产者/消费者分工（warp specialization）：一部分 warp 只负责发出异步加载，其余 warp 只负责矩阵与 softmax；两者通过环形缓冲与 barrier 交接。寄存器从生产者转移给消费者。
- 乒乒调度（pingpong）：两个消费者 warpgroup 交错，一个做 softmax 时另一个做矩阵乘，用 barrier 强制顺序；论文报告 FP16、head 维 128、序列 8192 时从 570 提升到 620–640 TFLOPS。
- 消费者内部两级流水（Algorithm 2）：第 j 轮的第二个矩阵乘（P̃V）与第 j+1 轮的 softmax 重叠，代价是额外保留一份 Br×Bc 的 FP32 分数片段（寄存器压力）；三级流水理论上更多重叠，实测因寄存器更紧而更差。编译器可能重排指令，论文用 SASS 检查确认重叠确实发生。

FP8 路径的两个额外要求：

- 布局：FP8 矩阵指令对 V 的操作数要求与 FP16 不同的内存排列，需要在 kernel 内对 V 块做转置（用寄存器高效的共享存储加载/存储指令，可在前两个矩阵乘的阴影里完成）；FP32 累加器交给下一个 FP8 矩阵乘时也要做布局变换。
- 精度：块量化（每个 Q、K、V 块各自一个 scale，而不是每张量一个）与非相干处理（在量化前用随机正交矩阵 M 同乘 Q 和 K，(QM)(KM)ᵀ = QKᵀ 不变，但异常值被分散；M 取随机 ±1 对角阵与 Hadamard 的积，O(d log d) 且可与 RoPE 融合）。论文报告在含异常值的输入上 FP8 误差比每张量缩放的基线低 2.6 倍。

## 结论

FP16 前向在 H100 上达到 740 TFLOPS（约 75% 利用率），比 FlashAttention-2 快 1.5–2.0 倍；FP8 接近 1.2 PFLOPS。消融实验表明分工与两级流水各自贡献加速。FP16 的数值误差与 FlashAttention-2 相同、低于标准实现（中间量保留 FP32）。基准固定 GPU 时钟为 1830 MHz。

## 本库据此推导

- 矩阵与非矩阵吞吐的量级差是"softmax 成为瓶颈"的来源；任何在矩阵单元很快的硬件上写 attention 的人都要回答指数运算往哪里藏，见 [GPU 流水](../hardware/gpu-kernel-pipelines.md)。
- 生产者/消费者与环形缓冲是依赖关系，不是 Hopper 专属；没有异步加载单元时同一依赖由普通线程指令承担，见同章"没有 TMA 与线程块簇时"一节。
- 更深流水换更多重叠、以寄存器为代价、实测可能反转，是 tile/级数设计轴的典型反例，见 [Triton Kernel](../frameworks/triton-kernels.md)。
- 块量化与非相干处理属于数值路径设计，见 [量化](../algorithms/quantization-contract.md)。

## 适用边界

算法依赖 Hopper 的异步矩阵指令、TMA 与 warpgroup 寄存器重分配；在 8.x 上这些单元不存在，需要换成软件流水。性能数字针对 H100 SXM5 与 2024 年 5 月的库版本。版本日期不证明模型训练实际包含该资料。
