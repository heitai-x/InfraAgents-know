# S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 原文

[原始 PDF](../../references/S-FLASH/original.pdf) · [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

固定版本：2205.14135v2。来源版本日期：2022-06-23；整理核对日期：2026-10-27。可直接阅读全文；下面是本库核对后抽出的算法与结论。

## 算法（论文 Algorithm 1，前向）

输入 Q、K、V ∈ ℝ^{N×d} 位于设备主存；论文二级存储模型中的 M 是片上可容纳的标量元素数，换算字节时须计入精度，不能直接用 KB 数值代替。

1. 取块大小 Bc = ⌈M/(4d)⌉，Br = min(⌈M/(4d)⌉, d)。
2. 在 HBM 初始化 O = 0 ∈ ℝ^{N×d}，ℓ = 0 ∈ ℝ^N，m = −∞ ∈ ℝ^N。
3. 把 Q 切成 Tr = ⌈N/Br⌉ 块，K、V 切成 Tc = ⌈N/Bc⌉ 块；O、ℓ、m 按 Q 的切法分块。
4. 对每个 K/V 块 j：载入 Kj、Vj 到 SRAM；对每个 Q 块 i：载入 Qi、Oi、ℓi、mi；片上算 Sij = QiKjᵀ；m̃ij = rowmax(Sij)，P̃ij = exp(Sij − m̃ij)，ℓ̃ij = rowsum(P̃ij)；mᵢⁿᵉʷ = max(mi, m̃ij)，ℓᵢⁿᵉʷ = e^{mi−mᵢⁿᵉʷ}ℓi + e^{m̃ij−mᵢⁿᵉʷ}ℓ̃ij；写回 Oi ← diag(ℓᵢⁿᵉʷ)⁻¹(diag(ℓi)e^{mi−mᵢⁿᵉʷ}Oi + e^{m̃ij−mᵢⁿᵉʷ}P̃ijVj)，以及 ℓi、mi 的新值。
5. 返回 O。

v2 版的循环是 K/V 在外、Q 在内，每个内层步都把 Oi 写回 HBM；后续工作把循环调成 Q 在外（见 [S-FLASHATTENTION3](S-FLASHATTENTION3.md) §1 对 FlashAttention-2 的概述），使每个 Q 块的 O、ℓ、m 全程驻留片上。循环顺序决定哪个对象驻留、哪个对象被重读，IO 复杂度的阶不变，常数与并行度改变。

反向传播不存储 N×N 的概率矩阵，而是保存 O 与 logsumexp L，反向时按块重算 P；用额外 FLOPs 换 HBM 访问。

## 结论

- 定理 1：Algorithm 1 返回 softmax(QKᵀ)V，O(N²d) FLOPs，输入输出之外 O(N) 额外内存。
- 定理 2：标准实现需要 Θ(Nd + N²) 次 HBM 访问；FlashAttention 需要 Θ(N²d²M⁻¹)。当可容纳元素数 M 相对 d² 较大时，模型预测设备主存访问显著减少；具体字节数与混合精度临时量需另算。
- 命题 3（下界）：不存在一个精确算法在 M∈[d,Nd] 的整个范围内一致只需 o(N²d²/M) 次 IO。其证明使用 M=Θ(Nd) 的输入读取下界；不是对每个固定 M 都成立的参数化最优性证明。
- 实验：HBM 访问次数是运行时间的主要决定因素；FlashAttention 的 FLOPs 比标准实现多（反向重算），运行时间却更短。块大小增大则遍历次数减少、运行时间下降；超过约 256 后被其他因素（算术）限制，且更大的块放不进 SRAM。

## 本库据此推导

- 在线 softmax 的三元组 (m, ℓ, u) 是可合并统计量：这是 split-K、分区 attention 合并与 ring 式分布的共同数学基础，见 [Attention 的 IO](../algorithms/attention-io.md)。
- 定理 2 给出该分块算法在声明的存储模型中的 IO 复杂度，命题 3 排除整个 M 区间上一致的更低阶算法。它们可作为数据流分析的参照，不能据此断言任意固定硬件上的自写 kernel 都不能低于某个数值流量，也不能把模型中的元素级 IO 直接当作实际显存事务。
- "块大小越大遍历越少，但受 SRAM 与算术限制"是 tile 选择的设计轴，具体拐点随硬件变化，见 [CUDA 平台](../hardware/cuda-execution.md) 的能力表。

## 适用边界

论文的实现与数字针对其所报告的设备、模型和编译器；算法与复杂度结论在声明的二级存储模型及输入条件下成立，实际事务、混合精度与驻留资源仍需核对。删除模型原本允许访问的 key 或采用低秩近似会改变问题，不能直接套用精确稠密算法的结论。版本日期不证明模型训练实际包含该资料。
