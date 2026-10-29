# S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 原文

[原始 PDF](../../references/S-FLASH/original.pdf) · [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

固定版本：2205.14135v2。来源版本日期：2022-06-23；整理核对日期：2026-10-27。可直接阅读全文；下面是本库核对后抽出的算法与结论。

## 算法（论文 Algorithm 1，前向）

输入 Q、K、V ∈ ℝ^{N×d} 位于 HBM，片上 SRAM 容量为 M。

1. 取块大小 Bc = ⌈M/(4d)⌉，Br = min(⌈M/(4d)⌉, d)。
2. 在 HBM 初始化 O = 0 ∈ ℝ^{N×d}，ℓ = 0 ∈ ℝ^N，m = −∞ ∈ ℝ^N。
3. 把 Q 切成 Tr = ⌈N/Br⌉ 块，K、V 切成 Tc = ⌈N/Bc⌉ 块；O、ℓ、m 按 Q 的切法分块。
4. 对每个 K/V 块 j：载入 Kj、Vj 到 SRAM；对每个 Q 块 i：载入 Qi、Oi、ℓi、mi；片上算 Sij = QiKjᵀ；m̃ij = rowmax(Sij)，P̃ij = exp(Sij − m̃ij)，ℓ̃ij = rowsum(P̃ij)；mᵢⁿᵉʷ = max(mi, m̃ij)，ℓᵢⁿᵉʷ = e^{mi−mᵢⁿᵉʷ}ℓi + e^{m̃ij−mᵢⁿᵉʷ}ℓ̃ij；写回 Oi ← diag(ℓᵢⁿᵉʷ)⁻¹(diag(ℓi)e^{mi−mᵢⁿᵉʷ}Oi + e^{m̃ij−mᵢⁿᵉʷ}P̃ijVj)，以及 ℓi、mi 的新值。
5. 返回 O。

v2 版的循环是 K/V 在外、Q 在内，每个内层步都把 Oi 写回 HBM；后续工作把循环调成 Q 在外（见 [S-FLASHATTENTION3](S-FLASHATTENTION3.md) §1 对 FlashAttention-2 的概述），使每个 Q 块的 O、ℓ、m 全程驻留片上。循环顺序决定哪个对象驻留、哪个对象被重读，IO 复杂度的阶不变，常数与并行度改变。

反向传播不存储 N×N 的概率矩阵，而是保存 O 与 logsumexp L，反向时按块重算 P；用额外 FLOPs 换 HBM 访问。

## 结论

- 定理 1：Algorithm 1 返回 softmax(QKᵀ)V，O(N²d) FLOPs，输入输出之外 O(N) 额外内存。
- 定理 2：标准实现需要 Θ(Nd + N²) 次 HBM 访问；FlashAttention 需要 Θ(N²d²M⁻¹)。d 取 64–128、M 约 100 KB 时 d²/M 远小于 1，因此 HBM 访问少很多倍。
- 命题 3（下界）：对 M ∈ [d, Nd] 的全部取值，不存在 o(N²d²M⁻¹) 次 HBM 访问的精确 attention 算法。
- 实验：HBM 访问次数是运行时间的主要决定因素；FlashAttention 的 FLOPs 比标准实现多（反向重算），运行时间却更短。块大小增大则遍历次数减少、运行时间下降；超过约 256 后被其他因素（算术）限制，且更大的块放不进 SRAM。

## 本库据此推导

- 在线 softmax 的三元组 (m, ℓ, u) 是可合并统计量：这是 split-K、分区 attention 合并与 ring 式分布的共同数学基础，见 [Attention 的 IO](../algorithms/attention-io.md)。
- 定理 2 与命题 3 给出精确 attention 的 IO 下界形状：任何自写 kernel 的 HBM 流量目标只能逼近 N²d²/M，不能低于它；衡量一个实现离下界多远，是比"比某库快多少"更稳定的尺度。
- "块大小越大遍历越少，但受 SRAM 与算术限制"是 tile 选择的设计轴，具体拐点随硬件变化，见 [CUDA 平台](../hardware/cuda-execution.md) 的能力表。

## 适用边界

论文的实现与数字针对 2022 年的 A100 与当时的编译器；算法与复杂度结论与硬件无关。精确 attention 的下界不约束近似 attention（窗口、稀疏、低秩）。版本日期不证明模型训练实际包含该资料。
