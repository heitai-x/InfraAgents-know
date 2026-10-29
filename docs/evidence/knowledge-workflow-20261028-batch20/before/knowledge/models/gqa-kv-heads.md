# K-MODEL-002：MHA、GQA 与 MQA：头映射和带宽复用

GQA 让多个 query head 通过明确的映射读取较少的 K/V head。它减少的是模型所需的 K/V 状态与潜在读取量；每个 query head 仍有自己的分数、softmax 和输出。逻辑共享能否成为系统收益，取决于物理副本、kernel 内的复用、分块重读以及并行和表示转换的代价。

## 问题、函数与状态

设一层 attention 有 Hq 个 query head、Hkv 个 K/V head，每 head 的 key 和 value 维度分别为 dK、dV，本次 query 行数为 M。各宽度和 head 数为正整数，M 可以为零。定义映射 μ：{0,…,Hq−1}→{0,…,Hkv−1}，以及 ρ(m)={h|μ(h)=m}。对一个请求、query 行 i、query head h，其可见 key 位置集合为 V(i,h)。无 dropout 的推理 attention 为：

z_(i,h,j)＝a·q_(i,h)·k_(j,μ(h))＋b_(i,h,j)，

p_(i,h,j)＝exp(z_(i,h,j))／Σ_(t∈V(i,h))exp(z_(i,h,t))，

o_(i,h)＝Σ_(j∈V(i,h))p_(i,h,j)·v_(j,μ(h))。

q 和 k 都是 dK 维，v 和 o 都是 dV 维；dK 与 dV 不要求相等。缩放 a 由模型规定，常见选择是 1/√dK。偏置 b 可以依赖 query、head 和 key。不可见位置不进入分子或分母，不能仅将其分数乘零。

最后把 h=0…Hq−1 的输出按模型规定的顺序拼成 Hq·dV 维，再乘输出投影。GQA 不会把输出 head 数改成 Hkv。本文为空可见集合选定零输出和 empty=true 的接口契约；这是执行约定，后端若采用其他空行合同，应在接入处明确转换。普通在线算法要求参与计算的数、分数及中间累加有限；NaN、正无穷或溢出必须拒绝或转入另行定义的数值路径。

MHA 对应 Hkv=Hq 且每个 qhead 读取自己的 KV head；MQA 对应 Hkv=1；通常所称 GQA 位于二者之间。论文的 GQA-G 按组数 G 命名，本文对应为 Hkv，均匀组内 qhead 数另记为 g=Hq/Hkv。连续均匀分组的映射是 μ(h)=floor(h/g)。显式 μ 也可以表达不等组大小和不连续成员，但这不表示所有后端都支持它。

缓存持久状态按请求、层、逻辑 token、KV head、通道和 K/V 种类区分。K 的逻辑形状是 [seq,Hkv,dK]，V 是 [seq,Hkv,dV]；batch 轴、物理页和排列方式由表示增加。qhead 不属于共享 K/V 对象的身份，却属于 query、归一化状态和输出的身份。逻辑位置、模型位置、有效长度和物理槽位也不能由 head 数替代。

输入包括 Q、已发布历史 K/V、本次候选 K/V、μ、逐 head 可见性、缩放与偏置；输出包括 Hq 个 head 的 attention 结果和空行标志。缓存追加和提交是外层状态操作，不能把 attention 返回当作缓存发布已经完成。已输出 token 数也不等于已提交 KV 长度，见 [K-MODEL-001](prefill-decode-state.md)。

## 合法映射与模型语义

连续均匀 GQA 要求 Hq 能被 Hkv 整除。这个要求来自所选映射和常见后端合同；一般显式映射只要求每个 qhead 恰好有一个合法 KV head。某个 KV head 没有消费者时可以不读取它，不能用局部数量之比重新定义原模型的 μ。

合法执行还须保持以下关系。

- K 和 V 都使用同一个 μ(h)，各自通道宽度分别为 dK、dV。显式展开为 Hq 个物理 head 时，每个副本必须来自 μ(h)，展开的容量和读写另计。
- 每个 (i,h) 只对自己的 V(i,h) 归一化。共享 K/V 不意味着共享可见集合、概率或 softmax 统计量。
- 输出按原 qhead 顺序拼接。TP 的局部索引、复制与 padding 必须恢复输出投影所需的全局 Hq 轴。
- 所有可见 key 恰好贡献一次。分块遗漏、重复和把 padding 放入分母都会改变函数。
- 布局、位置变换和表示解释与模型一致。浮点归约顺序改变后应按数值合同比较，实数等价不保证逐位相同。

S-GQA 的固定论文描述了从 MHA checkpoint 转换的方案：对每组 K/V 投影矩阵做均值池化，再按原预训练配方继续训练一小部分步数。其 T5-XXL 实验中，5% 继续训练得到的 GQA-8 质量接近 MHA、速度接近 MQA；实验针对编码器-解码器模型，未与从头训练的 XXL GQA 比较。这支持一种模型转换方案，不证明任意已有 MHA checkpoint 可以在推理时无损减少 Hkv。论文没有将 GQA 应用于 encoder 自注意力，因为该设置中的表示并行计算，KV 带宽通常不是主要瓶颈；这也不是禁止其他模型研究 encoder GQA 的普遍合法性规则。

原生分组 kernel、按 μ 展开的 MHA 参考、逐 qhead 的慢速参考可以计算同一个函数，但它们的状态驻留和交通不同。显式展开数值正确，只能证明展开路径可行。

## 一层分组 attention 的算法

下面描述一个请求的一层自注意力查询块。历史长 n，本次新增 M 个候选位置。Q 和新 K 已完成模型要求的位置变换，bias 也使用正确模型坐标。vis(i,h,j) 从逻辑位置与有效性定义：普通因果追加可要求 j≤n+i，窗口和逐 head 稀疏规则再增加条件。不能只凭拼接后的数组范围把所有候选位置都视为可见。

每个 (i,h) 保存状态 X=(m,ℓ,U)，U 为 dV 维。它代表已经消费的可见集合 T：m=max_(j∈T)z_j，ℓ=Σexp(z_j−m)，U=Σexp(z_j−m)v_j。空状态 E=(−∞,0,零向量[dV])。两个非空且互不重叠的状态合并时，令 m′=max(m1,m2)、r1=exp(m1−m′)、r2=exp(m2−m′)，得到 ℓ′=r1ℓ1+r2ℓ2、U′=r1U1+r2U2。它把两份贡献换到同一尺度后相加。

```pseudocode
算法 一层分组 attention
输入：Q[M,Hq,dK]
      已发布 Khist[n,Hkv,dK]、Vhist[n,Hkv,dV]
      候选 Knew[M,Hkv,dK]、Vnew[M,Hkv,dV]
      μ[Hq]、vis(i,h,j)、有限缩放 a、有限偏置 bias(i,h,j)
      正整数 query 块大小 Br、KV 块大小 Bc
输出：Y[M,Hq,dV]、empty[M,Hq]

函数 merge(X1=(m1,ℓ1,U1), X2=(m2,ℓ2,U2))：
    若 ℓ1=0：返回 X2 的副本
    若 ℓ2=0：返回 X1 的副本
    m′ ← max(m1,m2)
    r1 ← exp(m1−m′)；r2 ← exp(m2−m′)
    返回 (m′, r1ℓ1+r2ℓ2, r1U1+r2U2)

校验四个 K/V 数组各自的形状、μ 的长度和取值范围
Kall ← Khist 与 Knew 的逻辑拼接，沿 token 轴连接
Vall ← Vhist 与 Vnew 的逻辑拼接，沿 token 轴连接
建立 ρ(m)={h | μ(h)=m}
对 q0=0,Br,2Br,… 且 q0<M：
    R ← [q0,min(q0+Br,M))
    只载入 R 内真实 Q 行
    对每个 i∈R、每个 h：X[i,h] ← E
    对 k0=0,Bc,2Bc,… 且 k0<n+M：
        J ← [k0,min(k0+Bc,n+M))
        对每个 KV head m：
            N ← {j∈J | 存在 i∈R、h∈ρ(m)，使 vis(i,h,j) 为真}
            若 N 为空：继续下一个 KV head
            将 N 中每个 Kall[j,m,dK]、Vall[j,m,dV] 读入本组 tile 一次
            等待加载完成，使所有消费者可读
            对每个 h∈ρ(m)、每个 i∈R：
                C ← {j∈J | vis(i,h,j)}
                B ← E
                若 C 非空：
                    对 j∈C：z[j] ← a·dot(Q[i,h],Ktile[j])＋bias(i,h,j)
                    mb ← max_(j∈C) z[j]
                    对 j∈C：w[j] ← exp(z[j]−mb)
                    lb ← Σ_(j∈C)w[j]
                    Ub ← Σ_(j∈C)w[j]·Vtile[j]，Ub 为 dV 维
                    B ← (mb,lb,Ub)
                X[i,h] ← merge(X[i,h],B)
            等待本组 tile 的全部 QK/PV 读者完成，才允许覆盖输入缓冲
    对每个 i∈R、每个 h：
        若 X[i,h].ℓ=0：Y[i,h]←零向量[dV]；empty[i,h]←真
        否则：Y[i,h]←X[i,h].U／X[i,h].ℓ；empty[i,h]←假
返回 Y、empty；不发布候选，不改变缓存目录或长度
```

每个 KV tile 的索引区间互不重叠，并集覆盖 [0,n+M)；逐 head 的 C 再取各自可见子集。组内加载集合 N 是这些 C 的并集，它只决定共享输入的加载，不能替代某个 head 的归一化域。若后端用完整稠密 tile 加载代替 N 的稀疏 gather，数学仍可正确，但应按实际加载和发射量记账。

例如 Hq=4、Hkv=2、μ=(0,0,1,1)。同一 query 的 head 0 只看 key 0，head 1 只看 key 1，即使二者共享 KV 0，其输出也分别是 v_(0,0) 和 v_(1,0)。将 vis(i,h,j) 缩成组内共用的 vis(i,j)，会改变至少一个 head 的合法集合。将二者可见集合取并集并共同 softmax 也不能修复。

每完成一个 tile，X[i,h] 都应恰好表示已经遍历部分中的可见项。空块先构造完整 E，不能保留上一块的 lb 或 Ub。空状态在指数运算前处理，避免 −∞−(−∞)。M=0 时不创建 query 状态，直接返回空输出；n=0 时仍可由本次候选和 mask 完成 prefill。尾块只访问真实位置。

这是逻辑数据流，不指定 CTA、warp 或矩阵指令的 fragment 布局。“读一次”指同一个 R、m 下的一个逻辑 KV 向量对只加载一次；不同 query 块或 CTA 仍可能重新加载。在线合并的进一步推导见 [K-ALG-004](../algorithms/attention-io.md)。

## 缓存提交与最后读者

S-HF-CACHE 的固定文档说明，历史 K/V 逐层缓存，本次只产生新增 K/V，再将历史与新增状态共同用于 attention；增量 mask 必须覆盖二者。Dynamic、Static、StaticSlidingWindow 改变存储增长和更新方式，但不改变 μ 或有效位置的含义。

上面算法读取已发布历史及私有候选。整次前向成功、所有必要层和 rank 的新状态完成、消费者可见依赖建立后，外层才发布新目录与长度。第二层失败时，第一层私有候选可以丢弃，旧已提交状态仍然有效；如果已经覆盖旧内容，仅恢复长度字段不能回滚。

缓存对象至少绑定模型/权重版本、请求、层、逻辑 token、KV head、通道、K/V 种类和表示版本。分页或异步复用时再绑定物理页、页代次及读者快照。某个 qhead 输出完成，不代表同组其他 qhead 已结束读取；K/V tile 的覆盖条件是全部消费者完成。物理历史页的回收还需等待全部逻辑持有者和在途读者结束。

本章选用整次调用统一发布。按层流水发布是另一种方案，需要独立定义阶段可见性、失败边界和可恢复状态，不能同时宣称整次调用全有或全无。

## 条件化成本：容量、计算和读取

设同构层数 L，K/V 存储均为每元素 b 字节。一 token 的逻辑 KV 容量主项为：

C_KV＝L·Hkv·(dK＋dV)·b。

无共享的请求历史长度为 n_r 时，容量主项为 C_KVΣ_r n_r。它不含页尾、对齐、量化参数、残余区、物理副本和暂存。若 K/V 的 dtype 不同，应将 (dK＋dV)b 改成 dK·bK＋dV·bV。异构层则逐层求和。

以 L=32、Hq=32、Hkv=8、dK=dV=128、b=2 为例，每 token 为 131072 字节，即 128 KiB；4096 token 为 512 MiB。MQA 的 Hkv=1 时是 16 KiB/token、4096 token 为 64 MiB。这是结构算术，不表示已有 checkpoint 可无损转换。显式展开为 Hq 个持久 KV 副本后，相应主项按 Hq 计，不能继续使用原来的 Hkv 容量。

令 P_h=Σ_(i=0…M−1)|V(i,h)|，它已包含该 head 的因果、窗口与稀疏筛选，但不含 head 轴。有效 QK/PV 乘加主项为：

F_attn≈Σ_h 2(dK＋dV)P_h＝2(dK＋dV)Σ_h P_h。

只有各 head 的 P_h 相同、记为 P_pair 时，才可简化成 2Hq(dK＋dV)P_pair；dK=dV=d 时进一步成为 4Hq d P_pair。可见集合本身可以不同，只要计数相同就能用这个算术简式，但算法仍须保留 head 轴。不同计数时使用一个 P_pair 再乘 Hq 会漏算或多算。

这些是有效数学工作。稠密矩阵指令可能先计算整个 tile 再施加 mask，padding、空项和未跳过块增加执行工作。每个实际分数还要付缩放、偏置、指数和归约，每次状态合并还要重标度 dV 维累加；它们不在上述矩阵乘主项中。

Q/K/V 与输出投影也要分轴计。令本次 hidden 输入和输出宽度均为 D，不假定 D=Hq·dK 或 Hq·dV。采用行向量约定，四个矩阵几何是：

| 投影 | 输入与权重 | 输出 | FLOP 主项 |
|---|---|---|---|
| Q | [M,D] × [D,Hq·dK] | [M,Hq·dK] | 2MDHq·dK |
| K | [M,D] × [D,Hkv·dK] | [M,Hkv·dK] | 2MDHkv·dK |
| V | [M,D] × [D,Hkv·dV] | [M,Hkv·dV] | 2MDHkv·dV |
| O | [M,Hq·dV] × [Hq·dV,D] | [M,D] | 2MDHq·dV |

因此 F_proj≈2MD(Hq＋Hkv)(dK＋dV)；等宽 dK=dV=d 时为 4MDd(Hq＋Hkv)。这里一次标量乘法与一次累加各计一个 FLOP，包含向零累加的首项，是常用矩阵乘主项口径；bias、位置变换和激活另计。历史已缓存时不重新做其 K/V 投影。Q 或 O 错按 Hkv 缩宽，既改变矩阵几何，也少计了必须完成的工作。

读取有三种口径。唯一持久状态按 C_KV·n 计；每个消费者请求的逻辑字节按实际分块计；HBM/L2 事务则受缓存、对齐和调度影响，不能互相代替。

对上面的按需加载算法，令 A_(R,m)=∪_(i∈R,h∈ρ(m))V(i,h)。一个请求、一层的 K/V 向量读取主项是：

D_KV,use＝b(dK＋dV)Σ_RΣ_m |A_(R,m)|。

它明确保留组内可见集合并集和 query 块重读。若使用整块加载，则应按实际加载 tile 的位置数替换 |A|。若所有 head 都看相同 n 个位置，单 query 块的理想主项为 Hkv(dK＋dV)bn；Nq 个块各自扫描则乘 Nq。显式展开并逐 qhead 扫描的相应主项可变成 Hq(dK＋dV)bnNq；实际 HBM 不一定按该逻辑比增长，因为更近层级可能吸收重读。新 K/V 写入、Q/O、bias、mask 和 metadata 都须另计。

在每请求一行 decode、各 head 共享相同可见集合、每个 KV head 只读一次且忽略其他项时：

I＝F_attn／D_KV,use≈2Hq／(Hkv·b)，单位 FLOP/byte。

b=2 时约为 Hq/Hkv。一般逐 head mask 下，应使用前面的实际 P_h 与组内并集字节重算，不能机械沿用此式。这个强度是条件化资源模型，不是目标设备实测或层延迟预测。权重、投影、解包、非矩阵吞吐、同步和并行度都可能主导关键路径。

## 准备、同步与片上驻留

建立显式 μ 的逆分组可以按 Hq＋Hkv 的规模完成；逐元素任意 mask 或 bias 本身可能有 M·Hq·(n+M) 项，不能当作免费结构规则。分页间接访问、mask 打包、布局转换、计划选择及 workspace 写入进入准备账。S-FI 的固定机制说明将准备和执行分开，且计划复用依赖 batch、分段、KV 长度、mask 与变体条件；几何变化后不能只保留旧计划地址。

以一个 query 块、一个 KV 组在任务内持续驻留为候选，记该组 qhead 数为 g_m。若输入 Q、K/V 的元素分别为 bQ、bKV 字节，统计与输出累加为 bAcc 字节，单缓冲载荷主项为：

Q：Br·g_m·dK·bQ；

K/V：Bc·(dK＋dV)·bKV；

逐 head 状态：Br·g_m·(dV＋2)·bAcc。

若物化分数或概率，还需各自的 Br·g_m·Bc 个元素；按伪代码同时保存局部 Ub 时再增加 Br·g_m·dV 个累加元素。融合更新可以省掉独立 Ub，但必须实际改变数据流。K/V 双缓冲再增加一份输入载荷和相应生存期依赖。这些量分别可能落在寄存器、共享存储或其他片上区域，不能将其总和直接称作共享存储占用。

g_m 增大时输入复用更多，Q、统计和输出驻留也增加。资源分配粒度、寄存器压力和 spill 可能降低活跃任务数；没有编译器资源报告，不能由这份载荷账声称无 spill。分组跨任务后，原来的片上共享也可能变成重复加载。

同步至少包括加载完成后才消费、全部 QK/PV 最后读者结束后才覆盖，以及候选状态完成后才发布。具体 barrier、异步指令或矩阵接口需要目标硬件合同，本章没有从逻辑 tile 猜测 lane/fragment 映射。时间沿依赖与共同资源求关键路径，已重叠的准备、计算和搬运不能全部相加。

## TP 下的物理头副本

TP 先分配 query head，再从 μ 推导每个 rank 需要的 KV head。设 p 个 rank，Hq 能被 p 整除，每 rank 拥有 Hq/p 个连续 qhead，记集合 I_r。所需 KV 集合为 S_r={μ(h)|h∈I_r}。若采用本地复制而不远程读取，则每 rank 的 KV 主项按 |S_r| 计，全组按 Σ_r|S_r| 计；唯一模型状态仍只有 Hkv 个 head。

Hkv 能被 p 整除且组边界与 rank 对齐时，可以均匀分片。否则不能直接用 floor(Hkv/p) 或局部 qhead/KV 数量比代替原映射。

取 Hq=24、Hkv=6、p=4，原模型每 4 个连续 qhead 共用 KV，每 rank 分 6 个 qhead：

| rank | query heads | 必需逻辑 KV heads |
|---|---|---|
| 0 | 0…5 | {0,1} |
| 1 | 6…11 | {1,2} |
| 2 | 12…17 | {3,4} |
| 3 | 18…23 | {4,5} |

全组共 8 份物理 head 数据，容量为逻辑唯一状态的 8/6。rank 0 的局部两组分别有 4 和 2 个 qhead，局部 6/2=3 不是模型的 g=4。实现可使用显式 global-to-local 映射，或在支持的后端中按原组补齐 query、屏蔽填充输出并恢复顺序；后端仅接受均匀局部分组时，应改分片或并行度。

再取 Hq=32、Hkv=8、p=16，每 rank 两个连续 qhead，相邻两个 rank 需要同一个 KV head。本地各持一份时，全组 16 份，是逻辑 8 份的两倍。L=32、dK=dV=128、b=2 时，一个 KV head 每 token 全层为 16 KiB；每 rank 16 KiB，全组 256 KiB/token，逻辑唯一状态为 128 KiB/token。TP 增大不保证缓存继续按 p 缩小。

S-NXDI 的固定指南给出后端选择：CONVERT_TO_MHA 通过复制 K/V 与 Q head 对齐，容量更大；REPLICATE_TO_TP_DEGREE 通过复制使每 rank 一份 KV head，并交错补齐 Q 以保持对应关系，并非支持所有配置。指南说明 TP 度不能被 KV head 数整除时回退到 CONVERT_TO_MHA。这属于该版本的策略，不能推广成全部 GQA 后端的协议或支持保证。

若复制头由各 rank 的复制权重独立生成，会增加计算和权重驻留；若一个所有者生成后分发，则增加传输和完成依赖。不能在同一成本账中同时假定生成不重复、传输也不存在。模型状态跨 TP 布局迁移还需去重、重分片或复制，见 [K-SYS-009](../system-semantics/distributed-state-transfer.md)。

输出投影仍消费 Hq·dV。采用行并行输出投影、对 hidden 求和时，归约宽度不会随 Hkv 缩小。完整 hidden 激活为 A 字节时，理想 ring all-reduce 一次每 rank 发送量为 2(p−1)A/p，接收另计；另一处归约再单独计入。启动、拓扑、慢 rank、布局转换和通信缓冲不在这个 payload 式中。

## 分页、量化与头映射的组合

分页映射逻辑 token 到物理页，μ 映射 qhead 到 KV head，量化参数映射元素到其 scale/offset。这些映射必须同时成立。只把整数码复制正确、却使用另一个 head 或逻辑组的参数，会保持张量形状而改变数值。

KIVI v1 的 K per-channel 参数固定一个通道、跨 token 分组；V per-token 参数固定 token、沿通道分组。二者的归约轴不同，不能把 per-channel 理解成任意混合通道。论文保留高精度残余区处理流式追加，其固定实验使用组大小 32、残余长度 128；Falcon 这种 MQA 模型在该设置下的 2 bit 质量风险较大，4 bit 才较好保持。此结果不保证所有 GQA/MQA 的可用位宽，也不能直接套到任意压缩表示。

追踪一次只改变放置的迁移：逻辑 token 0…31 的一个已封存 K 量化组跨两个长度 16 的页，页表为 [9,4]。A、B 共享内容和该组参数。A 要把第二页迁到物理页 7：

1. 固定 A 的读取快照，保留旧页 4 及参数的读者保护；预留页 7 和必要描述符。
2. 复制第二页的码和其 V 参数；K 参数仍按逻辑组 0、KV head、通道查找，可以继续引用受保护的只读参数对象，也可复制一份，须与容量账一致。
3. 完成复制和消费者可见依赖后，发布 A 的页表 [9,7]；B 与旧读者仍可使用 [9,4]。物理页 7 不能被解释成新的逻辑量化组。
4. A 的 qhead h 先取 μ(h)，再按逻辑 token 定位页内元素，最后按该元素的逻辑量化组解码。所有 qhead 的可见性仍分别应用。
5. 旧页 4 在 B 和全部旧读者结束前不能回收；发布前失败保留 A 的旧映射，候选复制排空后再回收页 7。

若迁移第二页需搬运的码与随页参数共为 D_page 字节，单纯设备复制的逻辑读和写各 D_page，临时新增驻留至少是一份目的页；若额外复制共享 K 参数，再加其实际字节。不能把两页共同引用的同一参数对象按引用次数重复计容量。若此次不是迁移，而是追加导致重新拟合跨两页的量化组，旧码和参数可能都要隔离或重编码，不能沿用只复制一页的交通与失败语义。

一种编码组的有效位宽可按 w＋8m/G 估算，其中 w 为码位宽、G 为组内值数、m 为 scale/offset 合计字节。残余、padding、TP 副本和暂存另计。先完整恢复高精度历史再做 attention，与在 tile 内解码的容量和流量不同。

验证顺序是：先核对 μ 和输出拼接；再从逻辑内容独立恢复分页、组参数和解量化值；然后在同一解量化值上比较原生分组、展开 MHA 和逐 head 参考。相对原高精度状态的量化误差与模型质量另行评估，不能将身份错误吸收到总容差里。

## 条件变化后的保留、调整与放弃

| 条件变化 | 保留的机制 | 需要调整的对象和成本 | 改选条件 |
|---|---|---|---|
| 历史变长、小 batch decode | μ、逐 head 归一化、唯一 KV 身份 | 容量与扫描增长，重新比较权重、KV、计算和同步 | 计算或同步主导时，继续减 Hkv 不一定缩短关键路径 |
| query 块变大 | 组内 KV 共享 | Q、统计和输出驻留增加，query 块重读次数可能减少 | spill 或并行度损失超过收益时减小块 |
| Hkv 减到 MQA | attention 函数形式 | 重算模型质量、容量、投影和任务分配 | 质量不满足要求时改回 GQA/MHA 或重新训练 |
| TP 增大 | 全局 μ 和输出轴 | 复制、padding、生成或分发、输出归约及局部峰值 | 复制或同步过重时减小 TP 或改分片 |
| 后端只支持展开 | 展开可以提供数学参考 | 持久副本、写放大、转换和重读 | 容量或交通超预算时选择原生分组后端 |
| dtype、量化组或残余改变 | 逻辑 head 身份 | 参数、码、残余、解包和重编码 | 字节未降、转换更贵或质量不合格时放弃该表示 |
| 桶、padding 或执行计划变化 | 有效位置与 μ | 物理 query、mask、workspace、准备复用条件 | 填充和准备抵消收益时调整桶或程序族 |
| 各 head 的窗口或稀疏集合不同 | 逐 head 可见性与归一化 | P_h 分别计，读取按组内并集，准备和负载均衡另计 | 后端不能表达该规则时改执行路径，不能擅自统一 mask |

S-NXDI 的固定指南说明静态程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可以使用不同桶。桶改变物理执行范围，不应改变模型 μ、逻辑位置或有效可见集合。

## 诊断与区分实验

| 观察与竞争解释 | 可控区分实验 | 反证后的修正 |
|---|---|---|
| 单 head 对、多 head 错：映射或输出顺序错误 | 在单个 KV head/channel 放置非零探针，逐 qhead 对比实际读值和拼接位置 | 探针落错组先修 μ 或轴顺序，不扩大容差 |
| dK=dV 时通过，换宽度后失败：K/V 形状混用 | 用 dK=3、dV=2 和反向配置，分别验证历史及新 K/V 的合同 | K 必须与 Q 的 dK 收缩，U 和输出保持 dV |
| 同组 head 输出异常：共用 mask 或共用统计 | 让同组两个 head 分别只看不同 key，再加入不等可见数量及全空 head | 共享加载可用并集，softmax 域必须分别恢复 |
| 换 tile 后结果改变：归约舍入或状态缺陷 | 固定输入，改变 query/KV 块大小，插入空块并逐块核对已处理集合 | 漏项、重复、空块残留或非有限值先按算法错误处理 |
| 投影 FLOPs 随 Hkv 一起过度缩小：Q/O 轴错误 | 独立枚举四个实际矩阵的标量乘加，并核对各自几何 | Q 输出宽为 Hq·dK，O 输入宽为 Hq·dV；不能只比较两个预写总式 |
| 单卡对、多卡错：local μ 或归约错误 | all-reduce 前比较每 rank 的 KV 集合、逐 head 输出和全局顺序 | 修正复制与映射，容量改按物理副本计 |
| 缓存变小但时间未降：重读、转换或同步主导 | 固定有效请求集，分别记录持久字节、逻辑读取、实际事务、转换和完整调用 | 字节未降修数据流；事务下降而时间未降则转查其他关键路径 |
| 首 token 对、后续 decode 错：未消费压缩历史或组身份错误 | 实际执行多步，跨页、封组和残余边界，先独立解码再比 attention | 首 token 仅用高精度 prefill 时，不足以证明量化历史正确 |
| 换桶或迁移后错：metadata、布局或发布问题 | 保持逻辑内容与 μ，置换物理页并延迟一个复制完成事件 | 修单位、代次、组参数或发布条件；稳定地址不能证明内容正确 |

映射负对照可把连续分组 μ=(0,0,1,1) 错写为 h mod 2=(0,1,0,1)，让它实际进入同一 attention 路径，再由正确展开参考拒绝。可见性负对照则让组内所有 qhead 使用第一个 head 的 mask；同组不同单 key 的输入会直接暴露错误。两者分别检验头归属和归一化域，不能用一次最终文本比较代替。

投影检查需要独立于总公式的工作参考。例如 M=2、D=5、Hq=4、Hkv=2、dK=3、dV=2，按矩阵内部每次乘与累加计数，Q/K/V/O 分别为 240、120、80、160 FLOPs，合计 600。让 Q 的实际输出宽误用 Hkv·dK，计数路径会得到 Q=120；让 O 的实际输入宽误用 Hkv·dV，会得到 O=80。参考应分别拒绝它们，不能仅证明手写的 600 与另一个常数不相等。真实 O 若直接接完整 attention 输出，还会发生收缩轴不匹配；截去 head 以匹配错误权重则改变模型函数。

## 检查范围与剩余边界

随章文件 `tools/check_k_model_002_full.py` 是确定性、只读、离线的 Python 标准库检查。被测路径按 KV 组加载输入并在线合并，独立参考按正确 μ 展开成逐 qhead K/V，用 Decimal 整行 softmax 计算。覆盖 MHA、MQA、连续 GQA、显式不均匀映射，dK≠dV、偏置、逐 head 不等可见集合、空行、空历史、零 query、短尾块和不同分块。比较器拒绝非有限值；形状校验也实际拒绝误用 dV 宽度的历史 K 和新 K。

attention 的乘加计数发生在实际 QK/PV 标量循环中，与独立按可见项和通道枚举的参考及按 P_h 求和的公式核对。逻辑加载次数按组内可见集合并集核算，它计一对 K/V 向量的加载事件，不是设备事务。四投影检查实际构造小矩阵、执行标量乘加计数，再与按 token/head/channel/hidden 轴独立枚举的参考和推导式比较；错误 Q/O 宽度进入同一矩阵计数路径并被拒绝。

512 MiB、TP16 字节和有效位宽部分是资源算术核算；TP 局部集合另与人工推导的全局映射参考比较。这些检查没有分配相应设备缓存，也没有测量容量峰值。分页量化迁移轨迹用于解释组合身份与发布条件，脚本不实现其真实码格式或异步协议。

脚本交由协调者执行，实际执行状态以本轮证据记录为准。它只验证小尺寸 CPU 标量数学与成本模型，不验证 FP16/BF16/FP8 舍入、GPU/Neuron kernel、片上 barrier、内存可见性、物理事务、collective、模型质量或知识使用效果。目标设备实验还需固定制品、shape、布局、dtype、mask、预热和计时边界，并记录准备、资源、spill、同步和完整时间线。

<!-- infraagents:provenance -->

## 来源与推导

- [S-GQA：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](../sources/S-GQA.md)：固定 2305.13245v3 的 §2.1–2.2 定义 MHA、MQA、GQA 的 head 关系，GQA-1 等价于 MQA，GQA-H 等价于 MHA；转换时对组内 K/V 投影均值池化，再继续训练。T5-XXL 的 GQA-8 质量与速度结论来自其指定任务、TPU 计时和继续训练设置；论文没有与从头训练的 XXL GQA 比较，实验限于编码器-解码器族。本文将模型转换事实与 kernel 片上复用分开。
  [完整原文](../../references/S-GQA/reader.html) · [全文文本](../../references/S-GQA/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：固定文档支持逐层缓存、合并历史与新增 K/V、mask 覆盖组合长度，以及 Dynamic、Static、StaticSlidingWindow 的存储区别。它不定义 GQA head 映射或具体位置字段合同；原文以乘 mask 表达屏蔽的示意不能替代排除不可见 softmax 项的数学规则。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：提供的固定提交机制说明区分 query 分段、页索引区间和末页长度的单位，以及准备/执行、计划失效、workspace 和图缓冲生存期；部分布局转换是复制。本文据此说明准备、复用和物理交通边界。本轮只有来源机制说明，没有自行读取原始源码，也不据此确认任意 GQA、mask、量化或分页组合可直接接入该 API。
  [算法与机制说明](../sources/S-FI.md)

- [S-KIVI：KIVI](../sources/S-KIVI.md)：固定 2402.02750v1 的 §3 与附录 A 支持 K 固定通道跨 token 分组、V 固定 token 沿通道分组，以及高精度残余和流式追加。§4 的组长 32、残余长度 128 及 Falcon 的 2 bit/4 bit 差异限定质量观察；prefill 消费高精度 K/V、后续 decode 才读取压缩历史的区别，支持多步验证要求。本文没有将某模型设置推广成所有 GQA/MQA 的位宽保证。
  [完整原文](../../references/S-KIVI/reader.html) · [全文文本](../../references/S-KIVI/fulltext.txt)

- [S-NXDI：NxD Inference Features Configuration Guide](../sources/S-NXDI.md)：提供的固定全文支持 CONVERT_TO_MHA、REPLICATE_TO_TP_DEGREE、Q 的交错补齐与回退条件，也说明静态程序的桶和 padding。它们是该后端版本的选择。本文两个 TP 头集合与容量算例为独立推导，不是该后端或 Neuron 设备实测。
  [完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

函数、形状合同、在线算法、逐 head 成本、工作枚举、TP 副本及组合迁移轨迹是本库基于上述材料的独立推导，设定算例不冒充实测。方法、审查标准、所给当前正文与五份来源说明按提供全文阅读；S-GQA、S-HF-CACHE、S-KIVI、S-NXDI 使用所给完整提取文本，不声称阅读 PDF 版面。相关标为 excerpt 的章节仅按提供范围使用，S-FI 不列为本轮原始源码阅读。来源事实受 2026-04-30 cutoff 约束，编辑日期不改变来源发布日期。

## 延伸阅读

- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](mla.md)。
- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [KV 压缩：表示、近似保留与传输格式](../algorithms/kv-compression.md)。
- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [量化：误差传播、分组尺度与执行格式](../algorithms/quantization-contract.md)。
- [Attention 执行接口](../frameworks/attention-backends.md)。
