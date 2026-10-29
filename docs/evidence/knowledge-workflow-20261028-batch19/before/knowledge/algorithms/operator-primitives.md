# K-ALG-016：基础算子：从张量契约到合法融合

完整 forward 是张量变换和状态更新的组合。实现名字相同的算子并不充分：轴含义、广播、布局、累加精度、特殊值和原地写入关系共同定义它是否可替代参考。本章沿线性投影、索引、归一化、位置旋转和 softmax 展开这些关系，并把融合的收益放回真实消费者和状态生命周期中计算。

## 张量契约与执行状态

逻辑张量可表示为 T=(V,shape,stride,offset,dtype,device,meta)。V 是底层存储，meta 包括轴语义、量化 scale、zero point、有效位置和别名关系。对普通 strided 张量，设基址为 A、每元素 b 字节，offset=o、各轴 stride=sⱼ 均以元素计，则

addr(i)=A+b(o+Σⱼ iⱼsⱼ)。

shape 的元素数相同，不证明地址或函数相同。合法访问要求所有实际下标与地址在允许存储范围内，地址计算不溢出，后端接受实际 stride、offset 和对齐。本章的伪代码使用非负 stride；其他视图需要按消费接口另行证明。广播的零 stride 允许多个逻辑读者共享元素，却不能直接授权多个输出写者覆盖同一地址。

需要分开三种状态：逻辑状态说明张量、请求、有效元素和位置；数学状态说明累加器、归约统计和旋转结果；执行状态说明缓冲、就绪事件、全部消费者和最后读取。数学值已经定义，不表示异步生产已经完成；生产者完成，也不表示输入槽可以覆盖。

合法实现必须保持完整的归约域、位置对应、输出归属与数值路径。量化权重重排时，scale、zero point 和分组轴同步重排；原地写入时，必须证明不会覆盖任何尚需读取的值。tile、打包、split-K、缓冲深度和融合边界是方案选择，不能充当所有实现的统一条件。

特殊值与空域要单独定义。有限输入的乘加可以用零补齐归约尾部；softmax 的不可见位置则必须排除出归一化域。NaN、正无穷、空归约和全 mask 行不能由普通 max/exp/sum 公式自动处理。

## 线性投影：权重轴、分块与输出布局

令 X 为 M×K，W 为 K×N，bias 为 N 维，Y=XW+bias 为 M×N。M 可以是一个 decode token、一批请求的有效 token，或 prefill 展平后的 token。乘加计两次浮点操作时，主项为 2MKN FLOP。若输入、权重和输出每元素 s 字节，bias 每元素 s_b 字节，非空调用且不计重复搬运的理想流量为 s(MK+KN+MN)+s_bN。实际事务还取决于分块、cache、对齐和存储策略。

数学 W 的轴与制品轴应显式关联。例如本章设定一种文件契约：A_file[n,k] 保存输出通道 n 对输入通道 k 的权重，因而 W[k,n]=A_file[n,k]，Y[m,n]=ΣₖX[m,k]A_file[n,k]+bias[n]。这里的 N×K 是该设定制品的定义，不是从 S-SDPA 推出的通用框架接口事实。实际 checkpoint 必须从其配置与允许阅读的加载说明确定轴，见[权重加载](../frameworks/checkpoint-loading.md)。

一个可手算的反例是 X=[1,2]、A_file=[[3,4],[5,6]]，正确输出为 [11,17]。若直接将文件解释为数学 K×N，得到 [13,16]。方阵尤其容易掩盖这个错误，因为 shape 检查仍能通过。

```pseudocode
算法 TiledLinear
输入：X[M,K]、W[K,N]、可选 bias[N]，各张量的 base、offset、stride、dtype
      M、N、K 为非负整数；BM、BN、BK 为正整数
      独立输出 Y[M,N]；可选逐元素 epilogue E
输出：Y，每个有效位置恰好写一次
前提：输入在读取完成前不被覆盖；Y 与仍会读取的输入不别名
1 若 M=0 或 N=0：返回空输出，不发出输入读取
2 对 m0=0,BM,… 且 m0<M，对 n0=0,BN,… 且 n0<N：
3    对 i∈[0,BM)、j∈[0,BN)：acc[i,j]=0
4    对 k0=0,BK,… 且 k0<K：
5        对 i∈[0,BM)、kk∈[0,BK)：
6            若 m0+i<M 且 k0+kk<K：按实际 stride 加载 x_tile[i,kk]
7            否则：x_tile[i,kk]=0，不访问 X
8        对 kk∈[0,BK)、j∈[0,BN)：
9            若 k0+kk<K 且 n0+j<N：按实际 stride 加载 w_tile[kk,j]
10           否则：w_tile[kk,j]=0，不访问 W
11       等待本阶段加载与补零完整可见
12       对 i、j、kk：acc[i,j]+=x_tile[i,kk]·w_tile[kk,j]
13       全部输入读者结束后，本阶段输入槽才可覆盖
14   等待本 tile 的全部累加完成
15   对有效 m=m0+i、n=n0+j：
16       v=acc[i,j]
17       若指定 bias：v+=bias[n]，仅一次
18       若指定 E：v=E(v,m,n)，仅一次
19       Y[m,n]=按输出 dtype 转换(v)
```

处理完前 t 个 K 分块后，acc 等于对应输出在 [0,min(tBK,K)) 上的乘积和，这是循环不变量。尾部补零不改变有效和；输入访问必须先受边界保护，不能只掩码最终 store。K=0 时乘积为零，bias 和 E 仍应用一次。

沿 M/N 划分可以保持输出独占；沿 K 划分会产生同一输出的多个部分和。split-K 必须先合并部分和，再应用一次 bias 和非线性 E。若有 P 份完整部分矩阵、累加元素 s_acc 字节，工作区主项为 PMNs_acc，写出与重读约为 2PMNs_acc，最终输出另计。跨设备拆分还增加部分结果通信和完成依赖。并行度改善只有超过这些新增成本才有收益。

逻辑转置可以只改 stride，物理预打包则真的读写数据。设准备花费 T_p，每次调用节省 T_s，复用 R 次，严格摊销条件为 RT_s>T_p。固定权重可在加载后打包；每次改变的激活不能继承这项跨调用复用。极小 M 时，打包和提交也可能支配调用。

设连续 B×L×H×D 张量大小为 2×3×2×4，stride=(24,8,4,1)。交换 L/H 后，B×H×L×D 视图的 stride=(24,4,8,1)，相同逻辑 shape 的连续新缓冲则为 (24,12,4,1)。位置 (0,1,0,0) 在两者各自存储中偏移 4 与 12 个元素；正确重排应让它们表示同一个值。直接 flat copy 后只改 shape，会混淆 token/head。

QKV 合并投影还要保持真实分段。设 hidden=4096、Hq=32、Hkv=8、每 head d=128，则 Q 宽 4096，K/V 各宽 1024，拼接宽度为 6144。M=32、每元素 2 字节时，输出为 393216 字节，即 384 KiB，其中 Q 为 256 KiB、K/V 各 64 KiB；合并权重为 48 MiB。三个等宽 hidden 投影会错误地得到宽度 12288。

若输出按 token 行拼接，Q view 的有效行宽虽为 4096，行 stride 仍为 6144；K/V 也沿完整行前进。消费者接受该 stride 时可省复制；要求紧密排列时必须计入重排。让投影直接产生目标布局会改变 GEMM 的候选集合，应比较投影、布局适配与缓存写入的完整路径。

## 索引、广播与词表消费

Embedding 从 V×H 表按 token ID 取行，M 个输入的输出为 M×H，不是遍历整份词表的 GEMM。重复 ID 可以复用读取，输出仍必须保留出现次数和顺序。padding 行、越界策略、模型缩放和索引轴属于函数契约。

```pseudocode
算法 GatherRows
输入：T[V,H]、ids[M]，本例越界策略为拒绝
输出：独立 O[M,H]
1 在输出写入前检查每个 ids[t] 都满足 0≤ids[t]<V
2 对 t=0,…,M−1，对 j=0,…,H−1：O[t,j]=T[ids[t],j]
3 M=0 返回空输出；重复 ID 不删除输出行
```

```pseudocode
算法 ScatterIndexed
输入：目标 Z、等长的 ids 与 values，mode∈{assign,add}
输出：更新后的 Z
1 检查 mode、索引和全部值块范围
2 按明确的输入次序遍历 t：
3    若 mode=assign：Z[ids[t]]=values[t]
4    若 mode=add：Z[ids[t]]+=values[t]
5 并行化时，重复索引必须实现规定的次序或归约契约
6 全部写入结束后才向消费者发布完成
```

顺序 assign 的重复索引采用最后一次赋值；add 保留 multiplicity。浮点 add 的并行归约次序仍可能改变舍入。若实现仅提供无序竞争写入，就不能把它称为上述确定的顺序 assign。

Gather 的理想数据读取与输出写入各为 MHs 字节，另有索引；真实流量受随机访问、cache line 和重复 ID 命中影响。Scatter-add 还读取旧目标并可能支付原子操作或分组归约；大量重复索引可能使竞争而非有效字节成为限制。推荐 bag 和图邻居聚合中，去重读取不能变成去重语义，分别见[推荐推理](../models/recommendation-inference.md)与[图模型推理](../models/graph-inference.md)。

lm_head 是 H×V 的词表投影。只为实际需要 logits 的位置计算，可以减少位置维上的工作；若只需 argmax 或归一化统计，还可分块消费词表而不写出完整 logits。设 V=131072、H=4096、每元素 2 字节，权重为 1 GiB；若其余 decoder 权重为 6 GiB，在每步各读取一次的设定下，lm_head 占总权重流量的 1/7≈14.3%。tied embedding 节省唯一参数容量，不自动消除这次词表投影读取。这里的全权重读取是该执行设定，cache 与具体分块仍需另计。

## 归一化：完整统计域与活跃 residual

对 hidden 长度 H>0 的一行，LayerNorm 使用 μ=Σxⱼ/H、v=Σ(xⱼ−μ)²/H，输出 γⱼ(xⱼ−μ)/√(v+ε)+βⱼ。可重建的两遍算法先完整求 μ，再用中心化值求 v，最后应用仿射变换；若不保留输入，后两阶段需要重新读取。用 E[x²]−E[x]² 替代中心化平方，在大均值、小方差时可能发生消去。

本章选定 RMSNorm 为 yⱼ=γⱼxⱼ/√(Σx²/H+ε)，ε>0，不减均值。定义差异见 [S-RMSNORM](../sources/S-RMSNORM.md) §3–4。论文的理想公式与具体模型的 ε、bias、dtype 和转换顺序分开核对，不能推断已训练 LayerNorm 可在推理时无损替换。

```pseudocode
算法 RMSNormWithOptionalResidual
输入：x[M,H]、可选 r[M,H]、γ[H]、ε>0，M≥0、H>0
      keep_u 表示更新后的 residual 是否仍有外部消费者
输出：y[M,H]；keep_u 时另交付 u[M,H]
前提：输入及统计中间值在所选数值路径内有限；输出写区合法
1 对每行 t=0,…,M−1：
2    分配本行临时值 local_u[H]，q=0
3    对 j=0,…,H−1：
4        local_u[j]=x[t,j]+r[t,j]（无 r 时为 x[t,j]）
5        q+=local_u[j]²
6    q=q/H；denom=sqrt(q+ε)
7    对 j=0,…,H−1：
8        y[t,j]=cast(γ[j]·local_u[j]/denom)
9        若 keep_u：u[t,j]=按 residual 契约转换(local_u[j])
10   本行全部输出完成且可见后，才交给下游
```

local_u 是显式保存的本行值，不是离开第一次循环后仍神秘存在的标量。该组织需要 H 个临时元素；可以改为两遍读取或分段统计，但必须把重读、部分和、同步与驻留重新计入。若参考先将 u 舍入到 residual 存储精度再做 norm，步骤 4 后也必须保留该舍入点；直接对更高精度的和归一化可能改变函数。

归约 tile 有 P≥H 个槽时，无效槽可给平方和贡献零，分母仍为 H。误用 P 会在均方远大于 ε 时把非零输出放大约 √(P/H)。H=3072、P=4096 时约为 1.1547。全一有效输入、全零 padding 是闭式反例；普通随机输入和整体平均误差容易掩盖这一整行尺度错误。

residual 融合的流量由 u 的活跃性决定。令 N=MH，以下只计 x、r、u、y 的外部元素读写，忽略各方案共有的 γ。独立两步先读 x/r、写 u，norm 再读一次 u、保留本行值并写 y，合计 5N 次。融合并保留活跃 u 时，读 x/r、写 u/y，最低 4N 次。只有 u 没有外部消费者，才可省掉 u 的写出，降为 3N 次。

这些是明确数据流下的逻辑账，不是任意 norm 的事务数。独立 norm 若重读 u，要增加读取；融合若 spill，也有额外交通。M=32、H=4096、每元素 2 字节时，一份 u 为 256 KiB，融合保留 u 比上述独立路径少 256 KiB，u 死亡时再少 256 KiB。原地覆盖旧 r 还需证明旧 r 的其他消费者已结束。

一个关键反例是：融合内核在寄存器里正确计算 u 和 y，却只写 y。单独比较 y 会通过；后续 residual 分支读取 u 时，得到未初始化值或旧值。检查必须让该消费者实际读取 u，并核对其存储覆盖和数值。不能以预设的 4N 与 3N 不相等，声称已经检测了漏写。

## RoPE：行级契约、分量配对与缓存追加

对位置 p、频率 θ 的一对分量 (a,b)，RoPE 为 (a cos(pθ)−b sin(pθ), a sin(pθ)+b cos(pθ))。固定配对与频率时，R(p)ᵀR(q)=R(q−p)，使 QK 内积包含相对位置。[S-ROPE](../sources/S-ROPE.md) §3.2 给出二维旋转及其推广。范数保持只能验证正交性，不能独立锁定旋转方向、配对和位置。

下面的 RotateNewQK 明确限定为同一请求的一段新增自注意力 chunk。Q 与新 K 的 token 行数必须相等，M=T；第 j 行同时属于请求 req 的同一个新增 token，其逻辑位置均为 p0+j，j∈[0,T)。不是只要求两个数组长度恰巧相同。不同请求、不同 token 顺序、只取部分 query 的接口不属于该算法的合法域，应由调用方选择另一条显式位置映射路径。

每个 head 的向量宽度为 d，旋转前 R 个分量，0≤R≤d 且 R 为偶数。相邻配对为 (2i,2i+1)，半区配对为 (i,i+R/2)，i∈[0,R/2)。其余分量原样复制。Q 与 K 的 head 数可以不同；下述单 head 变换分别遍历所有 Q heads 与所有 KV heads，不要求 Hq=Hkv。

```pseudocode
算法 RotateNewQK
输入：同一请求 req 的 q_raw[T,d]、k_new[T,d]，共同起点 p0≥0
      两侧行身份均为 (req,p0+j)，j∈[0,T)
      d>0、偶数 R∈[0,d]、pair、θ[0:R/2]
      本请求 K-rot 缓存：已就绪前缀 [0,p0)，容量 C，私有可写尾区
输出：q_rot[T,d]、k_rot[T,d]；缓存新增 [p0,p0+T)
1 在任何写入前检查：Q/K 行数相等、行身份逐行对应、行宽为 d
2 检查配对、频率、缓存请求与旋转格式一致，p0+T≤C
3 若 T=0：返回空输出，缓存不变
4 对 j=0,…,T−1：
5    p=p0+j；先将 Q/K 本行复制到独立输出，保留未旋转分量
6    对 i=0,…,R/2−1：
7        若 pair=adjacent：(a,b)=(2i,2i+1)
8        若 pair=half：(a,b)=(i,i+R/2)
9        c=cos(p·θ[i])，s=sin(p·θ[i])
10       qa=q_raw[j,a]，qb=q_raw[j,b]
11       ka=k_new[j,a]，kb=k_new[j,b]
12       q_rot[j,a]=qa·c−qb·s；q_rot[j,b]=qa·s+qb·c
13       k_rot[j,a]=ka·c−kb·s；k_rot[j,b]=ka·s+kb·c
14   将 k_rot[j] 写入缓存中属于 (req,p) 的尾区
15 等待全部新增 K 写入成功且对消费者可见
16 发布本段 K 就绪范围到 p0+T；不据此单独推进整个模型的 consumed_len
```

M=1、T=2 在步骤 1 被拒绝，不能用共同循环的 min(M,T) 静默丢行，也不能循环到 max(M,T) 越界。等长但来自两个请求同样非法。元数据检查在写入前完成，避免不合法调用留下半段更新。

该伪代码选择 K-rot 缓存，旧前缀保持不变。若保存 K-raw，应写原始 K、记录其位置及格式，并在消费端恰好旋转一次；那是另一条完整数据流。把已经旋转的历史再次旋转会同时改变值和历史流量。若执行中失败，不发布新增就绪范围；尾区内容须按请求状态协议重算或隔离。本章不把单层 K 写入成功当成全部层状态已提交。

一次 p0=7、T=3 的调用使用位置 7、8、9。拆成 T=1 与 T=2 两次时，第二次起点为 8；在同一数值策略下应恢复同一组结果。物理搬页、压紧 batch 或请求退出不改变这些逻辑位置。用 j 代替 p0+j 会在 p0=0 的测试中隐藏错误。

[S-FA-IMPL](../sources/S-FA-IMPL.md) 的固定文档说明两种配对、按已有长度追加 K/V，以及缓存索引重复时的写入歧义。本章同请求、等长新增 chunk 是为讲清机制而限定的子算法，不是该后端的完整接口。该文档还为非因果且非局部模式规定不同的 query 位置处理，不能把此处的逐行位置推广到所有调用模式。

每个向量的旋转主项为 R/2 对、每对四乘两加，即约 3R FLOP，另有三角值计算或查表。对 T 行、Hq 个 Q heads 与 Hkv 个 K heads，旋转算术主项为 3TR(Hq+Hkv)。相同位置的三角值可跨 heads 复用；完整表需要为覆盖的位置保存每对的 sin/cos。融合投影写出或 K-cache 写入可减少新增表示的中间读写，收益随 T 增长，不自动减少 attention 对合法历史的读取。频率、配对或旋转维变化时，旧 K-rot 不再自动兼容，须重建或放弃复用。

## 激活、softmax 与可合并统计

门控 FFN 可写为 U=XWᵤ，G=φ(XW𝗀)，输出 (G⊙U)W𝒹。两条上投影的通道对应、bias、门函数和参数顺序属于模型定义。SiLU(x)=x·sigmoid(x)；为避免指数溢出，x≥0 时可用 x/(1+exp(−x))，x<0 时用 x·exp(x)/(1+exp(x))。精确 GELU 与近似 GELU 是不同的数值函数。

可重建的执行顺序是：两条完整上投影各自完成归约和 bias；对门分支逐元素应用 φ；按同一 token/FFN 通道相乘；下投影完整归约后应用自己的 epilogue。若上投影各为 M×F、中间元素 s 字节，分别物化 U、门结果及乘积会带来这些缓冲的写读；融合可让乘积直接成为下游输入，但仍须支付下游读取或合适的片上交接。更大的融合区域也可能增加驻留、spill 或限制矩阵内核。

对有限可见 logits，softmax 定义为 pᵢ=exp(zᵢ−m)/Σⱼ∈A exp(zⱼ−m)，m=maxᵢ∈A zᵢ。不可见项不进入分子或分母。下面选择空行返回零概率、empty=true、自然对数 LSE=−∞；这是一项明确的示例契约。

```pseudocode
算法 StableMaskedSoftmax
输入：z[N]、显式可见集合 A⊆{0,…,N−1}
输出：p[N]、empty、自然对数 LSE
1 检查 A 的索引有效；可见分数必须有限，否则拒绝或转特殊值路径
2 若 A 为空：返回全零 p、true、−∞
3 m=max_{i∈A} z[i]
4 l=Σ_{i∈A} exp(z[i]−m)
5 对 i=0,…,N−1：
6    若 i∈A：p[i]=exp(z[i]−m)/l
7    否则：p[i]=0
8 返回 p、false、m+ln(l)
```

N=0 由空集合分支处理。全 mask 行用有限负常数填充后调用普通 softmax，会得到非零均匀分布；这改变了本章契约。可见输入含正无穷时出现 ∞−∞，也不能依靠减最大值修复。

分块时，对已处理可见集合保存 m=max z、l=Σexp(z−m)、U=Σexp(z−m)v，v 可以是一个向量。空状态为 E=(−∞,0,零向量)。两个集合必须不重叠，并集恰好是需要的归约域。

```pseudocode
算法 MergeSoftmaxStates
输入：同一单位、维度和指数底数下的两个状态 (m1,l1,U1)、(m2,l2,U2)
输出：合并状态
1 若 l1=0：返回第二个状态
2 若 l2=0：返回第一个状态
3 m=max(m1,m2)
4 r1=exp(m1−m)，r2=exp(m2−m)
5 l=r1·l1+r2·l2
6 对每个 value 通道 c：U[c]=r1·U1[c]+r2·U2[c]
7 返回 (m,l,U)
```

空分支避免 −∞−(−∞)。新最大值改变尺度时，l 与 U 都要重标度；只修正分母会让输出错误。非空最终输出为 U/l，LSE=m+ln(l)。单独归一化各段再拼接概率会使质量和等于非空段数；直接平均各段 attention 输出也忽略各段质量。

普通 softmax 是 O(N) 归约与指数工作，保存整行需要 O(N) 临时元素；重算指数或分段统计可改变临时驻留与读取次数。融合 attention 保留局部概率和统计，可避免完整分数/概率矩阵写回，其 IO 推导与条件见 [S-FLASH](../sources/S-FLASH.md) 和 [Attention IO](attention-io.md)。

底数也是中间接口的一部分。使用 exp2 计算自然指数时，score 乘以 1/ln2；若保存 log₂Z，恢复自然对数 LSE 必须乘 ln2。固定 Triton 来源说明承载这一机制，不能把两类统计直接互换。

输出 dtype 不能概括整条数值路径。[S-TORCH-NUMERIC 的固定文档](../sources/S-TORCH-NUMERIC.md)说明，所述 SDPA math 默认路径会把 FP16/BF16 输入提升到 FP32，执行 FP32/TF32 计算再转换输出，这同时改变误差和内存成本。此事实限定于该文档所述版本与路径；本章不据 S-SDPA 断言 functional.softmax 的 dtype 参数语义。一般比较仍需明确转换发生在输入、乘法、局部和、概率还是输出边界。

## 成本账与条件变化

成本至少有六类：计算含乘加、归约、激活和三角值；访存含有效元素字节、重复读取、布局转换与实际事务；通信含 split-K 部分和与跨设备合并；准备含打包、scale 重排、位置表、编译和调优；同步含 barrier、事件、主机读取设备长度；驻留含权重、激活、KV、工作区和同时存活的缓冲。普通输出独占 GEMM 不需要跨输出块归约，但块内协作和输入可见性仍有成本。重叠阶段不能直接将耗时全部相加，关键路径由依赖与共享资源决定。

设 X 为 32×4096、W 为 4096×4096，输入、权重与输出均为 2 字节，忽略 bias。主项为 1073741824 FLOP，理想流量为 32 MiB 权重、0.25 MiB 输入和 0.25 MiB 输出，约 31.51 FLOP/byte；M=1 时约为 1 FLOP/byte。这解释了改变 M 为什么会改变重用条件，而非证明某个 tile 必定更好。

M 增大时可保留函数与归约不变量，调整输出块、聚批和权重复用；M=1 且权重大于 cache 时可比较 GEMV、窄 GEMM 与 split-K。stride 改变时保留逻辑轴，比较直接消费与重排的完整成本。dtype 改变时重算元素字节、scale、累加和舍入边界。u 从死亡变成活跃时，必须恢复其写出；不能继续沿用 3N 的融合账。片上资源不足时，应调整保留策略、分块或融合边界，必要时放弃该融合。

CPU 线程数、NUMA 和向量化影响实际资源，[S-CPU-TUNING](../sources/S-CPU-TUNING.md)支持这些条件关系；它不提供所有 CPU 的默认最佳线程数。GPU 的有效流量与事务不同，[S-CUDA-BEST](../sources/S-CUDA-BEST.md)中的访问段、对齐和共享存储模型可帮助推导，但不能把 CPU 事件计数解释为 HBM 实测。

局部收益也有组合上限。设某算子占无重叠关键路径的 10%，成本减半只能把总时间从 1 降到 0.95，加速约 1.053 倍。新增布局复制、降低驻留或增加准备可能反转结论；改变生产布局和融合边界又会改变这份成本模型，应重新列账。

## 诊断、负对照与验证边界

比较边界应覆盖真实生产者和消费者：QKV 包含切片、重排与缓存写入；norm 包含统计域、舍入点和活跃 residual；RoPE 包含请求、行位置、配对及缓存格式；GEMM 包含有效权重与输出布局。完整 forward 先定位首次分歧，再区分参数、布局、位置、归约和近似误差。

| 竞争解释 | 可区分预测 | 干预与反证修正 |
|---|---|---|
| 轴变换正确 | 按原轴语义独立重排后，每个逻辑值一致 | 使用非方形轴与非连续 stride；flat copy 错位时修正地址映射 |
| 差异仅来自舍入 | 改变归约树与精度会改变误差，不应出现稳定的漏项或身份错位 | 固定输入改变块界；误差随行置换稳定移动时先查索引 |
| bias 只应用一次 | K 分块数改变不应重复增加 bias | 使用非零 bias 与多段 K；出现整倍 bias 偏移时检查 epilogue |
| norm 使用真实统计域 | 全一有效输入的输出与 H 无关的闭式定义一致 | 改变 padding 宽度而保持 H；尺度随 P 改变时修正分母 |
| 融合保留活跃 u | y 和后续 residual 消费者均与独立参考一致，u 全部被写入 | 注入只写 y 的路径；消费者读到未初始化或旧 u 时拒绝 |
| RoPE 行位置合法 | 等长行身份对应，整段与分块使用相同绝对位置 | M=1、T=2 应拒绝；等长异请求也应拒绝；非零 p0 能检出位置重置 |
| RoPE 方向与配对符合模型 | 固定方向结果、范数和相对内积均匹配 | 同时反转 Q/K 方向可保范数，却应被独立固定参考拒绝 |
| softmax 状态合并正确 | 整行与分块的概率、U/l 和 LSE 一致，空块不改变状态 | 注入仅重标度 l 的路径；若输出偏差则修正 U 的尺度 |
| 融合收益来自交通减少 | 完整路径的相应事务或中间量下降 | 同时测资源与准备；没有预期下降时撤回单一带宽解释 |

随章检查 `tools/check_k_alg_016_full.py` 使用独立轴重排、逐输出 GEMM、Decimal 整行归一化与 softmax 参考，以及明确的存储契约。residual 三条路径逐元素执行并记录缓冲读写，按事件汇总成本；错误路径确实生成正确 y 但漏写 u，再交给后续消费者拒绝。初始化过的旧 u 也用独立数值参考检出，避免只靠未初始化标志。

RoPE 检查覆盖非零起点、同请求多行、跨 chunk、未旋转尾部、两种配对、空 chunk、不等行数与等长身份错误；用复数乘法参考和四分之一周的固定方向样例补充范数与群关系。softmax 同时核对标量质量、加权向量 U 和自然对数 LSE，不能只检查概率和为 1。

这些是确定性、只读、离线 Python 标准库检查。模型范围为 CPU binary64、Decimal 与顺序缓冲访问事件；不执行 PyTorch、CUDA 或 Triton，不模拟 FP16/FP8 舍入、GPU 内存可见性、真实 HBM 事务、线程竞争、性能或模型质量。实际设备还需核对生成指令、资源分配、同步与完整调用时间，没有这些记录就保留未测边界。

检查脚本、输入绑定与实际执行结果见[检查记录](../../docs/evidence/knowledge-workflow-20261028-batch10/K-ALG-016/round-2/check-0.json)。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：§3.2 定义 query-key 分数归一化后对 values 加权，decoder 在 softmax 前屏蔽未来位置；§3.3 给出逐位置 FFN，§3.4 讨论 embedding 与输出投影共享权重，§3.5 的位置编码加在网络入口。本章据此区分输入加性位置与逐层 RoPE，且不将共享权重容量等同于消除输出投影读取。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-RMSNORM：Root Mean Square Layer Normalization](../sources/S-RMSNORM.md)：§3–4 定义 LayerNorm 与不减均值的 RMSNorm，分析理想不变性。论文比较重新训练的模型，不能推出推理时替换既有 LayerNorm 无损。正文采用明确带 ε 的示例契约，统计域、融合与流量账是本库推导，不引用论文速度作本项目收益。此次原文阅读限于任务提供的字符摘录，不声称读完整原文或视觉核对 PDF。
  [完整原文](../../references/S-RMSNORM/reader.html) · [全文文本](../../references/S-RMSNORM/fulltext.txt)

- [S-ROPE：RoFormer: Enhanced Transformer with Rotary Position Embedding](../sources/S-ROPE.md)：§3.1–3.3 以二维旋转和分量对推广建立位置与相对内积关系。本章使用可直接核算的旋转矩阵，规避早期稿的下标笔误；该关系不保证任意 token 对的权重随距离单调下降，也不保证任意长度质量。等长同请求 chunk 与缓存发布边界为本章限定的算法契约。
  [完整原文](../../references/S-ROPE/reader.html) · [全文文本](../../references/S-ROPE/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：固定提交 647b6a2 的文档解释浮点非结合性、批量与分片路径差异、极端值、TF32、低精度 GEMM reduction，以及 SDPA math 路径的输入提升和输出转换。本章据此展开数值路径与成本，不将其扩展为 functional.softmax 参数或 linear 权重轴的接口依据。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../sources/S-FLASH.md)：固定 v2 的 Algorithm 1 使用分块和在线 softmax，避免完整分数与概率矩阵驻留主存；其循环为 K/V 在外、Q 在内。可合并统计支持本章的尺度推导，论文 IO 模型不证明任意固定硬件上的数值最优交通。此次只使用所给来源说明和原文摘录，不声称阅读全文。
  [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

- [S-CPU-TUNING：PyTorch Performance Tuning Guide](../sources/S-CPU-TUNING.md)：所给机制说明涉及线程、亲和、NUMA、融合和准备开销。本章采用其条件关系，没有读取未提供的原始教程或源码，也不采用示例配置作为默认最优参数。
  [算法与机制说明](../sources/S-CPU-TUNING.md)

- [S-SDPA：PyTorch：scaled_dot_product_attention](../sources/S-SDPA.md)：固定提交 647b6a2 的来源说明限定于 SDPA 的函数、mask、head 映射、dropout 与路径分派。它支持统一数学语义与执行路径分开的讨论，不承载 linear 的权重轴或 functional.softmax 的 dtype 参数事实。本章不再将这两项归因给它；N×K 制品是正文明确设定的示例。
  [算法与机制说明](../sources/S-SDPA.md)

- [S-FA-IMPL：FlashAttention 官方实现说明](../sources/S-FA-IMPL.md)：固定提交 c5f6ff4 的可读文档给出相邻/半区 RoPE 配对、新 K/V 原地追加、按已有长度确定位置及重复缓存索引的写入歧义。完整文档区分因果/局部与其他模式的 query 位置，本章不交付任意后端替换接口，也不将 V 解释为需要施加标准 Q/K RoPE 的对象。
  [完整原文](../../references/S-FA-IMPL/reader.html) · [全文文本](../../references/S-FA-IMPL/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 12.8.0 的有效流量与实际事务区分，以及访问段、对齐、共享存储和驻留机制用于条件化成本分析。此次只使用所给说明和原文摘录，没有目标 GPU 测量，也不把元素事件直接换名为显存事务。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-TRITON：Triton：Fused Attention](../sources/S-TRITON.md)：固定提交 042b2cd 的作者核验说明承载 exp2/log2 中间统计、概率转换与累加精度、执行域及 tile 与实际线程布局的区别。本章使用这些提炼关系，不要求读者打开源码，也不声称本轮读取了源码归档。
  [算法与机制说明](../sources/S-TRITON.md)

### 本章推导范围

正文设定算例、地址反例、资源账、行身份约束、残差消费者、缓存追加子算法和负对照均为本库解析推导。来源版本上限为 2026-04-30，编辑日期不改变来源时间。相关章节按本轮提供的完整文本或指定摘录使用，摘录不构成整篇阅读。本章没有目标 GPU 数值、性能、真实事务或 Agent 使用效果证据。

## 延伸阅读

- [从模型计算图推导 Prefill、Decode 与持久状态](../models/prefill-decode-state.md)。
- [Attention 的 IO、在线 softmax 与分块归约](attention-io.md)。
- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [CPU GEMM/GEMV：寄存器、缓存与线程划分](../hardware/cpu-linear-kernels.md)。
- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
- [Attention 执行：数学语义、分块归约、低精度与硬件数据流](../frameworks/attention-backends.md)。
- [正确性 Oracle：函数参考、状态变形与服务契约](../tools/correctness-oracles.md)。
