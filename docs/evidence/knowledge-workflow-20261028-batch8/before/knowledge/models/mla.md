# K-MODEL-006：MLA：压缩状态、矩阵吸收与解耦 RoPE

MLA 用一份低维内容 latent 和独立的位置 key 表示历史。缓存容量减少来自表示，decode 流量收益还要求计算真正沿该表示完成；若每步展开全部历史 K/V，持久状态小仍可能伴随大量计算和临时读写。

## 函数、输入与充分状态

普通 MHA 每层、每 token 保存各 head 的 K/V，共约 2H d_h 个元素。MLA 让 H 个 head 共用内容 latent，各 head 用不同的上投影解释它，并非把普通 KV head 简单合并。它是模型结构，不能凭下面的线性等式把任意已有 MHA checkpoint 无损改成 MLA。

采用列向量记法。h_t∈ℝ^d 是一层 attention 输入，H 是 query head 数，d_h 是内容 key 和 value 的每头维度，d_c 是 KV latent 宽度，d_q 是 query latent 宽度，d_R 是解耦位置宽度，L 是层数。本文采用来源公式中内容 K/V 同宽的情形；不同 value 宽度须分别改写投影和成本。各宽度为正整数，成对旋转要求 d_R 为偶数。MLA 的理想压缩要求 d_c+d_R 小于所比较缓存的每 token 宽度，但这不是吸收等式成立的条件，也不要求 d_c>d_h。

参数维度为 W_DKV∈ℝ^(d_c×d)、W_DQ∈ℝ^(d_q×d)、W_UK,i 和 W_UV,i∈ℝ^(d_h×d_c)、W_UQ,i∈ℝ^(d_h×d_q)、W_QR,i∈ℝ^(d_R×d_q)、W_KR∈ℝ^(d_R×d)，以及 W_O∈ℝ^(d×Hd_h)。W_O,i 是 W_O 对应 head i 的 d×d_h 列块。定义：

z_t = W_DKV h_t，s_t = Φ_KV(z_t)；
c_t = Φ_Q(W_DQ h_t)；
k^C_(t,i) = W_UK,i s_t，v^C_(t,i) = W_UV,i s_t；
q^C_(t,i) = W_UQ,i c_t；
k^R_t = R_(ρ(t)) W_KR h_t，q^R_(t,i) = R_(ρ(t)) W_QR,i c_t。

R_p 是制品指定的位置变换，ρ(t) 是模型坐标，不能由物理页号或 batch 行号代替。Φ_KV、Φ_Q 包括实际 latent 后的归一化与宽度缩放；简化论文图中的恒等映射只是一个特例。例如固定 gain g、宽度缩放 γ 与 ε>0 时，可有 Φ(z)=γ(g⊙z)/sqrt(Σ_a z_a²/width+ε)。实际参数和放置按制品核对。DeepSeek-V2 §3.1.2、DeepSeek-V3 v1 §4.2 都提到压缩 latent 后的额外 RMSNorm 和宽度瓶颈缩放，不能只为 KV 路径补 norm 而漏掉 query 路径。V2-Lite 不压缩 query，说明 query 压缩不是 MLA 的必要条件；这时直接按制品产生 q^C、q^R。

对 query t，A_t 是允许读取的逻辑 key 集合。有限输入域中：

e_(t,i,j) = α_model [q^C_(t,i)·k^C_(j,i) + q^R_(t,i)·k^R_j]，
a_(t,i,j) = exp(e_(t,i,j)) / Σ_(k∈A_t) exp(e_(t,i,k))，
o_(t,i) = Σ_(j∈A_t) a_(t,i,j) v^C_(j,i)，
u_t = Σ_i W_O,i o_(t,i)。

来源中的基础缩放 α_model=1/sqrt(d_h+d_R)。吸收后点积虽有 d_c 个内容通道，也不能改用 1/sqrt(d_c+d_R)。若 checkpoint 的长上下文策略还调整 attention 温度，应在 α_model 或所指定的分量中显式还原，不能重复施加。本文选择空 A_t 返回零 attention 输出并附 empty 标志；这只是接口契约，普通含当前 token 的因果 attention 通常非空。可见分数、参与值及投影中间量要求有限；NaN、溢出或正无穷须拒绝或另定义特殊值路径。

每层持久状态保存 s_j 和共享的 k^R_j，而不是每 head 一份位置 key；另保存有效逻辑范围、模型/权重/adapter 身份、位置规则、布局及量化参数。查询和归约状态不随历史持久保存。若缓存原始 z_j，需要能精确恢复 Φ_KV 的配置；逐 token 归一化因子可以重算或另存，其计算或字节均须计入。状态只支持继续计算，不自动包含末行 logits、已采样未消费的 token 或客户端输出记录，见[模型状态](prefill-decode-state.md)。

## 吸收成立的边界

内容 key 的吸收是

q^C_i·(W_UK,i s_j) = (W_UK,iᵀq^C_i)·s_j。

令 q̃_i=W_UK,iᵀq^C_i，历史内容 key 即可省去展开。对固定 head 的权重 a_j，先聚合 z̄_i=Σ_j a_j s_j，再求 o_i=W_UV,i z̄_i，与逐 token 上投影 V 后聚合在实数中相等。输出可逐头先算 W_UV,i z̄_i 再应用 W_O,i，也可预乘 G_i=W_O,i W_UV,i，得到 u=Σ_i G_i z̄_i。预乘是选择：G_i 有 d d_c 个元素，而原两级矩阵只有 d_h d_c+d d_h 个元素；预乘可能增加权重驻留和每 query 工作。

相同的 head 身份、可见集合、位置分量、缩放和 latent 数值是合法条件。吸收只跨越线性投影，不能把 Φ_KV(z) 换成 z。例如 z=[1,0] 与 [2,0] 在带 ε 的 RMSNorm 后既不是原值，也未必严格相同；省掉 norm 不是普通归约舍入。若 query latent 后有 norm，可先计算 c 再吸收 W_UK；不能预乘 W_DQ 与后续矩阵并顺带消去 Φ_Q。固定 gain/常数缩放可以纳入相邻矩阵，但依赖当前 token 的分母仍需计算。

解耦 RoPE 保留了内容吸收，同时单独计算位置内积。若在内容 key 上旋转，分数变成 qᵀR_pᵀR_j W_UK s_j，query 侧需要依赖历史位置 j 的矩阵；一般不能用一个固定 q̃ 对所有历史 token 计算。一个直接反例取 W=diag(2,1)、90° 旋转 R、x=[1,0]，RWx=[0,2]，WRx=[0,1]，二者不同。把 latent 当作普通 head 旋转不能从矩阵结合律推出。标准固定频率下，位置内积满足 (R_p q)·(R_j k)=q·R_(j−p)k；共同平移不变可用作参考，但只错改新增 query 的位置通常会破坏它。YaRN 等扩展还须核对频率、分量和温度；V2 §3.1.4 与 V3 §4.3 对共享位置 key 的说明不能代替实际制品的完整位置契约。

量化的非线性不服从同一重排。Q(Ws) 通常不同于 W Q(s)，Q(AB) 也通常不同于 Q(A)Q(B)。比较两条吸收路径时，先固定它们共同消费的反量化 s 和相同权重，再判断实现等价；与未量化模型的误差另外评估。latent 的误差 δs 会同时作用于多个 head：内容 logit 误差为 α_model q̃_i·δs，固定概率下 value 误差为 W_UV,i Σ_j a_jδs_j，随后还有概率变化造成的误差。不能把两者混成一个重建误差指标，也不能把 KIVI 对普通 K/V 的经验直接宣称为 MLA 的最佳轴。

## 从一行 decode 到分块 prefill

下面算法只描述一层无 dropout 的前向。当前 token 状态先进入私有候选视图，旧历史只读。层输出成功不等于完整模型步成功；调用者应等全部必需层与后继分布完成后共同发布消费长度，失败则保留旧视图。客户端是否已经收到 token 不决定这个提交点。

在线状态 X=(m,ℓ,U) 表示已处理可见 key：m 为最大分数，ℓ=Σexp(e−m)，U=Σexp(e−m)s，U 的宽度是 d_c。空状态 E=(−∞,0,0)。两个非空状态合并时，m′=max(m₁,m₂)，r_k=exp(m_k−m′)，ℓ′=r₁ℓ₁+r₂ℓ₂，U′=r₁U₁+r₂U₂。空状态先分支处理，避免 −∞−(−∞)。集合必须互不重叠且完整覆盖 A_t。

```pseudocode
函数 Merge(X1, X2)：
    若 X1.ℓ=0：返回 X2 的副本
    若 X2.ℓ=0：返回 X1 的副本
    m = max(X1.m, X2.m)
    r1 = exp(X1.m−m)；r2 = exp(X2.m−m)
    返回 (m, r1 X1.ℓ+r2 X2.ℓ, r1 X1.U+r2 X2.U)

算法 MLA_LAYER_STEP
输入：参数；旧视图 C，恰好覆盖逻辑 [0,n)；h_n；模型坐标 ρ；可见规则；块长 B>0
输出：u_n、逐 head empty 标志、未发布的候选视图 D
1 核对旧视图身份、表示、逻辑范围；预留私有追加空间
2 s = Φ_KV(W_DKV h_n)；kr = R_ρ(n) W_KR h_n
3 c = Φ_Q(W_DQ h_n)；检查 s、kr、c 及必要元数据有效且有限
4 创建 D：引用 C 的只读历史，私有槽 n 存 s、kr 及其必要元数据
5 对每个 head i：
      qC = W_UQ,i c；qR = R_ρ(n) W_QR,i c；qt = W_UK,iᵀ qC
      X = E
      对 J=[0,B),[B,2B),… 的真实尾部，直到 n+1：
          V = {j∈J∩[0,n+1)：可见规则允许 query n 读取 j}
          若 V 为空：Y=E
          否则：
              从 D 读取每个 j 的同身份 sj、krj，必要时先反量化
              e_j = α_model (qt·sj + qR·krj)
              检查可见 e_j 与参与 sj 有限
              mb=max(e_j)；w_j=exp(e_j−mb)
              Y=(mb, Σw_j, Σw_j sj)
          X=Merge(X,Y)
      若 X.ℓ=0：zbar_i=0；empty_i=真
      否则：zbar_i=X.U/X.ℓ；empty_i=假
      o_i = W_UV,i zbar_i
6 u_n = Σ_i W_O,i o_i；确认输出有限，返回 u_n、empty、D
7 任一步失败：停止发布；排空已提交访问后丢弃本候选，旧 C 不变

调用者：
    逐层消费本步 hidden 并收集各层私有候选
    全部必需层、最终 norm 和 logits 成功且可消费后，共同发布所有候选及长度 n+1
    否则不发布任何层的新消费位置；等待在途访问结束后回收私有对象
```

每完成一个 KV 块，X 必须等于已遍历可见集合的统计量；这是检查中间状态的独立不变量。全 mask 块保留旧 X，尾部不解引用越界地址。split-K 可把 key 范围拆给多个任务，每任务发布完整的 E 或 (m,ℓ,U)，等待完成后按 Merge 合并，再做一次除法。不能直接平均各分区的归一化输出；两个分区最大值不同而未重标度，也会改变结果。

prefill 的输入为 M 个已知 query 行及历史。先按各行绝对逻辑位置生产本块 s、k^R，块内未来条目虽已算出仍被各行 mask 排除。吸收路径为每行/head 运行上述归约，随后做值和输出投影。另一合法路径把本次所需 N 个 key 的内容 K/V 上投影一次，供多个 query 行复用，再进行分块 attention；共享 k^R 不必持久复制成 H 份。两条路径共同写入压缩持久状态，展开 K/V 作为临时工作区，在最后读者结束后释放。若 prefill 只留下展开对象，通常不能无损恢复原 latent，不能把未定义的逆投影交给 decode 补做。

分页只改变读取地址。量化组跨页时，码、scale/实数偏移或整数零点、有效组长与封组状态必须共同解释；尚未封组的高精度尾部不等于已量化组。可选逐 token 分组或跨 token 分组，但同一份 s 同时服务 K 和 V，不能不经推导复制普通 K/V 两套不相容的重建语义。

## 容量、算术与实际交通

未量化且每层同构时，N 个保留 token 的逻辑容量为 L N(d_c+d_R)b 字节，b 是每元素字节。位置 key 若错误复制成每 head 一份，容量会变成 L N(d_c+H d_R)b。按位宽估量量化容量时，对每个状态族分别计算码字节、组数乘元数据字节、高精度尾部、padding、页表及副本；不能把整个状态直接乘一个理想压缩比。

设定 L=32、H=32、d_h=128、d_c=512、d_R=64、b=2、N=8192：MLA 每 token 全层为 36 KiB，总 288 MiB；同设定 MHA 为 512 KiB/token，总 4 GiB。比例为 7.03125%。这只是结构算例；没有采用 DeepSeek 型号的层数/head 数，也没有把其 93.3% 部署降幅当通用比例。展开 K/V、图缓冲和 split-K workspace 若同时存活，要另加峰值。

下面比较单层两个明确候选，使用乘加次数 MAC，换成通常“乘和加各一次”的 FLOPs 时主项乘二。设 P=Σ_t |A_t| 为有效 query-key 对数，稠密非因果时 P=MN；实际 padded tile 可能执行更多。

| 阶段 | 本次展开 K/V 后 attention | query 侧吸收、latent 聚合 |
|---|---:|---:|
| 历史内容 K/V 上投影 | 2NH d_c d_h | 0 |
| query 内容吸收 q̃ | 0 | MH d_c d_h |
| 内容分数、位置分数和值聚合 | PH(2d_h+d_R) | PH(2d_c+d_R) |
| latent 聚合后的 value 上投影 | 0 | MH d_c d_h |
| 输出投影 | MH d_h d | MH d_h d |

共同的新增状态生产、query 生成另计：新增 token 数为 N_new 时，W_DKV、W_KR 主项为 N_new d(d_c+d_R)；压缩 query 的线性主项为 M[d d_q+H d_q(d_h+d_R)]，另有 norm、RoPE 和写入。比较表假定每次展开历史，未复用前一次调用的展开结果。若长期保存展开 K/V，可以只为新增 token 投影，却要付相应持久容量，已是第三个候选。

两个候选的 MAC 差“吸收减展开”为

Δ = 2H d_c d_h(M−N) + 2PH(d_c−d_h)。

因此 decode 的少 query 可免去随 N 增长的上投影，而长 prefill 中上投影被多行摊销；只有在 d_c>d_h 时，吸收的成对点积才比展开路径更贵。为讨论稠密交点，以下取 M,N>0 且 P=MN。吸收严格更省 MAC 的一般条件是

M[d_c d_h+N(d_c−d_h)] < N d_c d_h。

记 D=d_c d_h+N(d_c−d_h)，右侧 N d_c d_h 为正。若 D>0，可除以 D，得到 M<N d_c d_h/D；相等时两候选 MAC 相同。若 D=0，左侧为零，任意正 M 都满足严格不等式，不存在有限交点。若 D<0，左侧为负，任意正 M 也都满足；不能直接保留“小于负阈值”的方向。d_c>d_h 保证 D>0，所以常见宽 latent 配置可以直接使用正分母阈值。上述 d_c=512、d_h=128、N=8192 的设定属于该域，阈值约 167.18 行；这是这个算术模型的交点，不是 kernel 性能阈值。

因果或稀疏 prefill 必须把真实 P 代回 Δ，不能继续用 P=MN 的交点。完整 prefill M=N 时，Δ=2PH(d_c−d_h)：P>0 时，其符号分别随 d_c>d_h、d_c=d_h、d_c<d_h 为正、零、负；P=0 时为零。特别地，1≤M≤N 且 d_c≤d_h 时，Δ 的两项均非正，不存在该范围内由 latent 点积更宽引起的正差反转。即使 MAC 有严格优劣，实际时间仍取决于矩阵效率、流量和准备。

一个能触发旧结论错误的设定是 H=1、d_c=64、d_h=128、d_R=2、N=256。M=1、P=256 时，排除共同项后，展开为 4,260,352 MAC，吸收为 49,664 MAC，Δ=−4,210,688；此时 D=−8192，不能用打印出的负阈值 −256 排除吸收。完整稠密 prefill M=N=256 时，展开为 21,102,592 MAC，吸收为 12,713,984 MAC，Δ=−8,388,608，同样与 d_c−d_h 的负号一致。这是允许宽度域内的代数反例，不代表该配置的模型质量或设备性能。

唯一历史有效字节每层为 N(d_c+d_R)b，并非所有 query/head 的总读取。若每 CTA 联合 H_t 个 head、B_q 个 query 行，共享一份输入，忽略 cache 与尾部，消费者请求约为 ceil(H/H_t)ceil(M/B_q)N(d_c+d_R)b。一个 latent tile 可以同时供内容分数和 latent 值聚合，但在后者结束前不能覆盖。不同 CTA、query 块和请求可能重读同一历史；L2 命中、非合并访问和 spill 决定真实 HBM 事务。展开路径若写回临时内容 K/V，单层额外写 2NHd_h b，再由 attention 读；片上展开可省这次写回，却仍付计算和片上容量。

split-K 采用 C>1 个分区、FP32 的 (m,ℓ,U) 时，部分结果主项为 4MH C(d_c+2) 字节，写出与合并读取约两倍，另计输出。C=1 可直接消费结果，不必强行分配这份 partial。softmax 指数约按 PH 计，U 重标度约随 MH d_c 与有效块数增长，不能全部除以矩阵峰值估时。

准备包括页表/末页有效长度、量化组描述、mask、工作划分与计划构造。S-FI 的固定实现区分 query 行偏移、页索引数组区间和末页 token 数；单位相同的假设会产生形状看似正确的错误。计划可在满足条件时跨层复用；batch、长度、mask 或变体条件变化后须重建。图中的地址稳定也不能证明 metadata 仍属于当前请求。同步包括 tile 生产、全部读者完成、partial 合并及候选发布。权重、缓存、候选增量、临时展开、partial 和并发工作区按同时存活的物理对象计峰值，预留额度不再与其中已分配对象重复相加。时间沿依赖关键路径核算，资源下界可比较计算、设备交通和非矩阵吞吐，已重叠区间不机械相加。

## 一条可重建的线程与缓冲映射

给出普通标量运算的保守候选，便于检查地址和最后读者；这不是已调优的 Tensor Core kernel。每 CTA 处理一个 query 行、两个 head，64 线程，线程 τ=32w+λ，w∈{0,1}、λ∈[0,31]，warp w 独占 head i=2b_h+w。KV 块 B_s=16；以 d_c=512、d_R=64 为设定。超出 H 的末尾 head 不写输出，但所有线程仍参加 CTA 同步。

每层采用分开的、页内 token 优先的 S 与 KR 数组，页长 P_g。逻辑 token j 的页号为 table[floor(j/P_g)]、页内位置 r=j mod P_g。无 padding 的 BF16 元素地址为 baseS[页号]+2(r d_c+c)，位置 key 为 baseKR[页号]+2(r d_R+a)。有实际 stride、padding 或不同页表时必须改用对应描述符。末页有效长度和可见性检查先于解引用；跨页 KV tile 逐条映射，不能假设 tile 在物理上连续。

CTA 共享输入 S_tile[16,512]、KR_tile[16,64]，以及已生成的 FP32 q̃[2,512]、BF16 qR[2,64]。对各数组线性元素下标 x，线程 τ 搬 x=τ+64v 的有效元素，无效尾项填零；全部写入完成并建立 CTA 可见性后才能消费。q̃、qR 只在该 query 开始加载一次。它们的 dtype 是本候选的数值选择，代数等价并不承诺这一混合精度路径逐位相同。

warp w 的 lane λ<16 负责 key j=k0+λ：沿全部内容和位置通道顺序累加一个分数，应用可见性和 α_model。其他 lane 不产生有效分数。全 warp 参与有效项 max/sum；先判定该 head 本块是否全空。非空块得到新最大值 m′、旧贡献系数 r 和 p_j=exp(e_j−m′)；旧状态空则 r=0，本块空则保持状态、r=1 且 p_j=0。lane j 是共享 P_tile[w,j] 的唯一写者，warp 同步后，各 lane 才读取本 warp 的整行 P。

lane λ 独占 latent 通道 c=λ+32u，u=0…15；每块更新其 16 个 FP32 累加量 U_c←r U_c+Σ_(j=0..15)p_j S_tile[j,c]，并同步更新同尺度 ℓ。两 warp 都是 S_tile 的读者，各自也是 KR_tile 的读者；所有读者结束后 CTA 同步，才允许下一块覆盖输入。同步加载版本无需虚构异步完成事件；若替换为异步复制/矩阵指令，须另核对其完成契约。

结束时各 lane 将自己的 U_c/ℓ 写到全局 zbar[query,head,c]；空行写零。下一消费者等待这些写入完成，计算 W_UV zbar 后按原 head 次序拼接，应用 W_O。由此每个 zbar 元素只有一个写者。该候选在 kernel 间物化 MHd_c 个 FP32 latent 聚合元素，需要计写出与再读；可进一步融合投影，但应重算输出所有权和资源。

BF16 的 S_tile 占 16 KiB、KR 占 2 KiB，FP32 q̃ 占 4 KiB、BF16 qR 占 0.25 KiB、FP32 P 占 0.125 KiB，共享主项合计 22.375 KiB。U 的 4 KiB 分布在寄存器，平均每线程 16 个 32 bit 累加量；分数、统计、地址和编译器分配另计，不能将 26.375 KiB 全算作共享存储。双缓冲 S/KR 再增加 18 KiB，且需为每槽记录加载完成与两个 warp 最后读者。没有目标程序的资源报告，不能由这份账声称无 spill 或指定最佳 tile。矩阵指令版本要重新证明操作数布局、参与线程、完成依赖及输出交接，不能把这个 lane 映射直接当作内部 fragment 布局。

## TP、阶段交接与表示所有权

按 query head 分片的方案 A 让 rank r 拥有 head 集合 I_r 及相应投影，复制全部 s/k^R。各 rank 得到 u_r=Σ_(i∈I_r) W_O,i o_i，再对 hidden 求和，或按后继布局做等价归约。latent 的生产可由各 rank 重算，或由一个所有者生产后广播；两者计算和通信不同。TP4 时上面的单请求每 rank 仍为 288 MiB，全组 1.125 GiB。若对照 MHA 按 head 分片，MHA 每 rank 1 GiB、全组 4 GiB，全组 MLA 比例变为 28.125%。

按特征分片的方案 B 可以让四 rank 各存 d_c/4 个 latent 通道及 d_R/4 个完整 RoPE 成对通道，每 rank 状态降至 72 MiB。为了完整定义这个候选，所有 rank 复制 query heads；各 rank 计算自身通道的内容和位置 partial score，先 SUM 得完整 logits，才能对完整可见集合 softmax。每 rank 用相同概率聚合本地 s 通道，再以 W_UV 对应列算 value partial，应用 W_O 得 hidden partial，最后对 hidden SUM。不能在 feature partial score 上各做 softmax 再合并输出。若 Φ 的分母也在特征分片上计算，新增 token 还要归约平方和；RoPE 的一对通道必须按相同位置规则计算后分片，量化组跨 rank 时也需明确尺度归属。

例如 M=1、H=32、N=8192、完整 logits 为 FP32 时，score 归约张量大小 A=1 MiB。理想均匀 ring all-reduce 的每 rank 发送量为 2(p−1)A/p，TP4 得 1.5 MiB/层、32 层得 48 MiB/步；这是每 rank 发送量，全组为四倍，接收另计。最终 hidden 若 d=4096、FP32，归约又发送 24 KiB/rank/层。这个候选若先物化完整 logits，还需相应驻留；逐 tile 归约可减驻留，却增加集合次数和依赖。方案 A 已有自己的输出归约，比较方案 B 时不能把两者共有项重复当成新增项。实际 collective 算法、拓扑和延迟未测。

EP 只改变专家计算归属，不自动分散 attention 状态。attention 的请求所有者与专家派发/返回的 token 身份必须衔接，不能把 EP 度再乘到 attention 缓存分母里。

再追踪一次设定的 P/D 交接：源 TP2 采用方案 A，prefill 完成 8192 个 token，末行 logits 已选出但尚未消费 x_8192；目的 TP4 同样采用方案 A，模型、norm、位置和 BF16 状态格式相同。源暂时停止推进，固定所有层 [0,8192) 的压缩视图并保留读者保护；目标先预留四份状态。选一份源副本发送唯一的 288 MiB 给目的 root，再向三个目的 rank 分发，各再搬 288 MiB，共 864 MiB 的组内发送。不能从源的两个副本各重复复制同一范围，或仅传 s 而遗漏 k^R。

目标逐层收齐 s、k^R、有效范围和身份并建立消费可见性，完整包就绪后由唯一接管记录发布，同时交付 pending token x_8192 及其逻辑/模型位置。目的第一次 decode 消费它，生成位置 8192 的新状态；不能因为旧缓存完整就重复追加位置 8191，或把“无待 prefill token”误判为无工作。若没有 pending 且没有末行分布，需要恢复最后一行的后继分布，缓存长度本身不能产生 logits。

此轨迹在源尚未释放、目标四份已分配时，仅持久状态同时存活峰值为 (2+4)×288=1728 MiB，尚有发送暂存、页表和其他工作区。源确认目标接管及自身最后读者结束后才回收。发布前失败保持源为权威，目标停止新工作、排空已提交传输后回收；部分到达不构成可执行状态。此处选择完整包发布，逐层流水是另一种需展开失败语义的方案。以上字节与协议是设计推导，不证明真实网络或 GPU 可见性已经验证。

## 条件改变后怎样调整与诊断

保留吸收等式不等于保留原执行选择。M 增大时重新比较展开的摊销和吸收的 d_c 归约；是否出现算术反转须保留 d_c 与 d_h 的关系、真实 P 及交点分母的符号。batch 增大可以增加任务数，却不自动复用不同请求的历史。H_t 增大可少读输入，也增大 query/输出驻留；split-K 已补足并行度后，再增加分区可能只增加 partial 和合并。量化后的尺度、残余区和质量门槛可能抵消容量收益。模型/adapter、norm、位置频率或缩放改变时，只有明确等价转换才能保留旧状态；从展开 K/V 逆推 latent 不作为默认转换。

| 观察与竞争解释 | 可控区分实验 | 反证后的修正 |
|---|---|---|
| 缓存小却 decode 未加速：历史仍展开，或带宽收益被并行度/非矩阵工作抵消 | 固定形状记录持久对象、历史展开、实际设备事务与完整调用；独立扫 H_t、分区数 | 有历史展开先核对路径；字节确降但时间不降时，撤回单纯带宽解释 |
| 只有长历史/换位置后错：旋转、尺度、页表或 norm 不一致 | 显式展开参考逐项比内容/位置 logits、概率、head 输出；固定内容测试相对旋转，再只改新 query 位置 | 内容先分歧查 latent/norm/head；位置先分歧查坐标/频率；不用扩大最终容差掩盖 |
| 分区数改变就错：合并缺 rescale、漏项/重复项或空块残留 | 两分区设置不同最大值，插入空分区并置换顺序；对照整行 softmax；故意未缩放合并必须被检出 | 覆盖完整才讨论舍入；空状态不能参与无穷减法 |
| TP 单卡对、多卡错：head 拼接、feature 分数归约或 norm 分母缺失 | 固定张量比完整 logits、逐 rank partial 和输出；负对照漏掉一片特征 | 缺分片贡献是函数错误；head 复制导致每 rank 容量不降则修正容量账 |
| 量化首 token 对、后续错：首行未读量化缓存、组尺度错或共享误差放大 | 实际执行多个 decode，跨封组与页边界；固定码/参数先验证解量化，再对照高精度 logits 和任务质量 | 格式错误先修；格式正确仍超质量门槛则调整位宽/组轴/残余区或放弃候选 |
| 换后端或 P/D 后错：状态表示不一致、metadata 单位/代次错、包提前发布 | 固定输入比较交接前后同身份 s/k^R 和下一行输出；延迟其中一分量并注入失败 | 对象未就绪先修发布条件；不能把稳定地址当作当前内容证明 |
| 长 prefill 中吸收反而贵：点积宽度、投影和准备的占比改变 | 固定 N，扫 M 并分别量准备、投影、attention 与峰值；使用真实 P 重算 MAC，并区分 d_c 与 d_h 的大小 | 先检查交点适用域；曲线不随合法 MAC 预测变化时查矩阵效率/流量，不把算术交点当性能交点 |

随章独立检查以投影前 raw latent 为输入，使用 Decimal 高精度、显式逐 key K/V 展开和整行 softmax 作参考，比较 binary64 吸收与分块归约。覆盖双 head、两侧 norm、位置路径、非连续模型坐标、尾块、稀疏可见集合、单 key、空历史/空行和分区置换；故意交换 head、漏 norm/位置、改 scale、漏 feature 贡献及漏合并重标度，应与同一参考产生可检测分歧。脚本拒绝非有限比较，并核对容量、TP 复制和发送量的具体数值。成本检查另按标量乘加事件枚举两候选，对照推导出的 Δ；覆盖三种分母符号、严格交点、空可见集合和完整 prefill 的三种宽度关系，并用未判断分母符号的旧阈值作负对照。执行状态见交付记录，正文不预先宣称通过。

这只检验小宽度 CPU 数学模型和资源算术，不覆盖真实 BF16/FP8 舍入、GPU barrier/异步生命周期、真实分页/量化码格式、多卡 collective、迁移故障或模型任务质量。目标设备实验还应固定制品、shape、布局、dtype、mask、预热与计时边界，报告实际 HBM/片上事务、寄存器、spill、同步、workspace 与完整时间线。本章没有 GPU 实测或 Agent 使用效果证据。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：固定 2405.04434v5 的 §2.1 与附录 C 支持联合 KV latent、内容 key/value 投影吸收、共享位置 key 与 per-head 位置 query、基础 scale 和 (d_c+d_R)L 缓存元素。§3.1.2 支持 latent 后 norm/缩放边界，§3.1.4 说明其上下文扩展；附录 B.1 说明 Lite 不压缩 query。d_c=4d_h、d_R=d_h/2 是该文配置。93.3% 为其与 DeepSeek 67B 的部署比较，部署还包含低精度与缓存量化，不能只归因于本章结构比例。
  [完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：使用提供的固定提交机制说明，支持准备/执行分段、三类 metadata 单位、布局转换可能复制、workspace 和图缓冲生命期。本轮未提供可允许阅读的原文或源码，未自行核验其代码，也不据此宣称某种 MLA 组合已被该接口支持。
  [算法与机制说明](../sources/S-FI.md)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：所给完整数值说明支持实数结合律不能推出浮点逐位一致、批量/切片和平台差异，以及低精度中间归约与极端值边界。本文据此要求分清存储、投影、分数、概率和累加路径；原文一个切片求和例子的轴对应不能作为数学恒等式照搬。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

### 本次扩写补充来源

- [S-DEEPSEEKV3：DeepSeek-V3 Technical Report](../sources/S-DEEPSEEKV3.md)：固定 2412.19437v1 的 §2.1.1 给出 MLA 公式，§4.2 明确实际 latent 后 RMSNorm 和宽度缩放，§4.3 给出位置扩展条件。§3.4 的 P/D 分离、attention TP 与 MoE EP 是该部署的策略；本章四卡状态与通信算例为独立设计，没有采用论文集群吞吐作为本章结果。
  [完整原文](../../references/S-DEEPSEEKV3/reader.html) · [全文文本](../../references/S-DEEPSEEKV3/fulltext.txt)

- [S-KIVI：KIVI](../sources/S-KIVI.md)：固定 2402.02750v1 的 §3 与附录 A 支持分组量化、高精度残余和流式封组；K 的 per-channel 参数固定通道、跨 token 分组，V 的 per-token 参数固定 token、沿通道分组，不能照搬摘要含混的轴措辞。§4 的模型/任务与残余长度条件，以及 prefill 用高精度状态、后续 decode 才消费压缩状态的区别，限制了质量结论；未证明 MLA latent 的最佳轴或可用位宽。
  [完整原文](../../references/S-KIVI/reader.html) · [全文文本](../../references/S-KIVI/fulltext.txt)

正文中的公式展开、空状态算法、线程/地址映射、成本交点、TP 分解与阶段交接为本库独立推导；设定算例不冒充实测。当前正文、方法、审查标准和五份来源说明按提供的全文阅读；S-MLA、S-DEEPSEEKV3 原文及相关标为 excerpt 的条目只按本轮所提供字符范围使用，不声称读过其余原文或 PDF 版面。S-TORCH-NUMERIC、S-KIVI 为所给完整提取文本。来源事实受 2026-04-30 上限约束，编辑日期不作为事实发布日期；原章节身份与版本映射仍由既有版本清单管理。

## 延伸阅读

- [MHA、GQA 与 MQA：头映射和带宽复用](gqa-kv-heads.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [KV 压缩：表示、近似保留与传输格式](../algorithms/kv-compression.md)。
- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [分布式状态迁移：身份、传输与提交点](../system-semantics/distributed-state-transfer.md)。
