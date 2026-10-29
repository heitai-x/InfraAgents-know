# K-ALG-015：推理并行：张量分解、流水、专家与上下文

并行策略改变张量归属、同步边界和状态驻留。选择并行度之前，应先回答：哪部分计算可以独立完成，完整输出怎样重建，每个消费者等待什么，以及节省的计算和容量是否抵得过新增通信、准备与同步。

## 并行计划的对象与合法性

设层输入 H 为 T×d，T 是本轮有效 token 行数，d 是 hidden 维。并行计划 π 将请求、层区间、输入或输出通道、query/KV head、专家和上下文位置映射到 rank；rank 再映射到实际设备。TP、PP、DP、EP、CP 分别描述这些轴，但不同层可以复用同一设备集合。attention TP4 与 MoE EP4 可以在同四卡上依次执行，不能只把两个度数相乘便断言需要十六卡。

输入包含固定模型参数、请求 token、位置和 mask、已有状态及并行计划；输出是下一层 hidden、logits 或确认的生成 token。执行状态还包括权重分片、KV/MLA latent、微批激活、路由描述符、softmax 统计、通信区、工作区和未完成访问。每个对象应能对应模型版本、请求、层、逻辑位置、head/专家、表示和布局；有重试或地址复用时再绑定相应尝试代次。

合法性先于性能选择：

- 分片必须完整覆盖对应逻辑轴；除明确声明的副本外不得重复计算后再次累计贡献。
- 行并行的部分和需要归约，列并行的分片需要按全局通道顺序交给消费者；padding 不进入有效词表或 softmax 分母。
- 集合操作的成员、数量、datatype、归约方式和逻辑顺序必须匹配。本地没有有效 token 不授权某个成员跳过全组已经约定的操作。
- GQA 保留全局 query→KV 映射，MLA 保留实际 latent/位置表示，CP 保留全局位置与合法 key 集合。
- 消费者取得的是已经完成生产、布局相容且可见的数据。host 调用返回不能代替设备完成或消费者最后使用证明。

这些是函数和执行依赖要求。选择 ring 或树、均匀或不均匀分片、连续或交错层段、专家副本以及通信重叠方式，则是带前提的设计选择。实数中的等价重排也不保证浮点逐位一致。

## 张量并行：从矩阵分解恢复输出

对 Y=XW，X∈R^(T×d_in)，W∈R^(d_in×d_out)。列并行沿输出通道分割 W=[W_0,…,W_(p−1)]，rank r 得到 Y_r=XW_r。完整输出由按原列顺序拼接 Y_r 得到。行并行沿输入通道分割 X 和 W，rank r 得到 P_r=X_rW_r，完整输出为 Y=Σ_r P_r。

列并行后若接逐元素激活，再接对应的行并行层，中间分片可以直接消费。例如 Z_r=φ(XW_up,r)，Y=Σ_r Z_rW_down,r。这个性质要求 φ 在所切通道上独立；跨通道归一化或其他耦合运算不能未经推导直接留在局部分片上。

```pseudocode
算法 TP_LINEAR
输入：X、W、p 个合法且覆盖完整的通道区间 I_r 或 J_r
输出：列模式的 Y_r，或行模式的完整 Y
前提：本算法先描述无 bias 线性层

COLUMN：
    对每个 rank r：
        W_r ← W[:,J_r]
        Y_r ← X W_r
    若下一消费者接受 J_r 分片，直接交付 Y_r
    否则按全局 J_r 次序 gather 并恢复 Y

ROW：
    对每个 rank r：
        X_r ← X[:,I_r]
        W_r ← W[I_r,:]
        P_r ← X_r W_r
    在同一 TP 组对对应元素执行 SUM 归约
    按消费者需要保留完整 Y 或归约后的输出分片
```

输出 bias 在列模式按输出列切分；行模式的完整 bias 通常在求和后加入一次。residual 同样不能先由每个 rank 加入完整值，再将这些副本一起求和，否则得到 p 倍 residual。

Transformer 的一种常见组织是 QKV 和 FFN 上投影列并行，attention 输出投影和 FFN 下投影行并行。head 分配须与后续输出投影的全局列顺序相容。GQA 的 KV heads 少于 TP 或不能整除时，可能复制部分 KV、显式传映射或拒绝配置。MLA 按 query heads 分工而复制完整 latent 时，每 rank 的缓存不会按 TP 缩小。

设激活、权重元素分别占 b_a、b_w 字节。均匀分片的线性层每 rank 算术主项为 2T d_in d_out/p FLOPs，权重驻留为 d_in d_out b_w/p；这不包含 bias、激活、归一化和 padding。冷权重至少读取本地分片一次是逻辑读取账，不保证等于实际 HBM 事务。列模式产生 T d_out/p 个输出，行模式产生 T d_out 个部分和；输入是否复制、是否另需 gather，以及部分和能否原地归约，决定激活和通信区峰值。

令完整 hidden 为 A=T d b_a 字节。理想均匀 ring all-reduce 的 reduce-scatter 和 all-gather 各有 p−1 轮，每轮每 rank 发送 A/p 字节，因此一次每 rank 发送 2(p−1)A/p。接收是另一计量边界，不能在“发送量”中再加一次。两次归约合计每 rank 4(p−1)A/p。p=4、d=4096、b_a=2：T=1 时两次合计 24 KiB；T=2048 时合计 48 MiB。

若每轮启动成本为 α、单向有效带宽为 β，且收发全双工、无额外争用，则一次时间模型为 2(p−1)α+2(p−1)A/(pβ)。该模型不表示 NCCL 必定选择 ring。packing、布局转换、慢 rank、暂存和共享 HBM/链路另计。小 batch decode 容易暴露启动与同步；prefill 更可能暴露字节和带宽。p 增大时发送量趋近 2A，并不会与计算一样按 p 下降。

非整除维度可使用记录完整区间的不均匀分片、显式 padding 或更换并行度。均匀 count 的集合接口不能直接承接任意不均匀块，应补齐并屏蔽或使用相容的传输方案。容量与层时间取决于最大分片及慢 rank，不能只使用平均值。

## 流水与请求副本：不同的扩展目标

PP 把层区间分给 P 个阶段，每个微批依次经过全部阶段。阶段持有本层权重和本层请求状态；边界传递 hidden、请求/位置元数据及必要布局描述。下阶段接收完成后才能计算；源发送区和目标接收区分别按其最后使用结算。

以下只讨论前向推理流水，不使用训练 1F1B 的前反向调度公式。令 M 个独立微批按同一顺序通过各阶段，阶段 s 的时间为 c_s，忽略通信及资源争用：

```pseudocode
算法 FORWARD_PIPELINE
输入：P≥1，M≥0，阶段时间 c_s>0
输出：完成时间 F 及每个阶段、微批的完成表

若 M=0，返回 F=0 和空表
对 s=0..P−1：
    对 m=0..M−1：
        upstream ← 0，若 s>0 则为 finish[s−1,m]
        previous ← 0，若 m>0 则为 finish[s,m−1]
        start[s,m] ← max(upstream,previous)
        finish[s,m] ← start[s,m]+c_s
返回 finish[P−1,M−1]
```

等成本 c 时 F=(M+P−1)c。P 个阶段总可用时间为 PF，有效忙碌为 PMc，因此总体空闲比例为 (P−1)/(M+P−1)。它不是单请求节省的延迟。M=1 时仍需 Pc；一个自回归请求必须等上一 token 的最终结果后才能推进下一 token，不能把这些有依赖的 token 当作 M 个独立微批。

固定 c_s、无限缓冲且无通信时，完成时间为 Σ_s c_s+(M−1)max_s c_s；例如 [1,3,2]、M=4 得到 15。实际通信应接入产生、发送、接收和消费者依赖，不能把本已重叠的所有耗时再直接相加。增加微批可能降低气泡，却增加在途激活、接收槽与等待时间。阶段切分应按 attention、专家和通信成本，而非只均分层数。

DP 复制完整服务实例，或者复制一套内部已经使用 TP/PP 的执行组，并把不同请求交给不同副本。请求一经取得状态，后续迭代保持其状态所有者；迁移需要显式状态交接，不能仅修改路由指针。推理 DP 不要求每 token 跨副本求和，其代价是权重、工作区和部分前缀缓存复制。

若一个副本需要 W 字节权重，n 个独立副本的权重主项是 nW；共享同一物理只读页时才按实际共享计一次。理想独立资源下吞吐可随副本数增加，但共享内存带宽、NIC、存储和前缀亲和队列会破坏线性增长。模型能在更小执行组内容纳、请求数足够时，DP 应与继续增加单请求 TP 比较；低负载时，额外副本可能只增加驻留与准备成本。

## 专家并行：一次路由的完整覆盖

MoE 对 token x 选择专家集合 S_x，并计算 y_x=Σ_(e∈S_x) g_x,e E_e(x)，共享专家和 residual 按模型另加。Mixtral 的固定公式为 G(x)=Softmax(TopK(xW_g))，top-2 未选项为零。DeepSeek-V3 v1 使用 sigmoid affinity 和带 bias 的选择，但最终 gating 由选中项的原始 affinity 归一化；选择分数与加权数值不能混用。

EP 将专家函数放在不同 rank。item 的身份至少为 (origin,token,expert)，其中 token 是 origin 内的标识；两个 origin 都有 token 0，不能因此合并。模型层或尝试并发时还需加入相应身份。副本只改变同一专家函数的执行位置，不允许把一次逻辑专家贡献计算并累计多次。

```pseudocode
算法 EP_FORWARD
输入：各 origin 的 token、固定 router 契约、专家放置快照
输出：各 origin 的完整 MoE 输出

1 对每个 token 计算选择分数，按模型规则及确定的 tie 规则取得 S
2 用模型规定的原始分数或 affinity 得到 gating；固定本轮路由
3 为每个 (origin,token,expert) 选择一个合法执行副本并生成描述符
4 按目的 rank、专家计数，计算 offsets，交换必要接收计数
5 预留接收 hidden、描述符、专家激活和返回空间
   容量不足时分块或失败；不得删掉溢出 item
6 dispatch hidden 和描述符；本地项可引用本地 hidden
   同 token 同目的可共享一份 hidden，但保留所有专家描述符
7 接收数据可见且权重就绪后，执行每个非空专家桶
8 将每个专家输出乘对应 gating 一次
   可逐 item 返回，或按 (origin,token,目的 rank) 预合并
   预合并记录携带它覆盖的全部 expert 身份
9 origin 初始化 Seen=空、输出=0
10 对每个返回记录：
       验证身份与本轮路由一致
       要求 covered⊆S 且 covered∩Seen=空
       加入已加权向量，Seen←Seen∪covered
11 只有 Seen=S 才提交该 token 输出
   缺项、重复或错误身份均失败；未完成物理访问继续受保护
```

空专家桶无需执行 GEMM，但参与全组约定的计数/通信成员仍按计划推进。目的端预合并会改变浮点累加顺序，需要既定误差契约；不能仅凭覆盖完整便断言数值逐位相同。

设 hidden 维 d、专家中间维 f，三个 d×f 量级矩阵的门控专家，每 item 算术主项为 6df FLOPs，权重为约 3df b_w 字节。routed 总算术为 6df N_items，N_items=Σ_x|S_x|。shared 专家、router、激活和 padding 另计。稀疏激活不等于专家权重按 k/E 常驻：batch 可能触及大部分专家，副本还会增加驻留和分别读取的权重。

令 N_remote 为目的 rank 不等于 origin 的 item 数，D_remote 为远端 (origin,token,目的 rank) 对数。每个 hidden 和返回向量分别占 d b_h、d b_y 字节：

| 通信组织 | dispatch 发送主项 | 返回发送主项 |
|---|---:|---:|
| 逐 item 输入、逐 item 返回 | N_remote d b_h | N_remote d b_y |
| hidden 去重、逐 item 返回 | D_remote d b_h | N_remote d b_y |
| 逐 item 输入、完整目的端预合并 | N_remote d b_h | D_remote d b_y |
| 去重并完整预合并 | D_remote d b_h | D_remote d b_y |

这些是全 EP 组端点发送 payload，每次传输计一次；本地项不进入网络账。分 chunk 多次返回时，应以实际返回条数代替 D_remote。计数、索引、padding、重传、转发边及下一 TP 布局转换另计。

准备包含计数、前缀和、重排与描述符，计数分桶可接近 O(N_items+E)，比较排序则可达 O(N_items log N_items)。中间激活随每个专家实际桶长增长，hidden 去重不会让专家中间量消失。热点 rank 的桶长、grouped GEMM 效率、通信入口和返回等待共同决定慢端。冗余副本可分担热点，却也会拆小组长、增加冷权重读取和迁移准备，因此均衡不自动证明层更快。

DeepSeek-V3 v1 的 prefill 32 GPU 与 decode 320 GPU 是固定 H800 部署策略；prefill 双微批重叠是报告的方案，动态冗余及部分 decode 重叠仍属探索。训练 DualPipe 的前反向重叠不构成这里的推理时间证明。

## 上下文并行：分片必须先构成合法划分

CP 按上下文位置切分 K/V。固定一个 query、请求、层和 head，令其合法 key 集合为 V_q，分数 s_j=q·k_j/√d_k，value 维为 d_v。完整输出为 Σ_(j∈V_q) exp(s_j)v_j / Σ_(j∈V_q) exp(s_j)。这里以有限有效分数为前提；NaN、正无穷和完全空行需要另按模型/后端契约定义。

分片 G_r 必须满足并集完整且两两不交。可以先划分物理 key 域，再对每片使用同一全局 mask；也可以直接划分 V_q，但必须证明它的完整覆盖。空分片合法，漏 key 或重复 key 不合法。在线 softmax 合并只对已有贡献做代数重组，不会自动发现或补回漏项，也不会自动消除重复项。

对非空局部合法集合，定义

m_r=max s_j，l_r=Σ exp(s_j−m_r)，a_r=Σ exp(s_j−m_r)v_j。

a_r 是 d_v 维未归一化向量。合并两个非空统计时，令 m=max(m_1,m_2)，c_i=exp(m_i−m)，得到 l=c_1l_1+c_2l_2、a=c_1a_1+c_2a_2。最终输出 a/l。这来自 exp(s_j)=exp(m_r)exp(s_j−m_r)，共同缩放不改变最终比值。

```pseudocode
算法 CP_LOCAL_AND_MERGE
输入：query q、全局 key 域 J、分片 G_r、K/V、全局可见规则
输出：该 query 的完整 attention 输出

先验证 G_r 在声明域 J 上完整且互斥
对每个分片 r：
    m_r←−∞；l_r←0；a_r←零 d_v 向量
    对 j∈G_r：
        若全局位置 j 对 q 不可见，跳过
        s←q·k_j/√d_k
        若 s>m_r：
            c←0，若 l_r>0 则 c←exp(m_r−s)
            l_r←c·l_r；a_r←c·a_r；m_r←s
        w←exp(s−m_r)
        l_r←l_r+w；a_r←a_r+w·v_j

累计统计 S←EMPTY，EMPTY 表示 (−∞,0,零向量)
对每个分片统计 R：
    若 R.l=0，跳过
    若 S.l=0，S←R，继续
    m←max(S.m,R.m)
    c_S←exp(S.m−m)；c_R←exp(R.m−m)
    a←c_S·S.a+c_R·R.a
    l←c_S·S.l+c_R·R.l
    S←(m,l,a)
若 S.l=0，报告空可见集合并按已声明契约处理
否则返回 S.a/S.l
```

显式处理 EMPTY 避免计算 exp(−∞−(−∞))。它是合并单位元，不是应参与局部输出平均的零向量。对局部归一化输出 o_r=a_r/l_r，正确权重与 exp(m_r)l_r 成正比；直接平均 o_r，即使每片长度相同也通常错误。

本轮失败中的两组输入分别为 [[0,1],[2,3],[]] 和 [[0,2,4],[1,4],[3]]，声明 key 域为 0..4。前者漏 4，后者重复 4。它们不应作为“合并必须等于完整 attention”的正常分片，而应在覆盖检查中拒绝。扩大浮点容差不能修复这个错误。

causal 可见性使用全局位置 j≤q_pos；长度为 w 的窗口还要求 j>q_pos−w。局部重新从零编号会改变函数。图像/视频全局 attention 则不能擅自添加 causal mask；分片位置仍来自完整网格。分片顺序变化在实数中等价，在浮点中按声明容差比较。

### 两种通信方案的独立账本

一种完整候选是 stationary Q 的 K/V 环流：每 rank 保留自己的 Q 和输出累计统计，先处理本地 K/V，再进行 p−1 轮。第 k 轮将当前块发给下一个 rank、接收前一个 rank 的下一块，核对块身份和全局范围，完成接收后计算并合并；当前块的最后读取和发送都结束后才能复用其缓冲。每个 query 恰好处理全部 p 个 key 块一次。

设全组 K+V 字节为 S、均匀分片、每 rank 的 query 数为 Q/p，则每 rank 发送及接收分别为 (p−1)S/p。无争用的传输模型为 (p−1)α+(p−1)S/(pβ)。局部 K/V 持久存储为 S/p；保守的当前/下一块临时区可另预留最多 2S/p，不能把整个累计接收量视为同时驻留。具体原地别名可减少空间，但须验证最后使用关系。还需本地 Q、每 query 的 m/l/a 和 score tile，不要求物化完整 score 矩阵。

稠密计算主项每 rank 约为 2(Q/p)N(d_k+d_v) FLOPs，N 是 key 数；局部在线统计和跨块 merge 另计。causal 连续切片可能产生不同有效 query-key 对数，真实算术和慢端应按可见关系重算。K/V 被多个 query tile 重读、缓存命中和传输暂存读写决定实际 HBM 事务，逻辑 S 不能代替它们。

另一候选让 key 保持在所有者，先把所需 Q 分发到各 key 分片，各 rank 产生局部统计。对同一批 query，先全组求 m 的 MAX，再将局部 l/a 乘 exp(m_r−m)，对缩放结果求 SUM，最终在 query 所有者归一化。此方案搬 Q 和统计而非环流全部 K/V，但可能复制 query 计算及统计，并增加两段归约依赖；全空行必须单独分支。Q 分布、d_v 和归约实现决定收益，不能把该方案的低 K/V 流量与上一方案的局部 Q/统计容量混算。

## 四卡 MLA/MoE：归属、布局转换与容量一起展开

以下均为设计设定，不是 DeepSeek 型号规格或实测。任务是 16 个并发生成请求，prompt 2048–8192 token，输出预留 128 token，客户端 ITL 目标 40 ms；固定 MLA/MoE checkpoint 和 BF16 质量参考。四卡各有 24 GiB 供本任务使用，权重合计 48 GiB，其中 dense/attention 12 GiB、专家 36 GiB。32 层，每 token 每层缓存 latent 512 与 RoPE key 64 个 BF16 元素。

全层缓存为 32×(512+64)×2=36 KiB/token。最长 prompt 每请求为 288 MiB，16 请求为 4.5 GiB。三个候选明确复用同四卡：

| 候选 | rank 与层/参数归属 | 每 rank 的 prompt 缓存 |
|---|---|---:|
| A：全层 TP4 | ranks 0..3 各持全部层的四分之一权重；各自复制所有请求 latent | 4.5 GiB |
| B：attention TP4，MoE EP4 | dense 各 3 GiB；专家分八组，rank r 持 e=r 与 e=r+4，共 9 GiB；latent 仍复制 | 4.5 GiB |
| C：PP2×TP2 | ranks 0/1 持层 0..15，ranks 2/3 持层 16..31；各阶段内部 TP2 | 2.25 GiB |

这些专家组表示本设定跨层对应的专家参数集合，每组 4.5 GiB。C 还假定两段权重各 24 GiB，所以每 rank 权重也是 12 GiB；实际层大小不均时必须重分配。

另设每 rank 工作区/通信峰值预算 2 GiB，则 A/B 的 prompt 主项为 18.5 GiB，C 为 16.25 GiB。128 token 增长预留每请求 4.5 MiB，A/B 每 rank 再增 72 MiB，主项为 18.5703125 GiB；C 增 36 MiB，主项为 16.28515625 GiB。页尾、运行时、加载转换和其他并发生命期仍需另计，这不是完整峰值保证。

B 的一层 decode 轨迹可以具体恢复如下。rank 是 communicator 身份；例如 rank→device 为 [2,0,3,1] 时，root=0 指设备 2，不能把 root 当设备号。

1. attention 的输出投影归约完成后，各 rank 持相同 16×4096 hidden。令 token t 的唯一派发所有者为 t mod 4；其 residual 也在该所有者保留。其他副本不能再次 dispatch。
2. 为构造可复算路由，t<8 的两个专家目的为 origin 和 (origin+1) mod 4；t≥8 的目的为 (origin+1) mod 4 和 (origin+2) mod 4。这里使用 e=0..3，e=4..7 桶为空。这是固定 top-2 路由设定，实际 gating 仍来自模型。
3. 总计 32 item，其中 8 本地、24 远端；每 rank 发出 6 个远端 item。BF16 hidden 每项为 8 KiB，逐项 dispatch 发送总计 192 KiB，逐项返回同样为 192 KiB。接收端依照描述符执行专家，并返回到原 origin；每个 token 等待两个贡献完整后 combine，再加 residual 一次。
4. combine 后每 rank 只拥有四个完整输出行。下一 attention 若要求每 rank 复制全部 16 行，需要布局恢复。选择均匀 ring AllGather：每 rank 输入四行共 32 KiB，每 rank 发送 3×32=96 KiB，全组发送 384 KiB。
5. AllGather 输出按 rank 排列，即 [0,4,8,12]、[1,5,9,13] 等。消费者须按这张映射读取或排列回 t=0..15；不能直接当作原 token 次序。完成后，各 rank 再以同一请求/位置更新自己的 attention 状态。

因此 B 的 MoE dispatch/return 为 384 KiB，输出恢复又为 384 KiB，边界合计 768 KiB，尚有 metadata、准备和内存重排。若一个 block 恰有一次 attention all-reduce，A/B 的该项均为全组 768 KiB；A 的另一次 FFN all-reduce 也是 768 KiB。这个具体设定下，两候选的主要发送总量可以相同，差别仍在启动、专家矩阵形状、同步和偏斜，不能由“EP”名称推断一定省通信。

C 的阶段边界也需要明确组织：可由 rank 0 发送一份完整微批 hidden 给 rank 2，再在阶段 1 内广播给 rank 3，而非默认两份复制 hidden 都跨边界。阶段 1 必须等待自己的接收/广播完成，最终采样由唯一所有者确认并把下一 token 送回需要它的阶段。下一 token 的输入依赖不能靠增加微批数消除。

若为热点专家新增每卡 2 GiB 副本，B 的主项成为 20.5703125 GiB，同时重算每个副本桶长、迁移准备和在途峰值。布局切换先准备新权重与目录，旧轮保持原放置快照；旧访问结束后才释放旧副本。发生取消时停止新提交，已经开始的通信和计算仍需结算；某个 rank 失败后，部分输出不能作为完整层结果发布。

时间比较使用相同请求轨迹和质量门槛。某层计算从 8 降至 3 单位、新增不可隐藏通信 4，则总计 7；再分卡只省 1 却增加 2，选择便反转。这些单位是设定解释。40 ms 是目标，容量推导、理想带宽和一次局部层比较都不证明它已经达到。

## 条件变化与区分实验

| 条件或观察 | 竞争解释与可控实验 | 调整依据 |
|---|---|---|
| TP 增加但层不加速 | 固定 shape 扫 p，分别记录本地 GEMM、集合启动、实际字节及最慢 rank；可能是计算不足、启动或不均匀分片 | 计算下降而关键路径不降时，减少同步深度或改变组；不按卡数外推 |
| 请求少或进入小 batch decode | 比较较小执行组的请求副本与较大 TP，保持总资源和输入一致 | 若启动主导且模型可容纳，DP 可能优于继续扩大 TP；副本驻留与队列另算 |
| PP 利用率低 | 独立改变 M 和阶段切分，观察填充/排空、慢阶段、接收等待与活跃激活 | M 增大仍不改善时，修正阶段或依赖；容量接近上限时减少在途量 |
| EP 热点或层时延增加 | 固定 S/g 改同函数副本放置，分别记录桶长、权重读取、dispatch、专家和 combine | 队列缩短而层不降说明还需检查入口拥塞或小 GEMM，不能只追求均衡 |
| CP 结果随分片变化 | 先查完整互斥覆盖，再对照完整 softmax；分别注入漏项、重复项、空片、mask 位置重置和错误 merge | 覆盖错误先修计划；覆盖完整仍错再定位分数、mask、缩放与布局，不扩大容差掩盖 |
| 长上下文使复制缓存接近容量 | 比较 PP 层分散、CP token 分散及更小副本，分别算持久状态、交换与工作区 | 状态下降可能成为准入条件，但 CP 交换或 PP 依赖可能损害延迟 |
| dtype、量化或窗口改变 | 固定逻辑函数与质量门槛，重算实际字节、scale、合法可见集合和转换 | 不能沿用旧表示或全上下文的流量与误差结论 |

典型反例包括：行分片只返回一个部分和；每个 rank 加 residual 后再 SUM；AllGather 后把 rank 顺序误认为 token 顺序；EP 只按本地 token 编号合并不同 origin；CP 对各片输出取平均；PP 只降低理想气泡却漏计在途激活。它们可能保住张量形状，因此需要值、身份、覆盖与生命周期的独立检查。

真实多卡性能实验应扫描消息大小、并行度、微批、路由和上下文，固定 rank→device、拓扑、模型制品与质量标准，报告 TTFT/ITL、完成数、失败、慢 rank 和暂存峰值。重叠是否有效以关键路径缩短为准；两条时间线相交不证明共享 HBM 或链路可以同时全速工作。

## 检查范围与未测边界

独立检查脚本以整数矩阵核对 TP 拼接/归约，以消息模拟核对 ring 值与发送账，以前向完成递推核对 PP 的闭式不变量。CP 使用独立 Decimal 完整 softmax 参考，穷举小域分配并覆盖 causal、窗口、空片、全空可见集合与极端分数；两组原坏分片作为必须拒绝的负对照，同时证明绕过覆盖检查确会改变结果。EP 使用 Fraction 逐 token/专家参考，检查不同 origin、去重/预合并、缺项和重复项。四卡场景再核对容量、24 个远端 item 及 AllGather 的行身份。

上述是 CPU 算法模型，不能证明真实 NCCL 算法选择、CUDA 可见性、跨 stream 并发、故障排空、BF16/量化误差、真实 HBM 事务或 GPU 性能。来源和检查也不构成知识改善 Agent 构建能力的使用效果证明。

检查脚本、输入绑定与实际执行结果见[检查记录](../../docs/evidence/knowledge-workflow-20261028-batch6/K-ALG-015/round-2/check-0.json)。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。给定固定原文还说明相同 count/datatype、root 为 rank、AllGather 按 rank 排列、ReduceScatter 按 rank 分块；这些支持正文的归约和布局约束，不提供 ring 性能保证。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。本轮依据给定来源说明使用这些边界；提供的 CUDA 原文仅为摘录，不据此声称完整阅读多 GPU 章节或核验某条目标设备路径。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-MIXTRAL：Mixtral of Experts](../sources/S-MIXTRAL.md)：Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。给定全文 §2.1 支持 top-2 gating、SwiGLU 专家、dispatch 后返回原 token 位置及 EP 负载均衡挑战；它不证明本章四卡设定的速度。
  [完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

### 本次扩写补充来源

- [S-DEEPSEEKV3：DeepSeek-V3 Technical Report](../sources/S-DEEPSEEKV3.md)：DeepSeek-V3 v1 的 MLA、细粒度/共享专家、路由选择与 gating 区分、MTP，以及分阶段推理部署中的专家复制、微批重叠和在线偏斜；只使用该固定版本。给定摘录 §2.1.2 与 §3.4 支持 bias 只改变选择、原 affinity 决定 gating、32/320 卡部署及报告方案与探索策略的区别。训练成本、DualPipe 和 FP8 训练结论不转为本章推理性能。
  [完整原文](../../references/S-DEEPSEEKV3/reader.html) · [全文文本](../../references/S-DEEPSEEKV3/fulltext.txt)

CP 覆盖条件、在线合并代数、前向 PP 递推、ring 消息账、六类资源成本及四卡轨迹均为本库推导；没有将其归称为以上论文的完整实现。相关章节仅按本轮提供的全文或明确摘录范围用于衔接，未读取源码归档。

## 延伸阅读

- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [MoE：路由、Dispatch/Combine 与专家负载](../models/moe-routing.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](../models/mla.md)。
- [MHA、GQA 与 MQA：头映射和带宽复用](../models/gqa-kv-heads.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
- [集群放置：资源向量、拓扑、执行组与故障域](../workloads/cluster-resource-placement.md)。
