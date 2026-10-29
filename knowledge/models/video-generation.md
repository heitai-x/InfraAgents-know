# K-MODEL-010：视频生成：时空 Token、全局耦合与分块解码

视频生成在每个求解步更新整段视频的 latent。时间压缩决定实际状态大小，patch 化决定 Transformer 序列长度，全局 attention 决定跨时空依赖，VAE 和媒体交付决定最终哪些帧已经可消费。计算分块、状态分片和独立分段生成改变的是不同对象，必须分别判断函数、容量和质量。

本章以全序列 latent 视频生成为主，展开一个明确的教学用时空 Transformer、flow Euler 推进、精确分片 attention 和有状态局部解码参考。参考结构用于恢复机制，不声称复现某个 Wan checkpoint 的全部层序、位置编码或正式采样器。Wan 报告提供的结构与边界事实单独注明；DDIM 只用于说明预测和 scheduler 必须匹配。

## 请求对象、时间几何与预测契约

输入包括文本或图像等条件、目标帧数 F、像素高度 H 与宽度 W、播放帧率 f、checkpoint、求解时间表、guidance 和初始随机轨迹。输出是具有明确帧序、时间戳、色彩格式及终止语义的视频。帧率单位为 frame/s，求解时间是生成过程的参数，二者不是同一时间轴。

令实际 latent 为 z∈R^(T_l×H_l×W_l×c_z)，patch 为 (p_t,p_h,p_w)。整除时，N_t=T_l/p_t、N_h=H_l/p_h、N_w=W_l/p_w，token 数 N=N_tN_hN_w。每个 token 的原始 patch 向量有 v_p c_z 个元素，其中 v_p=p_tp_hp_w；投影后的 hidden 宽度 d 可以远大于它。latent 字节 T_lH_lW_lc_zb_z 与 hidden 字节 Ndb_a 必须分开计算。

非整除输入需要明确拒绝、裁剪或 padding。padding 后的 token 是否进入 attention、输出如何裁回、模型是否支持这种边界，均属于输入契约。不能把任意补零视为对原 checkpoint 的函数等价变换。

Wan 固定报告 §4.1 将 F=1+4q 帧映为 T_l=1+q 个时间 latent，空间每边压缩 8 倍、latent 通道为 16；首帧只做空间压缩。§4.2.1 再用 (1,2,2) patch。因此对报告支持的整除尺寸，N=(1+(F−1)/4)(H/16)(W/16)。81 帧、720×1280 对应 21×90×160×16 latent，以及 21×45×80=75600 个 token。F=1 时应得到一个时间 latent；80 帧不满足这里的精确公式，不能直接取整并宣称合法。

帧索引、latent 时间索引、patch 时间坐标与求解步索引分别保留。flatten 顺序可定义为 r=(aN_h+i)N_w+j，对应全局 patch 坐标 (a,i,j)。分片后仍使用这一全局身份。

| 状态 | 含义与消费者 | 更新或释放条件 |
|---|---|---|
| 当前 z_k、时间表位置 k | 本请求在第 k 个求解边界的完整逻辑状态，可分布在多卡 | 全部必要预测和 solver 更新成功后提交 z_(k+1) |
| 条件 E 与可复用投影 | 文本、图像、mask、processor、权重及表示版本共同决定 | 仅依赖固定的子图允许跨步复用 |
| self Q/K/V 与层激活 | 当前 z_k、求解时间、条件、层和 CFG 分支的函数 | 本次前向的全部读者结束后回收；通常不能跨步当作固定 KV |
| RNG 与 solver 历史 | 初始噪声、随机更新或多步求解器要求的请求私有状态 | 与 latent 和 k 一起提交，失败重试保持同一轨迹 |
| VAE 逐层 cache | 某请求最终 latent 的解码历史、层、有效范围和相位 | 顺序推进；不能混入其他请求或预览的 cache |
| 像素与编码缓冲 | 解码器、转换器、编码器和发送方之间的在途数据 | 所有实际消费者结束后复用 |

Wan §4.2.2 使用 flow matching：以 ξ 表示噪声、y 表示数据 latent，z(t)=(1−t)ξ+ty，目标速度 u*=y−ξ，t∈[0,1]。这里 t 从噪声端向数据端增加。若实际 pipeline 使用反向 sigma、缩放后的网络输入或另一种速度符号，必须先换算到其正式契约。

这与 DDIM 的 x_t=√a_t x_0+√(1−a_t)ε 不同。S-DIFFUSERS 的 epsilon、x_0、v 转换与累计 alpha 更新不能直接套在 Wan 的 flow 输出上。相同 shape 不能证明参数化相同；flow velocity 与 Gaussian diffusion 的 v 也不是同一个定义。

## 从时空 patch 到一次完整状态推进

下面给出可重建的教学结构。所有 token 矩阵以 token 为行、通道为列。输入投影 W_in∈R^((v_pc_z)×d)，输出投影 W_out∈R^(d×(v_pc_z))。参考输出仅包含与 latent 同形的速度；真实模型若输出额外方差、条件通道或使用不同 patch，必须按 checkpoint 适配。

设 head 数为 h，d=hd_h；条件 E∈R^(S×d_e)。self Q/K/V/O 投影均为 d×d；cross Q/O 为 d×d，cross K/V 为 d_e×d；普通两层 FFN 的权重为 d×d_ff 和 d_ff×d。这里为简洁省略投影 bias，真实模型的 bias、归一化与激活不能因此删去。

LN 对每行 d 个通道求均值与方差，并使用固定正稳定项 ε_n。参考 GELU(x)=x(1+erf(x/√2))/2。给定固定频率 ω_j，时间嵌入 e(t) 由 sin(ω_jt)、cos(ω_jt) 组成；第 l 层以 SiLU(e(t))A_l+b_l 产生六个长度 d 的调制向量。位置表 P[a,i,j,:] 是参考模型的固定输入，实际 checkpoint 可使用自己的三轴位置机制。

```pseudocode
算法 VideoPredict：教学用时空 Transformer 的一次完整前向
输入：z[T_l,H_l,W_l,c_z]，t，E[S,d_e]，固定参数与位置表 P
输出：速度 u，与 z 同形
前提：patch 三轴整除；d=h·d_h；每条 attention 行有合法 key

for 每个全局 patch 坐标 (a,i,j)：
    r ← (a·N_h+i)·N_w+j
    v ← 按 patch 内时间、行、列、通道顺序展平对应 z 区域
    X[r,:] ← v·W_in + P[a,i,j,:]
for l ← 1 到 L：
    γ1,β1,g1,γ2,β2,g2 ← SiLU(e(t))·A_l+b_l
    U ← (1+γ1)⊙LN(X)+β1
    Q,K,V ← SplitHeads(U·W_Q), SplitHeads(U·W_K), SplitHeads(U·W_V)
    对每个 head 和 query 行 q：
        A[q] ← 对完整合法 key 集合计算 Softmax(Q[q]Kᵀ/√d_h)·V
                可用下一节的精确分块算法替代显式 scores
    X ← X + g1⊙(ConcatHeads(A)·W_O)
    U ← LN(X)
    Qc ← SplitHeads(U·W_Qc)
    Kc,Vc ← SplitHeads(E·W_Kc), SplitHeads(E·W_Vc)
    对每个 head 和 query 行 q：
        C[q] ← Softmax(Qc[q]Kcᵀ/√d_h)·Vc，使用条件的合法 mask
    X ← X + ConcatHeads(C)·W_Oc
    U ← (1+γ2)⊙LN(X)+β2
    X ← X + g2⊙(GELU(U·W_1)·W_2)
Y ← LN(X)·W_out
按原 patch 坐标及 patch 内顺序还原 Y，得到 u
返回 u
```

该算法明确了各量的生产者和残差顺序，不把一个未解释的 DiT 调用作为全章算法。它不是 Wan 结构的逐项声明：Wan 报告采用共享时间调制 MLP 与逐 block bias，真实归一化、位置和激活仍须按实际模型核对。固定文本 cross K/V 只有在 E、该层权重、位置和格式均不依赖 t 或当前视频状态时才能缓存；原始 DiT 的某些 cross 变体把时间也放入条件源，不能照搬固定文本缓存结论。

参考 flow 路径采用显式 Euler。选定 0=t_0<t_1<…<t_M=1，M≥1；给定 CFG 的两个同状态预测 u_c、u_u，令 u_g=u_u+w(u_c−u_u)，再更新 z_(k+1)=z_k+(t_(k+1)−t_k)u_g。w 有限，w=1 得条件分支，w=0 得无条件分支。这只是一个完整且可验证的求解候选，不声明为 Wan 正式默认采样器，也不保证大步长质量。

```pseudocode
算法 AdvanceVideo：分片预测、CFG 与 flow Euler 提交
输入：合法时间表、初始 latent、条件与 null 条件、guidance、分片计划
输出：最终 latent z_M

编码条件；验证模型、shape、位置、预测类型和时间方向
建立完整互斥分片，给每个分片绑定请求、尝试、求解步和全局范围
state ← (z_0,k=0,请求私有随机与必要历史)
while k<M：
    固定 state 快照；所有 rank 使用同一实际 t_k 与模型版本
    分别完成 conditional/unconditional 的全部层与输出还原
        两分支可顺序执行并复用工作区，或合法合批执行
    验证两分支都属于同一请求、尝试、k 和全局范围
    对每个 rank 的本地 latent 元素：
        u_g ← u_u+w·(u_c−u_u)
        z_candidate ← z_k+(t_(k+1)−t_k)·u_g
    等待全部必要分片完成，验证没有缺片、旧步或非有限输出
    成功：以一致边界提交 (z_candidate,k+1,新随机与历史状态)
    失败：不推进 k；保留上一完整快照，结算在途访问后重试或失败退出
仅在 k=M 后，将最终 latent 交给最终解码路径
```

Euler 只需要当前状态，但其他 solver 可能需要历史预测、子步和额外模型调用；此时 M 个更新不等于 M 次 forward。分片迁移或迭代重组可以改变物理布局，不能合并请求的 k、条件或 RNG。并发预览必须读取受保护的快照；预览完成不授权覆盖仍被网络或 solver 使用的缓冲。

## 全局 attention 的精确分块与跨卡布局

对一个 head、query q，合法 key 集合为 J_q，分数 s_j=q·k_j/√d_h，输出 o=Σ exp(s_j)v_j/Σ exp(s_j)。Wan 基础全时空 self-attention 是非因果的；不能因为 VAE 因果就添加时间因果 mask。分片 G_r 必须在声明的全局 key 域上完整覆盖且两两不交，padding 不得进入有效分母。

对一个已经处理的非空集合维护 m=max s_j、l=Σ exp(s_j−m)、a=Σ exp(s_j−m)v_j。a 是未归一化向量。加入新分数 s 时，m'=max(m,s)、c=exp(m−m')、w_s=exp(s−m')，更新 l'=cl+w_s、a'=ca+w_sv。集合为空时用 EMPTY 表示，首次 c=0，避免 exp(−∞−(−∞))。

两个非空统计合并时取 m'=max(m_1,m_2)，并用 exp(m_i−m') 同时缩放各自的 l_i 与 a_i 再求和。这保证分母和分子处于同一单位。归一化只在所有合法 key 贡献完整后执行。实数下等价，浮点改变归约顺序时应按明确容差比较，不能要求普遍逐位相同。

```pseudocode
算法 ExactAttentionRow
输入：q，完整 key 域及其互斥分片 G，K/V，全局 mask
输出：该 query 的完整 attention 输出

验证 G 完整、互斥且索引合法
m ← −∞；l ← 0；a ← 零向量
for 每个 key 块，按计划恰好访问一次：
    for 该块每个全局 key j：
        若全局 mask 禁止 j，跳过
        s ← Dot(q,K[j])/√d_h
        若 s 非有限，按数值契约报告错误
        m_new ← max(m,s)
        c ← 0，若 l>0 则 c←exp(m−m_new)
        w ← exp(s−m_new)
        a ← c·a+w·V[j]
        l ← c·l+w
        m ← m_new
若 l=0：按模型明确的空行契约处理；本参考路径拒绝
否则返回 a/l
```

反例：两个 key 分片分别只有分数 0 和 ln 9，value 分别为 0 和 10。完整输出为 9；两个局部 softmax 输出的平均值为 5。块长度相同仍不能等权平均。漏掉 key、重复 key 或把全局位置改为局部位置，也不会被在线合并自动修复。

一种多卡候选固定本地 Q，让 K/V 环流。均匀分片时 rank r 的 token 区间为 [rN/p,(r+1)N/p)。若使用行主序本地 hidden，每个元素地址为 base_r+b_a[(j−rN/p)d+c]；head h 的通道为 hd_h+c_h。全局 j 仍通过完整 N_h、N_w 还原坐标。rank→device 是显式映射，rank 0 不必是物理设备 0。

实际计算可以将本地 query 和当前 key 块再切成 B_q×B_k tile：加载对应 Q/K/V 元素，计算 B_qB_k 个分数并更新各 query 的 m/l/a，最后将归一化结果写回该 query 的本地输出行。每个输出行只有一个逻辑拥有者。这里没有指定 warp、Tensor Core fragment 或共享存储 swizzle；具体 GPU 映射需要所选 backend 的公开契约与资源检查，不能由逻辑布局猜测。

```pseudocode
算法 RingAttention：每个 rank 的一层 self-attention
输入：本地 Q/K/V、分片目录、p 个成员的匹配执行计划
输出：本地 Q 对应的完整输出行

每个本地 query/head 初始化 EMPTY 统计；Seen←空
current ← 本地 K/V 块，携带请求、步、层、分支、owner 和全局范围
for k ← 0 到 p−1：
    要求 current.owner=(rank−k) mod p，且未在 Seen 中
    若 k<p−1：
        向下一 rank 提交发送 current，向独立 next 缓冲提交接收
    用 current 的全部合法 key 更新本地每个 query/head 的统计
    Seen ← Seen∪{current.owner}
    若 k<p−1：
        等待发送不再读取 current，接收完整且对计算消费者可见
        验证 next 身份与范围；交换 current 与 next
要求 Seen 覆盖全部 p 个块；归一化并按本地 query 次序写回
所有读者和发送结束后，释放相应临时缓冲
```

空 rank 可以没有 query 或 key，但仍参与约定通信次序。通信提交返回不等于 GPU 消费可见，失败后也不能立即复用仍被 DMA 或 kernel 访问的缓冲。单步内两缓冲足以表达上述交接；只有重试、并发或槽位复用时再加入相应尝试代次，避免迟到块写入新步骤。

另一候选先做 sequence/head all-to-all：每个 rank 从 N/p 行、全部 h 个 head，变为全部 N 行、h/p 个 head；本地完成这些 head 的全局 attention，再逆向交换恢复 N/p 行、全部 head。它要求 head 分配、整除或显式补齐、全局位置及逆重排均相容。它不是把每卡内部 attention 直接当作全局结果。两次 all-to-all 的准备、全组交换和屏障与 ring 的逐块流水不同，选择依赖 head 数和拓扑。

## 计算、流量、容量与关键路径分别列账

以下主项针对一个请求、一个 CFG 分支、一次前向的一层，假设 dense MHA、各 head 总宽为 d，乘加计 2 FLOPs。softmax、归一化、激活、位置、padding 和重算另计。

- self QK 与 PV 合计约 4N²d FLOPs；QKV 与输出投影约 8Nd² FLOPs。
- 普通两矩阵 FFN 约 4Ndd_ff FLOPs，激活中间量为 N×d_ff；门控 FFN 需按实际矩阵数重算。
- cross attention 配对工作约 4NSd，Q 与输出投影约 4Nd²，条件 K/V 投影约 4Sd_ed。源投影是否每步重复取决于合法缓存，而不是仅取决于源长度。
- 显式 self scores 为 hN²b_s 字节；精确 tile 消除了完整 scores 的驻留要求，没有消除 N² 配对工作，也没有保证 Q/K/V 各只从 HBM 读取一次。

全局 self-attention 在 N_t 翻倍而其他轴不变时，配对工作变为四倍；首帧规则使 F 翻倍未必恰好令 N_t 翻倍。若模型本身采用空间与时间因子化 attention，配对数可变为 N_t(N_hN_w)²+(N_hN_w)N_t²；局部窗口则按实际合法 pair 计数。这些是不同结构，不能把其成本直接借给已有全局 checkpoint。

令完整 K+V 字节 S_KV=2Ndb_a。均匀 p 卡 ring 每 rank 的发送量和接收量分别为 (p−1)S_KV/p，收发口径不互相重复。局部 K/V 常驻为 S_KV/p，保守地另预留 current/next 两块临时区至多 2S_KV/p；具体别名可以减少空间，但需证明最后读者关系。累计接收字节不是同时驻留字节。

每个本地 query/head 的统计含两个标量及一个 d_h 维向量；若全部用 4 字节存储，统计主项为 4(N/p)(d+2h)。它可在 query tile 内驻留或分阶段写回；不能既按全体统计计容量，又把同一分子缓冲另算成一份输出副本而不说明实际分配。

在无争用、有效单向带宽 β、每轮启动 α 的模型中，ring 传输项为 (p−1)α+(p−1)S_KV/(pβ)。若收发全双工且能与计算重叠，计算与传输的 max 只是理想必要时间；完整调度还有首尾、合并和依赖。共享 NIC、PCIe 或 HBM 时，应合并该资源的需求。观察到时间线相交并不证明两者能同时获得独占带宽。

均匀且全部 QKV、输出均为 b_a 字节元素的 sequence/head 方案，每 rank 两次交换的远端发送 payload 合计为 4Ndb_a(p−1)/p²：QKV 占三份，逆变换输出占一份。接收另列；packing、padding 与协议开销另计。这个字节账不包含权重 gather，不能把较低发送量直接当作更低端到端时间。

推理通常只需当前层工作区逐层复用，不必常驻训练的全部层激活；异步流水、跨步近似缓存或图池会改变这一点。模型权重、固定条件、当前/候选 latent、历史预测、通信区、FFN 中间量、VAE 和其他请求共同形成 C_peak=max_t C(t)。共享权重计一次；保留池与其中活跃对象不重复相加。驻留积分以 byte·s 计量，不能加进毫秒延迟。

准备成本还包括条件编码、可选输入图像 VAE、latent 生成与分发、位置表、分片目录、布局重排、权重加载/打包、编译和通信初始化。纯文本生成通常不需要输入视频 VAE 编码。FSDP 等权重分片可以节省驻留，但逐层 gather 会与 attention 交换争用链路和 HBM；必须按同一时间线计算。

## 分块解码必须保留历史、相位与输出归属

最终 latent 可以逐块解码，前提是当前解码图的所有跨块依赖均被覆盖。Wan §4.1.3 报告其普通 kernel=3 的因果时间卷积保留两个历史 feature，首次以两个零 feature 初始化；其 stride=2 时间下采样采用不同策略，在非初始 chunk 使用单帧 cache。后者不能推广为任意 stride=2 卷积都只需一帧，更不能直接作为所有 decoder 层的规则。

用一个独立的局部因果算子说明如何推导状态。设 dilation δ、kernel 长度 k、stride s，定义 y[n]=Σ_(j=0..k−1)w[j]x[ns−jδ]，负索引输入为零，有限输入长度 L 时仅产生 ns<L 的输出。最新已到达输入索引为 g；只在 ns≤g 时产生 y[n]。状态是该层历史、累计输入数和下一个全局输出 n，不是当前 chunk 内的局部编号。

```pseudocode
算法 StreamCausalConv：上述局部卷积的精确流式执行
输入：按原顺序到达的 chunk，参数 w、δ、s
输出：所有 y，顺序与完整序列参考一致

初始化 seen←0，next_n←0，history←空；过去负索引隐式为零
for 每个 chunk：
    for chunk 中每个新输入值 x：
        history[seen]←x；seen←seen+1
        while next_n·s<seen：
            v←0
            for j←0 到 k−1：
                idx←next_n·s−j·δ
                若 idx≥0：v←v+w[j]·history[idx]
            发布本层输出 (next_n,v) 一次
            next_n←next_n+1
        丢弃所有未来输出不再可能读取的历史
        保守做法：保留最近 δ(k−1) 个真实输入
空 chunk 不改变 seen、next_n 或历史；结束不产生额外右侧 padding 输出
```

例如 k=3、δ=1、s=2，完整输入索引 0..8 应输出最新输入位于 0、2、4、6、8 的五项。若在累计输入数为 1 的 chunk 边界从局部零开始，则下一块可能错误地从全局索引 1 输出，结果长度和相位都会错。缓存足够而相位错误仍不合法。

多层网络必须缓存每层自己的输入 feature。前层当前 chunk 产生的新输出是后层输入，后层不能拿原 latent 或最终 RGB 替代自己的历史。cache 还要绑定分辨率、dtype、解码器版本和请求；重新开始解码、切换请求或改 shape 时应重新初始化。

为完整说明首帧和逐层数据流，定义一个教学 decoder：时间 latent z[0] 仅展开一次，后续每个 latent 展开 r 次，得到 F=1+r(T_l−1) 个输入 feature；接着执行一个 kernel=3 的因果卷积、逐点 tanh、另一个 kernel=3 的因果卷积。该展开和层序是参考模型，不声明为 Wan 的真实上采样实现。

```pseudocode
算法 ReferenceChunkDecode
输入：最终 latent 序列 z，合法 chunk 划分，重复因子 r≥1
输出：与完整展开后两层卷积相同的 F 个参考输出

初始化 global_latent_index←0，conv1/conv2 的独立流式状态
for 每个 latent chunk：
    for 其中每个 latent v：
        count←1，若 global_latent_index>0 则 count←r
        将 count 个 v 顺序送入 conv1
        conv1 每产生一个输出，立即计算 tanh 并送入 conv2
        conv2 输出携带递增全局帧索引，交给有界下游队列
        global_latent_index←global_latent_index+1
要求 latent 恰好消费一次，最终帧索引覆盖 0..F−1
等待必要消费者完成后，清理解码状态与缓冲
```

真实 VAE 需要按其所有卷积、上下采样、残差和归一化逐层建立同样的依赖。对于一串局部卷积，输入尺度感受野可递推为 R_0=1、J_0=1，R_l=R_(l−1)+(k_l−1)δ_lJ_(l−1)，J_l=s_lJ_(l−1)。上采样需要先按实际索引映射展开，再推导依赖，不能套用一个统一 halo 常数。

空间 tile 或非因果时间 tile 可以扩展 halo，只发布核心区并按全局索引裁剪；输出核心区须完整且互斥。若依赖确为有限局部，halo 覆盖后核心区可与完整参考等价。若存在跨 tile 的 attention 或全局统计，单纯增加有限 halo 可能无法恢复函数。对同一空间面积，内部 halo 的处理放大近似为 (h+2r_h)(w+2r_w)/(hw)，边界裁剪和多层缩放需另算。加权 overlap 融合能减轻接缝，但缺失依赖时仍是近似，不会自动变成精确解码。

chunk 大小减少通常降低临时激活峰值，却增加调用、同步和边界工作；逐层 cache 容量近似 Σ_l C_lH_lW_lR_hist,l b_l，使用该层真实尺度。它不随视频总长线性增长的前提是解码图只有有界历史，并且下游队列也有界。

VAE 因果不意味着全序列 DiT 可以提前提交前部帧。在尚未完成的全局去噪中，后部 latent 可通过 attention 改变前部预测，后续求解仍会修改前部。只有最终 z_M，或模型明确支持且已稳定提交的流式状态，才有最终帧语义。Wan §5.6 的 Streamer 是经过适配和微调的窗口队列模型，包含不同噪声等级、warmup 丢弃和已生成 token 回注；它不是把基础全局模型切 chunk 就得到的等价执行方式。

## 四卡长视频从生成到交付的完整轨迹

沿用 81 帧、720×1280 的几何，设 d=1536、h=12、BF16 元素 2 字节、均匀四卡 ring，求解和 CFG 策略固定。hidden 与 head 是设定参数，不作为某个 Wan 型号的完整规格；以下均为推导，没有设备实测。

| 对象或工作 | 可复算主项 | 数值 |
|---|---:|---:|
| 完整 latent | 21×90×160×16×2 B | 9.2285 MiB |
| 完整单份 hidden | 75600×1536×2 B | 221.4844 MiB |
| 单 head 显式 scores | 75600²×2 B | 10.6457 GiB |
| 一层 self 配对工作，单分支 | 4×75600²×1536 FLOPs | 35.1152 TFLOPs |
| 完整 K+V | 2×75600×1536×2 B | 442.9688 MiB |
| 每 rank 本地 K+V | 上项/4 | 110.7422 MiB |
| 每 rank 一层接收，单分支 | 3×442.9688/4 MiB | 332.2266 MiB |
| 每 rank 全体 query 的 FP32 统计 | 4×18900×(1536+24) B | 112.4725 MiB |

完整 scores 的巨大容量解释了精确 tile 的必要性；它不能单独证明四卡 ring 最优。若 d_ff=4d，self 投影加普通 FFN 的主项为 24Nd²≈4.2807 TFLOPs/层/分支，cross、激活等另计。每卡配对工作理想降为四分之一，但网络字节不按同一比例下降。

设有效单向链路为 25 GiB/s，每 rank 纯接收时间项为 332.2266 MiB/(25 GiB/s)≈12.9776 ms/层/分支。发送另有同样 payload；全双工模型不直接把时间翻倍。若 M 次 Euler 更新、L 层、每步两个分支都完整执行，接收累计为 2ML×332.2266 MiB；是否合批和重叠会改变调度时间，不能把一次交换的 13 ms 当作整段生成的网络成本。

一次完整轨迹如下：

1. 准入前计算实际 token 几何、模型与工作区驻留、当前/候选 latent、通信区及解码/输出预算；建立显式 rank→device 和分片目录。条件编码只做需要的路径，准备并分发全局身份一致的初始噪声。
2. 每个求解步固定状态快照；所有 rank、层和 CFG 分支使用相同时间与条件身份。按前述环流完成每层全局 attention，再完成局部 cross、FFN 和输出还原。若权重分片，还需在相应消费者之前完成 gather。
3. 只有本步全部分片和必要 solver 状态完整时提交新边界。故障时保留旧边界或从已知起点重算；不能混合旧步和新步的 rank 结果。取消阻止未来提交，已在途的发送、计算和快照读者仍需排空。
4. 最终 latent 按解码器要求重新布局。若将完整 latent gather 到一个解码 rank，均匀四片时该 rank 需从其余三卡接收约 6.9214 MiB，其他 rank 各发送约 2.3071 MiB；这是末端一次性流量，与 DiT 每层交换不同。也可选择支持依赖传递的分布式 VAE，但必须另列边界状态与输出重排。
5. 解码 cache 初始化一次，按 chunk 连续推进，将每帧交给像素/色彩转换、视频编码、封装和发送。每个阶段对全局帧身份负责；编码器若有重排或 lookahead，解码完成不等于该帧对应的码流已可交付。
6. 编码器最终排空、封装完成且协议终止后，才记录完整交付。失败、取消和未完成请求留在资源消耗账中；有效产出以同时满足质量和服务目标的视频请求计数。

81 帧完整 RGB8 为约 213.5742 MiB，16-bit RGB 为约 427.1484 MiB。若只保留 Q 个待编码帧，RGB 缓冲主项为 QHW×3b_pix，还要加在途帧和编码器工作区。队列满时暂停上游解码或按协议结束请求，不能让慢编码器或客户端积累无限像素。暂停会延长最终 latent、VAE cache 和请求状态的驻留。

令 K 个解码 chunk 的串行解码、转换和编码阶段分别耗时 d、c、e，且资源独立、每个阶段一次处理一块、缓冲足够，理想流水完成时间为 d+c+e+(K−1)max(d,c,e)；完全串行为 K(d+c+e)。真实块大小、首帧、编码器重排和共享 GPU/HBM 会改变该模型。若编码落后，增加 Q 只能暂时吸收波动，不能修复长期服务率不足。

81 帧在 16 fps 下，首末帧 PTS 跨度为 (81−1)/16=5 s；若每帧完整展示 1/16 s，持续时间为 81/16=5.0625 s。生成吞吐 F/墙钟时间、播放帧率、首个可解码内容时间和完整交付延迟应分别报告。单个 DiT forward 无法代表其中任何完整服务指标。

## 条件变化怎样改变方案选择

| 变化 | 保留的机制 | 需要调整或放弃的假设 |
|---|---|---|
| F 或空间尺寸增加 | 全局坐标、求解状态和完整覆盖 | 重算首帧几何、N² 工作、N 规模缓冲及输出字节；重新检查模型支持 |
| 减小 patch | patch 与还原的身份关系 | 通常涉及输入/输出投影及模型选择，不是已有 checkpoint 的任意执行参数 |
| 视频变短或网络变慢 | 精确 tile 仍可保留全上下文 | CP 的启动和交换可能超过计算节省；可退回单卡或更小组 |
| head 数少于并行度 | ring 可按 token 分片的原则 | head all-to-all 的整除和局部 head 方案可能不成立，需改计划 |
| 采用窗口、因子化或独立片段 | 按真实合法 pair 重新计成本的方法 | 相对全局模型的函数和跨片一致性改变，需质量许可与独立评估 |
| CFG 顺序改合批 | 两分支同 z、同时间、同条件身份 | 工作区峰值、权重复用和 kernel 变化；容量不足时保留顺序路径 |
| 减少步数、跨步 attention 缓存 | 每次保留下来的更新仍须合法 | 轨迹或网络近似误差改变；重新评价运动、细节与一致性 |
| VAE chunk 变小 | 逐层历史、相位和一次发布 | 重新计算 cache、调用、halo 重算及输出流水首尾 |
| 编码改用 CPU | 帧归属、时间戳和有界队列 | 增加 D2H、主机带宽、CPU 配额与编码线程预算 |
| 条件图像、mask 或 prompt 改变 | 依赖不变才可复用的规则 | 条件表示、投影缓存和输入通道适配必须重新核对 |

Wan §4.4.2 报告按验证集选择跨步 attention/CFG 缓存策略并得到约 1.62× 推理加速。这支持把相似性缓存作为可评估候选，不能证明所有视频、guidance 和时间表都函数等价。单卡精确 attention、多卡精确 CP、独立短片拼接可以同时作为候选，但第三种只有在任务接受改变时间上下文时才参与质量合格后的比较。

## 用区分实验定位首个错误

先比较便宜的中间机制，再比较完整视频与设备关键路径。固定 checkpoint、条件、初始 latent、实际时间表、guidance 和随机项。只看最终视频可能把求解错误、位置错误、VAE 接缝和编码问题混在一起。

| 观察 | 竞争解释 | 可控实验及预期 | 反证后的调整 |
|---|---|---|---|
| CP 数值随分片变化 | 漏/重 key、错误 merge、位置重置或浮点顺序 | 小网格完整 softmax 对照；分别注入空片、不均匀片、漏项、重复项、全局坐标重置 | 覆盖错误先修计划；覆盖正确再查首次分歧，不能扩大容差掩盖缺项 |
| 增加 GPU 没有降延迟 | 启动、网络、权重 gather、慢 rank 或计算不足 | 固定 N 扫并行度，再扫 N；分开记录投影、attention、packing、传输与等待，固定拓扑 | 传输增加而计算已很短时减少组；若是 gather 争用，修正共同链路账 |
| chunk 边界出现伪影 | cache 被清空、层历史错误、stride 相位错或 halo 不足 | 固定最终 latent，对照完整解码；移动 chunk 边界，使用冲激/斜坡输入并逐层比较 | 周期性长度偏差先查相位；整体内部偏移需查非局部统计，不能只加 halo |
| 帧数多一批或少一批 | 首帧重复、末块裁剪或重复发布 | 检查 F=1、5、9 及尾部小块；核对全局帧索引恰为 0..F−1 | 数值正确但索引不全时修发布与裁剪，不能靠调 fps 隐藏 |
| 末端时延高且像素驻留增长 | 编码慢、D2H、封装等待或客户端慢 | 独立替换为有界快速消费者，再分别恢复转换、编码和发送，记录队列字节 | 若只替换消费者就改善，修下游服务率或准入，不把问题归给 DiT |
| flow 更新方向异常 | 速度符号、时间方向或网络输入缩放错 | 用已知 ξ、y 构造直线轨迹，常速度在非均匀时间表上逐边界应恢复插值并抵达 y | 代数通过但真实模型不符时，检查适配器与正式 scheduler，不能继续套 DDIM |
| 单请求正确而交错请求错误 | cache 串用、旧步迟到或缓冲提前复用 | 交错不同条件、shape、取消与重试；逐请求比较首次错误的层/步/帧身份 | 先修所有权和可见性，再做性能重叠 |

质量实验除单帧清晰度外，还应覆盖指令符合、对象身份、运动、物理合理性、时间一致性和边界伪影。改变近似策略时保留独立评估 prompt/seed，不能只挑成功视频。服务实验记录 offered、admitted、completed、failed、拒绝、质量与延迟的联合合格数；冷启动、排队、全部采样、VAE、编码和发送分段观测，再按依赖还原端到端。

## 随章检查的模型范围

唯一随章检查文件为 tools/check_k_model_010_full.py。它是标准库、离线、只读、确定性的标量模型，检查首帧几何、完整 softmax 与分片在线合并、全局位置、卷积历史与 stride 相位、两层参考解码、flow Euler 和 ring 字节账。

独立参考分别使用显式完整 softmax、完整序列卷积、先完整展开再逐层求值、解析直线插值以及枚举实际传输块。负对照包括错误平均各片输出、局部重置位置、chunk 重置相位、清空逐层 cache、重复首帧、flow 符号反转和把收发总量记成单向接收。每个缺陷由同一验证器执行正确接受、缺陷拒绝、恢复后接受；非法覆盖另用独立域错误拒绝。检查框架不会把任意异常都算成负对照成功。

检查脚本、输入绑定与实际执行结果见[检查记录](../../docs/evidence/knowledge-workflow-20261028-batch8/K-MODEL-010/round-2/check-0.json)。

这些结果不能证明真实 Wan VAE、完整 checkpoint、GPU 浮点、跨设备可见性、HBM/NIC 事务、通信重叠、编码器行为、视频质量、SLO 或 Agent 使用效果。真实 decoder 的逐层上采样与 cache 规则、正式采样器适配、目标 GPU/backend 和媒体协议仍是接入时必须验证的具体边界。

<!-- infraagents:provenance -->

## 来源与推导

- [S-WAN：Wan: Open and Advanced Large-Scale Video Generative Models](../sources/S-WAN.md)：固定 2503.20314v2 的 §4.1 与图 5–6 支持首帧仅空间压缩、4×8×8 时空压缩、16 个 latent 通道、普通时间卷积与下采样的不同 feature cache 规则；§4.2 支持 (1,2,2) patch、文本 cross-attention 和 flow matching 的插值/速度定义。§4.3–4.4 描述 ring/Ulysses 组合、模型分片与推理近似缓存；训练内存或论文加速结果不直接移作本章推理实测。§5.6 的 Streamer 有窗口假设、微调、warmup 和缓存回注，支持将其与基础全序列模型区别处理。
  [完整原文](../../references/S-WAN/reader.html) · [全文文本](../../references/S-WAN/fulltext.txt)

- [S-DIT：Scalable Diffusion Models with Transformers](../sources/S-DIT.md)：固定 2212.09748v2 的 §3.2 描述 latent patch 线性嵌入、位置表示、self/cross attention、adaLN/adaLN-Zero 和逐 token 输出还原；原模型输出含噪声与对角协方差。其 cross 变体的条件源包含时间和类别，附录还讨论部分 latent 通道 CFG，因此不能把普通固定文本缓存、输出通道或 guidance 范围未经核对套到所有 DiT。本文教学用视频网络是据这些机制组织的独立结构。
  [完整原文](../../references/S-DIT/reader.html) · [全文文本](../../references/S-DIT/fulltext.txt)

- [S-DIFFUSERS：Diffusers — Scheduler 与 DDIM（固定历史源码）](../sources/S-DIFFUSERS.md)：所供作者核验后的机制说明给出 epsilon、原始样本和 v 的不同转换，以及累计 alpha、指定前一时间点、方差与随机项的更新依赖。它用于限定预测和 scheduler 契约，不支持将 DDIM 当作 Wan flow 的通用更新公式。本轮没有读取未提供的源码归档，也不把来源说明记为源码或原始文档全文阅读。
  [算法与机制说明](../sources/S-DIFFUSERS.md)

正文中的在线合并推导、教学网络、flow Euler、通用局部卷积与参考解码、资源账和组合状态轨迹是基于所供材料的独立推导；设定算例不冒充设备实测。来源事实受 2026-04-30 上限约束，本轮编辑与检查日期为 2026-10-28，编辑日期不改变来源发布日期。原章节身份与版本映射继续以现有版本清单为准。

## 延伸阅读

- [图像扩散：迭代求解、条件复用与 CFG](../models/image-diffusion.md)。
- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [通信传输：注册、完成、可见与回收](../hardware/transport-and-rdma.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
