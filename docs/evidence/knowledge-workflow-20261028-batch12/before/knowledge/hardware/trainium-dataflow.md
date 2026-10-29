# K-HW-008：Trainium：HBM、SBUF、PSUM 与编译执行

Trainium 的高效执行需要同时满足三条链：逻辑张量到 partition/free 坐标的布局链，DMA、计算引擎与 PSUM/SBUF 生存期的执行链，以及固定形状程序与动态请求状态的编译和绑定链。布局错会读错元素，完成依赖错会消费尚不可见的数据或覆盖在途输入，程序绑定错会把输入送进不相容的程序。本章区分固定资料支持的硬件事实与本库的数据流推导；容量和时间算例均不是设备实测。

## 问题、对象与不变量

对单个无 dropout 的 attention head，令 Q[L,d]、K[S,d]、V[S,dv]，缩放为 a，query 行 i 的合法 key 集合为 A_i。定义 z_ij=a q_i·k_j，p_ij=exp(z_ij)/Σ_{t∈A_i}exp(z_it)，o_i=Σ_{j∈A_i}p_ij v_j。i、j 是逻辑位置，不能用 padding 后的物理偏移替代。padding、页表空洞和未命中范围必须由谓词排除；只把 K/V 填零不能防止它们进入 softmax 分母。

这里要求参与计算的有效分数和 V 有限。NaN、正无穷或计算溢出需要拒绝或另定特殊值路径，不能依靠稳定 softmax 自动处理。用负无穷编码 mask 时，先把它解释为不可见关系。空可见行输出零向量、自然对数 LSE 为负无穷，是本章明确选择的接口契约。

片上 tile 至少绑定以下对象：

- 逻辑范围：query/key 块、head、层、绝对位置和有效尾部。
- SBUF 对象：物理槽、partition/free 布局、dtype、槽位代次、写入完成依赖和读者。
- PSUM 对象：输出身份、累加是否结束、交接消费者以及允许覆盖的时点。
- 在线统计：m、l、U 分别表示已处理可见集合的最大分数、相对 m 的指数和、相对 m 的加权 V 和。
- 服务状态：程序身份、已提交 KV 长度、输入游标、块表/槽映射、采样状态和完成依赖。

每个合法 query-key 对必须恰好处理一次；位置、缩放、mask 和 head 映射不能因分块或分桶改变。DMA 写入及消费者可见性建立后才能读；PSUM 输出交接结束前不能覆盖；槽位的所有读者结束后才能复用。同一物理地址不证明仍是同一逻辑块，数学合并正确也不证明异步输入已经可读。

## 硬件坐标：从 HBM 到矩阵输入与输出

固定 Trainium2 NKI 资料给出每个 NeuronCore-v3 的 SBUF 为 28 MiB，即 128 个 partition、每 partition 224 KiB；PSUM 为 2 MiB。HBM/SBUF 之间主要由 DMA 搬运，PSUM/SBUF 之间由计算引擎指令搬运。因此 PSUM→SBUF 不是免费 view，也不能只按 HBM 带宽计费。

先展开普通 BF16/FP16 的单收缩块路径。设 query 块实际行数为 M，key 块实际列数为 N，query/key 起点为 i0、j0，元素字节分别为 s_Q、s_K、s_V。HBM 行连续布局的元素地址为：

addr_Q(i,p)=base_Q+s_Q(i d+p)，
addr_K(j,p)=base_K+s_K(j d+p)，
addr_V(j,t)=base_V+s_V(j dv+t)。

QK 的收缩轴是 d。形成 Q_s[p,r]=Q[i0+r,p]、K_m[p,c]=K[j0+c,p]，其中 p 为 partition，r、c 为 free，得到 PSUM 中的 Z[r,c]=Σ_p Q_s[p,r]K_m[p,c]，再施加缩放 a。若从 HBM 的 [L,d]、[S,d] 装入，源最内层 d 必须成为目标 partition；仅改变逻辑 shape 不完成这次物理转换。

普通 DMA copy 把源最内层维映射到 free，源行维映射到 partition。固定资料中的 HBM→SBUF DMA transpose 则把源最内层维映射到目标 partition，适合上述 Q/K 路径。Q 若已经由上一算子以目标布局留在 SBUF，可以省掉相应 HBM 装入或转换。

softmax 产生的权重暂按 W[r,c] 表示。PV 的收缩轴变为 key，必须先形成 P_s[c,r]=W[r,c]，即 key→partition、query→free；V_m[c,t]=V[j0+c,t] 可由行连续 V 的普通 copy 形成。然后 TensorE 才能计算 PSUM 中的 T[r,t]=Σ_c P_s[c,r]V_m[c,t]。矩阵完成后，将 T 通过计算引擎搬入 SBUF，建立后续向量消费者可见性，才能用它更新 U。QK 和 PV 的收缩轴不同，是概率布局必须实际生产出来的原因。

下文的单块执行路径选择 d≤128、Bq≤128、Bk≤128、dv≤512，并要求 dtype、实际 tile 和其他布局条件满足固定 ISA。这里 Bk≤128 同时满足 PV 的普通收缩块边界。更大的 d 要先跨多个收缩块完成同一个 Z，再做最大值和指数；更大的 Bk 要对同一个 tileU 跨 key 子块累加，再交接完整结果；更大的输出 free 轴要切分输出。不能把未完成的局部 Z 当成最终分数，也不能把单指令边界当作整个模型维度上限。

Double-FP8 是另一种指令输入契约。固定资料给出的最大 stationary/moving 输入为 [128,2,128]、[128,2,512]，输出为 [128,512]。张量第二维，即最外层（最慢变化）free 维必须为 2。令 partition 长为 P_rt，则收缩坐标可表示为 k=p+P_rt q，q∈{0,1}；两个输入必须使用相同配对，两半贡献合入同一个 FP32 输出。q 不是独立 batch 维，奇数收缩长度需要补零。该模式不能与 column tiling、sparse matmul 或 TensorE transpose mode 组合。

转置路径要比较整条消费者链。DMA transpose 直接读写 SBUF，但竞争 DMA 和存储端口；TensorE transpose 占用 TensorE、PSUM，通常还需 PSUM→SBUF 交接。固定资料给出的 HBM→SBUF transpose 最高约 90%、SBUF→SBUF transpose 最高约 50%、普通 copy 最高约 100% DMA 利用率是来源说明，不是本章测量。输出 partition 维为 128 的倍数、最内层 free 维为 16 的倍数是最佳吞吐条件，不是所有功能合法形状的必要条件。

DMA transpose 的 bit-accurate 说明覆盖 2/4 字节类型，不能自动推广到 1 字节 FP8。TensorE 显式 transpose mode 支持含 NaN/Inf 的转置；普通乘单位矩阵仍会执行乘加，特殊值可能沿累加链传播。FP8 布局转换必须另证合法路径，不能同时套用未覆盖的数据类型和互斥指令模式。

固定版本硬件 DGE 按需生成描述符，但不支持 indirect gather/scatter；分页式访问不能因此被算作零准备成本。VectorE 与 GpSimdE 可并行访问 SBUF，但部分 VectorE 模式共享总线；VectorE 与 ScalarE 并行全带宽访问 PSUM 的条件包括不冲突于同一 bank。引擎图上的并行连线不保证任意访问全速重叠。

## Attention tile 的完整状态与执行顺序

对已处理的可见 key 集合 T_i，非空状态定义为：

m=max_{j∈T_i}z_ij，l=Σ_{j∈T_i}exp(z_ij−m)，U=Σ_{j∈T_i}exp(z_ij−m)v_j。

空状态为 E=(−∞,0,零向量)。本块可见集合为 C_i=A_i∩[j0,j1)。若 C_i 为空，全部状态保持不变，不能计算 exp(−∞−(−∞))。否则令：

m_new=max(m,m_tile)，m_tile=max_{j∈C_i}z_ij，
alpha=0（旧状态为空），否则 alpha=exp(m−m_new)，
w_j=exp(z_ij−m_new)。

候选状态为 l_new=alpha l+Σw_j，U_new=alpha U+Σw_j v_j。推导来自 exp(z−m_new)=exp(m−m_new)exp(z−m)，因此旧分母和旧分子必须使用同一个重标度。最后共同提交 (m,l,U)←(m_new,l_new,U_new)。这里的共同提交是逻辑状态转换，不要求硬件提供一次原子写三组张量的指令；本块完成前不得让下一块读到混合新旧统计。

```pseudocode
算法 RunAttentionTile
输入：Q[L,d]、K[S,d]、V[S,dv]，有限缩放 a、可见谓词 visible(i,j)
      query 块长 Bq、key 块长 Bk、K/V 槽数 D
输出：O[L,dv]，以及请求的自然 LSE[L]
前提：采用上节普通单收缩块路径，形状、dtype 和布局满足所选指令

1 校验 L、S 为非负整数；Bq、Bk、d、dv、D 为正整数
  校验张量形状和所选硬件路径；非法零块长在任何循环之前拒绝
  L=0 时返回空 O 和空 LSE
2 对每个 query 块 [i0,min(i0+Bq,L))：
3   将 Q 装入 SBUF，形成 Q_s[d partition,query free]
    等待装入和布局转换完成，建立消费者可见性
4   每行初始化 m=-∞、l=0、U=零 dv 向量，统计量保留于 SBUF
5   对每个 key 块 [j0,min(j0+Bk,S))：
6     取得上一代全部写者和读者均已结束的槽 s
      建立本次调用/query 块身份 e、槽代次 g 和逻辑 key 块 j0 的票据
7     搬入 K、V；形成 K_m[d partition,key free]、V_m[key partition,dv free]
      尾部物理填充不属于有效 key 集合
8     等待本代全部加载/转换完成并对消费者可见
9     执行 QK 到本块 PSUM Z；等待矩阵完成
      将 Z 移到 SBUF，等待交接完成及归约消费者可见
10    按绝对位置施加 visible，逐行形成 C_i；有效分数必须有限
      若所有行 C_i 均空，可跳过后续矩阵工作并结束本代实际读者
11    为本块 W 初始化零值；对 C_i 为空的行标记 unchanged
12    对每个非空行，读取同一旧状态 (m_old,l_old,U_old)
      m_tile=max 有效分数；m_new=max(m_old,m_tile)
      alpha=0 若 l_old=0，否则 exp(m_old−m_new)
      W[i,j]=exp(z_ij−m_new) 当 j∈C_i；其他位置保持零
      l_new=alpha·l_old+Σ_j W[i,j]；此时尚未覆盖旧状态
13    由 W 形成 PV 输入 P_s[key partition,query free]
      完成所需类型转换、物理布局转换及其完成/可见依赖
      若使用 TensorE transpose，先完成其 PSUM 结果到 SBUF 的交接
14    确认 P_s 与本代 V_m 均可读，初始化独立 PSUM 输出 T 为零
      执行 T=P_s^T V_m；等待本块全部 PV 矩阵操作完成
15    将完整 T 从 PSUM 搬到 SBUF 的 tileU
      等待交接完成并建立更新 U 的向量消费者可见性
      此后才允许复用 T 所在 PSUM 区
16    对每个非空行，U_new=alpha·U_old+tileU_i
      共同提交 (m,l,U)←(m_new,l_new,U_new)
      对 unchanged 行保持旧 m、l、U 全部不变
17    结算该票据的 QK/PV 实际读者；最后读者结束后才能覆盖槽 s
      完成事件必须匹配 e、s、g 和逻辑块，旧通知不得释放新代
18  全部 key 块完成后：l>0 的行输出 O=U/l、LSE=m+ln(l)
    l=0 的行输出零向量、LSE=-∞；S=0 时直接走此分支，不等待不存在的 DMA
19  写回 O 和请求的 LSE；等待写入及全部实际使用结束后释放本 query 块资源
20 返回结果；失败时停止新提交，不发布未完成结果，排空已提交访问后释放资源
```

W 在这里是相对 m_new 的未归一化权重，PV 不应先把它按本块分母归一化。低精度 P 的舍入、矩阵累加和最终转换进入实现的数值契约；实数公式不保证不同分块浮点逐位相同。mask 后的零权重也要求相应填充 V 已按合法有限值初始化，不能用零乘 NaN 来替代初始化。

第 13 步必须生产并发布 PV 的合法输入，第 14 步才消费；第 15 步必须交接 PV 的完整结果，第 16 步才更新 U。只有 m、l、U 全部代表同一个已处理集合，下一轮递推才成立。遗漏 m 写回并非只影响可选 LSE：令分数为 [0,0]、V=[[1,0],[0,1]]、Bk=1，第一块后正确状态为 (0,1,[1,0])，第二块后为 (0,2,[1,1])，输出 [0.5,0.5]。若 m 始终为负无穷，第二块 alpha 又变为零，输出退化为 [0,1]。

这份伪代码串行处理 key 块，D 表示可用于重叠的槽数上限，本身不实现预取调度。加入预取时，下一块第 7–8 步可以与当前块计算重叠，但必须先取得另一个空闲槽并保留相应身份。以 D=2 为例，块 0 的槽在 QK 完成但 PV 尚读 V 时仍不可覆盖；块 1 可以在另一槽加载；块 2 必须等待槽 0 的最后读者结束并进入新代。迟到的块 0 通知不能改变块 2 的 pending 集合。更多槽可能减少等待，也可能因活跃量增加而降低驻留。

## 编译形状、程序身份与动态状态

固定 NxD Inference 指南将编译分成样例输入 tracing→HLO→Neuron Compiler→NEFF。依赖张量数值的 Python 分支只记录样例经过的路径，因此一次 tracing 成功不证明所有动态控制路径都覆盖。持久编译缓存仅在编译器版本、模型配置和 flags 相同时复用；编译默认不序列化分片权重，部署不能只检查 NEFF 是否存在。

分桶和 padding 把动态请求放进固定容量，实际长度和 mask 继续决定函数。context encoding 与 token generation 可用不同桶；前缀缓存的 context 编码有 suffix、prefix 两轴，prefix 轴含 0。下面是与固定资料选择顺序一致的绑定策略；完整命中的输出资格处理和私有写区检查是本库补充的服务前提。

```pseudocode
算法 SelectContextBinding
输入：请求集合 R，每项含长度 L_i、相容已发布 KV 命中 H_i、末行输出资格
      正整数 suffix 桶 B_s、含 0 的非负整数 prefix 桶 B_p
输出：ready 输出集合、context 绑定计划，或具体失败

1 校验两组桶严格递增；逐请求校验整数 L_i>0、0≤H_i≤L_i
  核对模型上限、计算身份、KV 完成/可见性及源对象持有资格
2 ready←空；work←空
3 对每个请求：
4   若 H_i=L_i 且有同一模型/位置/输出契约的末行 logits 或等价采样输入：
      加入 ready，继续
5   h_i←H_i；若 H_i=L_i 但缺少末行输出，则 h_i←L_i−1
6   s_i←L_i−h_i；加入 work
7 若 work 为空，返回 ready 与空计划
8 在 B_s 选能容纳 max_i s_i 的最小桶 b_s；不存在则失败，不截断历史
9 对每个工作请求：m_i=min(h_i,b_s−s_i)，a_i=h_i−m_i，q_i=s_i+m_i
10 在 B_p 选能容纳 max_i a_i 的最小桶 b_p；不存在则失败
11 核对 q_i≤b_s、a_i≤b_p、a_i+q_i=L_i、batch 和模型范围
12 核对 (b_s,b_p,batch,dtype,TP,LNC,compiler,model) 程序存在且已加载
13 保护可读源 [0,a_i)，为重算 [a_i,L_i) 取得私有或合法独占写区
14 绑定程序、有效 token 绝对位置、mask、块表/槽映射、输入游标和采样状态
15 返回计划；padding 不得进入 softmax，也不得被发布为已完成 KV
```

q_i 个 token 必须重新执行逐层模型，包括投影和相应 KV 生成；只复制旧 KV 不是重算。批次组成会改变共同 suffix 桶及每请求搬移量，单请求计划不能直接拼成合批计划。完整命中但缺少 logits 时，普通全注意力模型可以重算末 token；混合递归模型需要全部状态族的共同恢复条件，不能据此倒退递归状态。本算法只产出绑定计划，多层执行、状态发布与失败恢复由相关服务和缓存章节展开。

例如来源中的 L=900、H=640，原始 suffix 为 260，选 b_s=512，搬移 252 个已命中 token，得到 a=388、q=512、b_p=512。它减少 prefix 容量，却增加有效重算，不构成成本最优性证明。固定指南明确 NxD Inference 不自行实现前缀哈希、缓存淘汰或内存管理，需要外部提供块表和槽映射；说明路径为 block→flat→attention→scatter，转换和暂存必须计费。

服务侧程序身份至少依赖模型图、shape/stride/dtype、量化布局、TP、LNC、目标设备和编译器版本/flags；这不是 NxD 内部缓存键的声明。权重若为运行时参数且 ABI、表示和布局不变，不必因每次数值替换重编译；常量折叠、预打包、分片或尺度布局变化则须重查依赖。程序可复用不代表旧权重生成的 KV 可复用。

token 步消费位置 p 的输入，完成后才发布该位置 KV 并令已完成长度成为 p+1；采样得到 token 本身不使其 KV 自动完成。持久 KV 留在设备可以减少往返，但仍有输入、输出、采样和控制依赖。模型并行时，状态分片、GQA head 映射、collective 顺序和程序组必须一致。固定 NxD 指南要求 attention heads 可被 TP 度整除，TP 度属于目标平台支持集合。设备级 96 GiB HBM、8 个物理 NeuronCore 与 LNC2 下每逻辑 core 24 GB 是不同层级和单位，不能把设备总空闲量当作任意 rank 的局部可用量。

## 计算、交通、准备、同步与驻留成本

稠密单 head 的 QK、PV 主算术为 2LSd+2LSdv FLOPs，另有 mask、指数、行归约和旧 U 重标度。若完整矩形先算再 mask，数学有效对数减少不代表指令工作同比减少。旧 U 每块至多处理 Bq·dv 个元素，全序列约为 L·ceil(S/Bk)·dv 量级，空块和首块可减少工作。

query tile 在外、K/V 不跨 query tile 保留、无 spill 时，HBM 逻辑交通主项为：

D_Q=L d s_Q+ceil(L/Bq) S(d s_K+dv s_V)+L dv s_O。

这里 Q 只读一遍，每个 query tile 重读 K/V，输出写一次。该式忽略 cache、对齐、事务放大、padding、页表、显式 mask、尺度和 LSE，不是总 IO 或无条件 DRAM 下界。改变循环顺序或让 K/V 跨 query tile 驻留，必须重算 Q、统计量和输出的读取以及活跃集合。

以分数/P 共用一份逻辑存储、保留旧 U 为基线，片上载荷主项可写成：

C_base≈Bq d s_Q+D(Bk d s_K+Bk dv s_V)+Bq Bk s_P+Bq dv s_U+Bq(s_m+s_l)。

这不是完整分配证明。本节伪代码还保留独立 tileU，因此要按生存期增加 Bq·dv·s_tileU；转换同时保留 W 源和 P 目标时再增加对应副本。矩阵执行阶段的 PSUM 输出、对齐和 epilogue 也另列。数学上的 U_new 不一定要求独立物理数组：在 tileU 可读且旧 U 的使用已经完成后，可以就地写 U，随后完成该行逻辑状态提交。

取 BF16、d=dv=128、Bq=16、Bk=128，Q 为 4 KiB，K、V 各 32 KiB，FP32 分数/P 为 8 KiB，FP32 U 为 8 KiB，m/l 合计 128 B。C_base 单槽为 86,144 B＝84.125 KiB；K/V 双缓冲为 151,680 B＝148.125 KiB。独立 FP32 tileU 再加 8 KiB，在没有额外转换副本同时存活的设定下分别为 92.125、156.125 KiB。QK/PV 的 PSUM 输出和实际转换峰值仍须单独检查，不能把这些载荷直接除以 SBUF 或 PSUM 总容量推断驻留数。

同块关键路径是加载/布局可读→QK→PSUM 交接→mask/归约/指数→P 布局可读→PV→PSUM 交接→状态提交。计算下界受 TensorE，搬运受 DMA 和端口，非矩阵工作受 Vector/Scalar/GpSimd。独立工作可以重叠，但同一 DMA 上的 copy 和 transpose 需求应合并计账。槽位等待已包含在完整内核时间内时，不能再作为额外同步重复相加。更大的 tile 减少循环或 K/V 重读，却扩大分数、U、转换和单槽占用，可能减少并行驻留。

准备成本包括编译、权重布局、程序加载、桶搜索、请求归属、块表/槽映射和描述符。首次准备、按形状复用与逐请求准备有不同摊销边界。同步包括输入可见性、矩阵完成、PSUM 交接、collective、状态发布和失败排空；逻辑状态提交不意味着每块增加一次主机全设备同步。单核片上搬运没有网络 payload；跨核并行则另计激活、部分和或 KV 交换、通信缓冲和组完成依赖。

HBM 容量账必须使用同一执行组和时间范围。设 4 个逻辑分片，每片供服务使用的 HBM 为 20 GiB；48 GiB 权重均分后每片 12 GiB，加持久状态 4 GiB、程序/运行时 1 GiB、通信/workspace 峰值 2 GiB，每片共 19 GiB，剩 1 GiB。前缀执行再需每片 2 GiB flat 暂存时达到 21 GiB，配置失败；其他分片空闲不能补偿不可迁移的局部峰值。

再加载一套每片 0.5 GiB 的特化程序，会把原余量降至 0.5 GiB；并发加载新旧权重需要重新安排排空或放置。0.5 GiB 是设定值，实际 NEFF/运行时占用须测量。完整程序族可近似记作 N_prog=Σ_vΣ_batch(|B_s|·|B_p|+|B_tg|)，v 覆盖实际依赖变体；共享、裁剪或其他阶段会改变数量。二维桶组合是两轴数量的乘积，不能笼统称为随桶数指数增长。

并行度增加时还要重算 KV 复制、collective 和同步。设一层计算从 3 ms 降至 1.8 ms，新增不可隐藏通信 1.5 ms，层时间反升至 3.3 ms；若 prefill 可隐藏其中 1.0 ms，则为 2.3 ms。数字只是条件化说明，是否重叠要由依赖、链路和共享 HBM 争用确认。

## 条件变化、保留与放弃

| 条件变化 | 可保留的机制 | 调整或放弃的选择 | 重算内容 |
|---|---|---|---|
| batch、prompt 或历史变化 | 绝对位置、mask、在线状态定义 | 换桶；必要时重编译或拒绝超范围输入 | padding、物理 tile、程序驻留和尾延迟 |
| dtype 改为 FP8 | 逻辑矩阵与 FP32 累加目标 | Double-FP8 的张量第二维，即最外层（最慢变化）free 维必须为 2；不能叠加互斥模式 | 配对、补零、尺度、转换和误差 |
| d 或 Bk 超过普通收缩块 | 同一分数和加权和定义 | 拆收缩块；完成 Z 后才归约，完成 tileU 后才交接 | 累加次数、PSUM 生存期和输入重读 |
| DMA 已饱和 | 在线合并与输入可见性条件 | DMA transpose 可能加重等待；比较 TensorE 路径 | DMA、TensorE、PSUM 和交接 |
| Scalar/Vector 为慢端 | m/l/U 及最后读者协议 | 有 DMA 余量时比较 DMA transpose | 非矩阵占用、端口和暴露等待 |
| SBUF/PSUM bank 紧张 | 数学函数和合法坐标 | 缩 tile、减少槽、改变累加位置 | 分配、bank、临时量、spill 和驻留 |
| 前缀命中稀疏或转换昂贵 | 计算身份和合法状态范围 | 比较关闭缓存或更粗桶 | gather/scatter、flat 峰值和完整重算 |
| TP 或 LNC 改变 | 逻辑 head 映射和模型函数 | 重分片权重、状态和 collective | 局部容量、复制 head、通信和程序身份 |
| 编译器或 flags 改变 | 模型源定义 | 不假定旧程序可复用 | 编译、加载、验证和驻留 |

这些调整不能改变合法 key 集合或归一化。数据表示合法、误差/质量达标、完整执行更快是三个不同结论。没有脱离形状、依赖和资源的固定最优 tile、槽数或转置路径。

## 诊断、反例与区分实验

应固定输入、模型、dtype、质量契约和计时边界，只改变一个条件，同时记录有效工作、物理 tile、实际字节、资源等待与完整调用时间。

| 竞争解释 | 可区分预测 | 干预与反证修正 |
|---|---|---|
| HBM 带宽主导 | 相同工作下降低 HBM 字节能缩短时间 | 改 K/V 复用；字节下降而时间不变时转查归约、DMA 或同步 |
| DMA transpose 拖慢 | 计算端等待下降，DMA 等待上升 | 分别在 DMA 有余量和饱和时比较完整路径；饱和后变慢则修正选择 |
| TensorE transpose 更合适 | DMA 等待下降，但 TensorE/PSUM 占用上升 | 连同特殊值语义、PSUM 交接和 eviction 比较；局部转置快不足以证明整体快 |
| padding 浪费 | 相同有效 token 下，宽桶稳定执行更慢 | 分开冷加载和热执行；差异只在首次调用时先查加载 |
| 程序驻留/加载主导 | 窄桶少做计算但切换成本抵消收益 | 固定请求分布测冷、热及复用次数；重算准备摊销和容量 |
| 主机往返主导 | 设备工作不变，往返方式改变总时间和主机空档 | 分段记录准备、传输、执行和同步；不把总时间变化归于设备算术 |
| mask 混入 padding | 只改 padding poison 就改变输出 | 固定有效输入，修改 padding 分数与 V；有差异先修 mask |
| 旧 l/U 漏重标度 | 跨块最大值升高后偏离整行参考 | 用分数 [0,10]、V 单位向量；独立参考应拒绝 alpha=1 的负对照 |
| m 未共同写回 | 两个等分数块仍丢掉旧贡献 | 用 [0,0]、Bk=1 的闭式反例；同时查每块 m/l/U，不能只测最大值变化 |
| PV 消费早于布局或结果交接 | 操作顺序违反生产者→消费者依赖 | 删除布局可见事件或让更新早于 PSUM→SBUF，依赖检查应拒绝；设备上再核对实际事件与指令 |

独立数学参考应直接枚举整行或已处理前缀的有效 key，用稳定 softmax 计算统计与输出，不调用被测在线递推。逐块比较 m、l、U 能发现只比最终 O 可能遗漏的重复或空块残留。对布局，独立按逻辑 query/key/value 坐标做点积，再比较 partition/free 供数；非对称小矩阵可以使错误配对产生确定差异。

随章 tools/check_k_hw_008_full.py 是待执行的只读、离线 Python 标准库 CPU 检查：Decimal 整行/前缀参考核对在线状态、O 和 LSE，覆盖因果、padding、尾块、空 query、空 key、全空行、夹在非空块间的空块和极端有限分数；非法零块长及非正/非整数维度在循环前拒绝。负对照分别遗漏 m 写回、遗漏旧状态重标度、混入 padding，要求观察到偏差。整数矩阵检查只核对逻辑 QK/PV 坐标；顺序检查只核对布局、可见性、矩阵完成和 PSUM 交接的抽象依赖，包含故意交换生产消费顺序的负对照。

这些检查不加载 NKI/NxD，不执行 NEFF，不验证真实 FP8 编码、Trainium/GPU 数值、DMA 内存可见性、bank 分配、并发或性能。抽象依赖检查能发现非法次序，不能证明真实设备实现已经满足依赖。实际执行状态由检查记录说明；本轮没有设备测量或 Agent 使用效果证据。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NKI：Trainium2 Architecture Guide for NKI](../sources/S-NKI.md)：固定 Trainium2 NKI 资料区分 HBM、SBUF、PSUM、DMA 与计算引擎，给出每 NeuronCore-v3 的 SBUF 28 MiB（128 partitions×224 KiB）、PSUM 2 MiB、设备 96 GiB HBM 与 8 个物理 NeuronCore。Double-FP8 最大 [128,2,128]/[128,2,512] 输入、张量第二维即最外层（最慢变化）free 维必须为 2、FP32 累加、两半收缩配对，以及与 column tiling、sparse matmul、transpose mode 的互斥来自该资料。DMA transpose 的两种路径、2/4 字节 bit-accurate 边界、功能与最佳吞吐条件的区别、TensorE 特殊值转置、硬件 DGE 不支持 indirect gather/scatter，以及引擎并行访问限制也来自固定资料。源码式 API 示例未复制为正文实现；正文的坐标关系、attention 顺序、状态共同提交、依赖模型和成本账为本库推导。
  [完整原文](../../references/S-NKI/reader.html) · [全文文本](../../references/S-NKI/fulltext.txt)

- [S-NXDI：NxD Inference Features Configuration Guide](../sources/S-NXDI.md)：固定资料支持 tracing→HLO→NEFF、持久缓存的编译器/配置/flags 条件、默认不序列化分片权重、LNC2 每逻辑 core 24 GB、TP 支持度数与 head 整除条件、GQA 的 KV 复制策略、context/token generation 分桶、外部块表/槽映射和 block→flat→attention→scatter 路径，以及二维 suffix/prefix 桶选择与四个示例。正文将这些版本事实与服务侧程序身份、完整命中输出资格、绑定计划和容量推导分开；SelectContextBinding 不是 NxD 原子事务 API，也不是完整模型执行器。
  [完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

本轮来源阅读依据所供完整全文提取和来源说明，不代表检查过 reader.html 图版或上游源码归档。相邻章节仅按所供全文或摘录范围阅读，摘录不构成通读原章。正文公式、容量设定、反例和 CPU 检查是独立推导，不冒充来源实测。前缀管理边界只适用于固定版本，不代表所有 Neuron 后端。来源事实受 2026-04-30 cutoff 约束，编辑日期不改变事实日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [NKI：Partition 轴、片上生存期与 Attention 数据流](../frameworks/nki-tiling.md)。
- [Neuron/NxD：静态程序承接动态生成](../frameworks/neuron-serving.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
- [通信传输：注册、完成、可见与回收](transport-and-rdma.md)。
- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
