# K-MODEL-003：MoE：路由、Dispatch/Combine 与专家负载

MoE 用条件计算增加模型容量：每个 token 只执行少数专家，不同 token 可以访问不同专家。激活 FLOPs 的节省不等于相同比例的驻留或读取节省。专家权重、路由重排、跨 rank 往返、热点队列和在途缓冲共同决定真实服务成本。

## 一次 MoE 层的函数、输入与身份

设一个 EP 组有 R 个 rank。origin o 本轮负责 B_o 个 token，输入为 X_o[0..B_o−1,d]，组级 token 数 B=Σ_o B_o。本地序号 t 只在 origin 内唯一，完整 token 身份为 τ=(o,t)，origin(τ)=o。两个 rank 都可以拥有本地 token 0。若上游 TP 复制了 hidden，进入本算法前必须为每个逻辑 token 选定唯一 dispatch origin，不能把复制的 hidden 当成多个独立 token。

一层有 E 个 routed expert，每 token 选择 k 个不同专家，另有 N_s 个 shared expert。router 对 x_τ∈R^d 产生 E 个 logits 或 affinity，选中集合为 S_τ，最终门控乘数为 g_{τ,e}。routed 分支为

z_τ = Σ_{e∈S_τ} g_{τ,e} E_e(x_τ)。

shared 分支为 q_τ=Σ_a E_a^{shared}(x_τ)。若层边界包含 residual，输出为 h_τ=u_τ+q_τ+z_τ，其中 u_τ 是模型规定的 residual 输入，不一定等于归一化后送入专家的 x_τ。下文设定算例令 u_τ=x_τ。归一化位置、激活函数及额外缩放必须按模型制品定义；若存在 routed scaling，它应进入最终 g 一次。

一次层调用用不可复用身份 G=(attempt,layer,routing_generation,placement_generation) 标识。routing_generation 绑定 router 参数、选择 bias、tie 规则和本轮冻结的选择结果；placement_generation 绑定专家副本目录及权重版本。两种 generation 可以独立编号，不要求数值相等。对每个选中项再冻结目的 rank π_G(τ,e) 和权重版本 v_e。单副本时目的只依赖专家；多副本分流时，同一专家的不同 token 可以选择不同目的。

| 对象 | 内容与作用域 | 必须保留的关系 |
|---|---|---|
| logical item | I=(G,τ,e)，以及 g、目的 rank、v_e | 一个选中专家贡献的唯一身份 |
| dispatch packet | G、τ、hidden、一个或多个 item 描述符 | hidden 与所有描述符属于同一 token 和快照 |
| expert bucket | 接收 rank 上专家 e 的输入行及描述符 | 重排后仍能恢复 origin、t、e、g 和版本 |
| return partial | G、τ、sender、覆盖集合 C、版本和加权向量 | 明确代表哪些 logical item |
| origin 完成状态 | Expected[τ]、Seen[τ]、部分和和 shared 状态 | 预期贡献完整后才能发布 |

物理 slot、bucket 行号和本地 t 都不能单独作为跨 rank 返回身份。重新执行、取消后复用 slot 或切换放置表时，旧结果仍属于旧 G。

本算法假定一个目的端能完成一个完整专家函数。若专家本身做 TP，目的应扩展为专家执行组，先在组内完成必要归约，再返回一份专家贡献；其分片通信另计，不能把每个分片当成独立专家再次 gating。

## 选择、门控与合法的实现空间

Mixtral 固定版本采用 G(x)=Softmax(TopK(xW_g))：在 logits 中选择 top-k，其余置为负无穷，再做 softmax。对选中集合，稳定计算为 g_e=exp(ℓ_e−m)/Σ_{j∈S}exp(ℓ_j−m)，其中 m=max_{j∈S}ℓ_j。Mixtral 8x7B 有 8 个专家，每 token 选 2 个，专家为 SwiGLU；论文区分约 47B 总参数与约 13B 每 token 激活参数。

DeepSeek-V3 v1 使用 sigmoid affinity s_e=1/(1+exp(−ℓ_e))，以 s_e+b_e 决定选择，却用原始 s_e 在选中集合上归一化：g_e=s_e/Σ_{j∈S}s_j，再按制品要求应用额外缩放。选择 bias 不进入 FFN 输出的乘法权重。DeepSeek-V2 的 softmax affinity 与 V3 的 sigmoid 后选中归一化也不能混用。

普通 top-k 可以通过按“选择分数降序、固定平局键升序”排序并取前 k 个构造。全排序需要 O(E log E)，选择算法或小顶堆可减少工作，输出集合仍须符合相同契约。对于模型规定的分组限制，先按该规则构造允许专家集合，再在其中选择；不能把不受限 top-k 的结果随意替换成距离较近的专家。

合法性至少包括：

- 0≤k≤E，选中专家不重复。k=0 是空 routed 分支的算法边界，固定 top-2 或 top-8 模型不能为省计算临时改变 k。非有限分数、归一化分母为零等情况须按数值契约处理。
- 平局规则应固定并与参考一致，例如相同分数按 expert ID 排序。不同 rank 或副本不能各自改写已冻结的选择。
- shared 专家独立贡献一次。把它作为总会命中的 heavy-load routed 项参与部署统计，不改变函数语义。
- capacity factor 与 token dropping 属于模型允许的语义。固定形状装不下时，可以研究变长 grouped GEMM、padding 或分块；静默删掉必需项会改变输出。V3 v1 报告训练和推理均不丢 token，不能推广为所有 MoE 的规则。
- V3 v1 的 node-limited routing 将每 token 限制到至多 M=4 个节点，节点按其最高 K_r/M 个专家 affinity 的和选择。分组、评分、bias 与最终选择的具体执行次序仍须与制品一致。模型的专家分组和物理副本目录是不同对象；副本迁移后，应按实际 π_G 重新核对物理通信范围，不能机械沿用 M 的流量上界。
- 副本只能执行同一专家函数和同一权重版本。分流选择副本不等于改选专家；同一 item 发到两个副本并都合并会重复贡献。
- collective 必须满足对应操作的 communicator、count、dtype、root/归约方式和调用顺序契约。变长 dispatch 需要计数交换、固定容量或等价协议，不能把不等长桶直接交给等 count 操作。零 origin token 的 rank 仍可能执行远端专家项。

## 从路由快照到返回提交的完整路径

每个 origin 只为自己的 token 建立 Expected；接收 rank 则执行来自所有 origin 的专家项。下面采用先完整接收本轮 dispatch、再执行专家的概念算法。流式实现可以提前处理已就绪桶，但仍须保留同样的身份、覆盖集合和结束证明。

hidden 发送去重与返回预合并是两个独立开关。发送去重让同一 (τ,目的 rank) 只携带一份 hidden，仍保留全部专家描述符；返回预合并在目的端把同一 token 的多个加权专家结果合成一条 partial。两者均不允许删掉第二个专家。

```pseudocode
输入：各 origin 的 X_o、residual U_o，模型路由规则，冻结快照 G
      专家副本目录、权重版本、dispatch_dedup、return_premerge
输出：各 origin 的完整 H_o

初始化：
    各 rank 的发送箱、接收队列、专家桶、返回箱为空
    对本轮每个 token：routed_sum=零向量，Seen=空集
    本轮错误状态为空；保留输入、输出槽和快照的访问权

# 1. origin 路由并冻结预期贡献
for origin o:
    for t in 0..B_o-1:
        τ=(o,t)
        计算 logits 和模型规定的选择分数
        按模型允许集合、k 和 tie 规则构造 S_τ
        按模型公式计算最终 g_τ；选择 bias 不混入输出 gating
        Expected[τ]=空映射
        for e in S_τ:
            从 G 的合法副本中选定唯一目的 r=π_G(τ,e)
            I=(G,τ,e)
            desc=(I,g_τe,destination=r,weight_version=v_e)
            Expected[τ][e]=(r,v_e)
            将 desc 加入 origin 的本轮 item 清单
        冻结 Expected[τ]；启动模型规定的 shared 分支

# 2. 先构造完整 packet，再使每个 packet 入队一次
for origin 的每个 desc:
    若 dispatch_dedup：加入 packet[G,τ,r] 的描述符列表
    否则：为该 I 单独建立 packet
    packet 的 hidden 来自 X_origin(τ)[τ.t]
for 已构造完成的每个 packet:
    若 destination==origin(τ)：放入本地接收队列
    否则：放入 send_box[origin(τ),destination]

交换每个 peer 的 packet 数和 item 数，预留接收容量
按匹配顺序提交远端发送与接收；零项 rank 仍履行协议

# 3. 本地和远端接收项经过同一个入桶路径
for 接收 rank r 上的每个 packet:
    等待 hidden 与全部 metadata 完成并对专家消费者可见
    校验 G、发送 origin、τ、目的 r 和副本目录
    for desc in packet.descriptors:
        校验专家与权重版本；拒绝同一 I 重复入桶
        expert_bucket[r,desc.e].append((desc,packet.hidden))
        recv_plan[r,τ].add(desc.e)
等待本轮计数协议证明全部 dispatch 接收完成
若缺项、容量失败或元数据错误：本轮失败，不提交部分输出
for r:
    按专家组长计数并做前缀和，得到 offsets[r,e]
    将各桶 hidden 行打包为 X_r,e；同行保留 desc 映射

# 4. 专家执行与逐项加权
for rank r:
    for expert e hosted on r:
        n_r,e=该桶的有效行数
        若 n_r,e==0：继续
        等待 G 指定的 v_e 权重完整且可读
        V_r,e=对 X_r,e 执行专家函数，得到 n_r,e×d 输出
        for row j in 0..n_r,e-1:
            desc=该行原描述符
            c[desc.I]=desc.g×V_r,e[j]        # gating 恰好应用一次
            标记该 I 的专家计算完成

# 5. 目的端产生结果并显式发送回 origin
for rank r:
    若 return_premerge：
        for 出现在 recv_plan[r] 的 τ:
            C=recv_plan[r,τ]
            等待 C 内全部专家项完成
            v=按固定 expert 顺序累加 c[G,τ,e]，e in C
            建立 reply=(G,τ,sender=r,covered=C,versions,v)
    否则：
        for 本 rank 完成的每个 I=(G,τ,e):
            建立 reply=(G,τ,sender=r,covered={e},versions,c[I])
    对每个 reply：
        o=origin(reply.τ)
        若 r==o：交给 origin 的本地 combine 接收队列
        否则：加入 return_box[r,o]
交换返回计数并提交匹配的返回发送/接收，持续推进至完成或失败

# 6. origin 校验覆盖、累加和提交
for origin o 的每个本地或远端 reply:
    等待 reply 完整且可见
    校验 reply.G 属于活跃本轮，origin(reply.τ)==o
    C=reply.covered；要求 C 非空
    对 e in C：校验 Expected[τ][e] 的目的与版本匹配 sender、versions
    逐 item 模式要求 |C|=1
    预合并模式要求 C 等于 Expected[τ] 中分给该 sender 的完整专家集合
    要求 C 与 Seen[τ] 不相交；非预期、重复或重叠贡献使本轮失败
    routed_sum[τ]+=reply.v
    Seen[τ]=Seen[τ]∪C

收齐并校验本轮计划的返回后：
    对每个 token τ，只有 Seen[τ]==keys(Expected[τ])、shared 已完成且无错误：
        H_origin(τ)[τ.t]=U_origin(τ)[τ.t]+shared[τ]+routed_sum[τ]
        发布完整输出
    缺项者继续等待；截止或传输失败时使本轮失败，不能发布部分和

# 7. 交接与回收
按下一层 TP/EP 的复制或分片契约重分布已完成输出
停止新访问，等待已提交传输、专家读者及 combine 消费者全部结束再回收
取消使输出不可提交，但不能跳过其他 rank 仍在等待的匹配操作
```

origin 的完成条件是专家覆盖集合相等，不能只数消息条数。逐 item 返回有 k 份贡献；预合并后可能只有一条消息。收到覆盖 {e_1} 的结果却把它当成 {e_1,e_2} 的完整 partial，会在消息数看似正确时少算。覆盖集合可以编码成列表、位图或快照中约定的 group ID，不要求总是传显式集合。

一个 I 依次经过“已计划→已提交→接收可读→已入桶→专家完成→已返回→origin 已计入”。发送 API 返回、接收可读、专家完成和 token 完成是不同状态。G 防止旧轮结果进入新输出；版本字段验证权重身份，不能代替 DMA 完成和设备可见性。

这里的恰好一次约束针对本轮无重试执行。若增加重传，必须另定义重放和去重协议。流式发布时也不得让重复或迟到消息再次修改已提交槽。失败后的跨进程恢复不是这份有限状态算法已经验证的能力。

B_o=0 的 rank 没有本地输出，仍须运行接收、专家和返回路径。全组 B=0 时，只有全体共同约定后才能跳过相关通信。空专家组长为零，不产生有效 FFN 行；k=0 时 Expected 为空，routed 分支满足空集合完成条件，仍需等待其他分支。

浮点预合并改变加法树，实数等价不保证逐位相同。可固定组内 expert 顺序与 origin 的 sender 顺序，并按模型允许误差比较；精确有理数模型只验证代数、身份和覆盖关系。

## 两个 origin 都有 token 0 的组合轨迹

设 rank 0、1 各有一个 token，rank 2 的 B_2=0。令 A=(0,0)、x_A=1，B=(1,0)、x_B=2。专家函数设为 E_e(x)=(e+1)x，没有 shared，residual 等于输入。专家 1、2 放在 rank 2，专家 3 放在 rank 1。全部 item 绑定同一 G 和对应权重版本。

A 选专家 1、2，权重分别为 1/4、3/4；B 选专家 1、3，各为 1/2。

| 阶段 | rank 0 | rank 1 | rank 2 |
|---|---|---|---|
| 冻结预期 | A：{1→2,2→2} | B：{1→2,3→1} | 无 origin token |
| 去重 dispatch | 向 2 发 A，携带两个专家描述符 | 向 2 发 B 的专家 1；专家 3 本地接收 | 专家 1 桶接收 A、B 两行；专家 2 桶接收 A 一行 |
| 专家计算 | 等待返回 | B 的专家 3 加权贡献为 4 | A 的贡献为 1/2、9/4；B 的贡献为 2 |
| 目的端预合并 | 接收 (A,{1,2},11/4) | 接收 (B,{1},2) | 按两个完整 token 身份分别发回 origin 0、1 |
| 完成提交 | Seen_A={1,2}，h_A=15/4 | Seen_B={1,3}，h_B=8 | 最后消费者结束后释放 |

rank 2 的 partial 必须按 (G,origin,t) 分组。若按 t=0 分组，两份不同输入会混在一起。若交换专家 1 两条返回的 origin，两边的专家覆盖集合仍可能完整，却得到 h_A=21/4、h_B=13/2。覆盖检查因此需要与逐 token 直接公式配合。

若 B 的远端返回丢失，rank 1 只有本地专家 3 的贡献，Seen_B={3}；即使本地 FFN 和全部 dispatch 发送都已完成，也不能发布 h_B。若去重 packet 只保留 A 的一个描述符，接收 item 计数或 origin 覆盖检查必须失败。

在途时发布新 placement generation，不改变此轨迹中旧 G 的目的端。旧副本与输入要保护到最后消费者结束。取消 A 后，迟到结果仍属于旧 G，不能写入新请求同为本地 token 0 的输出槽。

## 成本账：计算、读取与真正的网络发送

对三矩阵 SwiGLU，按行向量表示专家函数：

E_e(X)=[SiLU(XW_gate,e)⊙(XW_up,e)]W_down,e，

其中 W_gate、W_up 为 d×f，W_down 为 f×d，SiLU(a)=a/(1+exp(−a))。忽略激活和 epilogue，单 item 主项为 6df FLOPs，单专家权重为 W_e≈3df b_w 字节，b_w 是权重元素字节数。

定义 N_items=Σ_τ|S_τ|。routed 算术主项为 6df N_items，固定 k 时为 6df Bk。相同宽度的 N_s 个 shared 专家另加 6df N_s B；专家宽度不同时逐项求和。router 线性层约为 2BEd FLOPs，另有选择、归一化和数值转换。

设 d=4096、f=14336、BF16 权重 b_w=2，则每专家为 336 MiB，8 个专家共 2.625 GiB。B=16、k=2 时有 32 个 routed item，算术主项为 11,274,289,152 FLOPs，约 11.274 GFLOP。任一专家最多接收 16 项；[16,4,3,3,2,2,1,1] 是合法偏斜分配，平均 4 不能代表慢端。

若每个 token 独立均匀选择 k 个不同专家，一个专家从未出现的概率为 (1−k/E)^B，因此期望触及专家数为

E_touch=E[1−(1−k/E)^B]。

E=8、B=16、k=2 时约为 7.9198。全部冷权重、每个触及专家只读取一遍的设定下，逻辑读取约为 2.599 GiB；按 active fraction 2/8 估计只得到 0.65625 GiB。这个账本不包含 tile 重读、cache 命中或多个副本分别读取，不能当作 HBM 事务。副本场景应按实际触及的 (rank,expert,version) 计权重读取。真实路由相关性也不能用独立均匀假设替代。

### 本地逻辑体积与跨 rank payload

以下统计整个 EP 组的端点发送 payload，每次传输在发送端计一次，不再把同一 payload 的接收重复加一次。跨 rank 包括节点内设备通信，不等于全部经过跨节点 IB；协议头、计数消息、padding、重传和下一 TP 布局转换另计。

定义远端 item 数

N_remote=Σ_τ Σ_{e∈S_τ} 1[π_G(τ,e)≠origin(τ)]，

以及远端 token-destination 对集合

P_remote={(τ,r)：r≠origin(τ)，且存在 e∈S_τ 使 π_G(τ,e)=r}，D_remote=|P_remote|。

设输入 hidden 每元素 b_h 字节，返回向量每元素 b_y 字节：

| 方案 | dispatch 网络主项 | 返回网络主项 |
|---|---:|---:|
| 逐 item 发送、逐 item 返回 | N_remote d b_h | N_remote d b_y |
| hidden 去重、逐 item 返回 | D_remote d b_h | N_remote d b_y |
| 逐 item 发送、目的端预合并 | N_remote d b_h | D_remote d b_y |
| hidden 去重、目的端预合并 | D_remote d b_h | D_remote d b_y |

返回列的 D_remote 假定每个目的端对同一 token 的完整组只返回一次。若按 chunk 分多次返回，必须用实际远端返回条数 D_ret，不能继续假设 D_ret=D_remote。

包含本地项的 N_items d b_h 只能称为“逐 item 逻辑输入体积”。令 D_all 包含本地 token-destination 对，则 D_all d b_h 是“共享 hidden 后的逻辑输入体积”。本地可能只建立引用而不复制，这些量既不是网络发送，也不直接等于本地内存事务。

全部专家均在各 token 的 origin 执行时，N_remote=D_remote=0，dispatch 和返回网络 payload 都为零。单 rank、一个 token、d=4096、b_h=2，即使 D_all=1，网络也不是 8 KiB。

前述双 origin 轨迹有 4 个 item，其中 3 个远端 item、2 个远端 token-destination 对。取 d=4096、b_h=b_y=2，每向量为 8 KiB：逐项往返共 48 KiB；只预合并返回或只去重 hidden 均为 40 KiB；两者同时使用为 32 KiB。rank 1 的本地专家 3 不进入网络账。

gating、origin、t、expert、G、版本、offset 和覆盖信息另占 metadata。hidden 去重后描述符仍随 item 数增长；返回预合并还需累加和完成状态。若 32 个 item 中 24 个远端且不去重，dispatch/return 各为 192 KiB，合计 384 KiB 加 metadata，这是带明确前提的另一个算例。

### 准备、同步、驻留与关键路径

计数分桶可接近 O(N_items+E)，比较排序通常为 O(N_items log N_items)，还需计数缓冲、前缀和、索引、输入重排和输出映射。接收 rank 的 n_{r,e} 来自全部 origin，不能用其本地 B_r 估计专家输入。动态接收容量不足时须分块推进或失败，不能静默删项。固定图重放与动态组长之间还需比较 padding、动态描述符和额外分支。

dispatch 接收可见、专家权重可读、专家完成、返回可见与 origin 提交是不同同步边界。发送调用返回不代表对端已可读取，也不代表发送缓冲立即可复用。小 decode 消息可能受启动和协调延迟支配；较大 prefill 更容易摊薄固定项。

驻留账包括可用专家权重、shared 与冗余副本、origin 输入、接收 hidden、metadata、专家中间激活、返回 partial、完成集合及 grouped GEMM workspace。未融合的两次上投影会产生约 2Σ_{r,e}n_{r,e}f 个中间元素；融合或分块可缩短生存期，但网络去重不能证明这些中间量消失。峰值取同一时刻存活对象之和的最大值，避免重复计共享 hidden，也不能提前释放仍有读者的对象。双缓冲使多个微批的生存期重叠。

时间由依赖图和共享资源决定。每 rank 的算术/有效算力、设备字节/有效带宽给出候选下界；拓扑上每条边的实际流量/有效带宽给出拥塞下界。总发送字节不能直接除全机总带宽。若采用 IB 到节点入口、再经 NVLink 转发，端点 payload 之外还存在转发边流量，须按具体路径映射。通信与计算重叠只隐藏有独立工作覆盖的等待，不消除字节、占用或同步。

MLA 减少 attention KV 状态，不减少 MoE 专家权重。腾出的容量可以用于副本或更大 batch，但专家并集、通信和热点需要重新计算。

## Grouped GEMM 与专家负载

专家 e 在 rank r 的输入为 X_{r,e}[n_{r,e},d]。grouped GEMM 统一组织不同组长的矩阵乘法，减少逐专家启动和小矩阵碎片；它不改变有效 item 的函数，也不会自动消除热点队列。

可比较逐专家小 GEMM、各组 padding 后执行和按真实组长调度 tile。设行 tile 粒度为 a：

R_slots=Σ_{r,e}ceil(n_{r,e}/a)a，R_true=Σ_{r,e}n_{r,e}。

空组贡献零槽；R_true=0 时不定义 padding 比。对 [16,4,3,3,2,2,1,1]、a=4，行槽为 44，真实行为 32，比值为 11/8；全部专家补到最大组长则为 128 行槽，比值为 4。这是两个不同候选，不能把全最大 padding 的成本套给逐 tile 取整路径。减少无效行可能增加描述符、间接访问和调度；固定 shape 则可能更利于图重放，需比较完整调用。

只计权重，单专家算术强度主项为 6df n_e/(3df b_w)=2n_e/b_w FLOP/byte。BF16 下约为 n_e：小组更像低复用 GEMV，大组更容易重用权重。该比值没有包含输入输出、中间激活、tile 重读和 epilogue。热点复制能缩短单副本队列，也会拆小组长并增加权重副本读取，因此负载更均衡本身不证明层更快。

## 副本、卸载与重叠的组合条件

副本目录属于 placement generation，实际 π_G(τ,e) 属于本轮 item 计划。部署调整先准备新权重并验证版本，再发布新目录；旧轮继续引用旧目录，最后消费者结束后才能释放旧副本。迁移不能让已经提交的 item 临时改目的，也不能丢失返回 origin。

DeepSeek-V3 v1 按在线统计周期调整冗余专家。prefill 最小部署单元为 4 节点 32 GPU，attention 使用 TP4+SP、DP8，MoE 使用 EP32；32 个冗余专家使各 GPU 在原 8 个专家之外再承载一个。decode 报告的最小部署单元为 40 节点 320 GPU，attention 使用 TP4+SP、DP80，MoE 使用 EP320，每 GPU 承载一个专家，64 个 GPU 承载冗余或 shared 专家。它们是该 H800 集群的部署策略，不是模型理论最少硬件或通用最优布局。

容量不足时，可以搬冷权重到 GPU，也可以把冷专家计算留在 CPU。前者传 W_c 字节权重再执行 GPU 专家；后者传输入和返回结果，并支付 CPU 计算、准备和合并。串行粗模型分别为 W_c/β+T_GPU 与 A/β+T_CPU+T_sync/combine，其中 A 按所选去重及返回模式计算。能否重叠再按实际依赖处理。小批可能使 A 远小于 W_c，但 CPU 算力、主存争用或 batch 增大可能让选择反转；若全部权重能常驻 GPU，搬运项又消失。

预测下一层专家只能作为预取提示。下一层 hidden 尚未得到时，预取覆盖率和无用流量要入账；真实路由选到未预取专家仍须需求加载或正确执行。等待冷专家期间，输入、返回槽和请求状态继续驻留。

通信重叠需要能独立推进的微批、足够缓冲和匹配操作顺序，每个微批仍等待自己的完整 combine。V3 v1 报告 prefill 同时处理两个计算量接近的微批，以一个的 attention/MoE 覆盖另一个的 dispatch/combine；动态冗余与部分 decode 两微批重叠仍属探索。训练 DualPipe 的前反向重叠不能直接当成推理收益。

取消先停止新提交，再排空已提交通信、CPU/GPU 计算和结果消费。业务请求终止不代表物理缓冲立即可重用。generation、引用关系与真实完成事件分别负责身份、访问权和已提交访问是否结束，不能互相替代。

## 条件变化后的选择与反例

| 条件变化 | 保留的约束 | 调整或放弃的候选 |
|---|---|---|
| B 很小 | 选中专家和 gating 不变 | 重算专家并集、启动开销及低复用读取；大 padding 可能不合算 |
| B 增大 | 每项贡献仍恰好一次 | 权重更易摊薄，也可能触及更多专家、扩大 KV 和接收缓冲、加重热点 |
| 副本分流改变 | 同专家同版本函数不变 | 按实际 π_G 重算远端 item、远端对、每副本组长和驻留 |
| 全部专家在 origin 执行 | 直接公式不变 | 跨 rank dispatch/return 应归零，本地重排与计算仍存在 |
| 链路变慢 | 模型路由规则不变 | 研究合法副本布局、hidden 去重和返回预合并；不能临时发明节点限制 |
| dtype 改变 | 身份、覆盖与完成关系不变 | 分别重算输入/返回字节、累加误差、scale metadata 和转换成本 |
| 在线热点漂移 | 训练平均不保证短批负载 | 更新副本候选，保留无副本对照，计入迁移成本和生效滞后 |
| 卸载或并发微批增加 | 输入输出依赖不变 | 重算冷权重、共享带宽和在途峰值；必要时减少并发或分块 |

## 区分错误归属、漏项与性能瓶颈

| 竞争解释 | 区分实验与观测 | 反证后的修正 |
|---|---|---|
| 返回按本地 t 混合 origin | 两个 origin 都有 t=0，hidden 与 gating 不同且命中同一远端；交换同 expert 返回并对照逐 token 公式 | 覆盖完整仍可能数值错误，必须保留完整身份与独立数值参考 |
| 接收项未入桶或结果未返回 | 记录 dispatch、入桶、执行、发送返回和 Seen 的逐项账；删除一个远端返回 | 输入发送完成不足以证明层完成，缺覆盖不能提交 |
| 预合并漏掉第二个专家 | 同 token 两专家同目的，保留全部描述符和 covered，对照逐专家和 | hidden 去重正确不证明专家覆盖完整 |
| 网络公式包含本地项 | 全本地和混合放置分别列发送账，要求每条 src≠dst | 全本地仍预测正流量时修正计量边界 |
| 小组主要受权重读取限制 | 固定路由扫描 B、每专家 n_e 和冷/热权重，记录字节及时间 | 读取摊薄而时间不改善时，调查启动、队列和通信 |
| 热点形成慢端 | 保持 S、g、权重不变，改变同函数副本分流，记录组长、最慢 rank 和返回等待 | 队列下降但层时间不降时，检查入口拥塞、combine 和更小 GEMM |
| padding 主导 | 同组长比较逐专家、tile 取整和全最大 padding，分别记录真实行和行槽 | 行槽下降而时间不降时，调查调度、元数据和资源驻留 |
| bias 混入 gating | 改 bias，分别检查选中集合及原始 affinity 归一化权重 | 选中集合不变但权重随 bias 改变时，修正门控函数 |
| 版本或缓冲复用错误 | 新旧 G 交错、取消或切换目录，检查迟到结果和最后读者 | generation 正确仍不能证明设备可见，需核对真实同步 |

配套检查路径为 `tools/check_k_model_003_full.py`。该路径以普通文字列出，检查文件由本轮独立交付；正文不建立指向尚未物化脚本的 Markdown 链接。脚本用 CPU 精确有理数的逐 token 公式对照多 rank 消息模型，并以实际建模发送事件对照远端计数公式。覆盖双 origin 同名 token、空 origin 的远端接收、hidden 去重、两种返回模式、全本地、混合、热点、副本分流、shared、空批和 k 边界。错误 origin、漏返回、重复贡献、旧 generation、错误版本、漏描述符、覆盖式 combine 和本地对误计网络均作为负对照。

检查另以组合概率递推验证独立均匀路由的专家并集期望，以显式行槽枚举验证 padding。专家为合成函数，affinity 为供给值；这些检查不验证真实 softmax/sigmoid 数值、SwiGLU kernel、浮点误差、NCCL 执行、GPU 可见性、取消后物理排空、HBM/cache 事务或服务性能。真实设备实验应保持路由与质量契约可比，再记录消息、计数器、资源和关键路径。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MIXTRAL：Mixtral of Experts](../sources/S-MIXTRAL.md)：固定 2401.04088v1。第 2.1 节定义选中 logits 上的 softmax、8 专家/top-2、SwiGLU 和输出返回原 token 位置；效率讨论区分约 47B 总参数、约 13B active，以及 routing 和多专家读取开销。第 5 节报告较深层相邻 token 选择的时间局部性，既提示缓存机会，也提示 EP 热点风险。本文的多 origin 身份、覆盖完成条件、网络账本和 grouped GEMM 算例为独立推导。
  [完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：固定 2405.04434v5。MLA 以低维 latent 联合表示内容 K/V，通过线性投影吸收减少历史展开；位置旋转不能任意与投影交换，因此采用解耦 RoPE。每 token 跨 l 层缓存主项为 (d_c+d_R^h)l 个元素。该机制减少 attention 状态，不减少 MoE 专家权重；V2 的路由与 dropping 定义也不能直接替代 V3。
  [完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：固定 NCCL 2.25.1 文档。集合通信由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义；参与者须执行匹配的逻辑操作。AllGather 按 rank 排序，ReduceScatter 按 rank 分块。本文据此要求动态桶具有计数/容量协议、零项 rank 不擅自跳过操作；消息模型没有声称执行了 NCCL。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

正文中的公式展开、容量算例、消息状态机与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

### 本次扩写补充来源

- [S-DEEPSEEKV3：DeepSeek-V3 Technical Report](../sources/S-DEEPSEEKV3.md)：只使用固定 2412.19437v1。所供架构与部署段落支持 sigmoid affinity、bias 仅用于选择、原 affinity 归一化 gating、shared/细粒度专家、node-limited routing 和不丢 token。部署段区分 prefill/decode 的布局、在线统计驱动冗余和微批重叠；动态冗余及部分 decode 重叠仍属探索。训练 DualPipe、FP8 训练和 16B 消融专家图不作为本章目标推理系统性能证据。本文的原文核对限任务供给的提取文本摘录，不把来源说明中的全 PDF 阅读或图片核对记录算作本作者本轮完成。
  [完整原文](../../references/S-DEEPSEEKV3/reader.html) · [全文文本](../../references/S-DEEPSEEKV3/fulltext.txt)

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
- [权重与专家卸载：重用、路由与在途版本](../algorithms/weight-expert-offload.md)。
- [CPU–GPU 计算放置：传数据、传结果与共同提交](../algorithms/heterogeneous-execution.md)。
