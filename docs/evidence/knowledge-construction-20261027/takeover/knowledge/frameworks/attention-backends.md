# K-FW-006：Attention 执行接口：可见性、几何、状态布局与执行模式的交集

一次 attention 调用的接口是四个独立维度的交集：每个 query 对哪些 key 可见、Q/K/V 的几何、K/V 在存储中的布局、准备与执行怎样分段。任何库、后端或自写 kernel 都要在这四个维度上与模型语义一致；它们之间的差别只在成本发生在哪个频率上。性能排名只对确定的运行点成立。

## 问题、不变量与设计轴

问题：引擎持有的是逻辑状态——每个请求的绝对位置、有效长度、页表或偏移、head 映射、量化尺度——而 kernel 接受的是张量、索引数组与标志。要回答的是：怎样把逻辑状态变成 kernel 输入，每次调用付出哪些工作，哪些工作能在多次调用间摊销，换一个 kernel 时什么必须原样保持。

任何解都要满足的不变量：每个 query 行实际读取的 key 集合恰好等于模型定义的集合，包括因果对齐、窗口、填充与自定义规则，多读或少读都是另一个函数；query 长度与 KV 长度、Hq 与 Hkv 的映射、QK 维与 VO 维各自独立表达，不能靠相等假设省略；每个逻辑位置的 K、V 与量化尺度一致，一个写入槽位在同一时刻只有一个写者；scale、softmax 稳定化、累加精度与分区合并在约定容差内保持同一函数；拆开计算的 key 子集互不重叠且并集完整。

成本项：转换（gather、转置、dtype 变换、mask 打包）、准备（消费 metadata、划分工作、选择实现）、执行（kernel 本身、输出与统计量的分配、事后缩放）、驻留（workspace、metadata 缓冲、布局副本）、同步（host 读取设备侧长度、图外提交）。每一项都要问它在哪个频率上发生：每任务、每模型加载、每形状、每层、每 batch 还是每 decode 步。

设计轴与已有方案所在的点：状态表示——稠密 padded、ragged 偏移、分页索引，padded 简单且图友好但付无效行，分页用间接寻址换容量；准备与执行的边界——每次调用全部重算、按几何缓存计划、把工作划分写成固定缓冲中的数据，摊销边界本身是设计变量；布局——页内 token 优先或 head 优先，K/V 合并或分离，生产者按消费者布局写出、消费前转换或 kernel 直接读生产者布局；可见性表达——布尔 mask、结构参数（因果、窗口）、块稀疏索引，结构参数最省也最弱；分区与合并——不拆，或沿 KV 拆再合并统计量。轴上未列点同样合法；下面各节按轴展开已知点的前提与代价。

## 先写出共同函数

接口必须表达 Q/K/V 的 head 数与通道、query 和 KV 各自长度、逻辑位置、mask/偏置、scale、dtype 和输出形状。分页、ragged 和稠密只改变存储表示；如果接口不能表达所需可见性，则速度再高也不是等价替换。

现有实现按"实现选择在哪里发生"分成三类，它们的成本结构不同，不是同一种东西的不同品牌。框架级算子把一个数学定义（参考实现见 [S-SDPA](../sources/S-SDPA.md)）交给运行时按输入条件选择融合路径或朴素路径：调用方不控制也不一定知道走了哪条，融合路径各有输入限制，不满足时静默回退到慢得多的朴素实现；它的代价是分派不透明，收益是零接入成本。融合 attention 库提供一族按特定数据流写好的 kernel（分块、在线 softmax、沿 KV 切分再合并），调用方选择入口并保证输入满足该入口的几何与布局；代价是每个入口有自己的对齐与布局约定，收益是数据流确定、性能可预期。面向服务的分页/变长库再把一次调用拆成依赖 metadata 的准备段与依赖张量的执行段，用索引数组表达分页与共享，用固定 workspace 承接动态划分；代价是准备段的 host 工作与 workspace 驻留，收益是动态批进入静态图。三类之外，自写 kernel 落在同一组轴上：它必须回答同样的问题，只是把选择权全部拿回来。库名不等于模型和平台的完整支持矩阵，具体函数比库总览更有决定性。

## 从零实现时的算法骨架

无论采用哪类实现，设备上执行的都是下面两种循环之一的变体；写自己的 kernel 或读懂别人的 kernel 都从这里开始。精确 attention 的 HBM 访问下界是 Θ(N²d²/M)（M 为片上存储容量，证明见 [S-FLASH](../sources/S-FLASH.md)），任何实现的流量目标只能逼近它。

prefill（query 多、KV 长）按 tile 做在线 softmax。以 Q 为外层：对每个 Q tile（Br 行）载入 Qi，初始化 Oi = 0、ℓi = 0、mi = −∞ 驻留片上；对每个 K/V tile（Bc 行）：Sij = QiKjᵀ，m_new = max(mi, rowmax(Sij))，P̃ = exp(Sij − m_new)，ℓi = e^{mi−m_new}ℓi + rowsum(P̃)，Oi = e^{mi−m_new}Oi + P̃Vj，mi = m_new；全部 K/V 遍历完后 Oi /= ℓi，连同 Li = mi + log ℓi 写回。每个 Q tile 读完整 K/V 一遍，因此 HBM 读取约为 Tr × 2Nd × b，Tr = ⌈N/Br⌉；Br 越大遍历越少，但 Qi、Oi（FP32）、Sij/P̃（Br×Bc）与 K/V tile 的多级缓冲都要放进片上容量与寄存器。以 K/V 为外层的变体让 K/V tile 驻留、每步把 Oi 写回 HBM，遍历次数不变但驻留对象不同，并行度也不同。

decode（query 很少、KV 很长）的并行度来不自 Q tile，而来自把 KV 切成块：每块独立算出 (Oc, Lc)，再按合并算子 O = Σ e^{Lc−m}Oc / Σ e^{Lc−m}、L = m + log Σ e^{Lc−m}（m = max Lc）合并。块数由并行需求决定：一种可参数化的划分是先定每块成本 cost = α·lq + β·lkv，取 KV 块长上界 Lkv = Σ_请求 ⌈lq/Tq⌉·lkv / CTA 数，把所有块按长度降序逐个分给当前累计成本最小的 CTA（见 [S-FLASHINFERPAPER](../sources/S-FLASHINFERPAPER.md) Algorithm 1）。代价是部分输出的 workspace（最多 2×CTA 数份）与一次合并 kernel；短 KV 的请求不切块、直接写最终输出。合并顺序固定则结果确定；用原子累加会丢掉这一性质。

在矩阵单元远快于指数单元的硬件上（比值可达两个量级以上），上面循环里的 softmax 成为串行瓶颈。缓解方法是让第 j+1 块的 QKᵀ 与第 j 块的 softmax 重叠、再让第 j 块的 P̃V 与第 j+1 块的 softmax 重叠（两级流水），代价是多保留一份 Br×Bc 的 FP32 分数片段；再深一级在实测中因寄存器压力反转。加载与计算分给不同 warp 并用环形缓冲交接，是同一依赖的另一种组织。这些属于硬件流水，展开见 [GPU 流水](../hardware/gpu-kernel-pipelines.md)，来源实例见 [S-FLASHATTENTION3](../sources/S-FLASHATTENTION3.md)。

分页 KV 对上面两种循环只改变一件事：K/V tile 的地址不再是 base + j·Bc·stride，而是 base + page_index[j]·page_stride + 页内偏移；把页表看成行块为 query tile、列块为页的块稀疏矩阵，稠密与分页就统一到一种索引结构。代价是索引读取、额外分支与末页变长处理，一篇来源在 2023 年的实现上测得 attention kernel 慢 20–26%（[S-PAGED](../sources/S-PAGED.md) §7.1）；收益是容量浪费被限制在每请求最后一页内。共享前缀可以作为更大行块的另一个块稀疏矩阵单独计算，再与各请求独有部分按合并算子合并，不搬动数据。

## 非等长 causal 是一个具体反例

设 Q 长 2，K 长 5，Q 对应历史尾部位置 3、4。期望第一行可见前 4 个 K、第二行可见 5 个。常用接口对非方形因果 mask 有两种对齐约定：右下对齐给出期望结果；左上对齐让第一行只看 1 个、第二行只看 2 个。同一个布尔参数在两类接口上含义不同，直接互换会静默改变函数。应以绝对位置关系构造等价 mask，或选择明确支持相应对齐的接口。

另一个常见差别是布尔 mask 的 true 含义：不同算子可能分别表示保留或屏蔽。dropout 概率也可能由算子参数直接控制，不能假定外部推理模式自动将它清零。模型语义优先于同名参数。

## 用契约表组织选择

| 场景 | 必须能表达的内容 | 决定额外成本的项 |
|---|---|---|
| 稠密 prefill | 因果/非因果、变长 query | padding、tile、是否物化 mask |
| 分页 decode | 页表、tail、Hq/Hkv | 准备段、间接读取、split-K |
| Cross-attention | 独立 Q/KV 分段与源可见性 | 投影缓存、独立几何 |
| MLA | latent 与位置分量 | 是否展开历史、专用归约 |
| 图执行 | 稳定地址及允许更新的 metadata | 捕获外准备、workspace |
| 窗口/稀疏 | 每 query 合法历史集合 | 索引和无效 tile |

## 适配器应承接什么

一个清晰的适配层可把引擎逻辑状态转换成 kernel 输入，说明是否发生 gather、transpose、dtype conversion 或额外 copy。这样的转换可能比 kernel 本身更贵，benchmark 应覆盖它们。fallback 也应保持同一数学契约，不能在不支持窗口时静默改为全 attention。

不同 prefill/decode 实现可共享持久缓存，但布局、量化 scale 和写入约定必须一致。跨实现比较使用固定张量与可见关系，再评价端到端成本，才能区分数学接入错误和资源效率差异。

## 分区 attention 合并的额外不变量

若同一 query 的合法 K/V 集合拆成互不重叠的 I、J，且每一部分使用相同 score 定义和位置/数值解释，可以保留归一化输出 O 与 log-sum-exp l。取 m=max(l_I,l_J)，权重 a=exp(l_I−m)、b=exp(l_J−m)，合并输出为 (aO_I+bO_J)/(a+b)，新的 l=m+log(a+b)。用 max 做平移避免直接 exponent 大 l 溢出。

互不重叠是必要条件：共享前缀已经在 I 中出现，就不能在 J 的完整历史中再计一次，否则它的概率质量被重复增加。只有输出 O 而没有归一化尺度，通常不能通过普通平均恢复整体 attention。空分区、全 mask 行、LSE 是自然对数还是其他底、scale/softcap 的顺序也必须统一。

共享 prefix 的多个 query 仍有各自 Q，复用主要是 K/V 的读取与存储；不能共享本应不同的 attention 输出。分页表示本身只说明存储间接寻址，稀疏可见性若删除了原函数允许访问的位置，则是另一个质量敏感算法，不能从“同为块稀疏表示”推出精确等价。

## 转换成本与可复算反例

设一层 K/V 各有 L=8,192、Hkv=8、d=128、FP16，合计 2×8,192×8×128×2=32 MiB。若每次调用先完整 gather 到连续布局，需要读取旧布局并写入新布局，单是转换至少产生约 64 MiB 有效字节，之后 attention 还会读取新缓冲。更快的连续 kernel 只有在抵消这些转换、额外驻留与同步后才构成收益。

独立小例中，对同一 query 设两个 logits 都为 0，V 分别为 0、2，则整体输出为 1。各分区输出为 0 和 2、LSE 同为 0，合并得 1；若第二分区同时包含第一个位置，错误合并会重复计数，得到 2/3。对 Q 长 2、K 长 5 的因果输入，再显式验证两行可见集合，而不只比较输出 shape。

接入测试应覆盖 GQA 的 head 映射、QK 与 VO 维度差异、非整页、窗口边界、prefix/suffix 不重叠、无可见 key、位置偏置和 quantization scale。对同一逻辑状态依次记录转换、准备、kernel、合并和输出路由的耗时/内存，并用独立稠密数学参考验证。长历史、低 batch 时分区可能增加并行度；短历史或转换占主导时，简单路径可能更快。选择应由接口交集与完整执行成本决定。

## 准备与执行：成本发生的频率

接入成本从逻辑请求状态可用时开始，到输出可被下一组件消费时结束。面向服务的接口普遍把工作分成两段：依赖 metadata 的准备——读取或整理分段与页表信息、划分工作、选择实现、更新缓冲——和依赖张量的执行——可能分配输出、准备布局、执行 kernel、缩放输出。把计时窗口只放在最后一个 kernel 上，不能代表这条路径。

准备段的成本来自 host 对 metadata 的消费：长度信息已在 CPU 还是必须从设备读回、mask 是否已预先打包、图模式下是否要把更新后的元数据复制到保留缓冲。一个“非阻塞”标志不会消除这些工作，只是改变它们与设备工作的重叠方式。

准备段能否复用，由它真正依赖的条件决定：几何、batch 成员、分段、KV 长度、mask 与实现选择。多层共享同一几何时，一次准备可以服务多层；条件改变后旧计划不再有效。这是一个摊销边界，是设计变量，而不是“初始化时发生一次”的常数。把每步成本当成一次性初始化会虚增收益，把一次性成本重复计入每步会错误淘汰方案；可复算的摊销算例见下文。

执行段的辅助工作同样可观察：未传输出缓冲时的分配与清零、额外统计量（如 LSE）的分配、某些路径在主 kernel 之后另行应用的缩放。它们影响完整调用时间与临时驻留，应与 kernel 时间分开报告。

## 数据驱动的工作划分与图执行

工作划分决定每个 CTA 处理哪些 query 行与哪些 key 块。若划分结果直接决定 launch 的 grid 形状，那么 KV 长度变化就会改变 grid，图重放失效。另一条路是把划分结果作为数据写入固定容量的工作队列，由固定 grid 的 kernel 读队列执行；动态任务由此进入静态图，代价是队列容量上界、空闲 CTA 以及分区合并的额外输出。

这条轴上有两个必须分开判断的性质：归约顺序是否确定（影响逐位可复现）与 launch 配置是否固定（影响图兼容）。固定划分粒度可以让归约顺序确定，却不保证 CTA 数不随长度变化；固定 grid 也不保证合并顺序确定。从一个标志推出另一个性质是常见错误。

容量上界属于计划：最大 query 行数、页索引数、末页长度与部分输出数量都要按所选路径建立上界；超界时切换到另一张图、普通执行或拒绝该计划，不应截断页表让重放勉强运行。

## metadata 的单位与实际分派

页表和分段数组不能仅凭名字与 shape 对接。query 的分段偏移可以表示 token 行偏移，另一条接口可能表示元素偏移；页偏移表示页索引数组中的区间，页索引再指向物理页，末页长度则以有效 token 计。把其中两个量当成同一单位，在小 batch 或某个特殊 head 数下可能偶然不出错，换形状才暴露问题。同一入口下的不同实现分支也可能用不同单位核对输入，适配不能只替换一个名字。

“自动选择实现”通常不是每次调用都重新寻找当前最佳实现：实现会在准备段解析设备、位置模式、归约方式、mask 与 dtype 条件，把结果固定在对象内部供后续执行使用。输入能力组合变化后，需要确认既有对象仍适用，必要时建立另一配置或显式选择路径；之前成功运行一次不构成对后来所有条件的支持。

可用路径还可能直接拒绝某些组合（布局、因果模式、构造方式）。这些是函数与版本的边界，不能推导成整个库或整个硬件不支持该语义；替代路径可能是另一接口、另一实现或受控的布局适配。反过来，某库存在该能力，也不表示当前调用实际选择了它。

## cache 布局、workspace 与并发必须共同设计

页内 token 优先与 head 优先只是轴顺序，K/V 的容器还可能是合并的五维张量或两个四维张量。逻辑页表中的索引关系不因 transpose 自动改变；变换时必须保持每个逻辑位置对应的 K、V 和 scale 一致。只转换数据、漏掉量化 scale 或末页有效长度，会产生无法用 shape 检查排除的错误。

“支持两种布局”不等于“两种布局的成本相同”。对普通可接受 stride 的数据，transpose 可以只是视图；对要求连续或带块级 scale 的量化分支，完整复制是真实成本。若 prefill 写入一种布局、每步 decode 又转换整段历史，转换量随已驻留 KV 而非本次新增 token 增长。

缓存写入有接口特定的别名条件：在原地更新缓存的接口上，同一槽位的重复写入是否有确定结果由该接口规定；共享只读 prefix 与多个逻辑更新写入同一槽位具有不同含义，不能从“接口接受索引数组”推断任意重复写入都有确定结果。独立尾槽、依序更新或另一种缓存更新实现可以改变写入依赖。

workspace 影响能否同时运行多个批次。分区归约的中间结果、整型与 pinned host 工作区、可变计划状态都占用容量；建议容量不是所有形状的硬下界，也不是空闲时没有驻留成本的承诺。共享同一 scratch 或复用同一可变对象时，需要以实际读写区域和完成顺序解释能否复用；不同 stream 的输出彼此独立不足以说明内部辅助状态独立。

图模式把这个问题放大：对象生命周期内 batch 容量固定、metadata 缓冲保留、准备段不能被捕获。活跃请求数仍可通过显式分桶或其他合法组织方式变化，但引擎必须保证 inactive 槽、实际长度和页表内容的约定一致，不能把固定外形当成固定请求状态。split 配置、输出分配与 workspace 地址也必须满足当前捕获路径，见 [Stream 与分配器](streams-allocator.md) 与 [CUDA Graph](cuda-graph-buffers.md)。

## 两个可复算的接入反例

第一个反例是 GQA 的物化扩展。设 Hq=32、Hkv=8，某层 FP16 K/V 长度 L=8192、d=128，则逻辑 KV 为 32 MiB。若通用路径把 K/V 真正复制为 32 个 head，临时 KV 表示成为 128 MiB，并增加生成该表示的读写。直接消费分组 head 的内核有机会避免这一扩展；但文档中的等价数学表达出现 repeat，并不能证明选定实现实际物化了它。需要通过实际分派、分配与设备流量判断，不能仅按概念公式给库记账。

第二个反例是准备段的摊销。设两个候选除公共工作外，A 每次 kernel 为 20 μs，B 为 12 μs，但 B 的一次合法准备为 160 μs。若同一计划只使用一次，B 为 172 μs；若确实能供 32 次相同几何调用使用，平均为 17 μs，才优于 A。若每次调用还伴随 6 μs 的不可摊销 metadata/输出工作，B 的平均又成为 23 μs。所有时长都是设定值，用于展示成本结构；真实多层或跨步复用是否合法，须先通过接口契约判断。

这两个例子说明适配器需要记录成本发生的频率：每任务、每模型加载、每形状、每层、每 batch 或每 decode 步。把一次准备成本重复计入会错误淘汰方案，把每步成本当成一次初始化则会虚增收益。

## 用同一逻辑状态比较实现

比较时先固定绝对位置、合法 key 集合、head 映射、K/V 内容和数值策略，使稠密参考与候选表达同一个问题。然后分别测已准备输入上的 kernel、真实适配后的完整调用，以及与投影、cache 更新和输出消费连接后的模型层。让 causal 对齐、GQA 分组、分页尾页和窗口边界逐项改变，可以区分数学接入错误与分派条件变化。

性能变化若只出现在首次形状，优先检查编译、准备、分配与缓存；若随历史长度线性增加而 kernel 时间未相应增加，检查 gather、布局转换及 metadata 路径；若并发执行后正确性失效，检查 workspace、可变对象状态和缓存写入是否共享。观测应对应可否定的解释，而不是把所有慢路径统称为性能差。

最后回到完整服务的吞吐、延迟与容量。某路径节省 attention 时间却多占一份历史缓存，可能减少可容纳的并发请求；另一条路径单次略慢却避免 CPU 同步，可能改善调度重叠。结论只能覆盖已测模型、硬件、长度与并发条件。短历史小 batch 下简单稠密路径可能占优，长历史小 batch 下分区并行占优，大 batch 下准备段摊销充分而转换成本显现；workspace 与副本对 KV 容量的挤占见 [已知张力](../system-semantics/known-tensions.md)。

## 诊断线索

| 现象 | 能区分解释的检查 | 对应机制 |
|---|---|---|
| 换接口后短序列正确、长序列或多分区输出漂移 | 固定输入比较分区合并前后的 m、ℓ；核对 LSE 底数与缩放方向 | 分区合并不变量 |
| 非方形 prefill 的首行或末行输出错误 | 用绝对位置构造期望可见集合，逐行比对 | causal 对齐约定 |
| 首次形状调用远慢于后续，形状变化时再次变慢 | 分别计时准备段与执行段；改变 batch 成员观察是否重新准备 | 准备段摊销边界 |
| 延迟随历史长度线性增长而 kernel 时间不变 | 计量 gather/转置字节是否随驻留 KV 增长 | 布局适配位置 |
| 显存按 Hq/Hkv 倍增长 | 检查实际分派是否物化扩展 K/V | 几何不变量与物化反例 |
| 并发两批后出现错误结果 | 单批正确、双批错误时检查 workspace 与可变计划状态是否共享 | workspace 并发 |
| 图重放后输出混入上一批 | 核对 metadata 缓冲的代次与在途读取 | 时间所有权 |
| 某组合直接报错不支持 | 区分函数边界与库/硬件边界；换接口、换布局或受控适配 | 分派与被拒绝组合 |

<!-- infraagents:provenance -->

## 来源与推导

- [S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../sources/S-FLASH.md)：分块在线 softmax 的算法、Θ(N²d²/M) 的 HBM 访问复杂度与精确 attention 的下界，是"算法骨架"一节与分区合并不变量的数学来源。
  [原始 PDF](../../references/S-FLASH/original.pdf) · [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

- [S-FLASHATTENTION3：FlashAttention-3](../sources/S-FLASHATTENTION3.md)：矩阵与指数吞吐的量级差、生产者/消费者分工、两级 GEMM-softmax 流水及其寄存器代价、三级流水的反转，是"算法骨架"一节硬件流水段的来源。
  [原始 PDF](../../references/S-FLASHATTENTION3/original.pdf) · [完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：块化 attention 的地址关系、块大小权衡与块化 kernel 20–26% 的额外开销，是"间接寻址换容量"这条交换的实测价格。
  [原始 PDF](../../references/S-PAGED/original.pdf) · [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-SDPA：PyTorch：scaled_dot_product_attention](../sources/S-SDPA.md)：框架级算子的参考定义（mask 极性、非方形因果左上对齐、GQA 整除条件、运行时在三种实现间分派并可能回退），是"实现选择在运行时发生"这一类的实例，也是对齐反例的一个端点。
  [完整原文](../../references/S-SDPA/reader.html) · [全文文本](../../references/S-SDPA/fulltext.txt)

- [S-FA-IMPL：FlashAttention 官方实现说明](../sources/S-FA-IMPL.md)：非方形因果右下对齐及其版本变更、短 query 时沿 KV 切分再合并、原地更新缓存时重复槽位写入结果不确定，分别是对齐反例的另一端点、decode 划分轴与"写入槽位单写者"不变量的来源实例。
  [完整原文](../../references/S-FA-IMPL/reader.html) · [全文文本](../../references/S-FA-IMPL/fulltext.txt)

- [S-FLASHINFERPAPER：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](../sources/S-FLASHINFERPAPER.md)：attention 状态与合并算子的定义、页表作为块稀疏矩阵、共享前缀的可组合格式、负载均衡划分算法与固定 workspace 的图兼容运行时，是"算法骨架"decode 段与"数据驱动的工作划分"一节的来源。
  [原始 PDF](../../references/S-FLASHINFERPAPER/original.pdf) · [完整原文](../../references/S-FLASHINFERPAPER/reader.html) · [全文文本](../../references/S-FLASHINFERPAPER/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：源码中准备段与执行段的实际分工、三类 metadata 的单位、布局转换何时是复制、workspace 构成与图模式条件，是"准备与执行""metadata 单位""workspace 并发"三节的来源实例。
  [完整原文](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：异步提交、stream 顺序与分配器生存期是两个独立关系、图私有内存池的共享条件，用于区分 host 提交、设备完成与缓冲的时间所有权。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

- [S-AITER：ROCm AITER 官方仓库](../sources/S-AITER.md)：一个库在实现层面是多条后端路径（块程序、模板 C++、手写汇编）的集合，各有形状、dtype 与设备条件，是"库名不等于支持矩阵"的来源实例。
  [完整原文](../../references/S-AITER/reader.html) · [全文文本](../../references/S-AITER/fulltext.txt)

正文中的对齐反例、合并反例、转换字节与摊销算例是基于这些来源的推导，全部设定值均明确声明，不冒充实测；引用的 20–26% 与两个量级吞吐差是来源论文在其硬件上的报告，不是本库实测。来源版本受 2026-04-30 上限约束，编辑日期不是事实日期。

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](../models/mla.md)。
- [基础算子：从张量契约到合法融合](../algorithms/operator-primitives.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
