# K-ENGINE-005：异步执行决策：结果确认、在途状态与回收

主机准备、设备计算、采样与外部状态传输可以部分重叠，但自回归依赖与存储生存期不会因异步提交而消失。需要分别记录设备访问是否结束、候选是否进入确认历史，以及每个外部消费者是否结束。一个互斥的 phase 不能同时表达这三件事。

TensorRT-LLM 的固定架构文档把入口、每 rank 的 PyExecutor、Scheduler、KVCacheManager、ModelEngine 与 Sampler 连成后台迭代。Overlap Scheduler 允许设备执行后继步骤时，主机处理前一步结果，并指出会多一个解码步骤的在途状态。下面给出满足这些依赖的一种保守设计；具体状态字段、提交事务、封闭协议和检查模型是本库推导，不声称等同于上游实现。

## 起始契约：已知、已算与已确认

核心算法限定为固定模型与 adapter、单条保留分支、普通 causal decoder、每次消费一个已知 token 的 decode。不覆盖 beam 重排、draft 接受、输入追加、原地更新的递归状态或跨进程恢复。多请求可以交错，但每个请求分别维护下述链，batch 行号不能代替请求身份。

位置从零开始，区间右开。规范化 prompt 为 x[0:n)，n>0；空输入先按模型合同补 BOS，或拒绝。新增 token 上限为零时直接结束，不发起模型工作。进入本算法时准备：

- 完整、可消费、受保护的全层 KV[0:n−1)。
- 最后一个已知 prompt token x[n−1]，以及位置、mask、页表、模型和表示身份。
- 第一次采样前的完整控制状态 Q₋₁，包括 RNG、重复惩罚历史和所有影响 token 选择的约束状态。

这是将 prefill 收敛到一个明确输入位置的入口，并不要求正常 prefill 必须逐 token 执行。若已获得 prompt 全部 KV 和合法末行 logits，可以直接采样首候选并记录等价入口票据；若全部 KV 命中但没有 logits，应只读历史 [0:n−1)，在私有写区重算最后一行，不能因为“没有缺 KV”而永久等待，也不能覆盖共享末页。这里展开的基线采用消费最后一行的路径。

请求 r 的步骤 E_g=(r,g) 消费位置

i_g=n−1+g，g≥0。

P₀ 消费 x[n−1]；P_g 在 g>0 时消费前驱候选 c_{g−1}。P_g 产生位置 i_g 的全层 KV 和预测下一位置的 logits，再采样得到位置 n+g 的 c_g。**c_g 自身的 KV 由消费它的 P_{g+1} 产生**；采样成功不会凭空多出一行 KV。

令 q 为已提交候选数，A 为确认 token 边界，K 为本算法已发布的全层 KV 边界。基线满足

A=n+q，K=n−1+q=A−1。

P_g 成功完成但未提交时，可以存在更远的私有 KV 与临时候选；它们不能推进 A、K 或对外输出序号。设备侧完整链的消费边界 B 可以大于 K，但 B 必须绑定具体候选链，不能当作公共确认边界。输出适配可能隐藏 EOS、缓冲停止串或合并多个 token，因此输出事件数量也不必等于 q。

## 从逐层前向到完整采样链

P_g 的输入快照包含请求实例与分支、g、i_g、输入 token、因果前缀身份、逐层 KV 视图、模型与数值身份、Q_in，以及本次私有写区和 workspace。g=0 时 Q_in=Q₋₁；否则 Q_in=Q_{g−1}，取自前驱已完成的临时票据，不能取尚未推进的正式 RNG。

下面以普通带残差的 decoder 表达数据流；norm、位置变换、attention 和 FFN 的具体顺序服从模型定义。

```pseudocode
算法 DEVICE_PACKAGE(g,snapshot)
输入：位置 i 的 token、全层历史 KV[0:i)、Q_in、私有段 W_g
输出：同链票据 T_g、逐层完整 KV 增量，或失败

h ← 该 token 在位置 i 的 embedding/输入表示
对模型规定的每层 l：
  等待本层输入 h、权重和历史 KV_l[0:i) 可消费
  由本层输入按模型规则得到 q_l、k_l、v_l，并施加位置规则
  将当前 k_l、v_l 写入 W_g 的本层位置 i；不覆盖共享历史
  attention 读取历史 [0:i) 加当前行一次，满足 causal mask
  经输出投影、残差、norm、FFN 等得到下一层 h
  保存本层写入及后继读取所需的完成依赖
末层 h 经 final norm/输出头产生位置 i 的 logits
按固定配置、Q_in 和完整 token 历史处理 logits、采样得到 c_g
Q_out ← 随机状态的实际推进及选中 c_g 后的约束/惩罚状态
汇合所有必要层、rank、KV 写入、采样与结果生产依赖
全部成功且对指定消费者可见后，产生 T_g：
  请求/分支/g、位置、父链身份、输入 token、Q_in、c_g、Q_out、
  模型/采样身份、私有 KV 描述、完整成功证书
必要访问真实结束的证据另用于结算设备责任
```

logits 就绪不能替代所有 KV 写入完成。例如输出支路已结束，而另一 stream 的 KV 拷贝还在写入时，不得把整步标为成功完成。多 rank 执行需要汇合所需参与者，不能用一个 rank 的 event 证明整请求完成；通信与设备可见性的适配条件见[通信传输](../hardware/transport-and-rdma.md)。

Q 不只是 seed。它至少包含 RNG 算法和状态或逻辑事件定位、采样配置版本、重复/频次统计、grammar 或其他会改变合法 token 集合的状态。一次采样可能消耗多个随机数，拒绝采样、过滤或实现变化都可能改变消耗数，不能固定假设“一 token 一次随机数”。票据应保存实际 Q_in→Q_out 转换及其身份。

后继继承的是完整临时状态链：

(input_g,Q_in,g的KV前缀) → (c_g,Q_out,T_g) → snapshot_{g+1}。

主机提交只将已验证的 Q_out 安装为正式 Q，不再采样或重新消耗随机数。主机可以决定停止、输出格式及提交资格；若它会修改候选、grammar、惩罚状态或随机轨迹，后继原输入链就失效，必须撤销该后缀并从确认边界重新建立。依赖外部可变状态的 CPU callback、无法提前复制的约束状态或不满足图安全随机状态合同的配置，应先关闭这种单请求候选重叠。即使 Dmax=1，跨请求的独立准备和计算仍可重叠。

## 正交状态与合法条件

每步分别记录以下状态，提交不改写物理完成事实。

| 状态 | 取值或内容 | 推进含义 |
|---|---|---|
| physical | SUBMITTED、ENDED | ENDED 只在必要访问真实结束后安装，并保留结束收据 |
| success | false/true | true 还要求本步所有必要产物完整、成功且可见 |
| business | PENDING、COMMITTED、DISCARD_PENDING、DISCARDED | 提交或丢弃资格，不能代替 physical |
| ticket | 完整 T_g 或已安全压缩的收据 | 后继输入与提交校验的依据 |
| accesses | 本包访问的对象和 uid | 每个真实结束只撤销自己的责任 |
| owners | 请求持有、临时链持有、缓存/传输持有 | 决定是否还允许未来访问 |

请求保存 next_submit、next_commit=q、正式 token 历史、Q、K、唯一终态和责任账本。LIVE 时 0≤next_submit−q≤Dmax；它限制尚未提交的候选步骤，**不代表已分配的全部 KV、输出或传输资源**。终止后 q 冻结，差值不再作为排空判断。

合法条件如下。

- 输入同一性：token、位置、mask、页表、模型/adapter、数值表示及 Q 属于同一请求实例和候选链。
- 临时依赖：后继可接受物理已结束且成功、业务状态为 PENDING 或 COMMITTED 的前驱；不能只接受名为 DEVICE_DONE 的互斥业务 phase。
- 顺序提交：只有 g=q 且输入链与当前确认历史、正式 Q 一致时才能安装结果。临时执行不是确认权。
- 完成覆盖：成功产物证书与最后访问结束证据分别核对；错误、取消、超时和主机 launch 返回都不自动结束访问。
- 存储隔离：临时 KV、采样输出、metadata 和 workspace 在最后使用前不能覆盖。共享尾页写入需要私有区域或 COW。
- 终止封闭：撤销新设备包和新业务提交权，但继续接收已有包、主机任务和消费者的结束事件。
- 回收门：对象的未来拥有权已关闭，且全部实际访问、输出借用、传输保护和隔离原因已解除，才能复用物理存储。代次检查不能阻止旧 DMA 实际写入。

T_g 的完整值至少保留到 H_g 完成且 COMMIT_g 或 DISCARD_g 已确定，并且后继快照已经取得所需值，或该链已关闭。后继快照可以复制 token、Q 和链描述；借用原地址则必须另登记读者。成功提交后的最后一个票据在尚未建立后继快照时仍须保留。旧票据压缩后保留身份、决定与必要摘要，不能使迟到通知再次获得启动或提交权。检查模型在有限轨迹中保留审计用合成值，不据此证明实际内存分配器的压缩策略。

## 有界推进、提交与独立排空

以下由一个状态拥有者串行处理事件；并行实现必须提供等价事务顺序。主机结果工作 H_g 在独立工作者上运行，不能占住唯一事件线程等待设备或消费者。快照和完成消息均携带请求实例、g、物理分配代次及责任身份。

```pseudocode
算法 BOUNDED_OVERLAP
输入：起始契约、Dmax≥1、停止规则、设备与主机工作者
输出：有序确认历史、唯一终态、已结算或明确未结算的资源

初始化：
  next_submit=0，q=0，LIVE=true，terminal=空
  正式历史=x[0:n)，正式 Q=Q₋₁，K=n−1
  steps=空；为入口历史建立请求持有

SUBMIT_NEXT：
  若非 LIVE：拒绝
  若 next_submit−q=Dmax：等待容量
  g=next_submit
  若 g=0：取入口 token、Q₋₁ 和 KV[0:n−1)
  否则：
    p=steps[g−1]
    要求 p.physical=ENDED、p.success=true、
         p.business∈{PENDING,COMMITTED} 且完整票据仍可取
    从 p.ticket 复制 token、Q_out、父链与逐层 KV 视图到新快照
    记录前驱已被后继取得；不要求前驱仍未提交
  原子预留本包完整可推进空间；不足则等待且不持有半份预留
  创建私有 KV 段、输出与 workspace，建立临时链持有
  建立 physical=SUBMITTED、success=false、business=PENDING 的记录
  先登记 device:g 对全部读写对象的责任，再授权一次提交
  next_submit 加一；前驱满足票据退休条件时压缩它
  确定未启动：用未启动证据结束本包责任，终止请求
  已接受或接受未知：保留责任；接受未知时终止并隔离

DEVICE_END(g,status,ended_certificate,result_certificate,ticket)：
  只处理身份匹配且 device:g 尚未结清的事件
  ended_certificate 不足：
    TERMINATE(ERROR)；保留 physical=SUBMITTED 与 device:g；返回
  幂等撤销本包登记的 device:g，physical=ENDED
  若 status 成功且 result_certificate 覆盖全部必要层、状态和采样：
    success=true；保存不可变 ticket
  否则：success=false；TERMINATE(ERROR)
  若 LIVE，唤醒后继提交和 TRY_HOST
  无论 LIVE 与否，执行 SETTLE_ALL

TRY_HOST：
  只考虑 g=q
  若非 LIVE、无成功完整票据或 H_g 已启动：返回
  为 H_g 的借用对象先登记 host:g；向工作者发出一次任务
  H_g 用候选与自己的快照计算停止、输出适配和提交提案
  H_g 不修改正式历史、Q、K 或输出序号

HOST_END(g,last_access_certificate,proposal)：
  匹配 host:g；证据不足时保留责任并 TERMINATE(ERROR)
  证据充分时只撤销 host:g
  若非 LIVE 或该步已失去 PENDING 资格：丢弃提案，SETTLE_ALL，返回
  在同一提交事务内检查：
    g=q；physical=ENDED；success=true
    输入位置=n−1+q，输入链等于正式历史
    Q_in 等于正式 Q，模型/采样/分支身份和父票据均匹配
    proposal 未改写设备候选或影响采样的 Q_out
  任一失败：TERMINATE(ERROR)，不安装部分结果
  全部成立则原子安装：
    追加 c_g；正式 Q←Q_out；K←i_g+1
    将本步 KV 的临时链持有转换为请求持有
    business=COMMITTED；q 加一
    登记连续不可变输出事件；借用缓冲先登记消费者责任
    若 EOS、长度或停止串决定终止：TERMINATE(对应原因)
  事务后才交付已提交事件；执行 SETTLE_ALL 并唤醒推进

TERMINATE(reason)：
  若已有终态：保留原终态；不恢复 LIVE
  否则 LIVE=false，保存唯一 reason，禁止新包与新业务提交
  对所有 PENDING 步设置 business=DISCARD_PENDING
  撤销尚未启动的任务；已有工作保留其访问责任
  关闭临时链的未来取得资格，撤销临时链持有
  本基线不保留缓存：撤销请求对私有 KV 的持有
  若有合法缓存或发送/保存交接，须先取得对应持有再撤销请求持有
  不改变 q，不删除任何尚未结束的 uid
  SETTLE_ALL

SETTLE_ALL：
  独立扫描全部已建立步骤，不要求 g=q 或前驱 COMMITTED：
    若 business=DISCARD_PENDING 且 physical=ENDED：
      business=DISCARDED；撤销候选的发布资格
    对已作出提交/丢弃决定且无未来取值者的票据执行退休
  对每个对象：
    只有未来拥有权关闭、实际访问/借用/传输与隔离原因均为空，才回收
  只有全部已提交设备包真实结束、业务决定均为终态、
       主机/输出/传输责任结清、待丢弃私有段已回收，才报告 DRAINED
  合法转交缓存的对象按声明基线另列；不能宣称它们已物理释放
  否则报告具体对象、责任、最后事件和解除条件，等待真实外部事件
```

错误结果可能已经结束全部访问，因此可以结算物理责任而拒绝产物；反之，错误码若没有访问结束证明，就不能把 physical 改成 ENDED。未知提交和部分启动也应按实际访问范围保留责任，不为未启动的消费者制造永远等不到的通知。取消与提交事务先后有明确含义：取消先发生则拒绝候选；提交先发生则保留历史，随后只撤销未来权利。

独立扫描不依赖业务前驱，是排空的关键。失败前驱可以从未 COMMITTED，后继仍必须接收真实结束并进入 DISCARDED。标记 DISCARDED 也不等于立即释放字节；后继、主机或外部读者仍可持有该对象。

## 持续生成与停止的一条组合轨迹

设 n=3、Dmax=2，入口已知 x[0:3)，全层 KV[0:2)。

| 事件 | 确认边界与临时状态 | 仍需保护的资源 |
|---|---|---|
| 提交 P₀ | 消费位置 2 的 x₂，Q_in=Q₋₁ | 入口历史、本步写区和 workspace |
| P₀ 成功结束 | 私有 KV 到 3；c₀ 位于 3，Q₀ 可用 | 票据及本步私有段保留链持有 |
| 提交 P₁，主机处理 H₀ | 消费 c₀、Q₀；读取 KV[0:3)，写位置 3 | P₁ 读权继续保护 P₀ 的段 |
| COMMIT₀ | q=1、A=4、K=3；正式 Q=Q₀ | P₀ 的物理 ENDED 事实保持；KV 转请求持有 |
| P₁ 结束并 COMMIT₁ | q=2、A=5、K=4 | c₁ 票据保留至后继取值 |
| 此后才提交 P₂ | 接受已 COMMITTED 且物理 ENDED 的 P₁ | 消费 c₁，写位置 4；不会因 phase 改名停滞 |
| 重复推进 | 每次提交归还一个候选额度 | 可以产生远超过 Dmax 的总 token 数 |

如果 COMMIT₀ 决定 EOS 或长度结束，而 P₁ 仍在运行，则 q 停在 1，P₁ 进入 DISCARD_PENDING。P₀ 与 P₁ 的私有段撤销请求/链持有，但 P₁ 的读写 uid 继续保护它们。P₁ 真正结束后只撤销自己的访问，进入 DISCARDED；若没有其他读者，两段才可回收。此时 next_submit−q 仍为 1，却可以已经排空，因此不能用该差值判断 DRAINED。

若 H₀ 校验失败而没有任何成功提交，处理相同：P₀、P₁ 都可被丢弃，后继结算无需等待 P₀ 成为 COMMITTED。反例是让排空复用提交前驱检查，或丢弃后不移动业务游标却继续仅查看该游标；两者都会留下永远没有后继动作的责任。

## 发送、保存与最后使用者

[S-TRTLLM-CODE](../sources/S-TRTLLM-CODE.md) 的固定机制说明描述了请求完成后的发送/保存保护：保留请求与传输元数据，每次实际开始登记使用者，完成时减少相应责任，全部结束后解除保护，并保留已有错误。加载走其他路径；该计数不能代表全部设备工作。

独立设计还需要明确“不会再登记新使用者”的时刻。若先进入保护阶段，send 完成后计数暂时归零，但 save 仍可能稍后启动，立即释放会使 save 读取被新请求覆盖的存储。

```pseudocode
算法 SEALED_TRANSFER_GUARD
状态：对象分配代次 epoch，open=true，guard_hold=true
      outstanding=空，completed=空，error=空

BEGIN(cid)：
  要求 open 且 guard_hold，cid 从未登记
  在启动前将 cid 加入 outstanding，登记它访问的对象范围
  确定未启动时，凭未启动证据结算这一项

SEAL：
  open=false；禁止登记新使用者
  未启动的计划项必须已撤销并保证不会启动
  TRY_UNGUARD

COMPLETE(epoch,cid,status,last_access_certificate)：
  旧代、未知或重复通知不改变其他责任
  若失败，保存最早错误，不覆盖已有错误
  若证据不足以证明本 cid 最后访问结束，保留 cid
  否则从 outstanding 移除 cid，加入 completed
  TRY_UNGUARD

TRY_UNGUARD：
  只有 open=false 且 outstanding=空时，撤销这一阶段的 guard_hold
  对象仍须等待其他拥有者、设备、主机、输出和隔离责任
```

失败但确实结束的传输可以结算自己的物理责任，同时保留失败结果；超时或失败通知本身不能充当最后访问证据。错误保留与是否仍在访问是两个轴。旧代通知不能解除新代保护，增加 epoch 也不能阻止仍在途的旧 DMA。

将原块复制到 staging 后提前结束原块保护是另一种选择：原块必须等复制最后读取结束，staging 必须等发送/保存结束。它增加一次复制和另一份容量，只有原块释放带来的准入收益超过新增关键路径及驻留成本时才值得。远端载入、SSD 副本和恢复所有权分别见[多级驻留](../algorithms/tiered-residency.md)、[状态传输](../system-semantics/distributed-state-transfer.md)和[服务恢复](../system-semantics/serving-recovery.md)。

## 重叠收益与六类成本

把每步设备前向及采样时间记为 D，主机结果处理记为 H，单位相同。假设阶段时间恒定、资源足够独立、输入快照可及时准备、Dmax=2、没有停止浪费或外部背压。N 步的两阶段理想完工时间为

T_pipe(0)=0，T_pipe(N)=D+H+(N−1)max(D,H)，N≥1。

串行基线为 N(D+H)。长序列吞吐增益上限为 (D+H)/max(D,H)，首步仍需 D+H。设 D=2 ms、H=5 ms，三步串行为 21 ms，理想流水为 17 ms；长序列周期为 5 ms，渐近增益约 1.4 倍。Dmax=1 强制前驱业务提交后才取得下一额度，基线退化为串行；更深缓冲只能吸收有限波动，不能消除持续的主机瓶颈。

实际 H 包括停止规则、结果校验和输出适配。必须在主机提交前产生的输入准备或采样约束应放回依赖链，不能藏在“异步”名称中。共享 HBM、PCIe、CPU 核或图实例会使 D、H 随重叠变化，常数阶段公式此时只是需要修正的模型。

| 成本 | 同一请求轨迹中的列账方式 |
|---|---|
| 计算 | 每个实际接受的 P_g 都执行前向和采样，包括停止后丢弃的步骤；重叠不减少模型算术。H、约束、日志和输出适配另计，padding 按无效行计浪费 |
| 访存 | 每步读取各层历史 KV，写入输入 token 对应的 KV，并读写 hidden、logits 和 workspace；临时状态复制、COW 与 staging 增加数据流。逻辑字节不能直接当作 HBM 事务 |
| 通信 | 按每次实际 send/save/load 与 collective 计载荷、转换及控制消息；两个消费者读取同一块仍可能付两次流量，加载路径单列 |
| 准备 | 快照、页表、私有段、图桶、采样状态、责任记录与输出预留；更深流水增加准备对象，完整 Q 的复制也有成本 |
| 同步 | 全层/rank 汇合、采样就绪、主机提交、借用结束及传输封闭；隐藏等待不等于删除依赖，不能把一个 event 扩大为所有消费者结束 |
| 驻留 | 确认 KV、临时 KV、候选/Q、logits、workspace、输出和传输保护的物理并集；共享对象只计一次，池与池内对象不重复加容量 |

普通未量化 KV 的每输入位置字节为

b_KV=2Σ_l H_KV,l·d_l·b_l，

其中 2 表示 K、V，H_KV,l 为该持有范围的 KV head 数，d_l 为通道数，b_l 为每元素字节。分片、复制、量化 scale、页表和对齐按实际组织另计，不能机械地把总量除以 TP。

若每步只消费一行，尚未提交步骤的 KV 数据主项至多为 Dmax·b_KV；它是相对确认 K 的额外段，不包含已确认历史、共享尾页 COW 或传输保护。设 32 层、8 KV head、128 维、2 字节，b_KV=128 KiB，Dmax=2 的这部分数据上界为 256 KiB/请求。页分配可能使实际预留更大。若另为每候选保留 V 个 logits、每元素 b 字节，则还增加约 Dmax·V·b；设备采样只回传 token 可以减少该项，但不能省去采样身份与控制状态。

传输保护峰值容量为同时受保护块的物理并集字节。字节时间积分可写为 R_transfer=Σ_b S_b·T_b^guard，单位是 byte·second，它描述占用持续时间，不能当作峰值字节相加。慢 save 即使不在 token 提交关键路径，也可能拖住准入容量。停止发生后，额外已接受包数量不超过当时未提交步骤数，但它们仍会贡献设备时间、KV 和排空尾部。

## 条件变化后的方案选择

| 条件 | 保留、调整或放弃 | 依据 |
|---|---|---|
| D≫H | 保留低深度重叠，优先减少启动和尾部 | H 有机会被隐藏，过深仍增加驻留 |
| H≈D 且存在抖动 | 比较一份额外候选缓冲与稍深缓冲 | 用实际到达轨迹判断设备空隙是否减少，同时核对峰值和尾延迟 |
| H≫D | 减少主机关键链或调整放置，限制深度 | 持续瓶颈不因增加等待候选而消失 |
| 短生成、EOS 或取消频繁 | 降低 Dmax，必要时回到确认后推进 | 额外计算经常没有有效产出，排空占比较大 |
| CPU grammar 或外部 callback 改变采样 | 在完整 Q 就绪后再推进，不能满足时关闭候选重叠 | token 转发不足以维持同一自回归分布 |
| send/save 长尾 | 按 cid 封闭保护，比较 staging 方案 | 提前释放可能污染旧数据；复制换取容量也付出访存和驻留 |
| 图桶或容量不匹配 | 比较合法桶、拆批和 eager | padding 减少提交却增加无效计算；固定地址不证明内容属于当前代 |
| 模型、adapter、数值或状态族改变 | 重建快照与合法性判断，放弃不兼容临时后缀 | 旧票据不能跨版本继续；递归和窗口状态需各自的版本化策略 |

CUDA Graph 的合法绑定、计划容量和跨代覆盖条件见[图缓冲](../frameworks/cuda-graph-buffers.md)。一个 graphExec 的自身执行限制、共享 workspace 或图安全 RNG 都可能约束实际重叠，不能仅通过增加 stream 或槽数推导并发收益。权重/专家搬运也会争用同一资源，成本应与[权重卸载](../algorithms/weight-expert-offload.md)的真实加载轨迹合并。

## 区分实验与反证修正

| 竞争解释 | 控制与不同预测 | 反证后的动作 |
|---|---|---|
| 提交后停滞来自容量，或来自错误 phase | 先完成并提交前驱，再建立后继，连续生成远超 Dmax 的序列 | 有额度和完整票据仍不能推进，检查物理/业务状态混用，不能只加缓冲 |
| 停止只抑制输出，或确实完成排空 | 在后继运行时停止，先检查责任仍在，再送真实结束；另外取消一个没有 COMMITTED 前驱的链 | 丢弃终态缺失或责任提前消失，修独立结算路径；拒绝提交不等于丢弃检查通过 |
| 同 seed 已足够，或 Q 继承错误 | 固定输入，用可变随机数消耗和非平凡约束状态，对照串行确认轨迹 | 候选或 Q 首次分歧时核对继承与事件定位，不能归因于正常异步顺序 |
| KV 长度记录错误，或采样差异 | 记录每步消费位置、逐层写入范围和候选位置；用重算完整前缀的参考对照 | 把 c_g 当作已算 KV 的实现会偏移一行，先修位置与状态边界 |
| 设备瓶颈，或主机结果瓶颈 | 固定设备输入，仅增加独立 H 工作，分记主机、设备与总关键路径 | H 增加但总时间不变可能仍被隐藏；总时间随 H 增长才支持主机进入瓶颈 |
| 缓冲改善空隙，或仅增加驻留 | 同一 offered 轨迹比较 Dmax，记录有效 token、浪费步骤、峰值和尾延迟 | 吞吐不变而驻留上升，降低深度；不以候选生成速度代替有效产出 |
| 传输完成可释放，或仍有另一读者 | send 先结束、save 延迟，插入重复及旧代通知，再检查内容代次 | 提前复用则修 cid 与封闭门；调换通知顺序不能修物理保护 |

设备实验还应记录每包实际接受状态、全层完成范围、消费者可见性、物理分配代次、失败/取消和最后访问证据。单次墙钟不能区分设备变快、主机等待被隐藏、padding 变化或测量边界缩短。对同一请求统计有效完成与待排空对象，不能丢掉未成功请求来改善表面吞吐。

## 检查覆盖与未验证边界

随章交付 `tools/check_k_engine_005_full.py`，定义确定、离线、只读的 Python 标准库 CPU 模型。步骤使用与正文一致的 physical、success、business 三个轴；检查前驱先提交再提交后继，以及不同深度下超过 Dmax 的持续生成。停止、取消、主机拒绝、缺层证书和缺失最后访问证据分别触发真实模型分支，断言额外候选进入丢弃终态、在途责任在结束前保留、最后访问后私有段才回收。

数据流检查采用合成整数 causal 层和带可变随机消耗、频次统计及约束状态的采样器。增量私有 KV 路径与每次从完整 token 历史重算的串行参考比较候选、控制状态、位置和逐层 KV；它检验链与位置关系，不实现真实 Transformer attention。独立责任审计从未结束步骤重建应有 device uid，检测提前清账。另有整数时间资源模拟与两阶段闭式公式对照，以及封闭传输保护检查。

故意错误的变体分别限制已提交前驱、让排空依赖 COMMITTED、取消时清空在途责任、重置后继 RNG、偏移 KV 位置、乱序提交和首个传输完成即解除保护；每个负对照必须实际触发参考或不变量失败。脚本中的有限审计收据保留合成数值，不等于真实物理内存仍有这些读者。

这些模型不证明 Python 交付脚本已经执行通过，也不调用 CUDA、TensorRT-LLM、网络或项目实现。不覆盖真实 kernel、浮点数值、stream/event 可见性、图捕获与更新、并发事务原子性、NIC/DMA 排空、跨进程持久恢复或目标硬件性能；没有 GPU 实测或知识改善 Agent 使用效果的结论。实际执行通过须由绑定当前脚本和输入的执行记录承担。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRTLLM：TensorRT-LLM Architecture Overview](../sources/S-TRTLLM.md)：所供完整文本的架构页将 LLM 入口、每 rank 的 PyExecutor、Scheduler、KVCacheManager、ModelEngine 与 Sampler 组织成后台循环，依次取请求、调度、准备 KV、执行前向和处理输出；介绍 CUDA Graph padding，以及设备后继工作与前一步 CPU 结果处理重叠、增加一个解码步骤在途状态的机制。KV Cache Transmission 页描述 context 完成后 KV 暂留设备、generation 请求具体 token 范围、context 发送、generation 继续计算及 context 释放，并说明布局映射和通信组件。本章不迁移网页性能数字、默认开关或未来计划到其他配置。
  [完整原文](../../references/S-TRTLLM/reader.html) · [全文文本](../../references/S-TRTLLM/fulltext.txt)

- [S-TRTLLM-CODE：TensorRT-LLM PyExecutor 固定提交](../sources/S-TRTLLM-CODE.md)：固定 Git 0343a9d9f2a8f69f9fe783834b042ff7687a28a4 的作者整理机制说明包含调度准备、模型执行、异步采样、请求更新，以及请求完成后的发送/保存管理。AsyncTransferManager 保留请求与传输元数据、保护 KV，每次开始增加使用者、完成后减少，全部结束后解除相应保护，并不覆盖已有错误；加载由其他路径处理。本章仅据这份机制说明引用实现事实，未独立读取源码归档。正交状态、完整 Q 链、open/seal、代次及失败排空协议属于独立设计推导，不能归为该组件已经实现的全局策略。
  [算法与机制说明](../sources/S-TRTLLM-CODE.md)

相关条目 [异步调度](../algorithms/async-scheduling.md)、[Prefill/Decode 分离](../algorithms/disaggregated-serving.md)、[CUDA Graph](../frameworks/cuda-graph-buffers.md)、[取消与完成](../system-semantics/cancellation-resource-completion.md)提供快照、首 token 与 KV 分离、图绑定和最后使用的交界。正文起始契约、逐层组合、状态算法、成本式、设定数值、反例与 CPU 模型为基于这些材料的独立推导，不是上游或目标设备测量。相关章节按所供摘录使用，不将摘录称作通读原章。

来源事实受 2026-04-30 cutoff 约束，整理日期不改变来源发布日期。原始章节身份与版本映射保留在知识库既有版本清单；本章不以编辑日期推断来源的新旧或平台能力。

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [通信传输：注册、完成、可见与回收](../hardware/transport-and-rdma.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
