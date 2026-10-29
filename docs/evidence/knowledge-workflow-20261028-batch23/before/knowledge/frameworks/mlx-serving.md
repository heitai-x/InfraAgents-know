# K-FW-008：MLX 服务：惰性求值、编译与请求状态

MLX 的数组表达式可以延迟执行。服务循环必须分别回答：何时构造图，哪些结果已经求值，哪些状态可以提交，以及哪些存储已经没有使用者。编译又增加一条身份边界：每次调用消费的是本次显式输入，还是 trace 时留下的常量。本章将这些边界连成一个可重建的服务算法；设备行为依据固定历史文档，服务协议与成本式是独立推导。

## 图、请求状态与完成证据

将计算表示为有向无环图 G=(V,E)，边 u→v 表示 v 消费 u 的结果。根集合 R 的依赖闭包 C(R) 包含这些根及其全部祖先。EVAL(R) 的成功完成只覆盖 C(R)，不能证明另一个独立分支已经完成。数组句柄存在、图已经构造、设备访问结束、业务状态提交和客户端消费，是不同事件。打印数组、转为主机可见表示、memoryview、保存、读取标量及标量数组控制流，都可能触发隐式求值。

请求的已提交状态为：

S_r=(x[0:n),k,ell,rho,v,KV,logits)。

x 是已确认 token，n 是已知长度；k 是所有必要层的 KV 都已完成的连续边界，0≤k≤n；rho 是采样状态；v 是业务版本。ell=m 表示 logits 来自精确前缀 x[0:m+1) 的最后一行，预测位置 m+1；本章只保存可直接采样的末行，因此有效缓存必须满足 k=n、ell=n−1。ell=−1 时没有缓存 logits。模型、位置规则、mask、数值表示和 adapter 身份也必须与这个快照一致。

对位置 t，先构造 h_t^(0)=Emb(x_t,t)。第 l 层由当前 hidden 产生该位置的 K/V，并结合该层历史 [0:t) 和当前位置的 K/V 得到 h_t^(l+1)。全部层完成后，LM(h_t^(L)) 才是该前缀的 logits。一个位置不能因为某一层先结束，就被计入 k。采样 x[n] 之后，n 增一，但新 token 的 KV 尚未计算，k 保持旧 n。

除业务状态外，请求拥有者保存 live、canceled、failed、commit_eligible、phase、active_execution 以及尚未结束的物理使用责任集合 P。每次执行有不复用的执行身份 I=(请求实例,g)。完成、错误和最后使用事件携带 I 与使用者 uid；版本 v 负责业务快照检查，不能替代执行身份和物理结束证据。

本章采用保守边界：每请求至多一个尚未结束的执行；失败撤销该请求实例的后续提交资格，不支持原地重试。排空后可以回收，但不能自动重新变为 live。若系统需要恢复，应在旧责任结束后，用新请求实例及新的执行身份显式准入，另行定义恢复和输出去重协议。

关键不变量如下：

- 图构造不证明完成；提交证书必须覆盖本步全部层 KV、必要的末行结果及其他独立状态根。
- 准备和采样只产生私有候选；已提交 token、KV 边界、RNG 和版本只由同一状态拥有者安装。
- 提交资格要求 live、commit_eligible、未取消、未失败、身份及版本匹配，且处于允许提交的阶段；FAILED_DRAINING、QUARANTINED 和终止状态均不能推进。
- 失败排空只解除物理责任，不恢复业务资格；迟到成功结果同样不能提交。
- 已提交读写的最后使用结束前，不覆盖或回收其存储。删除 Python 引用不证明这一条件成立。
- 编译区的动态状态必须是显式输入和输出；未声明输入的数组捕获按 trace 常量处理。

## 服务步与 KV 存储契约

普通推进取片长 q≥1、b=k、e=min(n,k+q)，消费已知输入 x[b:e)。若 e<n，只推进 prefill，不采样；若 e=n，末行才预测下一个 token。空输入由模型入口补 BOS 或拒绝；零新增输出直接结束，不建立图；非法 k、越界 ell 或不兼容缓存必须拒绝或进入明确修复入口。

本章选择的 local 不是整段 KV 的数据副本，而是“只读前缀引用＋私有增量”。其逻辑索引为：

local[l,t] = base[l,t]，当 0≤t<b；
local[l,t] = delta[l,t]，当 b≤t<e。

base 是已提交快照的不可变引用；delta 在准备前预留，只属于本次执行。WRITE(local,l,t) 只能写 [b,e)，任何前缀写入都必须拒绝。逐 token 推进时，位置 t 可以读取 base 的旧前缀，以及本执行已经产生的 delta[b:t)。提交前 delta 对请求不可见；成功提交通过新的范围描述符或块表安装这些对象，将其所有权转给请求，不搬运旧前缀，也不复制 delta 的有效载荷。失败或取消时保留 base；delta 等最后使用结束后丢弃。

这是服务候选的存储契约，不是对某个 MLX attention 实现已经支持分段 KV 的断言。后端必须能消费这一逻辑视图，或使用满足同样隔离条件的预留区间；若实际数组拼接、更新或 kernel 接口会物化连续副本，就必须采用后面的复制成本，不能沿用无前缀复制的预算。共享物理页的尾部若不能保证写入隔离，应使用私有增量或必要的写时复制，并单列其代价。

当 k=n 且有兼容末行 logits，可以只采样。当 k=n 但没有末行 logits，不能因 n−k=0 而停滞：在私有区重算最后一个已知 token，读取 [0:n−1) 的只读前缀，得到末行结果后丢弃重算 KV；既有 KV 对象及边界不变。无法建立安全私有区时返回 REPAIR_UNAVAILABLE。

```pseudocode
算法 PREPARE
输入：不可变快照 s，片长 q≥1
输出：prepared=(base,delta,b,e,k_after,logits,required_roots)

若 s.n=0：返回 REJECT_NO_INPUT
若 s.k 不在 [0,s.n] 或缓存身份不兼容：返回 REJECT_STATE

若 s.k<s.n：
  b=s.k；e=min(s.n,s.k+q)
  base=只读引用(s.KV)
  delta=预留 [b,e) 的全层私有存储
  local=分段视图(base 的 [0,b)，delta 的 [b,e))
  对 t=b 到 e-1：
    h=EMBED(s.x[t],t)
    对 l=0 到 L-1：
      K_t,V_t=PROJECT_KV(l,h,t)
      WRITE_PRIVATE(delta,l,t,K_t,V_t)
      h=BLOCK_REMAINDER(l,h,local[l,0:t+1],t)
  logits=空
  若 e=s.n：logits=LM(h)
  required_roots=delta 的所有必要层写入，以及非空 logits
  返回 (base,delta,b,e,k_after=e,logits,required_roots)

若 s.ell=s.n-1 且 s.logits 身份兼容：
  返回 (s.KV,空,s.n,s.n,s.n,s.logits,缓存完成证书)

若不能私有重算最后一行：返回 REPAIR_UNAVAILABLE
建立只读前缀 [0,s.n-1) 与私有单位置增量
按上面的全层循环重算 t=s.n-1，得到末行 logits
返回 (s.KV,不安装的重算增量,s.n,s.n,s.n,logits,重算全部必要根)
```

伪代码逐 token、逐层展开依赖，便于恢复机制；实际 prefill 可以将一个区间按层并行计算，只要因果 mask、位置和全层完成证书等价。没有被根引用的独立位置、RNG 或元数据更新必须另列为根；不能仅凭 logits 求值成功就宣布这些写入完成。

## 准备、采样和提交的同一条轨迹

状态拥有者串行处理准入、取消、失败、提交和完成事件。设备工作可以在外部执行，但只能写本执行私有区。入口检查与 COMMIT 检查都需要存在：入口阻止失败请求产生新工作，COMMIT 阻止已经准备好的旧候选在资格撤销后安装。

```pseudocode
算法 SERVE_AND_COMMIT
输入：请求 r，片长 q
输出：PREFILLED、COMMITTED 或拒绝/终止原因

由 r.state_owner 串行处理：
  若 r.canceled：返回 CANCELED
  若 not r.live 或 not r.commit_eligible 或 r.failed：返回 BLOCKED
  若 r.phase 不在 {READY,COMMITTED}：返回 BLOCKED
  若 r.active_execution 非空 或 r.P 非空：返回 BLOCKED
  校验输入、q、KV 边界、缓存身份及增量容量
  s=冻结已提交状态；I=分配新的执行身份
  prepared=PREPARE(s,q)
  r.active_execution=I；保存 s、prepared 及所有引用
  在启动访问前登记 (I,eval_use) 到 r.P
  r.phase=EVALUATING

执行 EVAL(prepared.required_roots)
若返回错误或缺少必要结果证书：
  交状态拥有者执行 ON_FAILURE(I,证据)
  返回 FAILED

由状态拥有者确认完成证书及身份：
  证书必须覆盖本执行全部相关访问结束
  幂等结束 (I,eval_use) 的物理责任
  若资格已撤销：丢弃候选；仅做完成清理；返回 BLOCKED
  r.phase=READY_TO_COMMIT

若 prepared 有 logits：
  token,rho_next=SAMPLE(prepared.logits,s.rho)
否则 token=空；rho_next=s.rho
将候选交给同一状态拥有者执行 COMMIT

COMMIT(r,I,s,prepared,token,rho_next)：
  要求 r.active_execution=I 且 r.version=s.version
  要求 r.live 且 r.commit_eligible 且未取消且未失败
  要求 r.phase=READY_TO_COMMIT 且 r.P 为空
  任一要求不满足：不改变已提交状态；返回拒绝
  验证区间、全层证书和末行前缀身份
  原子执行以下业务事务：
    以新描述符安装本次有效 delta，旧前缀对象保持只读共享
    r.k=prepared.k_after
    若 token 非空：
      将 token 追加到旧位置 n；r.n 加一
      r.rho=rho_next；清除末行缓存
      将带请求及输出序号的交付记录追加到响应对象
    否则：保持 token 序列与 rho 不变
    r.version 加一；r.phase=COMMITTED
  解除本执行记录及已结束的临时引用
  按响应记录执行输出交付
  返回 token 非空时的 COMMITTED，否则 PREFILLED
```

业务事务是线性化点。SAMPLE 只消费冻结的 rho，返回 token 和 rho_next；两者同时提交，失败候选不推进 RNG。若采样位于编译区，随机状态也必须是显式输入和输出。

取消由同一拥有者撤销 live 和 commit_eligible。取消先于线性化点时，COMMIT 必须拒绝，包括采样已经完成的情况；提交先于取消时，已提交记录不能回滚。响应记录的安装和网络交付分别计时：发送失败或客户端未收到，不改变内部已经提交的事实。实际系统需要另行定义重送、输出序号和去重规则；本章 CPU 模型只交付一次已提交记录，不证明网络行为。

## 失败、排空和迟到完成

失败不是一个可再次调用正常服务步的中间成功状态。本章在同一拥有者下同时设置 failed=true、live=false、commit_eligible=false。即使错误返回已经带有完整结束证据，进入 COMPLETED 也只表示物理责任结束，不表示恢复服务资格。

```pseudocode
算法 失败与最后使用

ON_FAILURE(I,evidence)：
  只处理属于本请求已登记执行的事件
  r.failed=true；r.live=false；r.commit_eligible=false
  若 evidence 证明某已登记使用者的全部访问结束：
    仅移除该使用者自己的责任
  若 r.P 非空：r.phase=FAILED_DRAINING
  否则：r.phase=COMPLETED；解除本执行的保留引用
  不安装 KV、token、RNG、版本或响应记录

ON_USE_DONE(I,uid,certificate)：
  若执行身份不匹配，或 (I,uid) 已结束/不存在：忽略
  若证书不能证明该使用者最后访问结束：
    保留责任及存储；r.phase=QUARANTINED；返回
  幂等移除 (I,uid)
  若资格已撤销且 r.P 为空：
    r.phase=COMPLETED；解除本执行的保留引用
  不恢复 live 或 commit_eligible，不执行候选提交

超时仍无结束证明：
  保持 QUARANTINED 和责任账，不宣称已经回收
```

完整轨迹为：旧状态冻结 → 私有增量建成 → 求值失败但 eval_use 未结束 → 请求终止并保留私有区 → 再次 SERVE 被入口拒绝 → 已准备候选的迟到 COMMIT 被资格检查拒绝 → 不完整结束通知保持隔离 → 真实 USE_DONE 只解除物理责任 → 排空后再次 SERVE 仍被终止资格拒绝。全过程旧 KV、token、版本、RNG 和输出记录都保持不变。

旧事件不能用来清除新执行的责任，重复事件也不能多扣一次。仅递增业务版本不足以保护还在写入的物理地址；存储仍要等最后使用结束。失败后无法证明访问停止时，应报告执行身份、对象和阻塞证据，而不是把等待超时改成完成。

## 编译、捕获与形状选择

固定 Compilation 文档说明：首次调用编译函数时用占位输入 trace、优化并生成代码，随后复用缓存。输入 shape 或维数、dtype、输入数量变化可能触发重编译；频繁创建并销毁新函数身份会破坏复用。缓存选择至少覆盖函数及模型身份、输入 dtype、数量、形状策略和影响静态分支的配置。缓存未命中时，只有编译准备和驻留预算允许才建立新版本，否则选择普通求值。

编译区域按纯函数使用。trace 中读取占位数组的真实值或打印会失败；向未声明的外部列表追加数组，可能把没有数据的占位对象留在列表中。网络发送、请求注册等副作用应由外部状态拥有者处理。动态 KV、位置和 RNG 应作为参数返回更新结果，或遵守该版本显式 inputs/outputs 捕获契约。

令编译函数为 F(I;C)→O。I 是每次调用的输入槽，C 是 trace 时保存的常量。未声明为输入的外部数组被当作 C：文档示例中 state[0] 初值为 1，外部替换为 5 后，编译的 x+state[0] 仍使用原常量；显式传入 state 或声明 inputs，才使新值进入调用。

这个机制需要区分 trace 和执行，不能用每次调用时的一个分支直接假定答案。最小解释模型如下：

```pseudocode
算法 TRACE 与 EXECUTE

TRACE(函数体，输入规格，外部对象)：
  为公共 x 建立 INPUT_SLOT(0)
  若 state 是显式输入：为 state[0] 建立 INPUT_SLOT(1)
  否则：把 trace 当时 state[0] 的值保存为 CONST
  用占位符执行一次函数体，生成有序操作表示
  保存操作、常量和输出节点；不保存重新读取外部容器的指令

EXECUTE(保存的表示，本次输入)：
  为本次调用绑定各输入槽
  按依赖顺序执行 INPUT、CONST 和算术节点
  返回输出节点；不重新执行 Python 函数体或重新 trace
```

配套 CPU 检查用 2x+s 作为非对称探针，既能区分常量与输入槽，也能暴露 x 和 s 错绑。检查者在 trace 外保存常量快照，随后分别改变 x、显式 s 和外部容器；预期由该快照及本次显式输入计算。故意把显式 s 固化、让隐式常量重新读外部对象、交换输入槽或每次重 trace 的变体，都接受同一判据。

shapeless 允许 shape 变化不触发重编译，但不把依赖首次 shape 的 Python 逻辑自动改成动态逻辑。来源例子的首次输入为 2×3×4，reshape 的首轴固化为 6；改成 5×5×3 后，75 个元素不能 reshape 为 (6,−1)。使用保持轴语义的 flatten(0,1)，结果应为 (25,3)，不是任意全展平都算等价。

自回归 KV 长度增长时，可以比较 shapeless、长度分桶、只编译稳定子图和普通求值。每个候选都要核对算子支持、输出与状态正确性以及真实编译次数。文档的一般规则不证明具体 attention 已支持任意动态长度。同一提交的 Lazy Evaluation 文档仍保留早期“不编译并重跑图”说法；该版本编译能力据专门 Compilation 文档判断，惰性建图与编译是两个事件。

## stream、统一内存与驻留

Using Streams 文档说明操作可指定 stream 或 device；未指定时使用默认设备的默认 stream。Unified Memory 文档说明 CPU/GPU 可访问同一内存池，并为框架跟踪的数组依赖建立顺序。例如 CPU 生产 c，GPU 随后读取 c，后者等待前者完成。该保证不自动覆盖框架外的原生指针写入、其他进程或未声明的别名访问。

统一内存减少显式设备间复制，不消除读取、同步和竞争。同一物理数组同时被 CPU/GPU 引用只计一份容量；量化前后权重、重新打包布局、私有增量、工作区、图缓冲、编译产物和输入队列若为不同分配，就分别计账。权重加载时源表示与目标表示共存的峰值见权重加载章节，不能只按最终量化权重预算启动。

CPU 预处理和 GPU 同时工作可能争用共同带宽，孤立测量后取两者最大值可能低估并发时间。来源在 M1 Max 报告过 matmul 与小 elementwise 分工的示例；它不证明任意服务具有相同比例的收益。CUDA Graph、迁移页和 allocator 的特定语义也不能直接当作 MLX 保证。

物理回收要求逻辑持有和所有执行、输出读者都结束。旧图、编译常量、缓存和失败排空中的使用者都可能继续持有对象。缓存驱逐或删除 Python 容器只是解除某些引用；尚未实际释放的对象仍占容量。

## 同一候选下的成本

令准备描述符和增量分配成本为 T_prepare，建图为 T_build，求值为 T_eval，采样为 T_sample，拥有者处理与提交为 T_owner，交付为 T_publish，编译未命中准备为 C_compile。按这些阶段串行的单步近似为：

T_step≈T_prepare+T_build+T_eval+T_sample+T_owner+T_publish+1{miss}C_compile。

实际允许跨请求重叠时，应沿依赖路径计算墙钟时间，并累计共享资源需求，不能把全部阶段时间机械相加。T_eval 包括固定提交、计算和访存；Q/B_eff 是相应资源下界，不能再与已经包含这段访问的测量时间重复相加。CPU/GPU 的隐式标量等待、跨 stream 依赖和结果可见性都应放入真实路径。

对全层 K/V，层数 L、KV head 数 H_kv、head 维 d_h、每元素字节 b_a，未压缩同精度的每 token 有效载荷为：

b_kv=2L H_kv d_h b_a。

设定 L=32、H_kv=8、d_h=128、b_a=2，则 b_kv=128 KiB/token。四个请求各持有 4096 token KV，共 2 GiB。量化权重 14 GiB、工作区 1 GiB、图及中间量 0.5 GiB，列项基线是 17.5 GiB，已经超过假想 16 GiB 设备；这些是容量设定，不是测量。

本章共享前缀候选的瞬时容量应写成：

M_peak=max_t |物理对象并集(t)|≈W+KV_committed+delta_live+workspace+graph+compile+queues+retained，

各项只有确属不同对象才相加；工作区预算若已经包含某增量或中间量，就不能再计一次。一次正常推进的私有 KV 有效载荷是 (e−b)b_kv，重算末行需要一个 token 的私有区但不安装。四个请求在各自 4096 token 的旧 KV 之外各新增一个私有位置，额外载荷为 512 KiB；若上述列项未包含它，峰值至少为 17.50048828125 GiB，另加描述符、舍入和在途保留。提交通过转移这些对象的持有关系完成，不再制造另一份 delta。

只读前缀引用不产生 b·b_kv 的 KV 数据副本，但需要描述符、地址映射和引用维护。按页描述时准备成本随触及的描述符数增长；预分配或分层表可以改变这项成本。CPU 检查为了便于审查复制字典的元数据引用，成本为 O(Lk)，不复制 KVCell 有效载荷，也不代表生产实现具有常数时间准备。

在假设每个查询逻辑上读取一次相应全层 KV、暂不计 head 重读及 cache 的口径下，处理 [b,e) 的逻辑 KV 读取量约为：

Q_KV,read≈b_kv Σ_{t=b}^{e−1}(t+1)，

新增有效载荷写入为 Q_KV,write≈(e−b)b_kv。q=1 的批量 decode 若每批读取一次权重，读取简式为 Q_read≈W_q+Σ_r n_r b_kv；包含新增 KV 写入后还要加 Σ_r b_kv。完整账还应包括激活、中间量、采样、元数据和主机访问。实际设备事务受 query head 重用、融合、cache、布局及分块影响，不能把这个逻辑口径当作 DRAM 事务测量。分块 prefill 的权重利用和计算密度也不能直接沿用单 token decode 估算。

若后端必须物理复制旧前缀，准备阶段额外 KV 读写约为 2b·b_kv，额外副本驻留约为 b·b_kv；复制时间下界受实际读写带宽限制，还需分配和提交成本。旧前缀在旧读者结束前不能释放。若 append 也复制已有 delta，或页尾写时复制按整页进行，应按实际复制范围继续增加。此时无复制的 Q_read 简式只覆盖模型读取部分，不能冒充完整服务流量。

编译是否划算取决于同一调用序列。设普通步骤 2.0 ms、编译稳态 1.5 ms、首次编译 200 ms，均为设定值：N=400 时总成本相等，N>400 才开始节省；N=1000 时分别为 2000 ms 和 1700 ms。令额外编译率为 eta，每次调用的期望额外准备是 eta C_compile，必要条件为：

T_plain−T_comp>eta C_compile。

该例 eta 必须小于 0.0025；eta=0.01 的额外准备已达 2 ms/调用。编译代码和捕获常量还会挤占 KV，故该式只是必要条件。片长 q 则在固定提交次数、图规模、首 token 延迟和私有区驻留之间取舍。

## 条件改变后的选择

| 条件变化 | 必须保留 | 调整候选 | 放弃或修正信号 |
|---|---|---|---|
| 形状稳定、调用多 | 纯函数、状态显式、完成证据 | 缓存编译函数并预热 | 编译仍频繁、稳态未缩短或驻留挤出请求 |
| KV 长度不断变化 | 正确前缀和末行身份 | shapeless、分桶、稳定子图或普通求值 | 编译减少但输出/状态错误；桶浪费超过收益 |
| 后端要求连续 KV 副本 | 前缀不被旧读者覆盖 | 计入复制，或改兼容的分段读取候选 | 2b·b_kv 流量和副本峰值使方案不可行 |
| 满 KV 无 logits | 不从中间行采样 | 私有重算末行或明确拒绝 | 覆盖共享页、没有合法后继动作 |
| 外部容器存动态状态 | 每次调用消费本次状态 | 显式参数或 inputs/outputs | 同形状仍使用旧常量 |
| 大 batch、长上下文 | 物理并集预算与因果依赖 | 限并发、减队列、比较量化及真实流量 | batch 增长只增加驻留和尾延迟 |
| CPU/GPU 并发 | 框架依赖及外部访问协议 | 调整分工、输入队列和并行度 | 两侧同时变慢，孤立时间无法预测并发 |
| 失败或频繁取消 | 禁止后继提交、保留最后使用责任 | 缩短在途，及时处理控制事件 | 失败重入或迟到结果改变业务状态 |

## 反例、区分实验与检查范围

只求值 loss 不能证明优化器更新已经完成。来源说明在完整求值前读取 loss 会只计算前向；它没有说明随后读取更新后的参数必然得到旧值。更新后参数句柄可能连接尚未求值的更新图，读取它会执行依赖。旧值应对应保留的旧参数对象或错误发布的旧快照。服务中的同类错误是以 logits 完成替代所有状态根完成。

把新 token 的 KV 边界提前推进到新 n，会让下一步读取尚不存在的层状态。采样使用旧 rho，却提交另一个随机值，会破坏后继序列。失败后只写 commit_eligible=false，却不让入口和提交检查它，仍可能在旧 eval_use 未结束时再次生成 token。只在入口检查也不足以拒绝已经准备好的迟到候选。

| 竞争解释 | 可区分预测与干预 | 反证后的修正 |
|---|---|---|
| 捕获过期、输入错绑或每次重新 trace | 保存外部 trace 快照；分别改变 x、显式状态和外部容器；执行同一保存表示并核对非对称探针 | 显式输入正确仍错，继续查槽位、结果根与采样；不能只比较两个预设函数 |
| 前缀被复制或被覆盖 | 比较准备前后前缀对象身份、私有区范围和同时存活对象；设备上再观察复制事务 | 数据相同但身份不同仍可能增加容量；身份相同却值变化应查写隔离 |
| 失败资格没有落实 | 注入求值失败、再次推进、迟到候选及最后使用事件；比较 token、KV、版本、rho 和响应记录 | 排空后又提交说明终止资格被错误恢复；不完整证书清账说明完成边界错误 |
| 编译未摊销 | 固定请求序列，分记首次编译、实际编译次数和稳态完成时间 | 编译次数低仍慢，转查建图、执行、主机或带宽 |
| 求值粒度不合适 | 固定总数学工作，改变片长和求值边界，记录图大小及固定提交次数 | 图缩小却更慢，考虑固定开销；不能只看单次求值变短 |
| 共同带宽或容量压力 | 固定模型，分别改变 CPU 预处理、并发和 KV 长度；分记驻留与真实完成时间 | 驻留不变且仅启动成本变化，容量解释不足 |
| 取消控制或提交顺序错误 | 在求值后、采样后以及事务后交付前注入取消 | 事务前仍提交说明资格检查错误；事务后未交付不能据此宣称内部回滚 |

配套检查保留图闭包和物理对象身份计账，并执行完整前缀串行参考与增量服务的多步比较。捕获检查先 trace 出操作表示，再执行；参考不由被测执行器调用。存储检查核对旧前缀对象共享、增量写入隔离和安装后的物理并集，故意复制前缀的变体接受同一判据。失败检查实际执行再次推进、迟到提交、错误身份、无结束证书、真实结束和重复结束；故意遗漏入口及提交资格的变体由相同终止不变量拒绝。提前 k、错误 RNG 和取消后提交也有负对照。

这些是确定性的 Python 标准库 CPU 模型，检查协议、对象关系和玩具代数；它们不证明真实 MLX 编译器、GPU 数值、stream 可见性、内存压力或性能。设备实验还需报告版本、shape、布局、并发、求值根、完成证据及测量边界。本轮没有真实 MLX/GPU 或 Agent 使用效果实验。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MLX：MLX：Lazy Evaluation](../sources/S-MLX.md)：固定文档说明数组操作先记录图，显式求值或访问数据才执行；打印、主机转换、memoryview、保存、标量读取及标量控制流可能隐式求值。过密求值有固定开销，过大图也增加开销。其 loss 示例只证明提前读取 loss 会产生前向 partial evaluation。本章据此推导根闭包、提交边界及旧对象与更新后对象的诊断区别。
  [完整原文](../../references/S-MLX/reader.html) · [全文文本](../../references/S-MLX/fulltext.txt)

- [S-MLX-MEM：MLX：Unified Memory](../sources/S-MLX-MEM.md)：说明 Apple Silicon 的 CPU/GPU 可访问共同内存池，设备在操作层选择；框架为所跟踪的跨 stream 数组依赖建立顺序。文档的 M1 Max 分工时间是来源报告。本章据此推导共享容量、共同带宽和外部写入协议，不将示例倍率外推为服务收益。
  [完整原文](../../references/S-MLX-MEM/reader.html) · [全文文本](../../references/S-MLX-MEM/fulltext.txt)

### 本次扩写补充来源

- [S-MLX-EXEC：MLX：Compilation 与 Streams（固定历史文档）](../sources/S-MLX-EXEC.md)：同一提交的 Compilation 与 Using Streams 全文说明首次 trace、编译缓存、shape/dtype/输入数量变化、纯函数、常量捕获、显式 inputs/outputs、随机状态捕获、shapeless 静态 shape 陷阱及 stream/device 选择。本章据此推导缓存条件、trace/执行解释模型与动态状态边界。服务终止资格、私有 KV 增量、响应事务和成本式是本库候选设计，不声称来源已经实现这些协议。
  [完整原文](../../references/S-MLX-EXEC/reader.html) · [全文文本](../../references/S-MLX-EXEC/fulltext.txt)

### 版本与推导边界

固定来源提交为 e8ebdebeeb655feaa85a51f6b24ece5b6d5518d1，日期为 2026-04-28，受 2026-04-30 来源上限约束；整理日期不改变事实日期。Lazy Evaluation 的早期不编译表述与专门 Compilation 文档并存，本章据后者判断固定版本的编译能力。公式、容量数值、失败轨迹、分段 KV 以及 CPU 检查是独立推导；是否能以真实 MLX 算子无复制实现该候选，仍需核对实际算子与存储。正文未使用源码归档，也未把文档、CPU 模型或进程成功解释为目标设备测量。

## 延伸阅读

- [Apple Silicon：统一内存、带宽与交互式推理](../hardware/metal-unified-memory.md)。
- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
- [权重加载：从制品身份到可执行布局](checkpoint-loading.md)。
- [共享内存路径：统一地址、物理驻留与显式迁移](../hardware/shared-memory-paths.md)。
