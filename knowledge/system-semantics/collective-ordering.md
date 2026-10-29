# K-SYS-005：集合通信：参与集合、操作顺序与失败传播

分布式前向是一组共同执行的协议。每个 rank 的局部控制流必须形成相互匹配的通信序列；输出还要经过设备完成、应用提交和消费者最后使用等边界。接入并行边界、研究新布局或验证通信原型时，需要同时恢复这些关系。

本章区分两层契约：NCCL 定义 communicator 内操作的参与、参数、顺序和发起边界；应用定义请求、微批、层、模型版本、取消和结果提交的对应关系。形状匹配只能满足其中一部分。下面的组级提交协议是一种保守的应用设计，不是 NCCL 自动提供的事务，也不要求所有生产系统逐次增加控制面握手。

## 集体操作的语义对象

定义 communicator 为 C=(P,π,e)，其中 P 是有序的 p 个 communicator ranks，π 把 rank 映射到进程和设备，e 是当前执行代次。共同决定 D 包含应用序号 seq、请求或微批、模型版本、阶段、按序排列的 slots，以及输出清单 publish_set。每个 slot 有共同逻辑键 k=(C 身份,e,seq,slot 序号)、描述符 d=(kind,count,dtype,归约方式,root) 和张量解释 h。h 指明每个偏移的全局轴、请求及分片归属。

本地分配另外有缓冲代次 g。完成票据绑定 (k,r,g,stream)，防止相同地址被重新分配后，旧 event 放行新数据。g 可以因 rank 而异；它不替代全组共同的 e 和 seq。

对 rank r 的输入 I_r 和输出 O_r，设 N 为语义块长，0≤j<N：

- AllReduce：O_r[j]=⊕_{s∈P} I_s[j]，每个 rank 得到相同结果。
- AllGather：O_r[sN+j]=I_s[j]，输出块按 communicator rank 排列。
- ReduceScatter：每个输入含 pN 项，O_r[j]=⊕_{s∈P} I_s[rN+j]。
- Reduce：仅 root 的 O_root[j]=⊕_{s∈P} I_s[j] 是该操作的结果。
- Broadcast：所有 rank 的 O_r[j]=I_root[j]。

这里的 root 是 communicator rank，不是物理 GPU 编号。例如 π=(GPU2,GPU0,GPU3,GPU1) 时，root=1 指 GPU0；AllGather 的第一个块仍来自 rank 0，即 GPU2。

本地状态包括 PREPARED、ATTEMPTED、ISSUE_PENDING、ISSUED、DEVICE_DONE、READY 和 PUBLISHED。ATTEMPTED 从进入通信调用前开始记录：调用失败或超时后，不能仅凭返回错误就断言它从未提交设备工作。失败状态 FAILED 或 UNKNOWN 与资源状态独立；逻辑终止后仍可能有在途访问。

## 合法条件、空工作与角色范围

固定 NCCL Collective Operations 文档要求所有 rank 使用相同 count 和 datatype 组成完整 collective；不匹配可能挂起、崩溃或损坏数据。应用还必须保证相同操作、归约方式、root、逻辑键及顺序，并使每个位置具有相同张量解释。group 不会修复这些错误。

每个参与者必须提供该角色实际需要的合法范围。下表限定输入输出分开分配；原位别名须另按具体函数核对。N 表示共同描述中的 count；AllGather 的 count 表示每 rank 的贡献长度，ReduceScatter 的 count 表示每 rank 的接收长度。

| 操作 | rank r 的有效输入范围 | rank r 的有效输出范围 | 主缓冲字节数，元素大小为 b |
|---|---|---|---|
| AllReduce | N 项 | N 项 | 2Nb |
| AllGather | N 项 | pN 项 | (p+1)Nb |
| ReduceScatter | pN 项 | N 项 | (p+1)Nb |
| Reduce | 所有 rank 均需 N 项 | root 需 N 项；非 root 无语义输出 | root 为 2Nb，其他为 Nb |
| Broadcast | root 需 N 项；非 root 无语义输入 | 所有 rank 均需 N 项 | root 为 2Nb，其他为 Nb |

表中的“无语义输入/输出”不授权向任意 API 参数传入无效指针。适配层须核对固定函数对忽略参数的要求；未核对时提供合法占位参数。本章不把来源正文未明确承诺的空指针行为当作事实。

空工作有两种不同情况。

1. **全组零长度。** D 中该 slot 的共同 count=0，所有 rank 使用这个描述。只有固定 API 及适配层已证明支持这一调用时才提交；否则在任何数据通信前共同拒绝，或共同决定删除整个 slot 并重新形成 D。数学上的空数组结果不证明某个库允许零长度调用。即使允许，也不能由某个 rank 单独省略。
2. **局部无有效数据，但共同 count>0。** 不能把本地 count 改成 0。先依据上表分配完整范围，再证明如何构造内容。对于应用明确将“没有贡献”定义为归约单位元的操作，可以填充单位元：例如精确整数求和填 0。ReduceScatter 要填充完整 pN 项；Reduce 的非 root 也需要完整输入。平均值不能只补零后仍除以固定 p；应同时归约有效计数，或重构数学定义。浮点 NaN、signed zero、溢出及自定义归约也需要单独确认等价条件。

AllGather 的缺失块没有通用单位元。若模型允许 padding，D 必须共同规定填充值、有效长度和消费掩码，消费者排除无效位置；否则共同拒绝或改用适合可变长度的交换。Broadcast 的非 root 没有本地贡献很正常，但仍需 N 项接收范围；root 缺少 N 项源内容时不能用一个任意填充块冒充原广播。Reduce 的非 root 无输出需求，也不等于它无输入需求。

例如三个 rank 要求和两个整数，输入分别为 [1,2]、无贡献、[7,8]。只有数学定义允许缺失贡献为零时，才能把第二个输入构造成 [0,0]，全组共同 count=2，结果为 [8,10]。把第二个调用改为 count=0 是非法反例；保留 count=2 却不分配输入同样非法。

还需保证生产者先完成输入准备，通信读取结束前不覆盖输入，所有消费者最后使用结束前不复用输出。非连续布局要么证明满足函数要求，要么计入 packing。单线程管理多个本地设备时，固定 Group Calls 文档要求使用 group；本章非 group 路径不适用于会在第一个设备调用上等待其他本地设备的循环。

## 发起单元与完成票据的契约

为贯通 group 和非 group，定义“发起单元” U：group 模式下 U 是本轮按声明次序暂存的所有本地 slots；非 group 模式下每个 U 只有一次调用。同一 C 的主机提交由一个 worker 串行拥有。本章执行算法使用一个本地通信 stream，且 D 中所有输入在发起前可准备完成；某 slot 消费本轮另一 slot 的结果时，须拆成两个共同步骤，不能在准备阶段等待尚未发出的结果。

适配层返回值的含义必须明确：

| 操作与返回 | 本章解释 | 后继动作 |
|---|---|---|
| group 内 ADD 返回正常 | STAGED，仅完成本地主机暂存 | 继续暂存；不能据此记录通信完成 event |
| GROUP_END 返回 SUCCESS | 该单元发起已确认 | 可以在对应 stream 后面记录 event |
| 非 group SUBMIT 返回 SUCCESS | 只有经过固定版本核对，证明该返回已确认入 stream，才映射为成功 | 随后记录 event |
| 单元返回 IN_PROGRESS | 发起尚未确认 | 查询涉及的所有 communicator，成功前禁止下一次依赖提交和相关 CUDA 操作 |
| 返回或查询错误 | 发起失败，可能已有部分工作 | 转交组级恢复，保留所有可能在途范围 |
| 等待期限耗尽 | UNKNOWN，不能证明成功或未提交 | 转交恢复；不能记录一个普通成功 event 来替代排空证明 |

给定 NCCL 2.25.1 Group Calls 原文明示：group 内调用可能尚未入 CUDA stream；nonblocking 的 GROUP_END 返回 SUCCESS 表示 kernels 已发起，IN_PROGRESS 表示后台仍在发起，应确认相关 communicator 状态成功后再调用相关 CUDA 操作。给定材料没有完整列出非 group nonblocking 接口的全部细节。因此该路径只有适配层另有固定版本证据、能够兑现上表时才准入；否则使用已核对的 blocking 非 group 模式或 group 模式。不能凭 group 文档猜测另一接口的契约。

完成票据 E 在准备阶段创建为 fresh、unrecorded 对象。确认 U 发起之后，在同一个通信 stream 上、U 的全部通信之后记录 E，并记录 covered_keys。由于 stream 顺序，E 的完成覆盖其前面的通信；group 后记录的一个尾 event 可以覆盖多个 slots，但必须明确这些 slots 的键和各自缓冲代次。一个 event 的名字或地址相同不能证明覆盖关系。

WAIT(E) 成功的后置条件是：E 确实已记录，身份及代次匹配，覆盖当前 slot，且对应设备事件已成功完成；同时未观察到相关异步错误。此时才更新该 slot 为 DEVICE_DONE。event 未记录、记录在无依赖的另一 stream、来自旧代次或不覆盖当前通信，都必须拒绝。

## 从准备到发布的完整执行算法

算法依赖一个独立于受损数据 communicator 的调用方控制协议 R。R 的具体承诺及故障边界在下一节展开。BEGIN 需要全体准备报告匹配；COMMIT 需要全体 READY。等待 R、发起状态和设备事件时都持续接收组级 ABORT，并使用有界轮询。可能无限阻塞的主机库调用需要外部 supervisor；仅在调用之后检查 deadline 无法抢占该调用。

```pseudocode
procedure CONFIRM_ISSUE(U, ret, R, deadline)
  if ret = SUCCESS then return ISSUED
  if ret is ERROR then return FAILED
  require ret = IN_PROGRESS
  mark every attempted slot in U as ISSUE_PENDING
  repeat
      if R reports ABORT then return FAILED
      if deadline expired then return UNKNOWN
      states ← query issue status of every communicator used by U
      if any query or state is ERROR then return FAILED
      if all states = SUCCESS then return ISSUED
  end repeat

procedure COLLECTIVE_STEP(r, C, D, R, deadline)
  ledger ← empty; reservations ← empty; tickets ← empty
  require one serialized owner of C and a supported submission mode
  try
      validate D epoch, participants, sequence, descriptors and tensor meaning
      validate publish_set maps each output to a slot, owner and valid subrange
      for slot in D.slots in declared order do
          sizes ← ROLE_RANGES(slot.kind, C.size, slot.count, slot.root, r)
          reserve every required input and output range with fresh generation
          construct payload or an explicitly justified neutral/padded input
          reject any local count different from the common descriptor
          finish input producers, packing and neutral filling
          tickets[slot] ← CREATE_FRESH_UNRECORDED_EVENT(slot.local_device)
          mark slot PREPARED
      end for
      begin ← R.PREPARED(r, D.digest, local range bindings)
      wait for common BEGIN or ABORT, receiving failures until deadline
      if no BEGIN then raise FAILED_OR_UNKNOWN

      if D.use_group then units ← [all local slots in declared order]
      else units ← [one unit per slot in declared order]

      for U in units do
          if R reports ABORT then raise FAILED
          if D.use_group then
              GROUP_OPEN
              for slot in U in declared order do
                  ledger.add(slot, ATTEMPTED, possibly_in_flight = true)
                  add_ret ← GROUP_ADD(slot, required role ranges, C.stream)
                  if add_ret is not STAGED then
                      R.REPORT_FAILURE(r, D.key, add_ret, ledger)
                      UNWIND_OPEN_GROUP_BY_FIXED_ADAPTER_CONTRACT
                      raise FAILED_OR_UNKNOWN
                  end if
              end for
              ret ← GROUP_END
          else
              slot ← the only slot of U
              ledger.add(slot, ATTEMPTED, possibly_in_flight = true)
              ret ← SUBMIT(slot, required role ranges, C.stream)
          end if
          issue ← CONFIRM_ISSUE(U, ret, R, deadline)
          if issue ≠ ISSUED then raise issue
          for slot in U do
              mark slot ISSUED
              E ← tickets[slot]
              RECORD_EVENT_AFTER_UNIT(E, C.stream, U)
              bind E to C.identity, D.epoch, D.seq, slot.index,
                        r, slot.buffer_generation, C.stream, U.covered_keys
              if event recording failed then raise FAILED
          end for
      end for

      for slot in D.slots do
          E ← tickets[slot]
          require E recorded and bound to this slot and generation
          repeat
              if R reports ABORT then raise FAILED
              if deadline expired then raise UNKNOWN
              reject any asynchronous communication or event error
              wait_state ← QUERY_EVENT(E)
          until wait_state = COMPLETE
          mark slot DEVICE_DONE
          ledger.mark_physical_communication_use_ended(slot)
      end for
      seal all outputs against mutation and further submissions for this seq
      R.READY(r, D.key, D.digest, completion proofs, publish_set bindings)
      decision ← wait or query R authoritative decision until deadline
      if decision unknown then raise UNKNOWN
      if decision = ABORT then raise FAILED
      require decision = COMMIT and manifest matches D and local bindings
      for output in manifest.outputs owned by r do
          slot ← D.slots[output.slot_index]
          require slot = DEVICE_DONE and output.range within its valid output
          install committed handle idempotently, tagged with slot key and generation
          mark that output PUBLISHED
      end for
      C.next_sequence ← D.seq + 1
      retain output references for every authorized consumer
      return COMMITTED
  catch FAILED_OR_UNKNOWN as cause
      stop this worker from starting new submissions
      decision ← R.REPORT_FAILURE_AND_QUERY(cause, D.key, ledger, reservations)
      if decision = COMMIT then
          hand committed manifest and owned buffers to committed-result recovery
          return COMMITTED_DELIVERY_PENDING
      end if
      revoke local unpublished handles and refuse consumers
      hand ledger, tickets and all reservations to recovery supervisor
      return RECOVERY_PENDING
```

这里的角色范围函数就是前表，不依据 has_payload 决定是否分配。进入 SUBMIT 或 GROUP_ADD 之前先登记可能在途，覆盖“调用开始后、返回前失败”的窗口。只有适配层能证明未入队时，恢复方才可把该项改为未提交。group 内错误的 unwind 是固定版本适配责任：停止暂存、报告最早错误、按该版本结束或撤销主机 group 范围并交接所有尝试；不能把一个无条件 GROUP_END 或直接 return 当作通用错误处理。

非 group 路径直接处理每次 SUBMIT 的实际返回，没有用一个人为 SUCCESS 覆盖它。若第一次调用 pending，就先确认该次发起，再记录覆盖它的 event，然后才允许下一单元。group 模式先结束整个 group，再确认该单元，之后记录尾事件。两种模式都经过成功、pending、错误和 UNKNOWN 分支。

正常通信状态为 PREPARED→ATTEMPTED→ISSUE_PENDING 或 ISSUED→DEVICE_DONE→READY。发布资格来自组级 COMMIT；PUBLISHED 是本地安装已提交句柄的状态。没有消费者的 slot 仍需结算通信，不能只等待 publish_set 涉及的 slots 而遗留在途访问。

## 失败传播、提交竞态与物理结算

R 接收参与集合 P、执行代次 e、seq、共同描述摘要、每 rank 的准备/READY/失败报告及资源账本，输出唯一的 COMMIT(manifest) 或 ABORT(reason)，并提供可查询的权威记录。R 可由既有调度和恢复控制面承担，但必须满足以下契约。

- 每个执行组有唯一、可串行化的决定权威；记录与消息绑定 e 和 seq，迟到旧消息不能获得新代次权限。控制面不可用时停止授予新提交权，保持未决状态。
- PREPARED 报告未齐或描述不一致时不发 BEGIN。准备失败也走共同 ABORT，不让其余 rank 继续等待数据 collective。
- READY 表示该 rank 所有本轮通信 DEVICE_DONE、相关错误检查成功、输出范围已封存，且在决定之前不再发出本轮新工作。READY 不是已经发布。
- 任一提交前失败或 UNKNOWN 进入权威记录时，若尚无 COMMIT，就选择 ABORT，取消本轮所有 READY 和未提交发布资格，通知所有参与者及其 supervisor 停止新增提交。已取得许可、正在进入库调用的尝试也必须入账，直到 worker 停止确认才能认为没有新的在途项。
- 只有全体 READY、相同 D 且没有先被排序的失败，才原子写入 COMMIT 和完整输出清单。逻辑发布的时间边界就是这个原子记录；各 rank 安装句柄可以稍后完成，并按同一清单幂等恢复。

```pseudocode
procedure R_ON_MESSAGE(message)
  serialize with all decisions for (execution_epoch, seq)
  reject stale epoch or inconsistent descriptor
  if decision already exists then return that decision
  if message is FAILURE or UNKNOWN or preparation rejection then
      decision ← append ABORT with earliest known cause
      cancel READY and unpublished permissions
      stop new grants; notify every member and supervisor
  else if message is PREPARED then
      save report
      if all members prepared with matching D then issue common BEGIN
  else if message is READY then
      verify BEGIN, sealed completion report and output bindings
      save READY
      if READY members = P then
          decision ← append COMMIT with complete immutable manifest
      end if
  end if
  return decision or PENDING
```

“任一 rank 失败后不能发布”需要这个时间边界。失败先于 COMMIT 被权威接受，则整轮不得发布；COMMIT 已经成立之后发现的故障不能追溯撤销此前已提交结果，而是停止后续步骤、恢复已提交清单的交付或明确处理提交后资源故障。物理故障发生但尚未被检测，与权威已记录失败是不同事实。本协议不声称在异步网络中瞬间知道所有节点的故障。

等待决定超时的 rank 也不能自行猜测 ABORT：它向权威报告 UNKNOWN 并查询记录。若 COMMIT 已先成立，转入已提交结果恢复；若 ABORT 先成立，禁止消费者；若记录仍不可查询，保持 RECOVERY_PENDING，保留资源和身份。这避免一方按超时放弃、另一方按已提交结果继续造成两个决定。

ABORT 后的物理结算单独进行：

```pseudocode
procedure RECONCILE_ABORT(P, attempt_ledgers, reservations)
  request every worker to stop; revoke future submission and publication grants
  for each member or its supervisor do
      acknowledge worker stopped, including retirement of outstanding host-call grants
      enumerate every possibly submitted communication and every authorized reader
      use fixed runtime error/termination procedure to end those accesses
      obtain physical last-use proof for each affected allocation
      if any access cannot be proven ended then
          retain that allocation in HELD state under supervisor ownership
          report cleanup PENDING; do not reuse its address
      else
          release never-submitted reservations and physically settled allocations
          acknowledge member ABORTED_AND_RECLAIMED
      end if
  end for
  if every member acknowledged ABORTED_AND_RECLAIMED then return CLOSED
  else return RECOVERY_PENDING
```

正常路径中，消费者取得已提交句柄后，要在自己的 stream 上建立对对应通信 event 的依赖，或使用已完成的 host 同步关系；仅传递一个指针不建立依赖。每个消费者结束其最后读取时产生独立的 last-use 证明并释放引用。输入在通信最后读取结束后可释放；输出在通信及全部消费者结束后才可回收。输入输出别名时取二者最后使用的并集。

逻辑 abort、调用 abort 接口、进程死亡通知和物理设备排空都不能直接画等号。需要固定运行时提供足够的终止或隔离证据，涵盖仍可能访问分配的 GPU、通信引擎及外部传输。若只能隔离而不能证明访问已结束，就保留 HELD，不能报告“资源已回收”。失联成员由 supervisor 代表交接也不免除这个要求。

本节保证的安全性以前述权威、可信完成报告和运行时最后使用证明为前提；它没有实现共识服务、设备重置或故障检测器。网络分区、不可中断的库调用、失效设备无法排空时，可以长期 PENDING。不能把这种未完成清理计为完整失败闭包通过。

## 一条组合轨迹与反例

考虑前述四卡映射，seq=7 的 slot 0 为 N=8、int32 求和 AllReduce。rank 1 在数学上没有贡献，D 明确允许补零；其输入和输出仍各预留 32 字节。其他 rank r 的输入为 100r+i，rank 1 输入全零，故所有输出第 i 项为 500+3i。有效输入总计 128 字节，分开分配的输出总计 128 字节；rank 1 没有省下通信范围。

各 rank 完成输入生产和补零，提交相同 D。若采用 nonblocking group，GROUP_END 返回 IN_PROGRESS 时，输入及输出全部保持占用；不能先记录 CUDA event。查询成功之后，在各自通信 stream 的尾部记录绑定 (C,e,7,0,r,g) 的 E_r。E_r 完成后更新 DEVICE_DONE，READY 报告齐全后 R 写入 COMMIT。消费者 stream 等待自己对应的 E_r，读取该输出，最后记录 L_r；E_r 完成只结束通信访问，L_r 完成才结束该消费者读取。缓冲从 g=11 复用为 g=12 后，E_r 的旧绑定不能再用于新对象。

若 rank 2 在发起查询时超时，R 在 COMMIT 之前收到 UNKNOWN，则共同 ABORT。某些 rank 可能已经 DEVICE_DONE，另一些仍 pending；它们都不能发布本轮结果。已完成者可在停止确认后结算相应范围，pending 者必须继续由恢复方处理。只收到 rank 2 的逻辑 abort 而没有其他参与者终态，或没有 pending 范围的最后使用证明，都不是 CLOSED。

另一个反例是两个同形状 AllReduce，分别属于逻辑序号 7 和 8。rank 0 跳过 7，其他 rank 正常执行 7、8。库可能按调用次序把不同逻辑输入组合，也可能等待；count 和 dtype 相同不能识别这种错配。因此共同计划需要检查逻辑序号和张量解释，而不是只比较形状。

EP 的可变路由数量不能直接套用本章固定 count collective 的规则。每对发送/接收可以有不同数量，但应先匹配元数据、对端及数据交换次序，再选择支持这种变长关系的通信组织。把局部零 token 直接变成一个缺席的固定 collective 不成立。

## 条件化成本与驻留

设每元素 b 字节、每 rank AllReduce 输入 D=Nb 字节，α 为每轮启动/协调秒数，β 为有效单向字节带宽，H 为主机提交时间。理想均匀 ring 先 ReduceScatter 再 AllGather，共 2(p−1) 轮，每轮每 rank 发送 D/p 字节。假设等分、全双工收发可并发、带宽稳定且无额外争用，则通信模型为

T_ring≈2(p−1)α+2(p−1)D/(pβ)。

每 rank 发送量为 2(p−1)D/p；同时发生的接收不重复加入这个发送计量或带宽时间。p=1 没有 ring 链路项。设 p=4、D=16 MiB、α=5 µs、β=40 GiB/s，得到 30+585.9375=615.9375 µs。这是设定算例，不表示 NCCL 选择 ring，也不证明实际零长度操作仍执行相同轮数。

归约本身也要付出计算和访存：精确标量参考对每个输出位置进行 p−1 次二元归约，AllReduce 全局结果需要 (p−1)N 次此类运算；实际分布式执行可改变重复计算与分配。理想 ring 的归约阶段每 rank 约处理 (p−1)N/p 个被合并元素。真实 HBM 事务、协议缓冲、缓存命中和融合路径不能由逻辑字节直接确定。网络、归约和访存可能重叠，设备时间应由实际调度及瓶颈决定，不能把三个已经重叠的耗时全部相加。

准备包括布局检查、分配或注册、packing、补单位元及生产者等待。对 D_pack 字节的独立 packing，最少逻辑读写量约 2D_pack；给局部空 rank 填充 N 项至少产生 Nb 字节逻辑写入。补零保持合法性，但不会自动把共同通信长度降为 0。AllGather 的每 rank 输出是 pNb，忽略额外的本地输入会少算容量。

完整路径还包括 H、发起查询、event 记录/查询、慢 rank 等待、应用提交及消费者依赖。若使用本章保守 R，每轮还有准备报告、READY 和决定的控制通信及可能的持久记录成本；这可以进入关键路径。在已有统一调度决定且故障语义允许的系统中，可批量形成共同计划、合并提交边界或复用已有权威，不必照搬逐 collective 握手。

设 group 聚合 g 个操作，可能把多次 launch 开销减少约 (g−1)H_launch，但也会等待最晚准备的输入、推迟最早消费者并增加存活范围。可写

T_group≈T_prepare_barrier+H_group+T_device_schedule+T_issue_wait+T_commit。

同一 stream 的多个操作不能仅因使用 group 就把 T_device_schedule 写成 max_i T_i。若设备执行串行，它接近各项之和；只有另外证明并行调度及资源独立时，才可用相应关键路径最大值。共享 NIC、链路、HBM 或 SM 会增加排队或降低各自有效吞吐。

对独立分配且同时存活的 g 组主缓冲，聚合峰值可能从逐项执行的 max_i R_i 增为 Σ_i R_i；实际峰值应沿分配到最后使用区间求并集，别名区只计一次。失败会延长这些区间，HELD 范围在清理前持续占用。使用预留池时，池容量与池内已计入的存活对象不能重复相加。

## 条件改变时如何保留或调整

消息大小变化保留共同身份、描述符匹配和生命周期规则，但会改变 α、准备、带宽和等待的相对占比。小 batch decode 可以研究减少调用或聚合；大 prefill 更应检查真实链路、归约和 HBM 瓶颈。group 减少 launch 却拖慢关键消费者、造成容量超限时，应缩小或放弃聚合。

动态 batch、EOS、取消、投机接受长度和新请求加入先进入共同 D。全组都空时可以共同选择合法的 count=0 或共同删槽；只有局部为空时保留共同正 count，并证明单位元或 padding 语义，做不到就拒绝或重构。改变缺失贡献的数学含义后，旧的补零证明失效。

成员、rank 映射或执行代次改变，需要重建相应 communicator 和张量解释；旧结果及旧 event 不能迁入新身份。拓扑改变后，保留数学结果契约，重新估计路径和共享边的流量。高频 TP 放在强互联域是候选；peer access 可达不证明每个方向等带宽，NUMA、NIC 亲和及共享上行仍需验证。

nonblocking 模式增加发起确认；若适配层无法兑现非 group 的返回契约，应退出该路径。跨 communicator、跨 stream、多线程提交和多设备单线程管理需要分别核对固定版本规则，并扩展顺序与票据覆盖证明。本章单 stream 算法不授权任意交错。

浮点归约顺序、精度或算法变化后，保留逐位置归约定义，但不能保留逐位相同的结论。整数参考用于发现布局、错序和丢贡献，浮点需要独立的误差及特殊值标准。恢复后从共同已提交前缀重建时，还要恢复请求、模型、随机及输出身份，不能只重建 communicator；相关责任见[推理服务恢复](../system-semantics/serving-recovery.md)。

## 区分实验、检查与证据范围

以下是可执行的诊断设计，未宣称在目标多卡环境实测。

| 观察 | 竞争解释 | 控制变量或注入 | 可区分预测与修正 |
|---|---|---|---|
| 一个 rank 空，其他非空后挂起 | 局部改 count；输入生产者未完成；链路慢 | 保持共同 count，显式分配并按允许语义填单位元；分别延迟生产者 | 参数不匹配应由计划检查拒绝；若只有生产者延迟影响完成，则应修正准备依赖解释 |
| 同形状结果错误 | 请求/层错配；rank 块顺序错；归约数值差异 | 用 100r+i 的精确整数输入，交换逻辑键或 rank 块 | 整数索引错误支持身份/布局问题；仅浮点变化不能归因于错序 |
| group 和非 group 行为不同 | 非 group 丢弃 IN_PROGRESS；消费者过早访问 | 对两种模式使用同样的 success、pending、error、timeout 返回轨迹 | pending 后出现下一调用或 event 记录是提交控制错误，不能解释为设备慢 |
| 消费者读旧内容 | 未记录 event；旧代次 event；event 未覆盖当前通信 | 分别注入这三种票据，仅改变票据绑定 | 三者都应被拒绝；补足正确 stream 因果边后仍错，应检查其他数据或可见性解释 |
| 超时后某 rank 仍发布 | 未查询权威决定；失败先后边界不清 | 分别令 FAIL 排在 COMMIT 前后 | 前者 ABORT 且无发布；后者保持已提交清单并恢复交付，不能追溯改成未提交 |
| abort 后复用地址导致偶发错误 | 逻辑取消被当作物理排空；消费者引用遗漏 | 延迟通信结束或消费者最后读取，立即尝试复用 | 最后使用前必须拒绝复用；若没有拒绝，修正资源结算而非扩大 timeout |
| 聚合后任务延迟变差 | 准备屏障；慢 rank；共享链路或 HBM 争用 | 分别缩小 group、固定准备时间、移除独立竞争流 | 预测与观测不符时修正对应瓶颈，不以 launch 数减少断言收益 |

独立 CPU 检查使用精确公式核对五类操作的值和角色范围，检查混合零/非零 count、允许的单位元构造及不允许的缺失块；执行 group/非 group 发起单元模型，以独立轨迹不变量检查发起确认之前不得记录 event 或继续依赖调用。票据负对照包括未记录、旧代次、错误 stream 和未覆盖本次通信。

失败检查显式接收参与集合，核对共同决定、每个成员终态及在途/消费者引用。原漏检轨迹 local_fail→局部 abort→另一 rank timeout，以及 timeout→另一 rank publish，必须拒绝；逻辑 abort 后未结算在途操作也必须拒绝。故意忽略 timeout 的旧式检查器作为负对照，展示为何它的通过不能支持失败闭包。

检查器中的 device_done、drain 和 consumer_done 是模型输入的抽象证明事件，不是实际 GPU 事件查询。CPU 检查不能证明 NCCL、CUDA 可见性、故障排空、跨 stream 并发、浮点行为或性能，也不证明知识改善了 Agent 的使用效果。真实多卡实验需保留固定版本、映射、输入、每 rank 最早错误、尝试账本、权威决定、完成证据及全部终态；超时保留为未知或失败，不因部分 rank 得到数值而记为全组通过。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NCCL-GROUP：NCCL — Group Calls](../sources/S-NCCL-GROUP.md)：NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。固定 NCCL 2.25.1 原文明示单线程管理多设备需要 group、组内返回可能尚未入 stream，以及 nonblocking GROUP_END 的 SUCCESS/IN_PROGRESS 边界。
  [完整原文](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。固定原文支持相同 count/datatype、root 是 rank、AllGather 按 rank 排列和 ReduceScatter 按 rank 分块。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-24 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。Group 提交边界来自固定 NCCL 2.25.1 原文，具体 CUDA 消费依赖来自下列固定异步执行说明；验证路径为建议，未宣称在目标多卡环境完成实测。

- [S-CUDA-ASYNC：CUDA 12.8 Asynchronous Concurrent Execution](../sources/S-CUDA-ASYNC.md)：来源说明中的 stream/event 关系支持区分通信提交、设备完成和消费者依赖。原有 §3.2.8.5 主题定位用于同 stream 顺序和跨 stream 依赖；本轮给定 CUDA 原文仅为摘录，不据此声称完整核对该节。摘录中的 CTA barrier、异步复制或 pipeline 也不能直接充当 NCCL 完成事件和跨 stream 可见性的实现证明。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

本轮完整阅读给定 NCCL 两份全文与四份来源说明；CUDA 两份原文仅限提供的摘录，未读取源码归档。局部缺失贡献的语义、角色范围账本、应用逻辑键、R 协议、事件绑定和成本模型是本库推导；不是 NCCL 的事务接口或自动恢复承诺。零长度、忽略指针、非 group nonblocking 返回及错误 unwind 的具体适配仍须固定接口证据，未核对时不准入相应路径。

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [分布式状态迁移：身份、传输与提交点](../system-semantics/distributed-state-transfer.md)。
- [通信传输：注册、完成、可见与回收](../hardware/transport-and-rdma.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
- [请求路由与迁移：命中收益、排队和追赶条件](../algorithms/routing-and-migration.md)。
- [失败分析：最早偏离、最小复现与机制归因](../tools/failure-evidence.md)。
