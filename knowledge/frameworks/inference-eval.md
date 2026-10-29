# K-FW-001：PyTorch 推理：模块模式、Autograd 与状态突变

推理行为不是由一次 eval() 决定的单一状态。模块模式决定部分层计算什么函数，自动微分上下文决定记录什么图，执行 stream 决定操作何时发生，对象所有权决定地址何时可以复用。关闭梯度只能回答其中一个问题。

## 问题与状态

将一次请求的前向写为 `y_q = f_{θ,T,P}(x_q; R_q)`。`θ` 是参数，`x_q` 是输入，`P` 是设备、dtype、归约和近似策略，`R_q` 是请求可变状态。输出契约还包含逻辑位置、别名关系、梯度范围和可读时刻，不能只描述张量的 shape。

`T = (t_m)_{m∈M_sens}` 是相关模块的模式配置。`M_sens` 包含 forward 会读取 training 或据其更新 buffer 的模块，包括 Dropout、BatchNorm 和自定义模式分支。根模块 training 正确不能证明子模块正确。`G` 表示梯度记录是否开启，`I` 表示当前线程有效推理上下文的深度；下文用深度表达嵌套恢复责任，并非声称框架公开了同名计数器。

| 状态轴 | 影响 | 不能据此推断 |
|---|---|---|
| `T`：逐模块模式 | dropout、批统计、模式分支 | 梯度是否开启、子模块是否服从根标志 |
| `G`：梯度记录 | 反向图和保存量 | 模块是否 eval、前向函数是否正确 |
| `I`：有效推理上下文 | view/version 等簿记、前向 AD 和张量使用边界 | 输出已完成、复制结果一定是普通张量 |
| `P`：设备与数值策略 | 舍入、溢出范围、后端路径 | 逐位一致或任务质量达标 |
| `R_q`：请求状态 | KV、位置、请求随机流 | 共享模型对象可以任意写入 |
| `E/L`：完成凭据与生命周期保护 | 可读时刻、在途访问、回收 | 请求不再提交就等于内存可复用 |

`G/I` 通常不改变理想实数函数，但可能改变执行路径与保存对象。`T` 可以直接改变函数：训练态 dropout 产生随机掩码，BatchNorm 的行为还取决于是否追踪运行统计。即使处于 eval，配置为不使用运行统计的归一化也不能被解释成固定统计量的仿射变换。

参数和不可变位置表可以共享；KV、位置计数和请求随机流应归属请求或明确的共享所有者。若合法混合模式会更新 buffer，必须使用请求私有副本或受独占保护的模型实例，并定义失败后状态如何恢复；只冻结 training 标志不能使 buffer 变成只读。下文服务算法要求共享模型在请求中没有未管理的写入，`R_q` 则由该请求独占。

异步输出包含两个不同事实：主机已经取得张量对象；设备是否已经完成它的最后写入。记录 event 只在队列中建立一个完成点。只有该点成功完成，才能给采用主机等待方案的消费者发放读取资格。

## 合法性与方案选择

模块模式可以统一设置，也可以按任务声明混合模式。启动时枚举相关模块，先完成可能递归覆盖子模块的统一设置，再应用混合模式例外，最后逐项核对模块身份、类与 training，封存清单。不能用只检查根标志的入口替代它。清单外的自定义模式分支需要补入声明；无法可靠枚举时，应对完整模块树建立统一模式契约并核对自定义 forward。

服务期间禁止未授权修改模式。请求前后比较清单可以发现检查点上的变化，但无法发现“中途改变又改回”的瞬态。要求更强隔离时，应限制 setter 的访问权或使用独立实例，不能声称两次快照证明了整个执行区间没有变化。

- eval 不会关闭梯度，no_grad 和 inference_mode 也不会自动执行 eval。`train + no_grad` 可以继续执行 dropout 或更新统计量。
- inference_mode 比 no_grad 进一步减少推理簿记，也关闭前向自动微分；no_grad 主要控制反向模式的梯度记录，工厂函数的 requires_grad 等特殊行为不能靠名称概括。
- 上下文在实际提交计算的线程内建立。初始化线程进入上下文不能代替 worker 的上下文；任务迁移到其他线程时必须重新建立相应契约。
- 仅对后处理求导时，模型可以不记录图。若 raw 是推理张量，必须退出所有有效推理上下文后，创建独立普通存储并复制；后处理从普通输出建立新的图。需要对输出值求导时，还要按 dtype 和任务要求建立普通 leaf 的 requires_grad。
- 需要梯度穿过模型时，模型前向必须保留所需目标的记录路径和保存量。事后复制、detach 或设置 requires_grad 都不能恢复已省略的模型图。record 区域只是必要条件，输入或目标参数的可微性、算子支持和图连通性仍需检查。
- 本章选择“主机确认生产完成后再校验和消费”的保守方案。它允许正常 pending，但不允许提前消费。也可以采用消费者 stream 等待事件的方案；那时 oracle 必须检查已建立的 happens-before 关系，不能沿用“事件已完成”的判据却省去等待。

普通存储要求实际的 is_inference 为 false，不能只改来源标签。下文转换要求有效 `I=0`，目标不与源存储别名，分配和复制均在非推理上下文中完成。外层 inference 仍生效时，选择拒绝转换；本章不假定存在通用的强制转换接口。

空 batch、空位置集合和全 mask 行按模型输入契约处理。没有合法选择位置时应返回规定的空结果或错误。即使设备工作量为零，也需要有来源明确的就绪凭据；不能把“没有元素”当作任意缓冲已经就绪。

## 服务前向的算法与状态变化

先规定执行与所有权边界。`S_f` 执行模型前向的主路径和输出选择，`S_c` 执行普通存储转换，`S_k` 执行消费者。三者属于本请求独占使用的执行域；实际实现可以复用 worker stream，但必须保证本次排空期间没有其他提交者继续向这些 stream 添加工作。模型内部若使用其他 stream，必须在提交前将它们登记到保护域。模型内部每次读取都由 forward 自己建立必要生产依赖；返回端汇合不能修复模型内部已经发生的提前读取。输入也必须带有生产者完成关系，下文在前向前等待输入就绪。

forward 返回 `y` 时，还须向事务提交与该输出版本绑定的生产者清单。清单覆盖选择实际读取的底层存储、版本和范围，而非仅记录外层张量对象；每项包含登记的 producer stream 及其最终生产提交点。返回时该版本的生产提交集合已经封闭，不能在返回后继续追加同版本写入。共享视图沿底层存储找到生产者，多个辅助 stream 的必要生产者全部列入；同一 stream 的多个写入可以由位于其最后必要写入之后的一个完成点覆盖。若无法建立完整清单，不能提交输出选择。

`tx` 是由监督者持有的请求事务对象，不依赖局部栈变量存活。它从提交前开始拥有输入、请求状态、模型版本引用和相关存储。每个新分配在任何异步使用前加入 `tx`；模型或消费者内部临时对象由框架分配器的已核验生命周期规则保护，或同样纳入受保护分配域。不能假定一个普通 Python 包装器可以自动接管未知扩展的内部地址。

这里显式持有对象直到相关工作完成，不依赖 Release 自动同步。持有强引用也不禁止另一个拥有者改写内容，因此输入与源数据还必须保持不可变，目标只能有本请求的写者。分配器必须保证新存储在分配 stream 上可安全使用；跨 stream 的首次使用仍需生产者或分配依赖。

```pseudocode
algorithm PrepareModelMode(model, mode_spec):
  modules = EnumerateDeclaredModeSensitiveModules(model)
  require mode_spec covers modules exactly
  ApplyUniformModeIfNeeded(model)
  ApplyExactPerModuleOverrides(model, mode_spec)
  entries = Snapshot(name, object_identity, class, training for modules)
  require entries agree with mode_spec
  return Seal(entries)

algorithm ValidateFrozenMode(model, sealed):
  live = EnumerateDeclaredModeSensitiveModules(model)
  require live names and identities match sealed
  for each entry in sealed:
      require live[entry.name].class = entry.class
      require live[entry.name].training = entry.training

algorithm PushMode(ctx, kind):
  require kind in {record, no_grad, inference}
  if kind = record:
      require ctx.inference_depth = 0
  saved = (ctx.grad_enabled, ctx.inference_depth)
  if kind = record:
      ctx.grad_enabled = true
  else if kind = no_grad:
      ctx.grad_enabled = false
  else:
      ctx.inference_depth = ctx.inference_depth + 1
      ctx.grad_enabled = false
  return saved

algorithm PopMode(ctx, saved):
  RestoreActualThreadContext(saved)
  require actual context agrees with saved
```

拒绝在有效 inference 内进入服务的 record 路径，是本章选定的服务规则，不是在声称框架一定会对对应上下文嵌套抛错。PushMode 的所有可预检条件先于状态修改；进入或恢复上下文本身失败时，worker 不再承接新请求，不能继续使用未知线程状态。

每份 `Ready(t,e)` 绑定存储身份、分配代次、最后写入版本和完成事件 `e`。事件必须位于该版本最后写入之后，并包含必要生产者的汇合。共享存储视图可以继承同一覆盖关系；新写入会使旧凭据失效。事件成功完成和设备错误也必须区分。

选择前的汇合建立读取顺序，选择后的事件证明 raw 已完成；二者承担不同责任。下面的生产者清单是本库服务适配器的合同，不是声称 PyTorch 张量自带同名字段。

```pseudocode
algorithm JoinOutputProducersBeforeRead(tx, y, S_f):
  producers = tx.SealedProducerManifestFor(y.required_storage_versions)
  require producers is sealed for this forward attempt
  require producers covers every storage/version/range that selection reads
  require no further write to these versions may be submitted
  require all producer streams and storage are already protected by tx

  for each producer p in producers:
      require p.final_submission is the last necessary production on p.stream
      if p.stream = S_f:
          require p.final_submission precedes the future selection in S_f
      else:
          e_p = RecordEventAfter(p.stream, p.final_submission)
          tx.ProtectEventAndProducerStorage(e_p, p)
          require adding wait S_f <- e_p preserves an acyclic dependency graph
          EnqueueWait(S_f, e_p)

  # 返回只证明等待已排入，未宣告生产者已在主机侧完成。
  # 所有等待必须先于选择的任何实际读取提交到 S_f。

algorithm BindReady(t, event):
  require event is ordered after all writes producing t.current_version
  return (t.storage_identity, t.allocation_generation,
          t.current_version, event)

algorithm AwaitReady(t):
  require t.ready exists and matches t storage, generation, version
  require t.ready.event covers all producing writes
  WaitEventOnHost(t.ready.event)
  require event completed successfully
  require t storage is still owned and valid

algorithm MaterializeOutsideInference(tx, raw, ctx, S_c):
  require ctx.inference_depth = 0
  require raw.is_inference = true
  require tx already owns raw and its producer dependencies
  saved = PushMode(ctx, no_grad)
  try:
      out = AllocateOwnedOrdinary(tx, S_c, raw.shape, raw.dtype, raw.device)
      require out.is_inference = false
      require out.storage_identity != raw.storage_identity
      # out 已被监督者持有；raw 的持有也尚未解除
      EnqueueWait(S_c, raw.ready.event)
      EnqueueCopy(S_c, source=raw, target=out)
      e_copy = RecordEvent(S_c)
      out.ready = BindReady(out, e_copy)
      out.model_graph = absent
      out.copy_source_identity = raw.storage_identity
      return out
  finally:
      PopMode(ctx, saved)
```

`RecordEventAfter` 在已封闭生产提交之后、对应 producer stream 上排入事件，不是把尚未完成的操作标成完成。同 stream 的生产和选择依靠队列顺序；其他 stream 通过事件等待建立先生产、后读取。生产者清单或事件登记失败时，不再提交选择，直接进入事务失败收尾。所有可能已提交的辅助工作仍在预登记域内，因而即使没有成功创建某个事件，也可以排空或隔离。

转换返回的是带 pending 凭据的普通张量，而非“复制已完成”的宣告。目标在 `S_c` 上分配并使用；`S_c` 等待 raw 在 `S_f` 上的最终生产事件。复制读源与写目标的生命周期保护在 EnqueueCopy 之前已经成立。若事件登记失败，函数不释放源或目标，也不负责伪造事件；异常交给事务失败收尾，通过已登记的 stream 排空取得完成证明。

no_grad 前向产生的普通 raw 不需要复制，它的就绪凭据直接绑定 `S_f` 在输出选择之后记录的 `e_f`。因此消费者不要求每个张量都有名为 copy_event 的字段，而要求每个输入都有覆盖实际生产过程的 ready。若输出选择只是视图，凭据仍必须覆盖底层数据生产。no_grad 不会把已有推理张量别名自动变成普通张量，分支必须检查实际张量属性。

```pseudocode
algorithm ValidateConsumer(t, grad_scope, ctx):
  require grad_scope in {none, postprocess_only, through_model}
  require t.ready exists and matches current storage/generation/version
  require t.ready.event covers producing writes
  require t.ready.event completed successfully
  require t storage is valid and owned
  if t has copy_source_identity:
      require t.storage_identity != t.copy_source_identity
  if grad_scope != none:
      require ctx.inference_depth = 0
      require t.is_inference = false
  if grad_scope = through_model:
      require recorded model graph covers requested differentiation targets
      require all graph-retained objects remain owned

algorithm ServeOne(q, model, cfg, sealed_mode, ctx, domain):
  tx = Supervisor.CreateOwnedTransaction(q, domain)
  # domain 在任何提交前登记 S_f、S_c、S_k 以及内部辅助 stream
  try:
      require q.state has exclusive writer q
      ValidateFrozenMode(model, sealed_mode)
      ValidateGradientScopeAndRequiredTargets(q, cfg, ctx)
      tx.AdoptAndProtect(q.input, q.state, model.version)
      AwaitInputProducers(q.input)
      q.execution_state = Running

      saved = PushMode(ctx, cfg.forward_region)
      try:
          x = NormalizeAndValidateOwnedInput(tx, q.input)
          y = model.forward(x, q.state) on S_f
          # forward 适配器同时向 tx 封存本输出版本的完整生产者清单
          tx.AdoptReturnedObjectsBeforeFurtherSubmission(y)
          JoinOutputProducersBeforeRead(tx, y, S_f)
          raw = SelectByLogicalPosition(y, q.output_positions) on S_f
          tx.Protect(raw)
          # 末端汇合覆盖剩余内部访问；不代替选择前的生产依赖
          JoinRegisteredInternalProducersInto(S_f)
          e_f = RecordEvent(S_f)
          raw.ready = BindReady(raw, e_f)
      finally:
          PopMode(ctx, saved)

      if q.grad_scope = postprocess_only and raw.is_inference:
          z = MaterializeOutsideInference(tx, raw, ctx, S_c)
      else:
          z = raw

      q.execution_state = Pending
      AwaitReady(z)
      ValidateConsumer(z, q.grad_scope, ctx)

      if q.grad_scope = postprocess_only:
          # 普通存储上的 detach 不执行数据复制，继承同一 ready
          z = MakeOrdinaryDetachedLeaf(z)
          ConfigureLeafGradForRequestedDifferentiableValues(z, q)
          tx.Protect(z)

      consumer_kind = record if q.grad_scope != none else no_grad
      saved = PushMode(ctx, consumer_kind)
      try:
          # 输入已在主机确认完成，因此 S_k 不需等待未完成的生产者
          result = q.consumer(z) on S_k
          tx.AdoptReturnedObjectsBeforeFurtherSubmission(result)
          JoinConsumerAuxiliaryStreamsInto(S_k)
          e_k = RecordEvent(S_k)
          result.ready = BindReady(result, e_k)
      finally:
          PopMode(ctx, saved)

      AwaitReady(result)
      ValidateResultContract(result, q)
      ValidateFrozenMode(model, sealed_mode)
      handle = tx.PrepareOwnedResultHandle(result, required_graph_roots)
      # 以下是单一原子提交点；失败必须没有任何可见提交效果
      AtomicPublishAndTransferOwnership(q, handle)
      q.terminal = Committed
      tx.ReleaseOnlyCompletedObjectsNotRetainedBy(handle)
      return handle
  catch error before atomic commit:
      FailRequest(tx, q, error)
      return failure
```

`SelectByLogicalPosition` 的输出分配和内部临时区也服从提交前保护规则；`tx.Protect(raw)` 保留返回对象，不能被解释成此前提交可以使用未保护地址。选择可以是视图，也可以在 `S_f` 上执行实际 gather 或复制。后两种情况会读取 `y`，因此必须先经过生产者等待；不能依赖选择恰好不发 kernel 的样例来省略该依赖。

例如 `S_a` 延迟生成 `y=(7,−3,1)`，选择位置为 `(2,0,2)`。合法顺序是 `S_a: produce(y)→e_y`，`S_f: wait(e_y)→gather(y)→e_f`，得到 `raw=(1,7,1)`。若先 gather，再等待 e_y，gather 可以读取未初始化内容或旧版本；之后生产者、选择和 e_f 全部完成也不会重新执行 gather。此时 e_f 可以证明 raw 的写入已经结束，却不能证明这个写入采用了正确的输入版本。正确性同时需要每个读取点的依赖和最终输出的完成凭据。

`ValidateGradientScopeAndRequiredTargets` 要求 through_model 使用 record 前向且无有效外层 inference；postprocess_only 使用 inference 或 no_grad 前向。它核对任务所需的可微目标，而不把一个来源字符串当作图连通性的证明。postprocess_only 的 detach 明确切断模型路径；若任务后来改为穿过模型求导，必须重建前向，不能继续复用这个 leaf。

归一化、前向及消费者内部也遵守提交前保护规则。对于可能在抛错前已经提交部分 kernel 的扩展，错误返回不能被解释成“没有提交”。`tx` 预先登记整个可能使用的执行域，失败后排空该域，而非只等待成功创建的事件。

PrepareOwnedResultHandle 在发布前完成所有可能失败的分配和所有权准备。原子提交把输出与必要图引用一起交给消费者；未提交结果不可见。提交之后的网络发送和外部消费者是后续责任，不能在已经可见后把本次请求改成“从未发布”。外部消费者在异步使用 handle 前取得持有权，直到其最终读取、反向及必要通信结束才释放；完成前不能复用输出或图保存量。

### 失败不等于物理可回收

```pseudocode
algorithm FailRequest(tx, q, error):
  require q has not atomically committed
  DisablePublication(q)
  q.terminal = Failed(error)
  q.cleanup_state = DrainPending
  Supervisor.StopNewSubmissions(tx.domain)
  # tx 已由监督者持有，栈展开不会释放其保护对象
  try:
      for each registered stream s in tx.domain:
          SynchronizeStreamOnHost(s)
      require all submitted work in this domain has physically finished
      require no external consumer has received unpublished objects
      tx.ReleaseOwnedObjectsAfterCompletionProof()
      q.cleanup_state = Reclaimed
      ReturnDomainOnlyIfWorkerContextAndModelStateRemainValid(tx.domain)
  catch drain_error or completion_not_proven:
      Supervisor.Quarantine(tx.domain, tx.owned_objects, drain_error)
      q.cleanup_state = Quarantined
      # 域和地址不再服务或复用；只有后续可信排空或域销毁证明才能解除
```

这里选择失败后排空，而非依赖 stream-ordered Release。排空完成前，源、目标、输入、请求状态和图引用继续被监督者持有；若同步失败或超时，隔离整个不能确认完成的域。取消提交资格可以立即发生，物理回收必须等待独立证明。不能在 finally 中仅恢复上下文就释放地址，也不能把等待超时当作排空成功。

如果实际系统改用事件驱动延迟回收，事件必须已经成功建立并覆盖所有访问；事件登记本身失败时仍需要可执行的回退。若改用分配器的 stream-ordered release，则必须分别证明源读取和目标写入均被其依赖覆盖，不能只登记目标而遗漏源。

正常组合轨迹如下。事件的 pending 是合法中间状态。

| 步骤 | 设备顺序与凭据 | 读取资格与持有者 |
|---|---|---|
| 前向生产 | `S_a: produce(y) → e_y`；其他必要生产者同样登记 | y 和底层存储已由 tx 保护；返回对象不代表设备完成 |
| 选择提交 | `S_f: wait(e_y及其他必要生产者) → select(y) → 末端汇合 → e_f` | 等待先于实际读取；raw 由 tx 持有，尚不可供主机消费 |
| 退出本次上下文 | 恢复先前 `G/I`；外层 `I>0` 时拒绝转换 | 模式恢复不代表设备已完成 |
| 普通化提交 | `S_c: wait(e_f) → copy(raw,out) → e_copy` | raw/out 都已受保护；out.ready 仍 pending |
| 主机等待 | 推进生产者、选择和复制，确认 e_copy 成功 | 此后同一 consumer oracle 才接受 out |
| 后处理 | `S_k: consumer(out) → e_k`，建立新图 | out、result 和图对象继续存活 |
| 发布 | 主机确认 e_k，检查模式及结果后原子提交 | handle 接管输出和所需图引用 |

只有 `S_f` 生产 y 时，选择依靠同 stream 顺序，不需要额外跨 stream 等待。no_grad 普通输出跳过复制行，直接等待 e_f 后进入同一消费者校验。through_model 同样等待 e_f，但必须保留模型图，并在普通记录上下文中继续消费。

具体失败反例：复制已入队，RecordEvent 抛错。此刻可能没有 e_copy，但队列仍含对 raw 的读取和对 out 的写入。请求立即变成 Failed，输出不可见；y、raw 和 out 继续由 tx 持有。排空覆盖辅助生产者及 `S_f/S_c` 的等待、选择和复制，随后才能释放。若故意把错误分支改成立即 Release(out)，独立生命周期检查应在复制尚未结束时发现无效地址。这个反例区分“失败结果不会发布”和“设备不再访问内存”两个条件。

## 成本：计算、访存、通信、准备、同步与驻留

### 保存量与底层存储

设 autograd 保存张量集合为 `S_q`，张量 `a` 有 `|a|` 个逻辑元素，每元素 `b_a` 字节，列举元数据按每个 `c_m` 字节估算，则

`B_logical(q) = Σ_{a∈S_q}|a|b_a + c_m|S_q|`。

这是逻辑载荷估算，不是物理驻留或实际节省的上界。小切片可以保留很大的底层存储，非连续视图可跨越较大范围，多个保存量也可能指向同一存储。

在时刻 `t`，令 `A_q(t)` 为保存引用所保留的不同物理分配集合，`r_s(t)` 为分配 `s` 的完整计费容量，得到这一口径的保留峰值：

`P_retained(q) = max_t Σ_{s∈A_q(t)} r_s(t)`。

每个物理分配只计一次。若只能取得被访问区间的容量下界，计算出来的也只是对应下界，不能称为完整物理峰值。共享 segment 的归属口径必须统一；跨请求共享的物理容量不能在全局账中重复计算。allocator reserved、碎片、workspace 和观察范围外的分配另列；保存引用保留的容量也不等于关闭记录后一定释放的增量容量，因为其他读者可能仍在持有同一存储。

设定 FP16 激活为 `8×1024×4096×2 = 67,108,864` 字节，即 64 MiB。12 个激活若各有独立紧凑存储且同时存活，数据为 768 MiB；每项再计 16 字节元数据，逻辑估算为 805,306,560 字节。若它们是同一 64 MiB 存储上的 12 个单元素视图，逻辑列表仅几百字节，保留容量仍可为 64 MiB；若都是同一存储的完整视图，逻辑求和可达 768 MiB，物理仍只计一份。该算例不含分配器取整、碎片、KV 和 kernel workspace。

### 模式、转换与同步的条件化成本

训练态 dropout 对 `E_d` 个元素生成随机掩码并缩放；若掩码写入内存，另有约 `E_d b_mask` 的逻辑载荷。eval 可使该层成为恒等。使用批统计的 BatchNorm 对每通道 `R=N·H·W` 个元素归约；使用固定运行统计的 eval 路径不再做该批归约，只读取每通道统计量。是否融合、是否落入 HBM 以及是否更新统计量由实际配置决定，不能把逻辑元素数直接当成 DRAM 事务数。

普通化复制同设备 `B=N·b` 字节输出，至少产生源读取 B 和目标写入 B 的有效数据搬运，并需要新目标容量 B。若数据都经同一内存带宽通路，理想带宽时间下界约为 `2B/β_effective`；实际可能由缓存、布局转换、kernel 启动和带宽竞争改变。跨设备或跨主机转换还应另计互联、IPC 序列化和同步，本章算法未执行这些通信。

例如转换一个独立 64 MiB 输出，源和目标在复制期间至少同时占有 128 MiB 数据容量，有效读写量为 128 MiB。它们何时能释放取决于后处理、图保存量及外部消费者的最后访问，而非复制函数返回。no_grad 已产生合格普通输出时可以省去这份目标及复制；是否值得放弃 inference_mode，应比较省下的簿记与新增复制、驻留和等待。

若 y 依赖 p 个外部 producer stream，按每个 stream 的最后必要写入合并完成点后，选择前需要至多 p 个生产事件及 p 条等待，清单与事件持有记录为 O(p)。等待本身不搬运 y 的载荷；实际 gather 的读写量另计。选择只能在所有必要生产者完成后执行，其起点受最晚必要生产完成点及既有 `S_f` 工作约束，不能把并行生产者耗时全部相加。把等待错误地移到选择之后并没有提供合法的同步节省。

本章的主机等待位于梯度上下文边界之外。若三个阶段存在严格数据依赖，设备关键链为 `D_forward + D_copy + D_consumer`，另有提交和主机等待后的唤醒间隙。已经包含在设备链里的工作不能因主机等待再加一遍。其他请求可以填补部分空隙；因此本章保守方案简化 oracle，但可能增加提交间隙，不能据此声称吞吐最优。需要重叠时可改用消费者 stream 等待方案，并同步改写就绪判据与分配器生命周期证明。

设备标量读取也会制造同步。设设备总工作 D、读取之后的主机决策 H、额外同步通知延迟 L。在决策阻塞后续提交且没有其他请求填补空闲的设定下，关键路径从约 D 变成 `D+H+L`。设 D=8 ms、H=1 ms、L=0.1 ms，则增加约 1.1 ms。这是设定解析算例，不是 GPU 实测。将读取移到主机已等待的完成点之后可能减少碎片化同步，但必须保持相同位置与选择语义。

编译、预打包、autotune、捕获和预热属于准备成本；变体、图私有池和 workspace 属于驻留成本。若准备耗时 C、每次合法命中节省 ΔT>0，只按累计执行时间摊销，调用数需超过 `C/ΔT`；冷请求延迟和额外容量会改变选择。shape、stride、dtype、模式配置和状态布局可能影响 guard，CUDA Graph 的固定地址与布局约束也不能由动态 shape 支持推断。

事件记录、跨 stream 等待、持有记录和失败排空都有主机或同步成本。故障排空可能阻塞整个借出的执行域；隔离则可能暂时损失域内全部保留容量。降低这些成本需要更细的依赖证明，不能靠删掉保护获得合法收益。

## 条件变化后的保留、调整与放弃

| 条件变化 | 可以保留 | 必须调整或放弃 | 区分依据 |
|---|---|---|---|
| 改统一或混合模式 | 权重的已验证布局 | 重建相关模块清单；管理新出现的 buffer 写入 | 子模块状态和实际更新对象 |
| 改为仅后处理求导 | 不记录图的模型函数 | 普通输出、就绪凭据、可微 leaf 与新图 | 无模型梯度但后处理梯度合法 |
| 改为穿过模型求导 | 输入、权重与数值契约 | 放弃省略所需模型图的前向，重新 record | 指定目标的图连通性和保存量 |
| 输出较大或生命周期变长 | 输出函数 | 比较 inference 加复制与 no_grad；调整在途预算 | 复制字节、同时存活容量、最后消费者 |
| 引入新 stream 或异步消费者 | 算子数学定义 | 重建每个读取前的生产依赖、allocator 使用与失败排空范围；输出版本清单必须覆盖新增写者 | 受控执行顺序、实际选择值与提前回收反例 |
| shape/stride/batch 变化 | 任务函数 | guard、变体、图桶、workspace | 冷热路径、边界输入和变体身份 |
| dtype、TF32 或归约策略变化 | 任务目标 | 数值容差和质量检查；必要时关闭近似 | 误差、首次分歧和 logits margin |
| 并发增加 | 共享只读参数 | 请求缓存、随机流和可变 buffer 的所有权 | 单独与交错执行的关系 |

若无法保证外层 inference 已退出，选择拒绝这条转换路径；若无法保留或重建所需模型图，不能承诺穿过模型的梯度；若无法确认失败域已排空，放弃立即复用该域。它们分别来自上下文、图和物理访问条件，不能用一次普通复制统一修补。若 forward 无法声明输出版本的全部生产者，也应放弃直接选择该异步输出，先补齐接口合同或采用明确完成生产后才返回的接口。

## 诊断：竞争解释与区分实验

先固定权重、输入、逻辑位置、数值策略和请求状态，再改变一个轴。观察只支持其覆盖范围内的解释。

1. **模块模式与记录状态。** 构造根为 eval、子 Dropout 为 train 的负对照。完整清单应拒绝，根标志检查会放行；声明该混合模式后则应按契约接受。纯线性模块在只改变 G 时应保持同一函数；加入 dropout 或批统计分支后才能将差异归到 T。若模式检查通过仍有随机性，再检查显式 functional dropout、前处理随机流及设备算法。
2. **pending、生产顺序与已完成。** 让登记的辅助 `S_a` 延迟生产 y，在 `S_f` 上执行实际读取数值的选择，再在 `S_c` 排入等待、复制和事件。选择前的等待应使 `S_f` 在 y 尚未产生时无法推进；逐步推进 `S_a` 的生产、e_y、选择、e_f、复制和 e_copy 后，结果必须匹配不读取待测缓冲的独立参考。同一 consumer oracle 在事件 pending 时拒绝消费。负对照只删除选择前等待，保留选择后的辅助 stream 汇合及 e_f；选择应因读取未初始化 y 被读取不变量拒绝。若缓冲预填了旧数据，则还需按读取版本或独立输出参考拒绝，不能把“非空且最后事件已完成”当作正确。另删除 `S_c` 对 e_f 的等待，让复制先运行，应触发独立的未初始化读取检查。这些干预区分选择前缺依赖、复制前缺依赖和单纯尚未完成。
3. **普通输出也需要凭据。** 对 no_grad 输出删除 ready，或换成其他存储、旧写入版本的事件；即使 is_inference=false，也必须拒绝。正确 no_grad 输出在 e_f 完成后接受。检查若只验证 is_inference，无法发现这一类提前消费。
4. **转换的上下文与梯度范围。** 保留外层 inference，错误转换器在其中分配复制目标；它可能有不同地址却仍是推理张量。同一 oracle 必须拒绝。合法普通化产物可供后处理求导，但不能满足 through_model；若需要穿过模型，改用记录前向并核对实际梯度目标，不能扩大 consumer 的放行条件。
5. **复制入队后的失败。** 在 EnqueueCopy 与 RecordEvent 之间注入登记错误。观察请求已经 Failed 且不可见，而源和目标仍被保护。排空辅助生产者、选择与复制后才进入 Reclaimed；排空无法证明时进入 Quarantined，禁止新提交和地址复用。立即释放目标的负对照应由“未完成操作仍访问无效存储”的独立不变量拒绝。另在消费者事件登记、消费者校验与发布前注入错误，验证相同失败收尾覆盖整条请求。
6. **内存口径。** 用大存储的小切片和同一存储的多个完整视图区分逻辑载荷与保留容量；按视图累计容量的负对照应重复计费。分别观察活跃 tensor、allocator reserved 和设备总占用。reserved 不下降不能单独证明图保存量没有减少，也不能证明存在泄漏。
7. **同步与主机调度。** 将设备标量决策改为保持语义的设备选择或完成点后处理，比较主机提交空洞与端到端时间。若设备链不变而空洞缩小，支持主机同步解释；若输出位置或数值改变，先修复函数关系。单次主机函数计时不能代表设备执行时间。
8. **数值、状态与编译。** 固定前缀逐层定位首次分歧；切换 TF32、半精度归约和 batch 组织，记录误差尺度与 argmax margin。若每项 logits 误差小于最大与次大间隔 Δ 的一半，argmax 保持；margin 很小时换 token 可以由小数值误差解释。错误轴、mask、权重排列或跨请求写入不能靠放宽容差修复。再比较同 T/P 的 eager、冷编译、热命中与 guard 失效路径，区分执行优化与模型语义变化。

请求随机流可以绑定 seed、请求身份与逻辑位置，作为避免调度改变抽样轨迹的一种设计。比较同一请求单独和交错执行；共享全局 RNG 会随插入请求改变序列。该关系检查调度独立性，采样分布正确性仍需要独立的小分布推导或统计验证。

## 接入与验证边界

权重加载、processor、forward、状态写入和输出解释构成可观察边界，不规定组件开发顺序。shape 相同不能证明权重轴、epsilon、激活近似或 tied weights 的共享关系相同。融合 QKV、RoPE 和 KV 写入后，可以沿 Q/K/V 分段、旋转值、写入逻辑位置、attention 与 logits 比较，定位最早分歧。

预热覆盖计划服务的模式、数值策略与代表性 shape。第一次输入创建位置表、算法缓存或共享 buffer 时，也必须满足设备、dtype、所有权和完整发布条件。共享 buffer 扩容不能让其他请求读到一半初始化的对象。

原生扩展必须说明 shape、stride、dtype、device、当前 stream、输出别名和 workspace 生命周期；内部辅助 stream 也要纳入读取前依赖、完成与失败协议。是否释放语言运行时锁、是否能并行执行 CPU 工作由实际实现决定。同步等待不能放进要求无主机同步的图捕获区域；可以将图作为前向阶段，在捕获区外建立本章完成边界。

配套检查是 Python 标准库的确定性 CPU 抽象模型：命令队列显式推进，事件由队列实际完成而非转换函数置为完成；检查同时使用数值参考、依赖覆盖和物理生命周期不变量。辅助生产路径明确包含延迟写 y、选择前等待、真实 gather、选择后的 e_f，以及保留末端汇合但删除选择前等待的错误变体。检查还覆盖普通化等待、pending 凭据和失败排空；它不导入 PyTorch，不执行真实线程、CUDA stream、分配器或反向传播，也不测 GPU 性能。未初始化读取负对照不等于已经模拟所有旧版本读取；目标版本的具体普通化 API、跨 stream 可见性、分配器行为和扩展异常责任仍需设备集成验证。知识内容、抽象检查、真实设备结果和 Agent 使用效果分别报告，不能相互替代。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TORCH-INFERENCE：PyTorch — inference_mode](../sources/S-TORCH-INFERENCE.md)：inference_mode 关闭推理不需要的 autograd 相关开销，并具有张量使用边界；它不会自动将模块切换到 eval 模式。来源说明还明确线程局部性和前向自动微分限制。
  [算法与机制说明](../sources/S-TORCH-INFERENCE.md)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。记录事件不等于完成，执行顺序和对象生命周期是两个独立关系。该文档的侧 stream 示例要求在读取前等待生产 stream；本文据此将依赖落实到输出选择首次读取之前，而非只在末端汇合。CUDA Graph 的重放依赖稳定地址与兼容执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致。
  [完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。模式清单、生产者清单、选择前汇合、就绪凭据、事务保护、失败排空及隔离是本库选定的服务算法，不是对 PyTorch 内部服务状态机的复述。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。S-TORCH-INFERENCE 仅依据供应的来源说明和固定提交身份，不声称读取其完整上游文档或源码。

### 本次扩写补充来源

- [S-TORCH-DYNAMIC：PyTorch — Dynamic Shapes](../sources/S-TORCH-DYNAMIC.md)：编译会基于 shape、stride、dtype、常量等条件特化，并用 guard 确定专门程序是否可继续使用；部分维度可以符号化，变化也可能触发重编译。动态尺寸不等于任意 rank 或图重放地址都可变化。
  [完整原文](../../references/S-TORCH-DYNAMIC/reader.html) · [全文文本](../../references/S-TORCH-DYNAMIC/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

## 延伸阅读

- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
- [权重加载：从制品身份到可执行布局](checkpoint-loading.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
- [正确性 Oracle：函数参考、状态变形与服务契约](../tools/correctness-oracles.md)。
- [框架 Profiler：算子归属、内存与图边界](../tools/framework-profiler-scope.md)。
