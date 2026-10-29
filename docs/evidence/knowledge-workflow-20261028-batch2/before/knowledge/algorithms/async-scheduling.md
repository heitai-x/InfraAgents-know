# K-ALG-008：异步调度：依赖图、代次与双缓冲

异步调度只重叠彼此独立的工作；自回归 token、缓存映射和资源回收仍由明确依赖连接。

## 可以重叠什么

串行迭代耗时可分成主机准备 C、设备执行 G、结果处理 O。理想流水的稳态周期接近 max(C＋O,G)，但下一 token 值依赖本轮采样，因此不能在未知输入时任意提交下一轮完整计算。可提前进行的工作包括新请求解析、独立 metadata 准备、已确定的内存规划；是否能把采样和下一轮输入留在设备上，取决于实际执行路径。

CPU 提交 kernel 返回表示工作已排队。若此时直接读取输出、检查 EOS 或将 token 转成 Python 标量，就可能同步并消除重叠。相反，无限向前排队会增加取消响应延迟、在途状态和回收压力。异步深度是延迟与资源的联合变量。

## 一张调度快照属于一个执行代次

第 n 轮使用的 token IDs、请求行映射、页表、lengths、采样参数和输出槽位应形成同一代快照。第 n＋1 轮可以更新另一组缓冲区；只有第 n 轮最后一个消费者完成后，旧组才能复用。双缓冲解决两代同时存活，不能自动支持任意多代流水。

概念时序是：准备 n → 提交 n → 设备读取 n → 完成 n → 消费 n 的输出 → 允许复用 n 的槽。准备 n＋1 可以插入设备执行 n 的区间，但不得覆盖尚未读取的 n。事件记录点若早于最后一次使用，会制造“已完成”的假象。

## 晚到结果如何回收

请求 A 在 slot 2 执行时被取消，随后 B 被安排到 slot 2。A 的结果回来后，仅按 slot 2 路由会污染 B。结果需要携带请求 ID 与代次；接收方可以丢弃已取消代次的业务输出，但仍要完成其资源账务。取消因此分为不再继续调度、抑制输出、等待在途完成、释放所有权四个语义阶段。

图执行也受同一规则约束。图绑定稳定地址，批次内容通过这些地址更新；稳定地址不等于内容可以随时覆盖。共享 workspace、多个 stream、CPU 的页表更新都要进入同一依赖模型。

## 如何推导有效异步收益

若 C＝2、G＝8、O＝1，完全串行是 11，理想重叠上限约 8，即约 1.375 倍；若结果依赖迫使 O 的 1 单位落在关键路径，则更低。设备已经饱和时，增加主机线程只会扩大排队。主机时间线、设备空洞和在途深度应共同解释当前属于可重叠区间还是依赖区间。

## 两个不同层次的流水

服务层可重叠输入准备、I/O、设备执行和输出；kernel 内部可重叠片上加载、矩阵乘加与归约。它们作用于不同资源和代次。[GPU 内核流水](../hardware/gpu-kernel-pipelines.md) 中的环形缓冲不能替代服务请求身份，服务的两个 stream 也不能证明 kernel 内部已经有效重叠。

每条可重叠边都需要已知输入。下一 token 未确定时，可以准备不依赖其值的元数据和容量，但不能把猜测输入当作正式下一轮计算。若采用投机分支，需要独立的确认/回滚协议；提前准备的工作在 EOS 或取消后可能无用，其成本也应计入。

## 流水深度怎样进入容量和尾延迟

在途 d 代可能同时占用 d 份输入/输出缓冲、各代 workspace、未回收页表和传输源保护。graph 的地址绑定还可能限制这些缓冲的复用方式。深度增加时，已排队任务可能推迟取消生效、模型切换或高优先级请求获得服务，即使稳态吞吐不变。

候选深度要同时比较启动、稳定周期、排空和异常路径。短任务可能在流水填满前就结束；长任务能摊薄启动，但也更容易积累队列。对一条 GPU 已持续满载的串行依赖链，继续增加提交深度通常不能增加该链的物理并行度。

## 计算分工的共同提交

CPU/GPU 同时计算同一层不同分片时，双方完成才能合成下一层的输入；输入传输与部分和合并可能仍在关键路径。空分片可以避免不必要同步，但必须基于该代真实工作集合。一个旧 CPU 结果迟到后，不得与新 GPU 结果拼接。

用可控制的延迟重排 I/O、CPU 和 GPU 完成顺序，检查输出身份、状态守恒与回收。再在真实运行中比较主机空洞、传输区间、设备时间和端到端结果，区分“代码能异步提交”和“关键路径已经缩短”。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。
  [完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

- [S-VLLM-SCHED：vLLM — V1 Scheduler source reference](../sources/S-VLLM-SCHED.md)：vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。
  [算法与机制说明](../sources/S-VLLM-SCHED.md)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-FLASHATTENTION3：FlashAttention-3](../sources/S-FLASHATTENTION3.md)：§2–3 以 Hopper 的 TMA、异步 WGMMA、warp specialization 和多级缓冲组织 attention；讨论寄存器、布局和 FP8 数值限制。
  [完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

- [S-POWERINFER：PowerInfer](../sources/S-POWERINFER.md)：§3–6 以激活局部性、预测器、神经元放置和 CPU/GPU 结果合并组织混合执行；§8.1 明确评估模型包含特定 ReLU/ReGLU 变体。
  [完整原文](../../references/S-POWERINFER/reader.html) · [全文文本](../../references/S-POWERINFER/fulltext.txt)

## 延伸阅读

- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
