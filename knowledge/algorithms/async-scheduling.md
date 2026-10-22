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

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。
  [完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

- [S-VLLM-SCHED：vLLM — V1 Scheduler source reference](../sources/S-VLLM-SCHED.md)：vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。
  [完整原文](../../references/S-VLLM-SCHED/reader.html) · [全文文本](../../references/S-VLLM-SCHED/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
