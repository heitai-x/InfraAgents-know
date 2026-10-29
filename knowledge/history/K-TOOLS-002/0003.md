# K-TOOLS-002：时间线分析：从请求路径定位执行瓶颈

时间线分析先建立关联和依赖，再用区间与关键路径解释延迟；热点累计时长只是入口。

## 由外向内的对象关系

一个慢请求对应若干调度批，一个批对应主机准备与设备执行，一个 forward 对应多个 kernel、复制和通信。NVTX/框架范围帮助定位阶段，API/kernel correlation 帮助连接提交与执行，request/batch 元数据连接业务身份。所有关系都要在进程和设备作用域内解释。

## 三种时间线形态

GPU 有长空洞且 CPU 忙于准备，支持主机供给不足；GPU 连续忙但单请求排队长，支持容量或调度问题；多个 stream 各自有工作却在同步点等待，支持依赖或负载不均衡。它们可能同时出现，且随并发变化。

对一个同步 API，长 host 时长可能只是等待之前工作；改这个调用位置可能转移等待而不减少总路径。对频繁小 kernel，间隙可能来自 host launch；融合或 graph 的预测是减少这些间隙，而不是让原 kernel 本身一定变短。

## 聚合后回到原始事件

按 kernel 名汇总找候选热点，再选择与慢请求关联的具体实例，保留开始/结束、stream、shape 和调用上下文。重叠区间先求并集再讨论设备忙碌时间；不同设备累计时间不能直接当墙钟。

采样窗口要包含问题触发前的状态与完成后的结果。只截中间 5 ms 可能看不见队列形成或缓存回收原因。最终无 profiler 的服务测量确认收益，短 trace 用来解释机制，二者互为补充。

## 跨主机和 I/O 队列的时间线

一段加载或迁移至少有请求创建、容量就绪、提交、传输完成、目标可见和消费几个事件。等待容量与等待传输属于不同资源；NIC 报告本地完成也不等于远端应用已消费。跨机可见事件的关联要依赖请求/对象/代次，时间轴相减则另需校准与误差界限。

块 I/O 还可能在软件队列、派发队列和设备服务之间等待。把文件调用整体画成“磁盘执行”会把调度、页缓存和唤醒开销混在一起；把 GPU copy 区间单独画完，也可能漏掉此前等待 pinned buffer 或 CPU 格式转换的时间。瓶颈解释应覆盖消费者真正等待的完整范围。

## 一组正反归因算例

设每步需要的权重读取为 10 ms、复制为 4 ms、计算为 6 ms。在完全串行且无额外成本的假设下，一步为 20 ms。若双缓冲允许下一步读取与当前计算重叠，稳态吞吐受最长阶段及共享资源约束；若读取和复制还必须串行争用同一个 staging slot，不能直接声称稳态为 10 ms。

一种支持预取有效的证据组合是：消费者等待减少、相同有效工作对应的物理读取不增加、驻留增加处于预算内，且无采集的端到端测量改善。相反，只观察到异步 API 更早返回，而完成事件和输出时间不变，支持的是等待位置移动。若低并发改善而高并发退化，应进一步检查预取占用容量与排队干扰，不能只保留改善区间。

## 共享工作怎样分配成本

一个批次的矩阵计算可能同时推进多个请求；一次远端前缀读取也可能服务多个会话。物理时间线中只出现一次工作的持续时间，逻辑请求各自的延迟却都可能受到它影响。把 8 ms 的共享阶段记入十个请求的等待解释是合理的，随后把这十份 8 ms 加成设备忙碌 80 ms 则改变了统计对象。

资源成本可以按 token、字节或实际计算量分摊，但必须声明分摊规则；任何分摊都不替代真实关键路径。比较调度方案时同时报告物理资源使用、请求分布和满足 SLO 的有效完成量，避免共享率提高却让长尾等待更差。

## 反事实预测与最终确认

对热点提出明确预测：若减少历史复制是原因，则相同上下文下复制字节随步数的增长应减弱；若降低调度开销是原因，host 准备间隙应缩短，而单 kernel 的计算量未必变化。再通过保持其他条件的候选对照观察预测是否出现。

只在 profiler 下出现的收益可能来自工具改变并发、cache 或 clock 状态。最终回到正常运行点验证；若性能改善而预测没有成立，应修正因果解释，即使该候选仍然值得保留。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NSYS：Nsight Systems 2024.6 — User Guide](../sources/S-NSYS.md)：Nsight Systems 2024.6 用户指南描述 host API、设备 kernel、复制、范围和相关事件的采集与查看。进程、线程、设备、采集范围和时间窗口共同限定观测含义。
  [完整原文](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt)

- [S-OTEL：OpenTelemetry — Context propagation](../sources/S-OTEL.md)：OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。
  [完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation](../sources/S-LINUX-IO.md)：Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。
  [完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

- [S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events](../sources/S-RDMA.md)：固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。
  [完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

## 延伸阅读

- [主机与设备时间：偏序、区间并集和关键路径](../system-semantics/host-device-timing.md)。
- [Kernel 计数器：吞吐、访存与 Replay 语境](../tools/kernel-counter-replay.md)。
- [性能实验：稳定态、对照与不确定性](../workloads/measurement-comparability.md)。
