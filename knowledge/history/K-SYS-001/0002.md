# K-SYS-001：主机与设备时间：偏序、区间并集和关键路径

服务延迟来自生产、提交、执行与消费之间的依赖路径；主机 API 区间与设备工作区间可以重叠。

## 事件并不在一条串行线上

host 调用开始、命令进入设备队列、kernel 开始、host API 返回、kernel 完成、CPU 消费结果，是不同事件。异步 launch 可以在 API 返回前就开始设备工作，也可能返回很久后才执行。设备开始时间减去 host API 结束时间因此可以为负，这不自动表示时钟错误。

跨 CPU/GPU 时间域比较需要工具提供的校准或可靠映射；跨机器更要考虑时钟偏移。correlation ID 表达因果关联，不能单独提供持续时间，也不保证跨进程唯一。

## 累计时长、忙碌时长与延迟

若两个 kernel 分别覆盖 [0,10] 与 [5,15]，累计时长为 20，区间并集为 15，整体经过时间也为 15。若某个请求只等第二个 kernel，相关关键路径还需追溯第二个的输入何时就绪。并发 memcpy、通信和多个 stream 使累计时长更容易超过墙钟。

一张执行依赖图中，从请求到达到可消费输出的最长依赖路径决定完成延迟。加速不在这条路径上的工作，可能只释放资源而不立即缩短该请求；释放资源又可能改善其他排队请求。局部与负载下整体效应应分开推导。

## 由合成事件说明统计口径

设一次异步 API 在主机时刻 0 开始、在 5 结束，关联设备工作在 3 开始、在 8 结束，则设备开始减 API 结束等于 −2。这里的单位任意，数字是为解释偏序关系而设定的算例，不是项目实测。负值与异步执行关系相容；能否对具体 trace 作同样解释，还需确认时钟域和事件关联。

若两个独立 kernel 区间为 [0,10] 和 [5,15]，累计时长为 20，忙碌区间并集为 15。这些可复算关系足以保留时间统计的推导能力，无需借用截止日之后的工程案例。

## 如何由时间线选择改动

若 GPU 空洞前 CPU 正在准备依赖数据，改进准备或重叠可能有效；若所有输入已就绪而设备等待另一 stream，需分析依赖；若 GPU 一直忙但目标请求排队，调度和容量更相关。API 时长大也可能包含同步等待，不能一律归因于驱动本身。

## 跨节点时钟：先判断可比较，再计算等待

同一主机的单调时间可以度量本机经过时间，但两台机器的时间戳相减还包含时钟偏移和漂移。跨服务 trace 上的父子关系表达因果，不会自动消除这些误差。[上下文传播](../sources/S-OTEL.md) 说明怎样传递关联身份；实际时钟校准仍是另一项条件。

设 A 的发送时间为 10 ms，B 的接收时间为 15 ms，B 相对 A 的时钟偏移 δ 只知道位于 −2 至 4 ms。传输经过时间为 15−10−δ，因此只能约束为 1 至 7 ms。不能把图上的 5 ms 当作精确单向延迟，更不能用它区分相差 1 ms 的两种实现。若偏移界限不能确定，应保留发送先于接收的偏序，使用本地持续时间、往返区间或更可靠的校准补充判断。

往返测量包括返回链路及对端处理，往返除二只有在明确的对称假设下才是单向估计。跨节点的事件排序也要携带进程、设备、请求和传输代次；重试沿用请求身份时，不能把前一次发送与后一次确认配成一次传输。

## 加载、传输与计算的完整路径

磁盘请求提交、块层派发、I/O 完成、用户可读、格式转换、H2D 完成、GPU 消费分别对应不同边界。[块 I/O 队列](../hardware/storage-io.md) 与 [传输完成协议](../hardware/transport-and-rdma.md) 解释为什么提交返回和接收完成都不应直接当成消费者开始。

设一个加载路径为：读取 0–40 ms，转换 40–48 ms，复制 48–60 ms，消费计算 60–75 ms；另一条独立计算覆盖 0–35 ms。所有阶段时长相加为 110 ms，但这个设定下的完成时间是 75 ms。将独立计算从 35 ms 缩短到 15 ms 不改变该结果；将读取缩短 20 ms，在其他资源和依赖均不变化时才预测完成时间降到 55 ms。

这个预测还含有资源条件：若独立计算争用转换所需的 CPU 或复制所需的内存带宽，原来的区间不能保持不变。依赖图说明哪些重叠合法，实际资源竞争决定合法重叠能否实现。既不能把各阶段孤立最优时间直接拼接，也不能以时间线重叠面积等同于吞吐收益。

## 用相反预测核对瓶颈

对 I/O 阻塞假设，预期增加合法预取距离能够减少消费者等待，但可能增加驻留峰值；对计算饱和假设，额外预取不应持续降低计算耗时。对跨节点偏移假设，改变校准只会改变显示对齐，不会改变客户端本地测得的完成时间。把这些不同预测放在同一请求边界下验证，才能区分机制和显示误差。

实际验收同时保留无采集服务测量、短窗口依赖证据和校准不确定性。平均阶段耗时之和既不等于请求尾延迟，也不能解释突发期间形成的等待队列。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-NSYS：Nsight Systems 2024.6 — User Guide](../sources/S-NSYS.md)：Nsight Systems 2024.6 用户指南描述 host API、设备 kernel、复制、范围和相关事件的采集与查看。进程、线程、设备、采集范围和时间窗口共同限定观测含义。
  [完整原文](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-OTEL：OpenTelemetry — Context propagation](../sources/S-OTEL.md)：OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。
  [完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)

- [S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events](../sources/S-RDMA.md)：固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。
  [完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

- [S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation](../sources/S-LINUX-IO.md)：Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。
  [完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

## 延伸阅读

- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
