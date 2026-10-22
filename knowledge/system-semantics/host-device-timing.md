# K-SYS-001：主机与设备时间：偏序、区间并集和关键路径

服务延迟来自生产、提交、执行与消费之间的依赖路径；主机 API 区间与设备工作区间可以重叠。

## 事件并不在一条串行线上

host 调用开始、命令进入设备队列、kernel 开始、host API 返回、kernel 完成、CPU 消费结果，是不同事件。异步 launch 可以在 API 返回前就开始设备工作，也可能返回很久后才执行。设备开始时间减去 host API 结束时间因此可以为负，这不自动表示时钟错误。

跨 CPU/GPU 时间域比较需要工具提供的校准或可靠映射；跨机器更要考虑时钟偏移。correlation ID 表达因果关联，不能单独提供持续时间，也不保证跨进程唯一。

## 累计时长、忙碌时长与延迟

若两个 kernel 分别覆盖 [0,10] 与 [5,15]，累计时长为 20，区间并集为 15，整体经过时间也为 15。若某个请求只等第二个 kernel，相关关键路径还需追溯第二个的输入何时就绪。并发 memcpy、通信和多个 stream 使累计时长更容易超过墙钟。

一张执行依赖图中，从请求到达到可消费输出的最长依赖路径决定完成延迟。加速不在这条路径上的工作，可能只释放资源而不立即缩短该请求；释放资源又可能改善其他排队请求。局部与负载下整体效应应分开推导。

## 原项目案例支持什么

[CASE-0001](../cases/CASE-0001-nsys-observability.md) 的原始清单保存微型 CUDA 采集与分析。在该记录中，首次关联设备开始减 host API 结束为 −1116 ns；3 个 kernel 累计 4352 ns，2 次 memcpy 累计 4896 ns。这些数字用于说明事件关系和统计口径，不能用来推断模型吞吐。原有版本和证据保留在历史中。

## 如何由时间线选择改动

若 GPU 空洞前 CPU 正在准备依赖数据，改进准备或重叠可能有效；若所有输入已就绪而设备等待另一 stream，需分析依赖；若 GPU 一直忙但目标请求排队，调度和容量更相关。API 时长大也可能包含同步等待，不能一律归因于驱动本身。

<!-- infraagents:provenance -->

## 来源与推导

- S-CUDA-ASYNC：[完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)
- S-NSYS：[完整原文](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt)
- S-CASE-0001：[完整原文](../../references/S-CASE-0001/reader.html) · [全文文本](../../references/S-CASE-0001/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。

- [S-NSYS：Nsight Systems — Post-Collection Analysis Guide](../sources/S-NSYS.md)：Nsight Systems 的原始事件与导出可用于查看 host API、设备 kernel、复制和关联信息。事件属于具体进程、线程、设备和时间窗口。

- [S-CASE-0001：InfraAgents CASE-0001 evidence manifest](../sources/S-CASE-0001.md)：本地证据清单记录微型 CUDA 采集、导出、缺表对照和事件关联。所记首次设备开始减 host API 结束为 −1116 ns；3 kernel 累计 4352 ns、2 memcpy 累计 4896 ns。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
