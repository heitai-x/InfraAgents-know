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

<!-- infraagents:provenance -->

## 来源与推导

- S-NSYS：[完整原文](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt)
- S-OTEL：[完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)
- S-CUDA-ASYNC：[完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-NSYS：Nsight Systems — Post-Collection Analysis Guide](../sources/S-NSYS.md)：Nsight Systems 的原始事件与导出可用于查看 host API、设备 kernel、复制和关联信息。事件属于具体进程、线程、设备和时间窗口。

- [S-OTEL：OpenTelemetry — Context propagation](../sources/S-OTEL.md)：OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [主机与设备时间：偏序、区间并集和关键路径](../system-semantics/host-device-timing.md)。
- [Kernel 计数器：吞吐、访存与 Replay 语境](../tools/kernel-counter-replay.md)。
- [性能实验：稳定态、对照与不确定性](../workloads/measurement-comparability.md)。
