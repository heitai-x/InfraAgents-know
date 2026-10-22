# 工具

本分类条目的适用条件、来源与未验证范围在各正文中说明。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-TOOLS-001：可观测性：执行事件与采集事件的映射](trace-observability.md) | v1 · seed | trace 是由采集范围和工具机制生成的观测，不是系统全部执行事实的无损副本。 |
| [K-TOOLS-002：时间线分析：从请求路径定位执行瓶颈](timeline-attribution.md) | v1 · seed | 时间线分析先建立关联和依赖，再用区间与关键路径解释延迟；热点累计时长只是入口。 |
| [K-TOOLS-003：Kernel 计数器：吞吐、访存与 Replay 语境](kernel-counter-replay.md) | v1 · seed | 硬件计数器帮助识别单 kernel 的资源限制，但它测到的执行可能与正常并发服务具有不同环境。 |
| [K-TOOLS-004：框架 Profiler：算子归属、内存与图边界](framework-profiler-scope.md) | v1 · seed | 框架 profiler 解释模型操作怎样落到设备工作，并通过 shape、调用栈与内存事件连接两层视图。 |
| [K-TOOLS-005：失败分析：最早偏离、最小复现与机制归因](failure-evidence.md) | v1 · seed | 失败记录的价值在于还原系统从哪一个状态开始偏离预期，并区分输入、实现、资源和外部环境。 |
| [K-TOOLS-006：正确性 Oracle：函数参考、状态变形与服务契约](correctness-oracles.md) | v1 · seed | 正确性验证需要一组互补的 oracle：数值函数、状态转换、采样分布和外部协议分别有可检查的关系。 |
