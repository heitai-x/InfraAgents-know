# K-TOOLS-001：可观测性：执行事件与采集事件的映射

trace 是由采集范围和工具机制生成的观测，不是系统全部执行事实的无损副本。

## 观测模型

系统先发生事件，采集器在被启用的 domain、进程和时间窗口中记录它，再由导出器转换成表或文件。任何一层的缺失都会让最终查询看不到事件。表不存在、表存在但空、窗口外、过滤后为空、无法关联，是不同状态。

缺少 kernel 表只支持“这份导出未提供该表”，不能推出“设备未执行”。证明没有执行需要覆盖相应进程、设备、窗口和事件类型的观测能力。相反，看到 kernel 也不说明它属于目标请求，仍需关联上下文。

## 覆盖矩阵怎样帮助推理

记录所需信号、采集是否启用、原始记录是否存在、导出是否成功和查询范围，可以明确每个结论依赖什么。主机 API、设备 kernel、memcpy、通信、NVTX 和 OS 调度各回答不同问题；一个工具的默认输出不能自动覆盖全部。

可以构造两份采集材料作区分：一份明确启用并取得目标设备事件，另一份未启用该类采集。后者缺少相应表，只能说明该观测路径未覆盖；是否执行须由其他独立信号判断。这里给出验证关系，不声称已经完成一次项目实验。

## 缺失也可形成下一步信息

若候选原因是 CPU 阻塞，但只有 GPU kernel，下一次观测需要补主机线程/调用区间；若候选是请求错配，单个 GPU 利用率指标不足，需要请求到 batch 的映射。高质量采集由问题的可区分预测决定，避免无目标收集庞大 trace。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NSYS：Nsight Systems 2024.6 — User Guide](../sources/S-NSYS.md)：Nsight Systems 2024.6 用户指南描述 host API、设备 kernel、复制、范围和相关事件的采集与查看。进程、线程、设备、采集范围和时间窗口共同限定观测含义。
  [完整原文](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt)

- [S-TORCH-PROFILER：PyTorch — torch.profiler](../sources/S-TORCH-PROFILER.md)：框架 profiler 可以采集算子、设备工作、shape、stack 和内存相关信息。更详细的采集会增加运行开销，部分选项也会延长张量引用。
  [完整原文](../../references/S-TORCH-PROFILER/reader.html) · [全文文本](../../references/S-TORCH-PROFILER/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
- [主机与设备时间：偏序、区间并集和关键路径](../system-semantics/host-device-timing.md)。
- [失败分析：最早偏离、最小复现与机制归因](../tools/failure-evidence.md)。
