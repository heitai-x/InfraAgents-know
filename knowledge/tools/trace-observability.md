# K-TOOLS-001：可观测性：执行事件与采集事件的映射

trace 是由采集范围和工具机制生成的观测，不是系统全部执行事实的无损副本。

## 观测模型

系统先发生事件，采集器在被启用的 domain、进程和时间窗口中记录它，再由导出器转换成表或文件。任何一层的缺失都会让最终查询看不到事件。表不存在、表存在但空、窗口外、过滤后为空、无法关联，是不同状态。

缺少 kernel 表只支持“这份导出未提供该表”，不能推出“设备未执行”。证明没有执行需要覆盖相应进程、设备、窗口和事件类型的观测能力。相反，看到 kernel 也不说明它属于目标请求，仍需关联上下文。

## 覆盖矩阵怎样帮助推理

记录所需信号、采集是否启用、原始记录是否存在、导出是否成功和查询范围，可以明确每个结论依赖什么。主机 API、设备 kernel、memcpy、通信、NVTX 和 OS 调度各回答不同问题；一个工具的默认输出不能自动覆盖全部。

原项目 [CASE-0001](../cases/CASE-0001-nsys-observability.md) 通过有记录与缺表对照验证了这些状态区分，并保存原始清单。该证据支持观测工具链的行为，不将微型案例外推为模型服务表现。

## 缺失也可形成下一步信息

若候选原因是 CPU 阻塞，但只有 GPU kernel，下一次观测需要补主机线程/调用区间；若候选是请求错配，单个 GPU 利用率指标不足，需要请求到 batch 的映射。高质量采集由问题的可区分预测决定，避免无目标收集庞大 trace。

<!-- infraagents:provenance -->

## 来源与推导

- S-NSYS：[完整原文](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt)
- S-TORCH-PROFILER：[完整原文](../../references/S-TORCH-PROFILER/reader.html) · [全文文本](../../references/S-TORCH-PROFILER/fulltext.txt)
- S-CASE-0001：[完整原文](../../references/S-CASE-0001/reader.html) · [全文文本](../../references/S-CASE-0001/fulltext.txt)

- [S-NSYS：Nsight Systems — Post-Collection Analysis Guide](../sources/S-NSYS.md)：Nsight Systems 的原始事件与导出可用于查看 host API、设备 kernel、复制和关联信息。事件属于具体进程、线程、设备和时间窗口。

- [S-TORCH-PROFILER：PyTorch — torch.profiler](../sources/S-TORCH-PROFILER.md)：框架 profiler 可以采集算子、设备工作、shape、stack 和内存相关信息。更详细的采集会增加运行开销，部分选项也会延长张量引用。

- [S-CASE-0001：InfraAgents CASE-0001 evidence manifest](../sources/S-CASE-0001.md)：本地证据清单记录微型 CUDA 采集、导出、缺表对照和事件关联。所记首次设备开始减 host API 结束为 −1116 ns；3 kernel 累计 4352 ns、2 memcpy 累计 4896 ns。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [时间线分析：从请求路径定位执行瓶颈](../tools/timeline-attribution.md)。
- [主机与设备时间：偏序、区间并集和关键路径](../system-semantics/host-device-timing.md)。
- [失败分析：最早偏离、最小复现与机制归因](../tools/failure-evidence.md)。
