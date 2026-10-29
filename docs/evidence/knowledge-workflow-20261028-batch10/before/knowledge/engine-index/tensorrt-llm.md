# K-ENGINE-005：异步执行决策：结果确认、在途状态与回收

主机准备、设备计算、采样与外部状态传输可以部分重叠，但自回归依赖与存储生存期不会因异步提交而消失。设计需要区分业务完成、输出提交、设备完成和各个外部消费者结束。

## 重叠怎样产生收益

若 host 工作与设备工作互相独立且有足够缓冲，稳态周期可从 tₕ+t𝒹 接近 max(tₕ,t𝒹)；这不是单请求时延的通用公式。设定 tₕ=2 ms、t𝒹=5 ms，理想流水周期从 7 ms 降到 5 ms，最多约 1.4 倍的该段吞吐改善；实际同步、采样依赖或共享带宽会缩小重叠。

提前安排下一代输入可以减少 host 等待，也会增加在途状态。取消到达时，尚未提交的可见输出可以被抑制，已经发出的设备工作则可能需要排空。同步确认每步更简单；允许多代在途可提高流水效率，但增加代次映射、预留和迟到结果处理成本。

## 传输使用者与缓存回收

[已核对的 TensorRT-LLM 实例](../sources/S-TRTLLM-CODE.md) 对请求完成后的异步发送/保存计数，在多个传输消费者均结束后解除相应保护；加载走其他路径。它说明一个完成标记不能代替所有使用者的证据，不证明该组件同时实现了所有驻留和恢复策略。

```pseudocode
开始一个将读取缓存的操作前，登记其身份并保护相关存储
业务终止时停止生成新的可见输出，记录已有在途操作
每个操作在约定完成点撤销自己的使用权；重复或旧代完成不能释放新代存储
业务持有与所有实际消费者均结束后，决定保留缓存或使旧身份失效并复用存储
异常时保存失败与未完成状态，按实际传输/设备语义决定排空和恢复
```

若两个发送者持有同一状态，一个完成后另一个仍未完成，保护不能因“收到一次完成”而撤销。设备 kernel、发送和存储可能使用不同完成证据；将它们统称为 future 完成会丢失回收条件。

## 如何验证重叠而不牺牲语义

固定相同逻辑输入，比较同步与重叠候选的有效产出、输出顺序、停止后额外工作、峰值状态和尾延迟。分别延迟采样、传输完成或取消通知，检查结果归属与保护是否仍成立。host 等待减少、设备空隙缩小但总时间不变，提示其他瓶颈接管；吞吐增加同时显著扩大状态驻留，则需要重新评价容量而非只报告单步收益。

远端预取、SSD 驻留和故障恢复仍需各自的身份、预留、数据可见性与接管机制，分别见[多级驻留](../algorithms/tiered-residency.md)、[状态传输](../system-semantics/distributed-state-transfer.md)和[服务恢复](../system-semantics/serving-recovery.md)。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRTLLM：TensorRT-LLM Architecture Overview](../sources/S-TRTLLM.md)：已读 TensorRT-LLM 架构页将入口、PyExecutor、scheduler、model engine 与 sampler 连接成后台迭代，并介绍 graph padding 和 overlap 思路。
  [完整原文](../../references/S-TRTLLM/reader.html) · [全文文本](../../references/S-TRTLLM/fulltext.txt)

- [S-TRTLLM-CODE：TensorRT-LLM PyExecutor 固定提交](../sources/S-TRTLLM-CODE.md)：固定历史提交的 PyExecutor 包含调度准备、_schedule、_sample_async 和异步传输管理。AsyncTransferManager 在发送与保存的全部使用者完成后解除 KV 保护；加载状态由其他路径处理。
  [算法与机制说明](../sources/S-TRTLLM-CODE.md)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
