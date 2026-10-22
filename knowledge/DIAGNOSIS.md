# 从行为回到机制

诊断是对系统关系的反向推理。先把现象定位到输入、状态、执行、调度或输出阶段，再选择能区分解释的观测。

| 实际行为 | 应还原的关系 | 专家正文 |
|---|---|---|
| 单请求正确，混合批错误 | batch 视图与持久请求状态的逆映射 | [身份](system-semantics/request-batch-identity.md)、[采样](algorithms/batched-sampling.md) |
| 前缀命中后结果不同 | 状态的计算等价类、共同恢复位置 | [前缀](algorithms/prefix-cache-identity.md)、[异构状态](algorithms/heterogeneous-state-cache.md) |
| 分块后 logits 分歧 | query 的绝对位置与历史可见集合 | [位置/mask](system-semantics/position-mask-cache.md)、[分块](algorithms/chunked-prefill.md) |
| 图重放或取消后串请求 | 地址稳定性、内容代次和最后使用者 | [图执行](frameworks/cuda-graph-buffers.md)、[生命周期](system-semantics/cancellation-resource-completion.md) |
| decode 随源长异常增长 | encoder、cross 投影与读源 attention 的成本分解 | [Cross KV](algorithms/cross-attention-cache.md) |
| MLA 容量小但带宽未降 | 是否仍把全部历史 latent 展开 | [MLA](models/mla.md) |
| MoE 某些 batch 特别慢 | 专家负载分布、dispatch 流量和慢端 | [MoE](models/moe-routing.md)、[拓扑](hardware/topology-parallelism.md) |
| 首 token 快但输出经常停顿 | 迭代时间片、排队与客户端缓冲 | [连续批](algorithms/iteration-batching.md)、[指标](workloads/metric-events.md) |
| CPU 忙而 GPU 有空洞 | 准备、提交、隐式同步和可重叠区间 | [异步调度](algorithms/async-scheduling.md)、[数据流水](hardware/host-device-transfer.md) |
| 某些长度第一次特别慢 | 编译 guard、图桶、autotune 与冷缓存 | [编译](frameworks/compile-dynamic-shapes.md)、[实验](workloads/measurement-comparability.md) |
| 吞吐提高但用户失败增加 | offered/admitted/completed 与联合 SLO | [Goodput](workloads/slo-goodput.md)、[到达](workloads/arrival-and-concurrency.md) |
| 音频平均足够快仍卡顿 | 生成时钟、解码 lookahead 和播放缓冲 | [语音](models/speech-generation.md)、[全模态](models/omni-multimodal.md) |
| 分布式卡住，最后只见超时 | rank 共同视图、操作顺序和首个失败 | [集合通信](system-semantics/collective-ordering.md)、[失败分析](tools/failure-evidence.md) |

[可观测性](tools/trace-observability.md) 说明当前材料能回答什么，[时间关系](system-semantics/host-device-timing.md) 说明累计时间为何不等于延迟。新的解释最终回到原机制章节，保持知识按系统关系组织。
