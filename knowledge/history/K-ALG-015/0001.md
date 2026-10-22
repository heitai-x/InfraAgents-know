# K-ALG-015：推理并行：张量分解、流水、专家与上下文

并行策略改变张量归属和同步边界；选择依据是每阶段的计算、状态与通信分解。

## 从一层矩阵乘法推导 TP

线性层按输出通道切 W 时，各 rank 得到部分输出；后续若接受同样分片就可继续，若需要完整向量则要 gather。按输入通道切 W 时，每 rank 计算部分和，必须 reduce 才得到完整输出。Transformer 的列并行、行并行常成对组织，以减少不必要的 gather。

TP 把权重和某些激活分散，但每层通信频繁。小 batch decode 的计算时间短，集合通信启动延迟容易占主导。KV heads 少于 TP 数时，还可能复制 KV 或采用不同分片，单卡 KV 不能机械地除以 TP。

## PP 与 DP 的不同扩展目标

PP 按层切模型，阶段之间传激活。多个微批可以填充流水；单请求逐 token 仍要走完整条流水，并等待下一 token。阶段不均衡和填充/排空产生气泡。PP 常有助于容纳大模型，不保证最低单请求延迟。

DP 复制整个服务副本，各自处理不同请求，减少单请求跨卡同步。模型能放入单副本且请求足够多时，DP 可以提升总吞吐；但权重复制、前缀缓存分散和负载不均衡影响实际效率。前缀亲和路由又会改变各副本队列。

## EP 与 CP 如何增加新轴

EP 将不同专家放到不同 rank，路由后按专家 dispatch，再将输出 combine 回原 token。流量由 token 路由决定，热点专家可成为慢端。专家均衡的训练统计不保证某个在线小批的均衡。

CP 将长上下文分片。attention 需要跨分片交换 K/V 或合并局部 softmax 统计；在线 softmax 的可合并性提供数学基础，但通信与临时空间仍有代价。窗口或稀疏可见性改变应交换的数据量。序列并行一词在不同实现中也可能指激活分片，阅读实现时应先确定到底哪一轴被切分。

## 组合策略的容量与时间模型

策略可以组合，例如节点内 TP、节点间 PP、请求间 DP、专家层 EP。每个并行组要说明成员、张量归属和操作顺序。算例中某层计算从 8 降到 3 单位，但新增不可隐藏通信 4，则净时间为 7；若再分卡只省 1 却增 2，就越过收益拐点。用实际通信路径的有效带宽和启动延迟代入，比按卡数线性估算可靠。

prefill/decode 的最佳组合可能不同，阶段分离则增加跨布局 KV 转换。应把转换代价一起计入整体路径，避免得到两个局部最优而整体更慢的配置。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-MIXTRAL：Mixtral of Experts](../sources/S-MIXTRAL.md)：Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。
  [完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [MoE：路由、Dispatch/Combine 与专家负载](../models/moe-routing.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
