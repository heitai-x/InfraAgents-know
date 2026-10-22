# K-SYS-005：集合通信：参与集合、操作顺序与失败传播

分布式前向是一组共同执行协议；每个 rank 的局部控制流必须形成匹配的通信序列。

## 集体操作的语义对象

communicator 定义参与 ranks，操作定义数据类型、数量、归约方式和必要的 root。all-reduce 需要所有参与者对同一逻辑步骤达成对应关系。一个 rank 提前进入下一层的 all-reduce，而另一个仍在上一层，可能让不同数据被匹配或形成等待。

group 调用帮助成组提交多个通信操作，不自动修正应用层顺序错误。异步返回也不表示输出已经可以由后续 stream 读取，通信本身要进入生产消费依赖图。

## 动态请求如何保持共同视图

TP ranks 通常共同处理同一逻辑 batch，EOS、取消、投机接受长度和新请求加入都需要一致更新。若 rank 0 因某请求取消跳过一层，而其他 ranks 仍执行其 collective，局部“省工作”破坏整体协议。可以统一调度决定或用明确的控制广播维护共同视图。

EP 的每对发送/接收数量可随路由变化，但双方对元数据和数据交换次序必须一致。零 token 专家/rank 仍需遵守协议要求，不能因为本地没有工作就跳过其他参与者等待的步骤。

## 失败是什么状态转换

一个 rank OOM、进程退出或通信失败时，其余 ranks 可能停在等待。任务级状态需要传播失败并终止相应执行组，保留最早错误和其他 rank 的上下文。超时只说明在时间预算内未完成，不自动区分网络慢、顺序错和对端已死。

恢复时，如果无法证明所有 ranks 的持久状态一致，可以从共同的已提交前缀重新建立执行，而不能让部分 ranks 沿旧状态继续。这里的提交点与 KV 迁移和请求代次相同，都是让局部结果变成全系统可依赖状态的边界。

<!-- infraagents:provenance -->

## 来源与推导

- S-NCCL-GROUP：[完整原文](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt)
- S-NCCL-COLLECTIVE：[完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)
- S-CUDA-TOPOLOGY：[完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-NCCL-GROUP：NCCL — Group Calls](../sources/S-NCCL-GROUP.md)：NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [分布式状态迁移：身份、传输与提交点](../system-semantics/distributed-state-transfer.md)。
