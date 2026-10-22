# K-HW-002：通信拓扑：从 Rank 组到瓶颈链路

多卡性能由数据经过的实际路径决定；rank 数只是拓扑模型的一部分。

## 拓扑是带容量的图

节点包含设备、CPU NUMA 域和网卡，边包含设备直连、PCIe、交换结构和跨节点网络。边的延迟、有效带宽、双向能力和共享上行影响集体操作。两个 GPU 虽然都支持 peer access，也可能经过带宽较弱或共享的路径；API 可达性不是等带宽保证。

一次通信可粗略写为 α＋n/β，α 表示启动/协调延迟，β 是路径有效带宽。小消息主要受 α，大消息主要受 β。多个流共享瓶颈边时，每流可获得的 β 下降；因此单链路 microbenchmark 不自动预测 all-to-all。

## 集合操作怎样使用链路

ring all-reduce 在理想均匀 P ranks 下，每 rank 传输量约为 2(P−1)n/P，阶段数量也随 P 增长。树形或其他算法改变延迟和流量组织。公式提供数量级，不代表某次 NCCL 一定选 ring，也不包含跨层网络争用。

TP 每层常有同步，小 batch 更容易暴露 α；EP 的 all-to-all 受路由偏斜和双向竞争影响；PP 更接近阶段间点对点；P/D 分离主要搬较大状态。把高频 TP 放在强互联域、把较低频通信放在更远层级，是可由这些模型推导的候选，而非所有部署的固定规则。

## NUMA 和网卡亲和

主机线程、pinned buffers 和网卡若落在远端 NUMA 域，数据可能额外穿越 CPU 互联。GPU 直连路径可减少某些主机搬运，但控制面与分配仍可能受 CPU 影响。通信发生在哪个进程、设备和接口，应与 rank 映射共同记录。

通信与计算重叠需要数据先就绪、接收方及时发起、硬件资源允许并行。独立 stream 只是提交关系，若两者争同一内存带宽，重叠也可能使各自更慢。判断收益看关键路径缩短，而非时间线上颜色相交。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-NCCL-GROUP：NCCL — Group Calls](../sources/S-NCCL-GROUP.md)：NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。
  [完整原文](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
