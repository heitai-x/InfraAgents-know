# K-HW-002：通信拓扑：从 Rank 组到瓶颈链路

多卡性能由数据经过的实际路径决定；rank 数只是拓扑模型的一部分。

## 拓扑是带容量的图

节点包含设备、CPU NUMA 域和网卡，边包含设备直连、PCIe、交换结构和跨节点网络。边的延迟、有效带宽、双向能力和共享上行影响集体操作。两个 GPU 虽然都支持 peer access，也可能经过带宽较弱或共享的路径；API 可达性不是等带宽保证。

一次通信可粗略写为 α＋n/β，α 表示启动/协调延迟，β 是路径有效带宽。小消息主要受 α，大消息主要受 β。多个流共享瓶颈边时，每流可获得的 β 下降；因此单链路 microbenchmark 不自动预测 all-to-all。

## 集合操作怎样使用链路

ring all-reduce 在理想均匀 \(P\) ranks 下，每 rank 的发送量为 \(2(P-1)n/P\)。这里 \(n\) 是每 rank 输入字节数，reduce-scatter 与 all-gather 共 \(2(P-1)\) 轮。若每轮启动成本为 \(\alpha\)，单向有效带宽为 \(\beta\)，各块均匀、带宽稳定且无额外争用，并假设全双工发送与接收可以并发，则

\[
T_{\mathrm{ring}}\approx 2(P-1)\alpha+\frac{2(P-1)n}{P\beta}.
\]

该带宽项以每 rank 发送量和单向带宽计算，不再将同时发生的接收重复加到时间里。设 \(P=4\)、\(\alpha=5\) 微秒、\(\beta=40\) GiB/s：\(n=16\) MiB 时约为 0.61594 ms，\(n=8\) KiB 时约为 30.286 微秒，其中启动项已占 30 微秒。这些都是解释公式的设定值；小消息字节少，仍可能需要降低调用次数或依赖长度。

树形、分层或其他算法改变延迟和流量组织。这个理想 ring 模型不代表某次 NCCL 一定选择 ring，也没有计入跨层网络争用、packing、暂存和慢 rank。要比较端到端收益，还需要把不能隐藏的通信接到实际消费者的关键路径。

TP 每层常有同步，小 batch 更容易暴露 α；EP 的 all-to-all 受路由偏斜和双向竞争影响；PP 更接近阶段间点对点；P/D 分离主要搬较大状态。把高频 TP 放在强互联域、把较低频通信放在更远层级，是可由这些模型推导的候选，而非所有部署的固定规则。

## NUMA 和网卡亲和

主机线程、pinned buffers 和网卡若落在远端 NUMA 域，数据可能额外穿越 CPU 互联。GPU 直连路径可减少某些主机搬运，但控制面与分配仍可能受 CPU 影响。通信发生在哪个进程、设备和接口，应与 rank 映射共同记录。

通信与计算重叠需要数据先就绪、接收方及时发起、硬件资源允许并行。独立 stream 只是提交关系，若两者争同一内存带宽，重叠也可能使各自更慢。判断收益看关键路径缩短，而非时间线上颜色相交。

## 每条共享边的流量要有明确方向

每 rank 发送、接收以及端点内存读写是不同计量边界。理想 ring 的常用字节公式不能在没有说明时既算发送又重复算接收；全双工链路与共享上行也有不同容量约束。按实际经过的边统计 TP、EP、P/D、缓存副本与制品加载，可以找到单个操作微基准看不到的竞争。

GPU 直接路径的可达性还受注册方式、驱动、PCIe 根/交换结构和平台支持影响。对某个设备对测得的带宽不能替代其他 GPU/NIC 配对；主机中转和直接路径分别测量，记录 CPU、NUMA 与接收端处理是否进入关键路径。

## 以真实消息形态校准

小而频繁的控制/集合消息更容易暴露启动与同步，大 KV 或权重流更容易暴露带宽与拥塞。扫描消息大小、并发流、双向传输和计算同时运行的情况，观察实际 bytes、队列与用户时延，而非只看总网络利用率。

热点缓存节点可以成为所有计算实例共同的瓶颈；将请求分散到更多 GPU 反而可能增加对它的读取。副本与迁移的拓扑选择，应与 [全局缓存](../system-semantics/distributed-kv-store.md) 和 [路由](../algorithms/routing-and-migration.md) 联合推导。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-NCCL-GROUP：NCCL — Group Calls](../sources/S-NCCL-GROUP.md)：NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。
  [完整原文](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-24 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events](../sources/S-RDMA.md)：固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。
  [完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

- [S-GPUDIRECT：NVIDIA GPUDirect RDMA — CUDA 12.8.0](../sources/S-GPUDIRECT.md)：CUDA 12.8 固定页面描述 GPUDirect 注册、缓存、映射撤销、平台路径和 GPU 内存排序；§2.7 区分外部 DMA 与 GPU 消费依赖。
  [完整原文](../../references/S-GPUDIRECT/reader.html) · [全文文本](../../references/S-GPUDIRECT/fulltext.txt)

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
