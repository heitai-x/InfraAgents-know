# K-HW-014：通信传输：注册、完成、可见与回收

推理通信既搬运 tensor，也传递执行责任。发送提交、传输完成、接收方可消费和请求状态提交是不同事件；只有把它们连接起来，才可以安全重叠通信与计算。

## 注册与操作类型限定访问

verbs 的 Memory Region 由范围、保护域与访问属性定义，local key 用于本地访问描述，remote key 用于允许的远端操作。key 或虚拟地址不是模型状态身份；同一地址被重新分配后，旧传输不得写入新对象。注册缓存可以摊薄注册成本，但必须检测释放、重分配和映射撤销。

SEND/RECV 与一侧 RDMA READ/WRITE 有不同的目的缓冲和通知约定。SEND 需要接收侧按相应协议提供接收资源；普通 RDMA WRITE 不能假定自动产生目标应用所等待的接收完成。带 immediate 的操作、传输类型与硬件能力也有具体限制，应按固定版本接口选用。[rdma-core v54.0](../sources/S-RDMA.md) 的原始 man pages 给出这些契约。

批量提交也可能只完成了前面一部分 WR 的提交，遇到首个可立即检测的错误时停止；已经提交的请求仍需结算。post 成功不是 CQE 成功。收到完成后先检查状态，失败完成中的部分字段不具有成功时的含义。

## 两个实例的逐事件时序

| 阶段 | 源实例 A | 目标实例 B |
|---|---|---|
| 准备 | 固定待发对象与逻辑代次 | 预留接收范围，建立有效注册/接收描述 |
| 协商 | 获取合法目的位置和传输 ID | 返回位置、范围、版本约定与容量结果 |
| 提交 | 发起对应 WR，保持在途保护 | 按实际操作类型等待数据与通知 |
| 传输完成 | 检查本地完成与错误 | 确认所需范围完整并建立消费者可见性 |
| 应用提交 | 保留协议要求的最后恢复责任 | 验证模型/布局/长度，发布可执行状态并确认 |
| 回收 | 在访问结束及责任移交后释放相应对象 | 消费者结束后允许复用目的槽 |

发送 staging 可以在其规定的本地使用结束后复用，而源端最后一份可恢复 KV 可能仍需保留到应用接管确认；不能把这两种对象的生命周期合并。固定 verbs 文档还明确 inline 发送有特殊的本地缓冲复用规则，应按实际操作解释，不能套用统一等待条件。

## CQE、通知和应用确认

CQE 对应具体工作完成；完成通道事件通知需要进一步 poll 并检查 CQ。通知可能与实际 poll 交错，不能假设每个通知恰好对应一个尚未消费的 CQE。一次性通知重置、事件确认和 CQ 容量也参与正确性；处理过慢导致 CQ overrun 后，不能继续把队列当作正常运行。

polling 可减少部分唤醒延迟但占用 CPU，事件等待可节省空闲资源但引入调度成本。批量 poll 与适度合并提交可以摊薄控制成本，又会改变尾延迟与在途容量。网络传输的完成保证应按 transport、flags 和 MR 属性理解；它不等同于接收方已经完成模型计算或提交外部响应。

## GPUDirect 还需要 GPU 可见性

[GPUDirect RDMA 的 CUDA 12.8 固定文档](../sources/S-GPUDIRECT.md) §2.7 说明，外部设备写入 GPU 映射与 GPU 正在运行的 kernel 之间需要明确的排序关系。NIC 完成不能被任意一个并发 kernel 当作读取新数据的充分条件；应先确认所需写入完成，再通过该版本支持的 CUDA 同步/提交关系启动依赖消费者。

直接路径还受 GPU/NIC、PCIe 拓扑、驱动、注册方式和平台支持约束；Unified Memory、物理共享内存与 GPUDirect 也不是同一承诺。比较显式主机 staging 与直接路径时，分别计入复制、注册/BAR 资源、CPU 控制开销及共享链路。可以建立路径不表示任意方向都有相同带宽。

## 小消息、大状态与共享拓扑

消息时间可先写为 α+D/B，但多个消息、并发流和分片会共享 NIC、交换链路、PCIe 与主存。小 metadata 可能受 α 和 CPU 处理支配，大 KV 更接近带宽限制。设定 512 MiB、有效带宽 20 GiB/s，纯传输下界为 25 ms；握手、布局转换与队列还未计入。

将 TP 通信、P/D 状态传输、缓存副本复制与制品加载放到同一拓扑图中，检查共享边的总需求。更深发送队列可以隐藏延迟，也可能加剧拥塞、拖延控制消息或占满目的容量。GPU 与 NIC 亲和、主机 NUMA 和接收方及时消费都影响最终路径。

## 取消、空工作和失败验证

没有 payload 的分片仍需按上层协议表达明确的空范围或完成状态；集合通信的参与顺序也不能因本地工作为空而随意跳过。取消先停止新增提交，再结算已提交 WR 和目的预留；超时本身不证明远端从未写入，因此重试要带同一逻辑传输身份与新的尝试代次。

验证包括提交部分失败、CQ 错误、接收资源不足、源退出、目的缓冲重用、通知与 poll 交错，以及 GPU 消费早于可见性的受控反例。记录实际字节、错误状态、排队和最后使用者，避免用“网络通了”替代完整状态传输协议。

<!-- infraagents:provenance -->

## 来源与推导

- [S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events](../sources/S-RDMA.md)：固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。
  [完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

- [S-GPUDIRECT：NVIDIA GPUDirect RDMA — CUDA 12.8.0](../sources/S-GPUDIRECT.md)：CUDA 12.8 固定页面描述 GPUDirect 注册、缓存、映射撤销、平台路径和 GPU 内存排序；§2.7 区分外部 DMA 与 GPU 消费依赖。
  [完整原文](../../references/S-GPUDIRECT/reader.html) · [全文文本](../../references/S-GPUDIRECT/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-NCCL-GROUP：NCCL — Group Calls](../sources/S-NCCL-GROUP.md)：NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。
  [完整原文](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [主机准备、DMA 与流水缓冲](../hardware/host-device-transfer.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [分布式状态迁移：身份、传输与提交点](../system-semantics/distributed-state-transfer.md)。
- [分布式 KV 存储：目录、有效副本与可恢复状态](../system-semantics/distributed-kv-store.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
