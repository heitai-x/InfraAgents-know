# K-SYS-009：分布式状态迁移：身份、传输与提交点

状态迁移是一项所有权和可见性协议；字节到达只有在布局和版本被共同解释后才成为可用模型状态。

## 迁移对象的描述

一个状态包需要模型/权重/adapter 身份、逻辑前缀、位置规则、各层状态几何、dtype/量化 scales、分片方式和有效范围。网络地址只是运输位置，不是内容身份；源指针值在目标进程通常没有相同含义。

发送方和接收方 TP/PP 布局不同，状态可能需要切片、汇集或重排。KV 布局 NHD/HND 不同还可能增加转置。传输协议可以选统一中间布局以减少组合数，代价是额外转换；也可选特定端点对的直接路径，提高效率但增加适配种类。

## 从预留到发布

目标预留资源，源开始传输，目标逐块确认完成，最后将完整状态以某个执行代次发布给调度器。发布前的页不能被新 query 读取；发布后源才能按约定释放最后责任。部分失败需要回滚未提交预留，或记录可恢复的已到达区间。

如果按层流水接收，目标可以在所需层状态完成后推进，但整个执行仍需保证后续层及时可用及失败能一致处理。优化传输与计算重叠不改变“读之前完成、释放之前无人使用”的基础关系。

## 内容复用与重试

目标已有相同前缀时可只接缺失状态，判断依据必须包含所有影响中间值的条件。重复网络提交需要幂等传输 ID，避免重复记引用或将晚到旧代次覆盖新状态。对混合模型，复用单位是共同恢复位置所需的状态包，并非任意 token 区间。

迁移总收益为避免的重算或排队成本，减去序列化、转换、传输和接管成本。迁移可以改善资源均衡，也可能把网络变成新瓶颈。将资源归属和提交点先定义清楚，才能安全探索这些策略。

## 把网络完成接到模型消费之前

[传输协议](../hardware/transport-and-rdma.md) 区分提交、CQE、GPU 可见性和应用接管。目的空间存在、目录已登记或 NIC 报告完成，都不能独立证明下一次 forward 可以读取整个状态包。每个消费者依赖所需范围的完整性和相应执行平台的可见性。

传输 ID 标识逻辑对象和目标范围，尝试代次区分重复或迟到的操作。重复到达同一区间不能重复增加引用；目标已回收并重新分配后，旧写入必须停止或失效。发送 staging 的最后使用与源端最后恢复副本的释放分开处理。

## 活跃状态与失败窗口

append-only KV 可以分阶段复制旧前缀并追赶增量；窗口覆盖、递归状态和投机回滚则需要其他快照条件。目标每阶段预留容量，源完成/取消或目的失败时撤销未提交工作。最后的停止、追赶、授权与恢复位置要形成同一条时序。

对确认消息丢失、源端退出和目标重启，明确由哪个权威记录判断接管是否已生效；不能让新旧两端分别按超时同时提交。若没有足够证据恢复，选择重算或明确失败，比使用部分未知状态继续生成更符合既定语义。

验证逐项覆盖状态几何不同、TP/PP 重分片、量化元数据、部分范围、旧模型版本和重试。性能同时报告完整迁移时间、可见停顿、转换及临时峰值，不能只给最小尾部复制时间。

<!-- infraagents:provenance -->

## 来源与推导

- [S-DISTSERVE：DistServe](../sources/S-DISTSERVE.md)：DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。
  [完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-VLLM-PREFIX：vLLM — Automatic Prefix Caching](../sources/S-VLLM-PREFIX.md)：所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。
  [完整原文](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events](../sources/S-RDMA.md)：固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。
  [完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

- [S-GPUDIRECT：NVIDIA GPUDirect RDMA — CUDA 12.8.0](../sources/S-GPUDIRECT.md)：CUDA 12.8 固定页面描述 GPUDirect 注册、缓存、映射撤销、平台路径和 GPU 内存排序；§2.7 区分外部 DMA 与 GPU 消费依赖。
  [完整原文](../../references/S-GPUDIRECT/reader.html) · [全文文本](../../references/S-GPUDIRECT/fulltext.txt)

- [S-LLUMNIX：Llumnix](../sources/S-LLUMNIX.md)：§4 通过分阶段 append-only KV 复制和预留握手支持活跃迁移，采用全局实例级调度与本地请求选择、虚拟使用量。
  [完整原文](../../references/S-LLUMNIX/reader.html) · [全文文本](../../references/S-LLUMNIX/fulltext.txt)

## 延伸阅读

- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
