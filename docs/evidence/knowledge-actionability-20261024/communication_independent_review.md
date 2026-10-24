# Communication 内容独立审查

审查者：`/root/knowledge_use_audit`；作者：`communication_content`；日期：2026-10-24。
基线：`f640148464929f803a53b55e27ee4cbb83dc8657`；固定来源上限：2026-04-30。

## 最终结论

**修订后通过，未留下未解决意见。** 四篇在公式、数量与字节单位、NCCL nonblocking 提交与实际完成、CUDA stream 与 allocator 生存期、verbs/GPUDirect 可见性方面均通过实质核对。首轮提出的一处测量窗口表述和一处 ring 带宽假设，均由根代理修正并经独立复读关闭。

本审查没有执行 GPU、NCCL 或服务 benchmark；全部数值核验为设定算例的独立算术复算。

## 发现与关闭记录

### R1：medium，已关闭

文件：`knowledge/hardware/cpu-gpu-overlap.md`；段落：用可区分实验选择同步、重叠或减少搬运；本次快照行号：78。

首轮意见：将有限批完成时间和稳态吞吐窗口合并为一个强制口径。有限批计时必须直到最后批完成；持续服务的稳态窗口可以与停止接入后的排空分开，但须定义窗口和在途请求的归属、追踪结局。当前绝对表述容易把稳态与有限 workload 的指标混用，也与 service-protocol 对稳态窗/排空的区分不一致。

处理结果：已分别定义有限任务完成时间包含排空、明确稳态窗口及窗口两端在途工作/计数口径/窗口后结局；不再强制将稳态窗与排空混为一个口径。

修订后 SHA-256：`1a4eb23a214d12081e475035460f98082f3855739e2312ae1a1a41b2be8c76a0`。

### R2：low，已关闭

文件：`knowledge/system-semantics/distributed-execution-contracts.md`；段落：成本模型要从实际不能隐藏的路径计算；本次快照行号：64。

首轮意见：理想 ring 公式和数值正确；建议显式说明发送与接收方向可同时工作，B 是单向可持续带宽而不是双向共享总带宽。若方向竞争同一资源，需要重新合计该共享资源流量。

处理结果：已显式限定全双工、发送与接收可并发、B 是单向有效带宽；原公式与计算在此假设下成立。

修订后 SHA-256：`ae97eb296bb0528ed2100677e6d50d066a9dedb8245a9f9e3034a7149488e5c2`。

## 核对结果

- **four_stage_pipeline：pass**。P0=max(tp,th,tg,td); T_N=sum(t)+(N-1)P0；假设：固定阶段时长、独立资源、足够有界缓冲、跨批无额外依赖；不是自回归单请求承诺；独立复算：{"serial_8_ms": 112, "pipeline_8_ms": 70, "first_batch_ms": 14}
- **slot_lifetime_bound：pass**。n_j >= ceil(L_j/P)；假设：固定生存期、启动周期、完成边界即刻复用；正文明确不能用平均值保证安全；独立复算：{"input_slots": 2, "output_slots": 2, "input_output_pool_MiB": 34}
- **shared_bandwidth_bound：pass**。P >= Q/B；独立复算：{"Q_MiB": 128, "B_GiB_per_s": 8, "period_ms": 15.625}
- **ring_cost：pass**。2(p-1)alpha+2(p-1)A/(pB)；假设：根代理已补足全双工、收发并发与单向 B 的适用条件。；独立复算：{"p": 4, "alpha_us": 5, "B_GiB_per_s": 40, "A16MiB_ms": 0.6159374999999999, "A8KiB_us": 30.28610229492188}
- **collective_buffer_sizes：pass**。AllReduce N->N，AllGather N->pN，ReduceScatter pN->N；表中明确分离主缓冲，库暂存另计；元素数乘 dtype 字节得到容量。
- **rank_distinct_values：pass**。四 rank 的第 i 项 100r+i，sum_r=600+4i；ReduceScatter 对 32 项输入取 rank r 的 8r..8r+7 块。
- **nonblocking_nccl_group：pass**。ncclGroupEnd 的 ncclInProgress 是后台提交进行中；各相关 communicator 成功才可依赖其 CUDA 提交；成功 issue 不等于通信设备完成。与固定 NCCL 2.25.1 Group Calls 原文一致。
- **stream_and_allocator：pass**。wait_stream 处理生产消费依赖；record_stream 处理 allocator 存活期。CUDA allocator 的跨流分配/使用/释放条件、legacy/PTDS 默认 stream 区分有原文支持。
- **rdma_visibility：pass**。一般 WR 缓冲完成后复用、inline 例外，以及本地 completion 与 GPU 可见性分开；与固定 verbs/GPUDirect 原文一致。
- **engineering_actionability：pass**。四篇均给出触发场景、输入上下文、候选选择、工程产物、反例及局部—集成—系统验证。异步故障和取消验证是建议，未宣称已在目标环境运行。
- **source_and_noncode_boundary：pass**。固定来源日期全部不晚于 2026-04-30；CUDA/NCCL 版本和非通用化限制明确；未增加来源；正文无实现代码、命令、伪代码或 code fence，本地链接存在。
- **finite_vs_steady_measurement：pass_after_revision**。已分别定义有限任务完成时间包含排空、明确稳态窗口及窗口两端在途工作/计数口径/窗口后结局；不再强制将稳态窗与排空混为一个口径。

## 正文最终快照

初稿全文阅读四篇，并核对两篇旧文相对基线的完整 diff。修订后再次逐句读取 CPU–GPU 重叠的系统验证段，以及分布式执行的 ring 成本段。四篇最终 SHA 均重新计算，代码围栏为零，本地链接不存在缺失。

- `knowledge/hardware/cpu-gpu-overlap.md`，101 行，最终 SHA-256 `1a4eb23a214d12081e475035460f98082f3855739e2312ae1a1a41b2be8c76a0`。
- `knowledge/system-semantics/distributed-execution-contracts.md`，106 行，最终 SHA-256 `ae97eb296bb0528ed2100677e6d50d066a9dedb8245a9f9e3034a7149488e5c2`。
- `knowledge/hardware/host-device-transfer.md`，93 行，最终 SHA-256 `8d8a8ccc4f67e89cbab3c4baf9896d549e115a7503bcaa61b6e05b58cc20c162`。
- `knowledge/system-semantics/collective-ordering.md`，82 行，最终 SHA-256 `74e9e7c694171602d09518dc7639298c4fd07109304c9918fb580501a4c25ad2`。

## 原文阅读范围与 SHA

以下 SHA 属于实际读取的源文件；范围仅表示所列段落，未将摘读冒充整本手册全文。版本与日期取同目录固定 provenance。

### S-NCCL-COLLECTIVE / fulltext.txt

路径：`references/S-NCCL-COLLECTIVE/fulltext.txt`。

SHA-256：`03fee76f12d44b950f31ce903bf28d4268bab2e65f805d62523c6dff3ddc59eb`。

版本：`NCCL 2.25.1`；可用日期：2025-01-27；行范围：1–51。

全文；collective 匹配条件、各操作 count/rank 语义。

### S-NCCL-GROUP / fulltext.txt

路径：`references/S-NCCL-GROUP/fulltext.txt`。

SHA-256：`75d2257017fc2e2409f7a672b9f2bc5fc01a54a4c098612f3abf8d83b5691ba5`。

版本：`NCCL 2.25.1`；可用日期：2025-01-27；行范围：1–416。

全文；组内调用与提交、嵌套组、多 stream、nonblocking group 的 ncclSuccess/ncclInProgress。

### S-CUDA-ASYNC / fulltext.txt

路径：`references/S-CUDA-ASYNC/fulltext.txt`。

SHA-256：`1f45872f110694d59f7e9b55ea37150648376f92ba85b0c3922b712a73736a32`。

版本：`CUDA 12.8.0`；可用日期：2025-01-27；行范围：3980–4085, 4359–4430, 4608–4705。

完整相关段落：锁页/映射主存、异步执行/复制能力、stream、legacy 与每线程默认流及显式隐式同步；未阅读全文 Programming Guide。

### S-CUDA-ALLOC / fulltext.txt

路径：`references/S-CUDA-ALLOC/fulltext.txt`。

SHA-256：`1f45872f110694d59f7e9b55ea37150648376f92ba85b0c3922b712a73736a32`。

版本：`CUDA 12.8.0`；可用日期：2025-01-27；行范围：46639–46755。

§11.3 相关 API Fundamentals 段落：分配返回指针、跨 stream 使用与释放；未阅读全文 guide。

### S-CUDA-BEST / fulltext.txt

路径：`references/S-CUDA-BEST/fulltext.txt`。

SHA-256：`11b3bf6520b02e730c2ce3a875c198563a5f432b405bec29350cf8b2dd4cffe2`。

版本：`CUDA 12.8.0`；可用日期：2025-01-27；行范围：560–680, 2795–2825。

§9.1 主机设备传输、锁页、复制计算重叠的相关完整说明；§9.4 NUMA 相关完整说明。

### S-TORCH-CUDA / cuda.rst

路径：`references/S-TORCH-CUDA/upstream/docs/source/notes/cuda.rst`。

SHA-256：`a292f6da55ca6d91d0f45444e449e669152422e12f644a7e617b0c16f12aa725`。

版本：`647b6a2ff9a83a886464932dfad78781ceb61cb6`；可用日期：2026-04-30；行范围：280–376, 1260–1298。

Asynchronous execution、CUDA streams 中 wait_stream 与 record_stream 的完整相关说明；Use pinned memory buffers。

### S-LINUX-RESOURCES / cgroup-v2.rst

路径：`references/S-LINUX-RESOURCES/upstream/Documentation/admin-guide/cgroup-v2.rst`。

SHA-256：`ad3bd37498a57cc249326fd5c2d609669b32611fe69d126df3c3abec1539f572`。

版本：`Linux v6.12 commit adc218676eef25575469234709c2d87185ca223a`；可用日期：2024-11-17；行范围：2297–2353, 2350–2410。

cpuset.cpus.effective、cpuset.mems/effective、内存迁移代价与不完整性。CPU 资格相关范围在本轮作者任务已读，内存节点范围在独立审查补读。

### S-TORCH-NUMERIC / numerical_accuracy.rst

路径：`references/S-TORCH-NUMERIC/upstream/docs/source/notes/numerical_accuracy.rst`。

SHA-256：`061656553a7f8ff52dea30cd57e13f4463dfb7597fb2385e838ed9da96dc34c4`。

版本：`647b6a2ff9a83a886464932dfad78781ceb61cb6`；可用日期：2026-04-30；行范围：1–90。

浮点非结合性、逐位结果限制和批处理差异的相关说明；只核对章节实际使用的数值边界。

### S-RDMA / ibv_post_send.3

路径：`references/S-RDMA/upstream/libibverbs/man/ibv_post_send.3`。

SHA-256：`7492a3d1a7438bf6922b59a70dc6e1bde8c1b74c69c86d6aff13913a4b5a0ec2`。

版本：`rdma-core v54.0 commit 77128e6b6d32a1aad7c1bec9d6eb1f061b63cf39`；可用日期：2024-10-06；行范围：125–190。

支持操作、flags、返回与 NOTES；发送源复用与 inline 例外。

### S-RDMA / ibv_poll_cq.3

路径：`references/S-RDMA/upstream/libibverbs/man/ibv_poll_cq.3`。

SHA-256：`1c11bde1b5fb8e6eb11927ea3094eeeecd8122aab724f4d7437fdb26267a36c7`。

版本：`rdma-core v54.0 commit 77128e6b6d32a1aad7c1bec9d6eb1f061b63cf39`；可用日期：2024-10-06；行范围：1–93。

完整 man page；CQE 状态、有效字段、消费与 CQ overrun。

### S-GPUDIRECT / fulltext.txt

路径：`references/S-GPUDIRECT/fulltext.txt`。

SHA-256：`657c37b1e20c78292dddfc81c172f676ec5868c3f39884127f0f636082fced8e`。

版本：`CUDA 12.8.0`；可用日期：2025-01-21；行范围：350–395。

完整 §2.7 Synchronization and Memory Ordering：第三方 DMA、CPU initiated CUDA 排序与并发 kernel 可见性。

## 限制

- 审查固定本地原文和 provenance，没有把此次阅读说成整个 CUDA/PyTorch/Linux 仓库全文。
- host-device-transfer 与 collective-ordering 的既有来源映射只核对本次新增或修改的内容；未重新审查所有未修改旧来源。
- 没有新增网络来源、GPU/NCCL 执行、真实请求或服务 benchmark；算例是独立算术复算。
- 审查者未修改被审查四篇正文；根代理两处修订已经独立复读。结论绑定本报告的最终 SHA，后续内容变更应另行核对。
