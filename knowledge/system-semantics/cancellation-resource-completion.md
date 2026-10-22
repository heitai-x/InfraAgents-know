# K-SYS-004：取消与完成：业务终止、设备排空和资源回收

取消撤销未来服务意图，完成证明既有工作已结束；物理资源重用必须等待最后一个使用者。

## 请求终止不是一个瞬间

请求可以从接收进入排队、运行、生成、结束。取消可能在任意阶段发生：尚未提交时可以直接移出队列；已经提交时通常只能阻止后续推进与输出，等待在途操作完成。EOS、长度上限、用户取消、超时、内部错误是不同的业务终止原因，但都需要一致资源账务。

网络断开不总会即时送达执行线程，发送错误和断开通知也可能竞态出现。执行生命周期应由内部请求状态管理，而不是由某一个网络回调独占全部清理权。

## 三种引用

逻辑引用表示哪些请求或共享节点拥有状态；执行引用表示哪个 kernel/graph/传输仍在使用；输出引用表示哪个响应消费者还需要结果。只有相关引用都结束，物理块才可重用。Python 引用计数、KV 页引用计数和 GPU event 分别只覆盖其中一部分。

概念状态可表达为活跃 → 停止继续调度 → 等待在途完成 → 可回收 → 已重用。允许多个终止事件落到同一请求，因此释放操作应具备幂等语义；重复通知不能把共享页计数减成负数。

## 代次保护与所有权转移

slot 释放给新请求时递增代次，晚到输出按旧代次处理。跨设备 KV 迁移中，目标确认完整状态可用前源仍保有责任；迁移失败后不能让双方都认为对方负责。取消正在传输的请求需要处理发送、接收和已到达未提交页。

## 对性能的影响

过度提前回收造成数据污染；无限延迟回收则造成容量压力和抢占。更精确的最后使用事件既提供正确性，也缩短资源占用。服务在高取消率下的可用吞吐取决于取消响应时间、无效在途工作量和回收滞后，正常完成流量的平均成绩不能预测这些行为。

## 不同失败阶段的回收条件

| 触发点 | 立即停止的工作 | 仍需等待或确认的事实 |
|---|---|---|
| 输入尚在排队时取消 | 撤销准入与未来调度 | 解析/分词任务若已提交，结束后才能重用它持有的缓冲 |
| GPU 计算中取消 | 停止后续迭代与业务交付 | 最后 kernel/graph 的使用结束；业务取消通知不证明设备停止 |
| 权重/KV 从存储读取中失败 | 不发布部分对象，不再提交依赖它的计算 | 读取与转换的完成范围、在途写入和共享下载的其他等待者 |
| 跨设备传输时取消 | 撤销接管或新调度，保持唯一责任方 | 网络/设备完成与旧代次失效；应用超时不等于 DMA 已停止 |
| worker 进程退出 | 从路由移除对应执行代次 | 共享池、外部对象和对端注册状态由各自拥有者完成清理 |

一个请求取消不应取消其他请求共用的已封闭前缀、基座权重或下载。减掉逻辑引用后，还需检查共享对象是否有其他拥有者和在途读者。失败数据进入未发布状态；部分 I/O 成功不能被填零后冒充完整缓存继续运行。

## 超时、卸载与迟到事件

超时表示期限已过，是否能中止底层操作由具体接口决定。若无法取消某次传输，目标缓冲在真实完成之前继续隔离，完成后再丢弃其业务结果。仅递增请求代次可阻止错误提交，却不能阻止旧 DMA 写坏已经交给新请求的物理地址；两种保护都需要。

模型排空时，请求归零之后还可能存在预取、graph、编译或响应引用。卸载按照对象最后使用事件推进，失败重建也不继承旧进程未确认的完成状态。分布式组不能各自跳过局部失败的 collective；受影响组按[恢复协议](serving-recovery.md)共同结束或重建。

## 压力下的最终守恒检查

使用反复准入、取消、错误和 slot 重用的长序列，在页边界、传输分段与完成回调交错处注入事件。停止产生新请求并等待已定义的清理边界后，检查请求私有引用和在途引用归零、共享引用等于仍存活的拥有者、借用额度归还、旧代次不再拥有提交权。

分配器缓存或预留 arena 可以保留物理内存，因此不能要求驱动报告的总占用为零。应先定义空闲基线和允许保留的公共缓存，再检查多轮压力后没有持续增长的不可解释占用。若还有未结束的操作，报告其对象、最后使用者和阻塞事件，不能把等待超时记为已经成功清理。

<!-- infraagents:provenance -->

## 来源与推导

- [S-ASGI：ASGI — HTTP and WebSocket Message Format](../sources/S-ASGI.md)：ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。
  [完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。
  [完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events](../sources/S-RDMA.md)：固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。
  [完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

- [S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation](../sources/S-LINUX-IO.md)：Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。
  [完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

- [S-SERVERLESSLLM：ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](../sources/S-SERVERLESSLLM.md)：§4 的加载格式、块池和多级流水；§5 的 token 重算迁移；§6 的局部性与启动时间模型。
  [完整原文](../../references/S-SERVERLESSLLM/reader.html) · [全文文本](../../references/S-SERVERLESSLLM/fulltext.txt)

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
