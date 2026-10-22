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

<!-- infraagents:provenance -->

## 来源与推导

- [S-ASGI：ASGI — HTTP and WebSocket Message Format](../sources/S-ASGI.md)：ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。
  [完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。
  [完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
