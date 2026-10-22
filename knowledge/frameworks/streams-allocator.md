# K-FW-003：Stream 与分配器：依赖顺序和对象生存期

stream 解决执行顺序，内存生存期解决地址何时可重用；正确的异步程序需要同时满足两种关系。

## 生产者和消费者

同一 stream 内的命令按其定义的顺序执行；不同 stream 没有自动的数据依赖。若 A 产生张量，B 消费它，需要一个从生产完成到消费开始的顺序边。host 已经拿到 tensor 对象，仅表示它持有描述符，不能证明设备内容完成。

event 可以表达特定位置的完成，consumer 等待该 event 可以避免全设备同步。event 记录必须在最后需要等待的生产操作之后。默认 stream、库内部 stream 和框架调度存在具体语义，不能把自己只创建一个 stream 等同于系统只有一个执行队列。

## 顺序正确仍可能地址失效

假设张量在 A 分配并写入，B 等待后读取；CPU 很快删除最后一个 Python 引用。若分配器只知道 A 的使用，就可能把内存交给 C 重写，而 B 尚未读取完。PyTorch 的跨 stream 使用需让 allocator 知道实际消费 stream，或显式持有对象并建立回收同步。顺序边保护内容何时有效，生存期保护内容还存在多久。

stream-ordered allocator 把分配/释放也放进执行顺序，跨 stream 使用仍需要可达的依赖关系。不能将“异步 free”理解为立即物理回收，也不能在未建立关系的 stream 上使用尚未生效的分配。

## 共享 workspace 的隐式依赖

两个算子输入输出独立，若共用同一个暂存区，仍然有写写或读写冲突。attention plan、graph replay、通信工作区尤其容易出现这种共享。可采用每在途代次独立 workspace、串行使用或事件保护；容量成本必须计入并发预算。

由对象关系推导事件，而不是随意增加同步，能同时解释正确性和性能。全设备等待使问题消失，说明存在时序相关线索，却不能定位是哪条依赖缺失；逐步缩小到具体生产、消费与回收关系，才能保持有效重叠。

## 多消费者需要完成关系汇合

同一权重或 KV 块可能被多个 stream、图实例或通信操作读取。释放前需要等待所有最后使用者；只跟踪最近发起的一个事件不足以证明其他路径结束。引用保护负责阻止新的回收，事件关系负责已提交访问的完成，两类账务需要同时成立。

把对象迁移到其他层也会产生消费者：D2H 仍在读取源，磁盘写入仍在读取主机缓冲，预取目的还可能等待转换。迁移发布后，原副本是否可释放取决于所有权和恢复协议，而不是“目的地址已经分配”。

## 操作取消与执行顺序分开处理

业务取消可以抑制输出和后续调度，却未必能撤销设备队列中的工作。先禁止新访问，再让在途任务汇合，最后回收；缓冲复用必须与旧代次完成记录区分。为使错误暂时消失而添加全设备同步，可能破坏本来可以保留的并行关系，应缩小到具体对象和依赖。

验证包括两个消费者不同步结束、跨 stream 临时变量销毁、共享 workspace 的并发复用，以及迁移与取消交错。比较独立工作区与事件串行化时，既测吞吐和等待，也测额外容量；容量压力本身可能让更多异步执行失去收益。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator](../sources/S-CUDA-ALLOC.md)：stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。
  [完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
