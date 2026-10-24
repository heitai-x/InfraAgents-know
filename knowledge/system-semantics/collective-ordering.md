# K-SYS-005：集合通信：参与集合、操作顺序与失败传播

分布式前向是一组共同执行协议；每个 rank 的局部控制流必须形成匹配的通信序列。接入并行边界、研究新的布局或验证通信原型时，都可以使用这些关系；具体要表达的字段和检查取决于所选数学分解、执行库与任务，不要求先形成固定数量的候选或统一计划文件。

## 集体操作的语义对象

communicator 定义参与 ranks，操作定义数据类型、数量、归约方式和必要的 root。all-reduce 需要所有参与者对同一逻辑步骤达成对应关系。一个 rank 提前进入下一层的 all-reduce，而另一个仍在上一层，可能让不同数据被匹配或形成等待。

group 调用帮助成组提交多个通信操作，不自动修正应用层顺序错误。异步返回也不表示输出已经可以由后续 stream 读取，通信本身要进入生产消费依赖图。

## 动态请求如何保持共同视图

TP ranks 通常共同处理同一逻辑 batch，EOS、取消、投机接受长度和新请求加入都需要一致更新。若 rank 0 因某请求取消跳过一层，而其他 ranks 仍执行其 collective，局部“省工作”破坏整体协议。可以统一调度决定或用明确的控制广播维护共同视图。

EP 的每对发送/接收数量可随路由变化，但双方对元数据和数据交换次序必须一致。零 token 专家/rank 仍需遵守协议要求，不能因为本地没有工作就跳过其他参与者等待的步骤。

## 失败是什么状态转换

一个 rank OOM、进程退出或通信失败时，其余 ranks 可能停在等待。任务级状态需要传播失败并终止相应执行组，保留最早错误和其他 rank 的上下文。超时只说明在时间预算内未完成，不自动区分网络慢、顺序错和对端已死。

恢复时，如果无法证明所有 ranks 的持久状态一致，可以从共同的已提交前缀重新建立执行，而不能让部分 ranks 沿旧状态继续。这里的提交点与 KV 迁移和请求代次相同，都是让局部结果变成全系统可依赖状态的边界。

## 调度代次与通信代次共同推进

动态 batch、专家路由和迁移改变本轮数据量，但 communicator、参与集合、操作序号和数据解释仍需一致。控制面应先形成共同执行决定，再让各 rank 提交匹配工作；本地取消或空分片不能独立改变其他参与者期待的操作序列。

并发通信与计算是否安全取决于固定库版本、communicator/stream 关系和全局调用顺序。某个历史系统选择另一条传输路径，不证明所有版本的 NCCL 都具有相同限制；反过来，有多个 stream 也不保证任意交错都合法。

## 错误传播与恢复证据

保留每个 rank 最早失败、已提交操作、完成状态及被阻塞的依赖。设备失效或无法证明共同状态一致时，应终止受影响执行组并从共同恢复点重建；不能让幸存 rank 独自跳过失败步骤。新的执行代次不能继续使用旧组未确认的局部结果。

故障注入覆盖一个 rank 在 metadata 交换前后、数据传输前后或提交前退出，以及零工作 rank 仍需参与的边界。检查所有参与者最终进入一致终态，资源完成关系不悬挂，并把诊断中的超时与根因分开记录。

## 张量解释、count 与每 rank 容量

NCCL Collective Operations 的固定定义要求各 rank 使用匹配 count 与 datatype；即便这些值相同，应用仍可能把不同请求顺序或 hidden 通道的元素错误地相加。调用还要保持全局轴、rank 布局、逻辑推进与生产消费关系一致。非连续输入是否需要 packing、允许何种原位使用，应按所选函数与固定版本核对；形状一致不自动建立这些保证。

以下以参与者数 \(p\)、每元素 \(b\) 字节及语义块长 \(N\) 为符号，列出输入与输出分开分配时的主缓冲关系，不含库内部暂存。

| 操作 | 每 rank 的输入与输出 | 容量与解释 |
|---|---|---|
| AllReduce | 输入 \(N\) 个值，各 rank 得到同一 \(N\) 元素归约结果 | 输入/输出各 \(Nb\) 字节；同一位置应有相同全局含义 |
| AllGather | 每 rank 输入 \(N\) 个值，各接收 \(pN\) 个值 | 输入 \(Nb\)、输出 \(pNb\) 字节；输出块按 communicator rank 排列 |
| ReduceScatter | 每 rank 输入 \(pN\) 个值，归约后各得到 \(N\) 元素块 | 输入 \(pNb\)、输出 \(Nb\) 字节；rank 决定取得哪一块 |
| Reduce | 各 rank 提供相同逻辑位置的 \(N\) 个值，仅 root 得到结果 | 仅 root 需要该操作的归约输出；root 是 communicator rank |
| Broadcast | root 的 \(N\) 元素内容分发给参与者 | root、内容版本及接收后的解释必须一致 |

这里的 \(N\) 是数学块长，不能不经核对就当成每个 API 的 count 参数。AllGather 的总接收数与每 rank 发送数不同，ReduceScatter 的总输入数与每 rank 输出数不同；字节数还需乘 dtype 大小。改变 rank 到设备的映射后，解释层仍需知道每块对应的全局张量；物理 GPU 编号不是 root 的同义项。

执行机制需要维持请求/微批、层、阶段、模型版本与通信代次的对应。诊断或交接时，可按需要记录这些关系及相关 communicator、元素解释、生产者和消费者。详细检查可以置于 test/profile 路径，不要求每次生产通信前增加全局校验，但正常执行所依赖的身份和顺序关系不能因此关闭。TP、PP、EP 的张量分解与逐层账本见 [推理并行](../algorithms/parallelism.md)。

## Group 的提交完成不同于通信完成

固定 NCCL 2.25.1 的 Group Calls 规定，组内通信调用返回时可能尚未把操作加入 CUDA stream，因此不能在 group 尚未合法结束时用 stream 完成检查代替组提交。聚合还会延后部分操作的启动；即使减少了 launch 数，若关键消费者等待更久，端到端可能变慢。

nonblocking communicator 增加另一层边界。`ncclGroupEnd` 返回 `ncclInProgress` 时，kernel 仍可能在后台入队；应按 `ncclCommGetAsyncError` 等该版本规定的状态检查确认相关 communicator 进入成功，再进行依赖这批提交的 CUDA 操作。这里的成功仅表示提交成功，之后仍需等待相应设备工作完成，后续 stream 才能在正确依赖下消费结果。

因此，一次调用的事实应区分“组已形成”“相关操作已提交”“设备通信已完成”“应用结果可提交”。失败可能发生在这些边界之间，局部函数成功不能抹掉稍后报告的错误。将不同 communicator 或 stream 混用时，也应核对当前固定版本的具体并发规则；本文的提交模型并不授权任意跨组交错。

## 用可辨识反例检查所选通信边界

一个可以手工复算的输入是四个 rank 各提供 8 个整数，rank \(r\) 的第 \(i\) 项为 \(100r+i\)，其中 \(r=0,1,2,3\)、\(i=0,\ldots,7\)。AllReduce 求和后每个 rank 的第 \(i\) 项为 \(600+4i\)；AllGather 保留四个按 rank 排列的完整块。检查 ReduceScatter 时，让各 rank 输入 32 项并采用同一生成关系，归约后 rank \(r\) 得到索引 \(8r\) 到 \(8r+7\) 对应的八项。全零或全一输入可能掩盖重复、交换和位置错误；这个精确整数构造用于检查值与布局，不替代浮点或模型质量验证。

局部原型可独立比较归约、拼接和 packing；还可以检验同形状但不同请求顺序、空专家、非整除维度及 padding。能够提前识别的参数矛盾应由适当的检查拒绝，未识别的问题仍须通过原型或实验暴露；不能把 benchmark 超时直接判为某一种根因。

涉及相邻生产与消费时，可让某个生产者晚就绪，检查通信不会读未完成输入；让消费者晚完成，检查缓冲不会提前复用。在取消、单 rank 失败和新代次恢复中，旧结果不能进入新请求。故障注入在有界、隔离执行组中完成，保留最早失败及各 rank 最后已知状态。未完成或被终止的检查保留失败/不确定结果，不能因其他 rank 得到了数值就记为全组通过。

完整系统应按 Task 原定的质量、性能与资源边界评估，所比较方案的任务条件应保持可比。生成服务可观察任务相关的 TTFT、ITL、尾延迟或合格吞吐；通信次数、暂存峰值、重排和慢 rank 等待用于检验当前机制，是否需要记录由问题决定，不额外成为所有 Task 的验收项。若局部 collective 更快但任务指标变差，可据此区分聚合等待、重排、GPU 资源竞争和慢 rank。局部、交互与完整任务的检查可以服务于不同阶段的探索，其结论限于实际覆盖范围。通信启动与带宽模型见 [通信拓扑](../hardware/topology-parallelism.md)，RDMA 完成和 GPU 可见性见 [传输协议](../hardware/transport-and-rdma.md)。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NCCL-GROUP：NCCL — Group Calls](../sources/S-NCCL-GROUP.md)：NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。
  [完整原文](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-24 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。新增 Group 提交边界来自固定 NCCL 2.25.1 原文，具体 CUDA 消费依赖来自下列固定异步执行说明；验证路径为建议，未宣称在目标多卡环境完成实测。

- [S-CUDA-ASYNC：CUDA 12.8 Asynchronous Concurrent Execution](../sources/S-CUDA-ASYNC.md)：§3.2.8.5 说明同 stream 顺序、跨 stream 依赖及不同同步边界；支持区分通信提交与消费者执行。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [分布式状态迁移：身份、传输与提交点](../system-semantics/distributed-state-transfer.md)。
