# K-TOOLS-005：失败分析：最早偏离、最小复现与机制归因

失败记录的价值在于还原系统从哪一个状态开始偏离预期，并区分输入、实现、资源和外部环境。

## 不同失败有不同语义

无法启动进程、运行返回非零、超时、被信号终止、数值检查失败和 SLO 未达标不是同一事件。前四种解释执行状态，数值失败解释任务函数，SLO 失败解释性能可行性。完整 stdout/stderr、退出状态、环境与输入身份提供追溯基础。

设备错误可能异步在后续同步点出现，traceback 的最后一行未必是错误的首次触发位置。分布式场景中一个 rank 的首个 OOM 可能表现为其他 ranks 通信超时。保留全组原始事件并定位最早相关偏离，优于只读取最后一条报错。

## 最小复现保留什么

最小化应保留触发机制：页边界错误需保留跨页访问，取消竞态需保留在途与重用，数值问题需保留敏感形状和精度，编译错误需保留触发 guard 的条件。删除所有并发后错误消失，不代表已经复现了原问题。

可以按输入、状态、执行并发和环境四轴逐步缩减，每次说明移除后哪种关系不再存在。确定性种子有助于固定随机输入，但异步竞态往往需要记录调度/代次而不只固定 seed。

## 修复形成什么知识

当修复改变了某条依赖或布局关系，并使预期现象恢复，记录这条关系及其条件。临时路径、一次机器不足和偶发网络故障通常留在案例。反复出现且可解释的模式才适合提炼成机制知识，例如“分配器不知道跨 stream 使用会提前重用地址”。

## 将故障注入绑定到状态边界

故障实验先定义预期契约和恢复点，再选择能区分原因的注入。下表是一组验证设计，不表示这些实验已经在目标服务运行。

| 故障 | 保留的触发条件 | 需要的证据与判定 |
|---|---|---|
| 主机 OOM/内存节流 | 限额与加载峰值、页缓存或 IPC 同时存在 | cgroup 有效限制、memory.events、最早失败分配与进程终态；区分 memory.high 压力和 OOM kill |
| GPU 容量不足 | 权重、KV 增长、adapter 换入或 workspace 峰值 | 各对象有效/预留/在途字节与首次失败；不能只看请求结束后的空闲显存 |
| 存储超时/截断 | 正在读取一个固定版本的特定分片或范围 | 已完成范围、后端错误、发布状态与缓冲引用；未完成数据不能进入计算 |
| 部分 rank 退出 | 全组已提交的操作顺序与动态 batch 元数据 | 所有 rank 的最早事件、失败传播和共同终态；通信超时可能只是后果 |
| 网络断开或确认丢失 | 已交付位置、在途状态和唯一提交权 | 重试/接管记录、是否重复内容、旧代次是否失效及迟到操作是否安全 |
| 滚动更新中失败 | 新旧版本同时存在，部分实例已切流 | 版本绑定、实际就绪数、排空请求和回退后的容量；不能只检查进程重新启动 |

资源故障分为环境限额、容量设计不足和泄漏，不应一律通过扩大资源后宣布修复。I/O 超时还要区分排队、设备服务、网络传输、重试与应用等待；超时阈值变大只改变表象，未必解释根因。

## 重复、压力与恢复的证据

在隔离测试环境内使用相同输入与状态边界比较故障前、故障中、恢复后；故障概率或持续时间改变时，保留相同 offered workload。统计失败请求、拒绝、恢复时间、无效计算和受影响的其他租户，不把失败样本从吞吐分母中删掉。

反复注入取消、分片失败与重试后停止新输入，检查私有状态与在途引用结束，共享池回到已定义的空闲基线。允许保留的编译缓存、arena 或公共权重应有解释；无法解释的逐轮增长才是进一步查找泄漏的直接信号。

修复验证同时包含触发故障的小例、仍保留并发关系的压力例和未参与诊断的代表负载。修复若改变重试、路由或缓存策略，还要检查输出提交与版本身份，防止把崩溃变成静默错误。一个故障点通过只能支持该路径与条件，不能被写成全部故障恢复已经正确。

<!-- infraagents:provenance -->

## 来源与推导

- [S-PY-SUBPROCESS：Python — subprocess](../sources/S-PY-SUBPROCESS.md)：Python subprocess 区分启动失败、返回码和超时等执行结果，stdout/stderr 与进程状态属于独立记录。
  [完整原文](../../references/S-PY-SUBPROCESS/reader.html) · [全文文本](../../references/S-PY-SUBPROCESS/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-LINUX-RESOURCES：Linux 6.12 — cgroup v2 and powercap](../sources/S-LINUX-RESOURCES.md)：Linux v6.12 的完整 cgroup v2 文档定义资源层级、CPU quota/权重、内存压力/上限、cpuset、进程和设备/RDMA 控制；完整 powercap 文档定义计量域、能量单位/范围和功率约束窗口。
  [完整原文](../../references/S-LINUX-RESOURCES/reader.html) · [全文文本](../../references/S-LINUX-RESOURCES/fulltext.txt)

- [S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events](../sources/S-RDMA.md)：固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。
  [完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

- [S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation](../sources/S-LINUX-IO.md)：Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。
  [完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

## 延伸阅读

- [工程认识如何成为可迁移的专家知识](../system-semantics/evidence-and-independent-checks.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
- [正确性 Oracle：函数参考、状态变形与服务契约](../tools/correctness-oracles.md)。
