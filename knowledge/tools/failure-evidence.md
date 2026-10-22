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

<!-- infraagents:provenance -->

## 来源与推导

- [S-PY-SUBPROCESS：Python — subprocess](../sources/S-PY-SUBPROCESS.md)：Python subprocess 区分启动失败、返回码和超时等执行结果，stdout/stderr 与进程状态属于独立记录。
  [完整原文](../../references/S-PY-SUBPROCESS/reader.html) · [全文文本](../../references/S-PY-SUBPROCESS/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [工程认识如何成为可迁移的专家知识](../system-semantics/evidence-and-independent-checks.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
- [正确性 Oracle：函数参考、状态变形与服务契约](../tools/correctness-oracles.md)。
