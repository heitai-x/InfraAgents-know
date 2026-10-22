# K-SYS-002：请求身份：逻辑序列、批次视图与结果归属

请求是长期语义对象，batch 行与物理槽是临时执行位置；正确性依赖它们之间可追溯的映射。

## 身份的四个层次

请求 ID 关联输入和用户输出；序列/分支 ID 关联 beam、投机或多样本；逻辑位置关联 token 历史；执行代次关联某次具体提交。batch index、压紧 token 行和 KV 物理页是这些对象在一轮的视图，不应成为持久身份。

一个请求可参加多个 batch，一个 batch 可包含多个请求，prefill 请求可占多行，投机可产生多条临时分支。分布式 trace 的 parent-child 树并不天然表达所有多对多关系，需要附带执行批与请求贡献映射。

## 压紧和逆映射

假设上一轮顺序为 A、B、C，B 完成后下一轮压紧为 A、C，C 的物理行从 2 变为 1。KV 句柄、length、采样参数、随机状态和输出目标都必须随 C 映射。只移动 token tensor 而其他数组维持原顺序，会把同一行拼成多个请求的混合状态。

变长 prefill 中，请求 A 占 0–127 行，C 占 128 行，末 token logits 的抽取位置与请求级数组又不同。可以用段起止偏移表达这种映射；所有消费者应基于同一轮快照，而非各自重新按当前活跃列表猜测。

## 晚到输出与槽位重用

物理槽重用需要结果携带原请求与代次。旧代次已取消时，业务结果可以丢弃，但其完成仍触发资源释放。若只按 slot 号判断，旧输出可能被当成新请求的第一 token。版本计数帮助区分同一槽的不同生命期。

## 可迁移的实现推导

长期请求状态以稳定身份组织，一轮生成不可变执行视图，执行返回带原映射的结果，再按当前语义状态决定提交、丢弃或回滚。这种分层同样适用于多模态块、分布式子任务和 CPU 小模型。它说明需要什么信息，但不限定具体容器或编程语言。

<!-- infraagents:provenance -->

## 来源与推导

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。
  [完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

- [S-VLLM-SCHED：vLLM — V1 Scheduler source reference](../sources/S-VLLM-SCHED.md)：vLLM V1 的调度入口根据请求所需 token 与已计算进度分配本轮工作，结合计算预算、encoder 输入和 KV 资源形成 SchedulerOutput。
  [完整原文](../../references/S-VLLM-SCHED/reader.html) · [全文文本](../../references/S-VLLM-SCHED/fulltext.txt)

- [S-OTEL：OpenTelemetry — Context propagation](../sources/S-OTEL.md)：OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。
  [完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [批量采样：概率变换、请求随机性与行映射](../algorithms/batched-sampling.md)。
- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
