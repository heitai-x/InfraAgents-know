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

## 会话、模型版本与恢复代次

会话 ID 连接多轮任务，但一轮请求仍需独立的输入版本、父回合和提交位置；工具等待结束后到达的内容必须归属相应回合。多样本/beam 在请求内产生分支，分支重排改变物理位置而不改变父路径。[分支生成](../algorithms/branch-generation.md) 将这一关系延伸到 grammar、RNG 和 KV 的共同提交。

模型服务别名与具体执行版本也要分开。请求准入时绑定基座、adapter 与 processor 的有效版本；[模型切换](model-lifecycle.md)后，新请求可以使用另一版本，旧请求的身份保持原样。租户作用域还决定同一内容的缓存是否允许共享，不能只凭 token 相同合并拥有者。

迁移会改变执行地点，恢复会改变执行尝试代次，都不应改变已经对外承诺的请求身份。状态包同时携带逻辑恢复点和源/目标的尝试身份；目标只接收与当前授权相容的结果。连接 ID、worker PID 或 slot 号因重试变化，不能成为判定逻辑重复请求的唯一依据。

## 一个贯穿重排、取消和迁移的验证

构造 A、B、C 三请求，其中 B 含两个分支，C 属于另一租户且使用不同 adapter。让 prefill 长度不同、batch 压紧重排、B 淘汰一支、A 取消、C 迁移后原 worker 迟到返回。独立事件账逐次检查“结果行→请求→分支→逻辑位置→模型/adapter→尝试代次”的完整映射。

合法终态应是：A 不再交付新内容但在途资源正常完成；B 只有保留分支更新正式状态；C 只由当前有提交权的代次继续输出，旧结果不进入新 slot。分别核对采样参数、grammar、RNG、KV、计费和 trace 归属，不能用最终文本看起来合理替代身份正确性。

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
