# K-WORK-010：多轮与分支负载：依赖、状态驻留和到达过程

一组独立 prompt 不能完整描述聊天、工具调用和分支推理。后续请求的输入、到达时间与缓存身份可能取决于先前输出；这些依赖会改变可并行工作、资源驻留和性能评估。

## 会话与请求是两层身份

会话记录已提交历史、模型/processor 版本和用户可见顺序；请求记录本次输入、deadline、生成配置与执行状态。一个会话可以产生顺序回合，也可以 fork 为多个分支再 join。分支共享前缀不意味着后续写入可以共享同一可变状态；取消一个分支也不应撤销其他分支已提交的结果。

用有向依赖图描述生成、工具、检索和用户等待节点。边界必须说明依赖的是首 token、完整输出还是通过质量检查的结果。工具输出进入下一轮后，会改变 processor 输入和缓存身份；只按原会话 ID 命中缓存不足以保证计算等价。[SGLang 固定版本](../sources/S-SGLANG.md) §2–3 给出多调用、fork/join 与前缀复用的具体研究入口。

历史压缩、截断、模板变化或模型切换会改变后续模型函数。它们可以是合法应用策略，但必须成为 workload 条件；不能为获得更好性能，在基准中悄悄删除历史 token 或把原本依赖输出的调用提前发出。

## 回合长度和重用距离

第 k 轮的输入长度可以写为 Lₖ=S+Σⱼ<ₖ(Uⱼ+Aⱼ)+Uₖ，其中 S 是固定系统前缀，U 是用户输入，A 是已提交回答；模板与工具内容需要另外计入。这个式子只适用于完整保留历史的设定。缓存命中省去可复用部分的 prefill，decode 对可见历史的访问仍然存在。

设定系统前缀 100 token，每轮用户输入 20、回答固定 30，共三轮。各轮 prefill 长度为 120、170、220，完全重算合计 510；若前轮完整 token 前缀的状态均可保留并正确补齐，三轮需处理的新输入为 120+20+20=160 token。自回归执行中“最后生成的 token 已发送”不一定意味着该 token 的 KV 已写入，因此下一轮还可能需要推进这个边界 token。基准应同时记录逻辑历史与真实可复用的计算前缀。

思考时间或工具等待决定缓存保留多久。近似地，若会话完成率为 λₛ、等待期平均 τ、每会话保留 K 字节，则等待会话占用约 λₛτK；这依赖稳定到达和相应的平均定义。高命中率的代价可能是长时间占用容量，不能只统计省下的 token。

## 闭环和开放到达不可混用

闭环用户等待本轮响应后再思考并发出下一轮。稳定条件下，N 个用户满足交互响应关系 N≈λ(R+Z)，其中 R 是平均响应时间，Z 是平均思考时间。服务变慢会自动降低该负载生成器的请求率；因此它适合描述这类用户行为，却不能单独证明系统在固定外部到达率下能保持稳定。

开放到达由外部时钟产生，不随系统变慢而自动减压。回合间有因果依赖时，可以固定会话开始到达过程，并在会话内部保留依赖；这比把所有轮次完全独立重放更接近原问题。比较缓存或调度策略时，要同时报告会话到达、请求到达、分支数、完成率及未完成会话，避免只对成功样本计算平均。

## 分支预算与系统目标

设一个共享前缀产生 b 个分支，每个分支预计生成 d token。即使前缀存储共享，新增状态和执行工作仍随分支增长；join 可能等待最慢分支，提前完成的结果也会占缓冲。更大的分支数可能提高应用质量，也可能恶化延迟和成本，不能只把生成 token 数当作成功产出。

需要定义会话成功条件：所有必要回合完成、合法分支被选择、工具结果版本正确，以及最终质量达标。请求级 TTFT/ITL 与会话级完成时间分别度量不同约束。优先处理能解除下游依赖的请求、按租户公平分配，或按短工作优先推进，都是候选策略，选择依赖质量目标与长请求饥饿约束。

## 设计可区分的对照

构造相同模型、token 分布与资源预算下的四类负载：独立请求、短等待多轮、长等待多轮、分支后合并。比较关闭缓存、按容量保留和按重用估计保留时的计算量、真实物理占用、命中等待与会话完成率。再改变思考时间、分支宽度和模板/模型版本，检查策略是否按预期反转。

正确性覆盖分支写时复制、同会话并发修改、上一轮末 token 状态、工具失败、会话取消和迟到响应。可复算轨迹用于验证调度与容量解释；实际质量和收益仍需在独立任务上评估，不能由“前缀 token 相同”一项代替。

<!-- infraagents:provenance -->

## 来源与推导

- [S-SGLANG：SGLang: Efficient Execution of Structured Language Model Programs](../sources/S-SGLANG.md)：SGLang 的 RadixAttention 利用 token 前缀树组织复用状态，结构化执行场景可以复用重复前缀计算。
  [完整原文](../../references/S-SGLANG/reader.html) · [全文文本](../../references/S-SGLANG/fulltext.txt)

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：Orca 以生成迭代为调度边界，让完成请求退出并让新请求进入；不同算子依据是否适合聚批采用相应执行组织。请求不必等待原批全部结束才获得资源。
  [完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

- [S-VLLM-PREFIX：vLLM — Automatic Prefix Caching](../sources/S-VLLM-PREFIX.md)：所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。
  [完整原文](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt)

- [S-MLPERF-SCENARIOS：MLCommons — Inference Mobile Scenarios](../sources/S-MLPERF-SCENARIOS.md)：不同推理场景使用不同请求驱动方式，例如前一响应完成后再发与独立到达。相同平均吞吐并不表示相同负载压力。
  [完整原文](../../references/S-MLPERF-SCENARIOS/reader.html) · [全文文本](../../references/S-MLPERF-SCENARIOS/fulltext.txt)

正文中的设定算例、资源账和机制组合是基于上述资料的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；本次编辑日期不改变来源时间。

## 延伸阅读

- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
