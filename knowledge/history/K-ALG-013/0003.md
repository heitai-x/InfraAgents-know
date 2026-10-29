# K-ALG-013：Prefill/Decode 分离：状态传输与队列耦合

分离式服务把两个阶段的资源策略解耦，同时引入 KV 传输、布局转换和跨节点状态提交。

## 分离为什么可能有效

prefill 的较多 query 行适合矩阵计算，decode 则反复读取权重与历史状态。共置时，长 prefill 可能打断 decode 的稳定节奏，两个阶段也被迫共享相同并行度和容量分配。独立池允许分别选择 batch、并行策略和扩容比例，但每条请求必须把足够的中间状态交给 decode 池。

端到端首 token 路径变为排队 P＋prefill＋传输/转换＋排队 D＋必要的首步工作。若在 prefill 侧已生成首 token，其网络交付与 decode 接管可以重叠，但第二 token 的时间仍受接管约束。不能只缩短 prefill kernel 就假设 TTFT 和 ITL 同时改善。

## 传输字节从模型状态得出

普通 GQA 的原始 KV 约为 2LNHd b，具体分片后每节点发送多少取决于 TP/PP 布局。沿用 32 层、8 KV heads、128 通道、2 字节和 N＝4096 的算例，总状态约 512 MiB。理想有效链路 25 GiB/s 下，仅传输下界约 20 ms，尚未计排队、转换、握手和其他流量竞争。

若目标方使用不同 TP 数，源分片到目标分片需要重排；若 dtype 或量化尺度不同，则可能还要转换。传输格式必须包含状态几何、有效长度、位置规则、模型/adapter 身份和源执行代次。只发送裸页地址没有跨进程含义。

## 资源预留与提交

目标先预留完整状态容量，源才能有确定的接收目的地。分块传输可以与后续层计算重叠，但目标只有在其下一步所需的全部状态到齐且完成可见后才能消费。中断时，部分到达的页属于未提交传输，应可回滚；源不能因为发起发送就释放最后一份状态。

前缀复用让这个过程更有趣：目标已有部分有效前缀时可以只传缺失部分，但双方必须对同一逻辑身份和恢复位置达成一致。窗口、SSM 和多源 context 让“缺失部分”不再是一个简单连续 token 区间。

## 何时值得分离

采用分离需要节省的干扰与资源配置成本超过传输和新增排队。短 prompt、低到达率、弱互联下，共置可能更好；长 prompt、严格 decode SLO、强互联可能更有空间。P/D 实例配比由到达率、输入输出长度分布和阶段服务率决定，不能简单按相同 GPU 数各分一半。瓶颈转移后应重新优化整体 goodput。

## 场景：P/D 接管与远端 KV 的完整时间轴

以下均为设计设定。固定模型采用 32 层、8 个 KV heads、128 维、2 字节，prompt 为 4096 token；需传输的全层 KV 为 512 MiB。P/D 布局可转换且模型/位置一致，有效传输带宽设为 20 GiB/s，故纯传输下界为 25 ms。目标为客户端 TTFT≤150 ms，随后 token 间隔≤30 ms。

候选 A 共置：排队 70 ms、prefill 80 ms、首次发送 5 ms，首内容在 155 ms 可见。候选 B 分离：P 排队 10 ms、prefill 80 ms，在 t=90 ms 产生首 token；状态传输 25 ms、转换 5 ms、D 排队 10 ms，接管就绪在 t=130 ms。D 消费首生成 token、计算下一 token 需 10 ms，发送需 5 ms，所以第二 token 在 t=145 ms 可见。

若 B 在 P 完成后立即发送首 token，它在 t=95 ms 可见，TTFT 改善但第一次间隔为 50 ms，违反设定 ITL。若协议允许先等待 D 就绪再发送首 token，则首 token 在 t=135 ms 可见，第二 token 仍在 t=145 ms，两个设定约束均有可行空间。这个例子说明提前发送与接管延迟需要共同设计；它不保证后续 decode 或真实系统已满足 SLO。

第三个候选可以保留共置并使用受时间预算约束的 chunked prefill，减少长输入对已有 decode 的干扰，避免跨池传输；其首 token 和队列时间要在相同外部到达轨迹下校准。不能将 B 的更少干扰与 A 的零传输成本同时计入同一个候选。

## 容量和全局缓存进入接管

若到达 8 请求/s、每次均需 512 MiB，则平均 KV 传输需求为 4 GiB/s，额外副本、重试和 TP 流量另计。平均低于 20 GiB/s 不保证尾部无拥塞。目标既要保留接收中的状态，也要保留当前 decode 与下一步增长；两个池的配比由输入/输出分布与服务曲面决定。

目标已有正确前缀时可以只接缺失范围；远端 KV 命中也可能减少 prefill，但加上读取与转换后未必更快。目录旧记录、不同 adapter、不同量化或重分片都需要显式检查。P 产生并发送的首 token 是否已有自身 KV，按已计算边界说明，避免 D 接管时漏算或重复推进。

提前评估 D 的负载可以减少做完 prefill 后再拒绝的浪费，但负载预测到实际接管之间有时滞。新的请求和输出增长可能使原决定失效，因此目标需要本地再检查或有效预留。[Mooncake v1](../sources/S-MOONCAKE.md) 的提前拒绝讨论给出了这种反馈问题的实例，不能把当前利用率直接当成未来接管容量。

## 验证与选择反转

语义对照比较共置与分离的固定前缀状态、首 token、后续 logits、输出序号和终止事件，再覆盖部分命中、跨分片转换、取消、源失败和重复接管消息。性能记录首内容与第一次间隔、P/D 两侧队列、状态字节和无效 prefill，不能只统计 P 池吞吐。

短 prompt、低负载或弱互联可能偏向共置；长输入干扰显著且互联足够时，分离才更有空间。更高前缀命中率、长输出或不同精度又会改变池配比。保留这些条件与区分实验，才能在任务变化后重选方案。

<!-- infraagents:provenance -->

## 来源与推导

- [S-DISTSERVE：DistServe](../sources/S-DISTSERVE.md)：DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。
  [完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-MOONCAKE：Mooncake: Kimi’s KVCache-centric Architecture for LLM Serving](../sources/S-MOONCAKE.md)：§3–6 描述分离式 KV 池、块身份、层间传输、缓存感知调度、热点副本和过载提前拒绝。
  [完整原文](../../references/S-MOONCAKE/reader.html) · [全文文本](../../references/S-MOONCAKE/fulltext.txt)

## 延伸阅读

- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [分布式状态迁移：身份、传输与提交点](../system-semantics/distributed-state-transfer.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
- [分块 Prefill：服务时间切片与历史可见性](../algorithms/chunked-prefill.md)。
