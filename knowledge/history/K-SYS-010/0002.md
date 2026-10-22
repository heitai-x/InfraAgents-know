# K-SYS-010：系统设计推导：把模型、缓存、执行与目标组合起来

专家指导的产物是一条从任务条件到系统结构的推导链：状态模型、资源模型、候选设计及其可区分的结果。

## 统一问题表述

输入条件包括模型函数与状态、硬件资源、到达与长度分布、质量和时延目标。设计变量包括状态布局、调度粒度、执行后端、并行策略和阶段划分。输出应解释为什么这些变量组合在当前条件下有价值，以及改变哪个条件会促使设计改变。

可以用四类关系串联：模型决定可复用状态；状态和 workload 决定容量与流量；硬件和 backend 决定执行成本；调度与队列决定用户看到的延迟。只优化其中一环可能把瓶颈推向另一环，设计需要顺着完整链重新求解。

## 场景一：长系统提示、高前缀复用的 GQA 聊天服务

先从 Hkv 计算 KV/token，再用长度分布得到驻留需求。大量相同系统前缀满足计算等价条件，适合共享前缀状态。分页让共享完整块与各请求私有尾部共存。前缀命中减少 prefill 工作，剩余新输入仍可分块，避免阻塞活跃 decode。

当命中后 query 很短，主机调度、采样和 launch 的比例可能上升，因此固定缓冲的图执行与异步准备值得评估。图前缀页表在每轮内容变化，稳定地址与代次隔离一起设计。若加入多 LoRA，缓存身份与图形状还要包含 adapter 依赖；共享收益可能下降，应重新估算容量和路由。

本设计的预测是 prefill 计算量按命中 token 减少，KV 物理驻留按实际共享页减少，单 token decode attention 仍读取其可见历史。若观测只有 hit rate 上升而队列更长，可能是亲和路由集中负载，应比较排队增加和计算节省。

## 场景二：语音 Encoder-decoder 多请求服务

源音频先经过明确的 processor 与 encoder，固定源可缓存各层 cross KV。decoder self KV 随目标长度增长，cross KV 由源长度决定，两个容量预算分开。encoder 批次按音频长度聚合，decoder 按迭代连续推进，阶段间传递带身份的源状态。

若要流式输入，先判断 encoder 是否保证已提交表示稳定。双向 encoder 不具备该性质时，任意追加源并重用旧 cross KV 会改变函数。可选按模型支持的块协议接入，或将独立窗口识别作为另一种明确任务。首个稳定文本时延需包括输入窗口等待，不能只测 decode。

## 场景三：MLA＋MoE 的多卡长上下文服务

MLA 减少持久 attention 状态，但是否避免历史展开决定实际带宽收益；MoE 全部专家权重仍需驻留或按策略移动。容量账必须分别计 latent、专家、临时分组激活与通信区。EP 改变专家归属，TP 改变 attention/线性层分片，两者通信频率和消息分布不同。

节点内外链路不同，先给高频同步选择强互联组，再计算远端 dispatch 的路径和偏斜。若 prefill/decode 分离，还要计 latent 状态在不同分片之间的传输与转换。较小 KV 不等于网络完全无代价；也不意味着专家热点被解决。

## 场景四：静态编译平台上的混合长度任务

模型函数固定，服务形状变化。按 prompt、query、context 和 batch 的联合分布选择有限桶，设备保留 KV，控制面只推进真实长度。短高频请求应有合适的小桶；稀少长请求可以用较粗桶，避免过多 executable 与 workspace 驻留。

前缀命中会改变待算 query 分布，连续批会改变 batch 分布，投机会改变验证长度分布。桶选择因此应在这些机制组合后的 workload 上推导。若先按无缓存独立请求选桶，再加入其他机制，原配置可能失去意义。

## 从推导到 RSI 增量

规划说明当前链条中哪个关系最不确定，执行实现对应机制，诊断对其预测和实际状态进行核对。进化把新认识写回受影响的机制章节：补足成本项、修正可恢复条件或新增组合分支。成功不必产生新条目；更精确地理解一个现有机制，往往比增加一个孤立提示更有价值。

## 候选空间、依赖和反证

把设计变量按其影响对象组织：模型接入决定合法算子与状态；表示/布局决定容量与可用内核；执行调度决定同时存活对象；部署与路由决定访问路径；质量与 SLO 决定可接受结果。优化之间的依赖和冲突必须进入候选描述，例如图重放需要稳定缓冲、异步双缓冲增加在途容量、低位表示需要相容内核、跨实例复用需要一致版本与迁移语义。

先用不依赖精细拟合的容量和资源下界排除不可行组合，再选择少量能区分竞争解释的实验。记录一次性成本、稳态工作和失败恢复，避免候选只比较最顺利的一次执行。一个优化被拒绝时保留使选择反转的条件，例如带宽提高、复用距离缩短或质量门槛改变；这使探索可以回退到有依据的分支。

## 场景五：CPU 多请求生成的完整推导

以下全部参数为可复算设定，不是项目机器或模型的实测结果。任务是固定版本的稠密 GQA decoder 文本生成，质量以独立固定前缀 forward 和任务评估为准；8 个活跃交互请求，平均外部到达 2 请求/s，每请求生成 64 token，输入平均 512 token，缓存计算历史上限 2048。目标客户端 ITL 为 60 ms；主机与响应预算暂设 4 ms，实际网络与排队是否满足此预算仍需测量。

硬件设定为一个 32 核、32 GiB 可用主存的服务域，主存有效读带宽假设为 96 GiB/s；权重和已选布局合计驻留 3 GiB，固定与临时工作区预算 2 GiB。模型状态参数设为 24 层、4 个 KV head、每 head 128 维、每元素 2 字节。算子 dtype、累加和指令路径仍由具体实现确认，不能由权重大小推断。

### 模型与容量

每 token 全层 KV 为 2×24×4×128×2=49,152 字节，即 48 KiB。2048 token 历史为每请求 96 MiB；8 个请求为 768 MiB，权重+工作区+KV 合计 5.75 GiB，另需页尾、输入/输出队列和在途增量余量。64 个同长请求的该项合计为 11 GiB，仍低于 32 GiB；这只证明所列驻留项未超容量。

加载先按 [制品与布局](../frameworks/checkpoint-loading.md) 建立固定参数，避免保留不必要的完整转换副本。请求状态由 [分配器](memory-allocators.md) 管理，初版不要求前缀共享，也不要求使用分页；连续池与固定块池可以保持同一模型语义，先比较长短混合时的碎片和增长方式。已输出 token 与已计算 KV 的边界按模型章节分别记录。

### 三个执行候选

| 候选 | 每次推进的组织 | 主要收益与代价 |
|---|---|---|
| A：逐请求单独执行 | 每次一个请求推进一个 token，再轮转 | 路径简单，但权重在多个请求间重复读取 |
| B：单实例连续批 | 同轮合并 8 个就绪 decode，prefill 受时间预算约束 | 摊薄权重读取，需要批次身份、状态与公平调度 |
| C：同域两个副本 | 两个实例各处理 4 个 decode，请求相互隔离 | 复制权重和工作区，竞争同一内存带宽 |

在“每次调用读取全部 3 GiB 权重、每请求读取其全历史 KV、忽略其他流量”的设定下，A 一次的流量为 3.09375 GiB，带宽下界约 32.23 ms；8 个请求轮转一次约 257.81 ms，已经无法支持每个请求 60 ms 的间隔。B 一轮流量为 3.75 GiB，下界约 39.06 ms，对应这一资源模型下的吞吐上界约 204.8 输出 token/s。

C 若把该内存域带宽平均分成每实例 48 GiB/s，每实例流量为 3.375 GiB，下界约 70.31 ms，仍超过目标。这个结论依赖共享同一带宽域的设定；若两个副本分别拥有独立内存控制器带宽，或权重常驻足够大的 cache，就必须重新计算，而不能照用此结论。

64 个活跃请求若全部进入一轮，权重加 KV 流量为 9 GiB，带宽下界为 93.75 ms；可见“内存容得下 64 请求”与“64 请求都满足 ITL”不同。将 64 请求轮转成多个小批也不能消除它们等待其他轮次的时间。因此 admission 不能仅按内存上限设置。

### 将 prefill、主机和质量接入同一方案

对 B，60−4−39.06≈16.94 ms 是在当前理想 decode 下界之外剩余的预算，并不保证真实实现还有这么多。计算、额外流量和同步可能先消耗它。需要测量混合批的时间曲面，再按 [分块 prefill](../algorithms/chunked-prefill.md) 选择合法块；若所有块都不满足，应调整运行点、准入或资源，不能宣称增大 batch 已解决服务目标。

平均到达产生 128 输出 token/s，低于 B 的上述理想上界，但还要支付每秒 1024 个新输入 token 的 prefill 和其他开销，所以仍不能据此断言系统稳定。主机侧给分词、调度和响应留独立预算，限制待处理 token 与输出字节；慢客户端与取消通过明确身份回传到调度和资源回收。

CPU 线性层比较无打包/预打包、合适线程数和不同 M 形态。若考虑低位权重，再加“固定量化参数下的执行正确性”和“相对原模型质量”两层验证；新的字节下降应重新进入资源模型，不能把原有性能估计直接乘以理论压缩倍数。

### 区分实验与反转条件

先验证单请求固定前缀、完整/分块/增量 forward，再比较不同 batch 排列、不同槽位、跨页和取消后复用。小型可穷举状态轨迹检查同一逻辑请求不受物理布局影响；长运行检查空闲、活跃、共享与在途资源守恒。采用独立采样参考和协议事件验证输出，不只检查最终字符串。

性能实验分别扫描活跃请求数、历史长度、prefill 块、算子线程和副本数，测实际权重/状态流量、CPU 阶段服务时间、TTFT/ITL、拒绝与队列增长。用 [统计方法](../tools/statistical-experiments.md) 在独立区组确认候选，保留首次加载和稳定态两种口径。

若低负载仅有一个请求，A 的简单路径可能更合适；若独立 NUMA 域容量和带宽足够，C 的隔离价值可能超过权重复制代价；若长历史主导流量，继续增加 batch 的收益会下降；若任务转为离线且放宽 ITL，可以选择更大批和更粗 prefill。该场景给出的结论是条件化设计与待测量项，不是未经执行的性能承诺。

<!-- infraagents:provenance -->

## 来源与推导

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。
  [完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

- [S-DISTSERVE：DistServe](../sources/S-DISTSERVE.md)：DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。
  [完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。
  [完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

- [S-NXDI：NxD Inference Features Configuration Guide](../sources/S-NXDI.md)：NxD Inference 说明静态形状可执行程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可配置不同桶；持久状态与模型执行共同受配置约束。
  [完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-CPU-TUNING：PyTorch Performance Tuning Guide](../sources/S-CPU-TUNING.md)：PyTorch CPU 调优资料涉及线程组织、NUMA、向量化与执行配置；线程数和内存归属影响真实可用资源。
  [完整原文](../../references/S-CPU-TUNING/reader.html) · [全文文本](../../references/S-CPU-TUNING/fulltext.txt)

- [S-GOTO：Anatomy of High-Performance Matrix Multiplication](../sources/S-GOTO.md)：§3–6 从多级存储模型推导 GEMM 的分解、panel 打包、cache/TLB 条件和实现参数。
  [完整原文](../../references/S-GOTO/reader.html) · [全文文本](../../references/S-GOTO/fulltext.txt)

- [S-INT8：Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](../sources/S-INT8.md)：§2 给出仿射量化、整数点积、零点补偿、bias 和输出重新缩放；附录 A–B 讨论组合算子、舍入及 NEON 中间累加条件。
  [完整原文](../../references/S-INT8/reader.html) · [全文文本](../../references/S-INT8/fulltext.txt)

- [S-STATS：Quantifying Performance Changes with Effect Size Confidence Intervals](../sources/S-STATS.md)：§6–7 连接多层随机影响、实验重复、效应量和置信区间，并明确方法假设。
  [完整原文](../../references/S-STATS/reader.html) · [全文文本](../../references/S-STATS/fulltext.txt)

## 延伸阅读

- [连续批处理：从自回归依赖到迭代调度](../algorithms/iteration-batching.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [容量设计：驻留、服务率与阶段配比](../workloads/capacity-planning.md)。
