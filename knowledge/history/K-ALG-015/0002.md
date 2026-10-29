# K-ALG-015：推理并行：张量分解、流水、专家与上下文

并行策略改变张量归属和同步边界；选择依据是每阶段的计算、状态与通信分解。

## 从一层矩阵乘法推导 TP

线性层按输出通道切 W 时，各 rank 得到部分输出；后续若接受同样分片就可继续，若需要完整向量则要 gather。按输入通道切 W 时，每 rank 计算部分和，必须 reduce 才得到完整输出。Transformer 的列并行、行并行常成对组织，以减少不必要的 gather。

TP 把权重和某些激活分散，但每层通信频繁。小 batch decode 的计算时间短，集合通信启动延迟容易占主导。KV heads 少于 TP 数时，还可能复制 KV 或采用不同分片，单卡 KV 不能机械地除以 TP。

## PP 与 DP 的不同扩展目标

PP 按层切模型，阶段之间传激活。多个微批可以填充流水；单请求逐 token 仍要走完整条流水，并等待下一 token。阶段不均衡和填充/排空产生气泡。PP 常有助于容纳大模型，不保证最低单请求延迟。

DP 复制整个服务副本，各自处理不同请求，减少单请求跨卡同步。模型能放入单副本且请求足够多时，DP 可以提升总吞吐；但权重复制、前缀缓存分散和负载不均衡影响实际效率。前缀亲和路由又会改变各副本队列。

## EP 与 CP 如何增加新轴

EP 将不同专家放到不同 rank，路由后按专家 dispatch，再将输出 combine 回原 token。流量由 token 路由决定，热点专家可成为慢端。专家均衡的训练统计不保证某个在线小批的均衡。

CP 将长上下文分片。attention 需要跨分片交换 K/V 或合并局部 softmax 统计；在线 softmax 的可合并性提供数学基础，但通信与临时空间仍有代价。窗口或稀疏可见性改变应交换的数据量。序列并行一词在不同实现中也可能指激活分片，阅读实现时应先确定到底哪一轴被切分。

## 组合策略的容量与时间模型

策略可以组合，例如节点内 TP、节点间 PP、请求间 DP、专家层 EP。每个并行组要说明成员、张量归属和操作顺序。算例中某层计算从 8 降到 3 单位，但新增不可隐藏通信 4，则净时间为 7；若再分卡只省 1 却增 2，就越过收益拐点。用实际通信路径的有效带宽和启动延迟代入，比按卡数线性估算可靠。

prefill/decode 的最佳组合可能不同，阶段分离则增加跨布局 KV 转换。应把转换代价一起计入整体路径，避免得到两个局部最优而整体更慢的配置。

## 逐层账本以张量所有者为起点

设本轮有效 token 数为 T、hidden 维 d、元素字节 b，完整 hidden 为 A=Tdb 字节。对一个常见的 dense Transformer TP 组织，QKV 与 FFN 上投影按输出维切分，attention 输出投影和 FFN 下投影按输入维切分，后两者各需要一次归约得到完整 hidden。此描述是该候选的账本，融合、reduce-scatter 或 sequence parallel 会改变具体通信操作。

| 对象/边界 | 本地内容 | 额外工作或缓冲 |
|---|---|---|
| 层输入与 residual | 复制或明确分片的 T×d | 直到 residual 最后消费者完成 |
| Q/K/V 与 attention | 本 rank heads，KV 可能分片或复制 | 页表、量化参数、split attention workspace |
| Attention 输出投影 | 输入维分片产生 T×d 部分和 | 归约发送/接收及输出生存期 |
| FFN 或 MoE | 分片中间激活或按专家重排输入 | 门控中间量、专家 offsets、dispatch/combine |
| PP 边界 | 下阶段所需 hidden 与 metadata | 至少说明发送持有与接收预留；不能提前释放 |
| CP 边界 | 本地上下文及当前 query | 远端 K/V、局部 softmax 统计与合并输出 |

理想 ring all-reduce 对一个 A 字节张量，每 rank 的发送量约为 2(p−1)A/p；接收为另一计量边界，不在“发送量”中再重复相加。上述两次归约合计每 rank 4(p−1)A/p。设 p=4、d=4096、b=2，T=1 时 A=8 KiB，两次发送共 24 KiB；T=2048 时 A=16 MiB，两次共 48 MiB。decode 虽 bytes 少，启动和同步仍发生两次；prefill 更可能受数据量约束。

EP 按远端分派项计算 hidden 与返回输出流量，并加入索引/计数；同 token 的多个专家同设备时可研究去重传输，但返回聚合也要相容。CP 可交换 K/V 或合并局部 softmax 统计；不同方案的通信次数、查询复制及 workspace 不同。不能同时取一种方案的低流量与另一种方案的低暂存作为同一个候选。

## 并行组组合与非整除维度

TP、EP、DP、PP、CP 描述不同轴，但有些组在不同层复用同一批设备。应逐阶段写 rank→请求/token/head/专家/层区间映射，再求实际设备集合；TP4 与 EP4 可以在同 4 张卡的不同层执行，也可以被设计为不同轴，单写两个度数不能判断需 4 还是 16 张卡。

非整除 hidden、FFN、head 或 vocabulary 维度可以通过不均匀分片、padding 或改变并行度处理，前提是后端支持。padding 不得进入有效词表采样、归约分母或模型输出；以均匀 shape 分配后静默舍弃尾部则不合法。GQA 需要全局 query→KV 映射，MLA 可能复制 latent，专家数非整除时则按每 rank 实际权重与最大负载计算，不能只看平均。

PP 的阶段长度应按实际 attention、专家和通信成本而非仅按层数均分。对等时长阶段和充分独立微批，理想填充/排空气泡可以由流水槽数推导；逐 token 自回归、不同长度和慢阶段会破坏理想条件。通信重叠还要求有可计算的独立工作、足够缓冲，以及传输与计算没有把同一 HBM/链路/计算单元耗尽。

## 场景：四卡 MLA/MoE 的三个候选

以下全部为设计设定，不是 DeepSeek 型号规格或性能实测。任务为 16 个并发生成请求，prompt 为 2048–8192 token、输出目标 128 token，客户端 ITL 目标 40 ms；使用固定 MLA/MoE checkpoint 和 BF16 参考质量。资源为 4 张各 24 GiB 可供本任务使用的 GPU。假设权重共 48 GiB，其中 attention/其他 dense 部分 12 GiB、所有专家 36 GiB，模型 32 层、latent 512 维、RoPE key 64 维、状态 BF16。最长输入下，每请求逻辑缓存 288 MiB，16 请求共 4.5 GiB。

候选 A 在四卡上做 TP4，专家矩阵也切分；候选 B 保持 attention TP4，MoE 改为 EP4；候选 C 为 PP2×TP2，每个阶段各持一半层。A/B 若各 rank 复制所服务请求的整份 MLA latent，则每卡权重 12 GiB、prompt 结束时 KV 4.5 GiB，另设 workspace/通信 2 GiB，主项 18.5 GiB。C 按等层切分使 prompt 结束时每卡 KV 2.25 GiB，在相同设定下主项 16.25 GiB，但逐请求增加跨阶段依赖及流水气泡。不能仅因 C 占用更低便断言它的 ITL 最小。

为输出的 128 个 token 预留新增状态时，每请求再增 4.5 MiB；16 请求在完整 latent 复制的每个 rank 上增 72 MiB，即 0.0703125 GiB，A/B 主项成为 18.5703125 GiB。C 每 rank 只持一半层，增 36 MiB，主项为 16.28515625 GiB。这仍不是完整峰值保证：页尾、程序与运行时、量化元数据、加载和其他在途缓冲必须继续预留；若最后输出 token 尚不进入模型状态，预留可略多于实际使用，不能反过来省掉必要增长预算。

| 候选 | 主要机会 | 需要区分的代价 |
|---|---|---|
| A：全层 TP4 | 权重均匀分担，避免专家目的端偏斜 | 每层归约、小 expert 分片 GEMM、latent 复制 |
| B：attention TP4 + MoE EP4 | 专家矩阵更完整，grouped GEMM 与副本可调 | 分派/返回、热点专家、边界 token 所有权 |
| C：PP2×TP2 | 每卡状态下降，TP 组较小 | 阶段激活、气泡、慢阶段及单请求完整往返 |

B 的 attention 输出在组内复制时，MoE 边界先为每个 token 选定唯一派发所有者；不能让四个 rank 都重复发送同一完整 hidden。设一层 top-2、16 token、hidden 4096、BF16，远端分派比例为 3/4，且未做同目的去重，则 24 个远端项各发送 8 KiB，返回同样大小，组级专家 dispatch/return 主流量合计 384 KiB，另有 metadata。所有者完成 combine 后，还要按下一 attention TP 的复制或分片契约重分布输出；相应广播、all-gather 或其他转换需另计，384 KiB 不是 B 的全部层通信量。这一小流量也可能受消息启动限制；在较大的 prefill 中，专家组长、字节和计算比又不同。

若热点专家使一个 rank 成为慢端，可用额外 2 GiB/卡的设定预算研究冗余副本，此时 A/B 含输出增长的主项增到 20.5703125 GiB。副本放置只分担相同专家函数，路由仍保持模型规则。若用低位专家或 KV 量化腾出空间，应另检质量和执行格式；不能与 BF16 候选只比较 bytes。

验证先用少量 token 对照单设备/逐专家逻辑参考的选中专家、权重、MLA 吸收结果和逐步 logits，再覆盖 nonuniform 路由、空专家、KV 页尾、TP 输出拼接、PP 不同长度与单 rank 失败。性能在相同到达轨迹和质量门槛下测每层关键路径、最慢 rank、真实通信/暂存、TTFT、ITL 和合格请求数。40 ms 是目标，容量算例不证明任何候选已经达到它。

当拓扑连接快、batch 较大而 expert GEMM 更有效时，B 有机会胜出；低并发、消息启动昂贵或路由极偏时，A 可能更稳；最长上下文把复制 KV 推至容量边界时，C 的较低状态量可能成为必要条件，但需更多独立请求填流水。若模型能在更小副本内完整容纳，请求级复制又应进入候选。选择通过这些可区分的瓶颈反转，而非固定认定更多并行轴总是更好。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

- [S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs](../sources/S-CUDA-TOPOLOGY.md)：多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。
  [完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

- [S-MIXTRAL：Mixtral of Experts](../sources/S-MIXTRAL.md)：Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。
  [完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-DEEPSEEKV3：DeepSeek-V3 Technical Report](../sources/S-DEEPSEEKV3.md)：DeepSeek-V3 v1 的 MLA、细粒度/共享专家、路由选择与 gating 区分、MTP，以及分阶段推理部署中的专家复制、微批重叠和在线偏斜；只使用该固定版本。
  [完整原文](../../references/S-DEEPSEEKV3/reader.html) · [全文文本](../../references/S-DEEPSEEKV3/fulltext.txt)

## 延伸阅读

- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [MoE：路由、Dispatch/Combine 与专家负载](../models/moe-routing.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [集合通信：参与集合、操作顺序与失败传播](../system-semantics/collective-ordering.md)。
