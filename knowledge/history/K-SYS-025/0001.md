# K-SYS-025：已知张力：单实例交互式服务中的机制冲突

单实例交互式生成服务里，若干机制在同一资源与同一时延预算上相互拉扯。下列条目是有来源支撑的探索起点：各写机制、来源指针、可观测信号与反转条件。清单不穷尽，也不构成任务必须处理的缺陷列表。

## Decode 带宽下界与 ITL

机制：每轮 decode 若读取大块权重与各请求可见 KV，则设备主存流量给出迭代时间下界；客户端 ITL 还要覆盖主机、采样、同步与排队。当流量下界已接近或超过目标 ITL，继续堆叠同轮工作只会恶化间隔，吞吐与单请求间隔不能用同一除法解释。

来源指针：[容量与 roofline](../hardware/memory-and-roofline.md) 中权重/KV 流量分解；[ORCA](../sources/S-ORCA.md) 的迭代批语义说明一轮推进多个请求仍共享同轮等待。

可观测：活跃 decode 数增加时，每请求输出间隔近似不降或上升，而设备 HBM 字节/迭代接近表征带宽×时间；主机空闲比例高。

反转：权重驻留被更高层缓存命中、KV 读取因共享或压缩真实下降、目标 ITL 放宽，或阶段分离使本实例不再承担该流量形态时，张力缓解或形态改变。

## Prefill 与 Decode 干扰

机制：长 prefill 的大矩阵与长上下文 attention 占用 SM、HBM 与工作区；与短 query decode 同实例交错时，decode 的迭代被拉长，表现为已有流式用户的 ITL 尖峰。分块把 prefill 切成可调度片，但每块仍须按可见性读取其依赖的历史，块长受剩余 ITL 预算约束，不是任意变小就无代价。

来源指针：[SARATHI](../sources/S-SARATHI.md)；[分块 Prefill](../algorithms/chunked-prefill.md)。

可观测：新长大请求到达后，无关会话的 token 间隔同步变差；时间线上出现长 prefill kernel 与 decode 交替，decode 间隙被拉宽；TTFT 改善若来自抢占 decode，常伴随 ITL 恶化。

反转：prefill/decode 分实例且状态迁移成本可接受、到达以短输入为主、或时间片与准入使 decode 预算硬隔离时，同实例干扰下降。分实例引入的传输与调度是另一组成本。

## 投机收益随 Batch 增大而消失

机制：投机用草稿成本换目标并行验证；单位已确认 token 成本须低于普通目标 decode。连续批摊薄每 token 权重读取后，普通 decode 变便宜，同一 γ 与草稿配置可能从加速变为减速；额外验证 token 还占用其他请求预算。

来源指针：[投机解码](../algorithms/speculation-budget.md)；[S-SPEC](../sources/S-SPEC.md)。设定算例见该章：普通每 token 10 单位、投机后 8 单位有益；普通降至 6 单位后同配置变慢。

可观测：小批或单请求加速比明显，批增大后接受长度未变但端到端 token/s 低于关闭投机；验证占用的 token 预算挤出其他就绪请求。

反转：批变小、草稿更准或更便宜、验证与 decode 能重叠且不挤占关键资源，或普通路径因别的原因变慢时，投机可再次有利。关闭投机是合法运行点。

## KV 量化的质量悬崖

机制：降低 KV 位宽削减驻留与部分流量，但误差经注意力传播；分组轴、尺度与残余区配置在某些位宽/模型/任务上出现断崖式质量下降，而不是平滑可插值的曲线。容量账按理想位宽估算会忽略 scale、残余与 workspace。

来源指针：[KIVI](../sources/S-KIVI.md) §3–4 与适用边界；[量化契约](../algorithms/quantization-contract.md)。

可观测：某位宽下任务指标相对基线突然恶化，相邻更高位宽仍接近基线；仅测 TTFT/首 token 正常，长生成后偏离；消费压缩历史的 decode 与仍用高精度缓存的 prefill 结果不一致。

反转：换分组轴或残余策略、提高关键层精度、任务对误差不敏感，或长度/批使流量不再主导时，可重新评估。悬崖位置随模型与任务移动，不能当跨模型常数。

## 图桶、Workspace 与 KV 争容量

机制：CUDA Graph 与固定计划缓冲需要稳定地址与多桶驻留；注意力/矩阵 workspace、流水双缓冲同样占用设备容量。它们与 KV 争同一显存池：桶与 workspace 增大可降提交与峰值临时，却直接减少可驻留历史或并发。

来源指针：[CUDA Graph 缓冲](../frameworks/cuda-graph-buffers.md)；[内存分配器](../system-semantics/memory-allocators.md)；[S-TORCH-CUDA](../sources/S-TORCH-CUDA.md)。

可观测：启用更多图桶或更大 workspace 后，最大并发或最大上下文下降；OOM/驱逐与“计算变快”同时出现；页回收成功但图仍引用旧代次导致错误。

反转：形状集合收窄使桶数下降、改为 eager 且提交不是关键路径、或扩容设备内存时，争用缓和。地址不稳定的负载下图路径本身可能先失效。

## 小 Batch 下主机侧占比

机制：设备迭代很短时，分词、调度、采样、JSON/协议与 launch 的主机时间在端到端中占比上升。优化仅针对 HBM 或 GEMM 时，壁钟改善受 Amdahl 限制；图与异步可摊薄提交，但不消除采样与协议成本。

来源指针：[主机运行时](../system-semantics/host-runtime.md)；[CUDA 执行中的 eager/图](../hardware/cuda-execution.md)；[S-NSYS](../sources/S-NSYS.md) 对 host/device 时间线边界的说明。

可观测：设备利用率低但 ITL 仍高；时间线显示 kernel 很短、host 线程或运行时跨度占间隔大部分；增大批后设备占比上升、主机占比下降。

反转：批变大、输出更长使设备工作占优，或主机路径与设备重叠且无假同步时，张力形态改变。主机变快不自动提高已饱和的 HBM 路径。

## 张力之间的耦合

上述条目常耦合：为缓解带宽–ITL 而增大批，可能削弱投机；为保护 ITL 而切细 prefill，可能增加 launch 与主机占比；为省 KV 而量化，可能撞上质量悬崖；为降提交而加图桶，可能挤掉并发。要区分单因素效应，可以控制其他条件；要识别交互作用，则比较交叉配置或建立因素实验。改动多个机制本身合法，关键是证据能否区分各解释，不能把组合效应任意归给一项。

设计轴包括：同实例混合 vs 阶段分离、批大小与准入、投机开关与 γ、KV 表示位宽、图/eager、主机/设备重叠程度。轴上未列点同样合法。

## 用成本关系判断张力是否真正成立

设已确认某轮必须跨设备主存读取 D=8 GiB，匹配访问形态的可持续带宽 B=0.8 TiB/s，则只计该流量的时间下界为 D/B=9.765625 ms。若本例要求这一轮完成才能生成下一个 token，且 ITL 预算为 5 ms，这个运行点不能满足预算。以上均为设定值；张量的逻辑大小不能未经计量直接当作 D，cache 命中、共享读取和压缩都可能改变实际跨边界字节。

若有人将延迟归因为提交空隙，另一解释是主存流量，区分方法是保持语义与长度不变，分别改变提交组织和读取字节，记录设备事务、关键路径与完整 token 间隔。图重放降低 host 时间而设备流量和 ITL 几乎不变，支持流量主导；减少合法读取量后设备时间下降，进一步支持这个解释。干扰、频率、并发和误差也要记录；单一利用率不能排除竞争解释。

这个下界约束该运行点，不规定分离服务、量化、改变批或专用 kernel 中的哪一个必须采用。若优化只是重叠同一个共享带宽上的消费者，D/B 的总流量账不会因此消失。

<!-- infraagents:provenance -->

## 来源与推导

- [S-ORCA：Orca: A Distributed Serving System for Transformer-Based Generative Models](../sources/S-ORCA.md)：迭代级批处理使多个请求共享调度轮次；同轮等待与选择性批影响延迟形态。
  [完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：分块 prefill 与 decode 的组织影响二者干扰；块长需结合时延预算选择。
  [完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

- [S-SPEC：Fast Inference from Transformers via Speculative Decoding](../sources/S-SPEC.md)：投机的成本–收益随草稿、验证与接受长度变化；条件改变可翻转符号。
  [完整原文](../../references/S-SPEC/reader.html) · [全文文本](../../references/S-SPEC/fulltext.txt)

- [S-KIVI：KIVI](../sources/S-KIVI.md)：KV 分组量化与残余区改变容量与误差传播；位宽与任务相关的质量行为需独立评估，2 bit 非无条件保证。
  [完整原文](../../references/S-KIVI/reader.html) · [全文文本](../../references/S-KIVI/fulltext.txt)

- [S-TORCH-CUDA：PyTorch CUDA Semantics](../sources/S-TORCH-CUDA.md)：CUDA Graph 与静态缓冲在地址/形状稳定时摊薄提交，并引入额外驻留与失效条件。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：KV 以块映射驻留；容量与并发受物理块与映射约束，与其他固定缓冲共享设备内存。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

正文中的张力条目、可观测信号与反转条件是基于上述来源及机制章节的推导；设定算例指向既有章节中的演算，不冒充实测。来源版本上限为 2026-04-30；本次编辑日期不改变来源时间。此独立 base 版本内的章节修订号从 1 开始。本清单是探索起点，不是验收要求。

## 延伸阅读

- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [分块 Prefill：服务时间切片与历史可见性](../algorithms/chunked-prefill.md)。
- [投机解码：分布保持、状态提交与收益模型](../algorithms/speculation-budget.md)。
- [量化：误差传播、分组尺度与执行格式](../algorithms/quantization-contract.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
- [专用化杠杆：固定模型、负载与硬件时多出的自由度](../system-semantics/specialization-levers.md)。
