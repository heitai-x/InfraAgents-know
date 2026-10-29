# K-WORK-006：容量设计：驻留、服务率与阶段配比

容量设计把每请求资源需求和阶段服务时间连接到到达过程，得出实例数、并发和缓存预算。

## 先解驻留约束

可用设备内存 M 扣除权重 W、固定工作区 F、峰值激活 A 和图池 G，剩余才可用于请求状态。普通 KV 请求 i 约占 cNᵢ，加上页尾碎片；共享前缀要按物理唯一页计，而不能把所有逻辑长度相加。递归/多模态状态按各族分别计。

若剩余 16 GiB、每请求平均状态 512 MiB，理想平均最多 32 个；但长度尾部、峰值并发和分配粒度使这个平均估计不构成硬 admission 上限。可以按当前实际占用及下一步最坏增量预留，或用明示的概率模型管理风险。

## 再解服务率约束

prefill 需求近似为每秒到达×每请求未命中输入工作，decode 需求近似为每秒到达×平均输出步工作。两阶段成本依赖 batch 与历史，所以应通过少量代表形态建立服务曲面，而非一个固定 tokens/s 常量。

分离池的实例数比例由两个需求除各自有效服务能力得出，再留出满足尾延迟的余量。平均利用率接近 100% 时，轻微波动就会堆积队列；目标 SLO 越紧，可接受利用率区间通常越受限制。

## 网络和主机也是容量约束

每请求需要迁移 K 字节时，平均传输需求 λK；与模型并行通信共用链路时还要合并流量。CPU tokenization 或媒体处理有独立服务率。扩 GPU 数而不扩这些阶段，可能只让设备等待。

## 从估算到决策

已知前提下的资源下界可以排除不可行配置；未知的有效算力、带宽和阶段时间可以通过原型或运行测量取得。解析估算与实验可交替修正，不要求下界筛选先于所有实验。若估算误差很大，优先解释缺失成本而不是堆更多参数搜索。更准确的成本模型本身就是 RSI 应写回知识库的高价值结果。

## 从一个吞吐常数走向服务曲面

在少量有解释的坐标上测服务能力：query token 数、历史长、batch、dtype、缓存命中、模型版本和并行组。对每点记录计算时间、主存/设备流量、工作区峰值和主机开销。需要插值时说明适用区域，遇到 kernel 切换、桶边界、重编译或跨 NUMA 放置应另建区段；不能把连续线性假设越过结构性变化。

一个资源 j 的必要条件可写成 λ·E[dⱼ]≤nⱼ·cⱼ，其中 dⱼ 为每请求资源需求，nⱼ 为实例或资源份数，cⱼ 为相应能力。CPU 秒、设备计算秒、链路字节和存储 I/O 使用各自单位。这个平均约束只能排除明显不足的配置，尾延迟还依赖方差、排队、拓扑与相关性。

## 阶段配比必须包括状态交换

设定每秒 4 个请求，未命中输入平均 1000 token、输出平均 100 token；某已声明形态的每实例 prefill 能力假设为 2500 输入 token/s、decode 为 250 输出 token/s。平均需求分别为 4000 和 400 token/s，两个池都至少需要向上取整的 2 个实例。若每请求迁移 0.5 GiB 状态，网络需求又是 2 GiB/s；可用链路低于此值时，池的数量满足算式也不能保证可行。

这个设定还没有证明 SLO，因为额外传输延迟、两个池的排队、实例负载不均和失败余量都未解决。CPU 预处理、模型加载与共享存储同样可能限制整体；新增 GPU 时重新计算所有约束，避免只扩已经不是瓶颈的阶段。

## 冷启动、故障与弹性迟滞

扩容时间包含资源分配、环境启动、制品读取、转换、编译与预热。若突发持续时间短于实例就绪时间，响应式扩容可能赶不上，需要预留热实例、提前预测或准入控制。热实例保留成本与冷启动等待可以比较，但必须使用目标 workload 的到达与模型切换分布。

缩容先停止新准入，再排空或迁移在途请求，最后释放模型与状态。迁移会占用网络和目标容量；不能把发出迁移命令当作源端容量已经释放。故障容量还要按故障域计算：多个实例共用同一主机、NIC 或存储路径时，不能将它们视为独立冗余。

## 校准与选择反转

容量边界、各资源平均必要条件、到达轨迹与尾部目标分别检验驻留、持续供给、排队和故障恢复；根据当前问题选取相应证据，并在完整交付时满足 Task 的实际要求。模型偏差出现时，核对实际完成工作、重复流量、重算、加载、同步和在途资源，而不是只拟合一个更复杂的回归式。

前缀命中提高会减少 prefill 需求，输出变长会增加 decode 和驻留需求；更快存储改变冷启动成本，但不一定改变常驻模型吞吐。把这些敏感性保留在设计记录中，才能解释为什么同一集群在 workload 改变后需要不同的 P/D、CPU、网络和存储配比。

<!-- infraagents:provenance -->

## 来源与推导

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-DISTSERVE：DistServe](../sources/S-DISTSERVE.md)：DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。
  [完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)

- [S-MLPERF-LLM：MLPerf Inference — Llama 2 与 LoadGen（固定历史版本）](../sources/S-MLPERF-LLM.md)：MLPerf 的 Llama 2 服务基准将特定任务、到达模型、质量和延迟条件组织成评测协议，服务能力依赖这些共同条件。
  [完整原文](../../references/S-MLPERF-LLM/reader.html) · [全文文本](../../references/S-MLPERF-LLM/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-SERVERLESSLLM：ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](../sources/S-SERVERLESSLLM.md)：§4 的加载格式、块池和多级流水；§5 的 token 重算迁移；§6 的局部性与启动时间模型。
  [完整原文](../../references/S-SERVERLESSLLM/reader.html) · [全文文本](../../references/S-SERVERLESSLLM/fulltext.txt)

## 延伸阅读

- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [到达过程与并发：排队、反馈和饱和点](../workloads/arrival-and-concurrency.md)。
- [系统设计推导：把模型、缓存、执行与目标组合起来](../system-semantics/design-synthesis.md)。
