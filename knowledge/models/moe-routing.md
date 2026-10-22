# K-MODEL-003：MoE：路由、Dispatch/Combine 与专家负载

MoE 的条件计算节省激活 FLOPs，但全部专家的驻留和路由后的不均衡决定真实服务成本。

## 一次 MoE 层的数据关系

输入 token hidden states 经过 router 得到专家分数，每 token 选择 k 个专家。每个 token 形成 k 个带权分派项；按专家重排后执行各专家 FFN，再将结果按原 token 聚合。典型形式 yₜ＝Σₑ gₜₑ Eₑ(xₜ)，其中只有选中专家具有非零权重；共享专家分支可以独立叠加。

重排必须保留原 token ID、专家 ID、路由权重和返回位置。combine 的语义通常是加权归约，不能用后到达的专家输出覆盖先到达结果。路由是否在 top-k 后重新归一化、是否有共享专家、是否限制设备组，都由模型定义。

## 激活参数不等于驻留参数

假设 8 个同尺寸专家，每 token 选 2 个。专家计算量接近 2 个专家的工作，但通常要保留 8 个专家权重以响应下一批任意路由。小 batch 下，每专家 token 很少，GEMM 变成许多碎片；增大 batch 可以增大各组，却提高排队和 KV 容量。

若 B＝16、k＝2，总 dispatch 项为 32。理想均匀分给 8 专家，每个 4 项；若热点专家拿到 20 项，其执行会决定本层完成时间，而其他专家空闲。在线语料和小样本的路由分布可能与训练平均不同。只报告平均 tokens/expert 会隐藏慢端。

## 专家并行的通信形态

EP 让 token hidden states 被发送到持有专家的 rank，再把结果发送回去。发送流量近似随远端分派项×hidden size×dtype 增长，另有索引和权重元数据。同 token 的多个专家在同设备时，可以减少重复网络传输，但仍需本地分派。拓扑感知路由只有在模型允许相应路由规则时才成立，不能运行时随意改 expert 选择。

容量因子与 token dropping 常见于训练或特定模型实现；推理是否允许丢弃必须由模型语义决定。为让固定形状 kernel 装下而静默丢弃溢出项，会改变函数。可变长 grouped GEMM、padding、分批执行是不同的实现选项。

## 可推导的优化路径

当主要成本是小专家 GEMM，融合 dispatch、分组计算和减少 padding 有意义；当主要是跨节点传输，调整 EP/TP 组合与专家布局更直接；当主要是权重容量，量化或卸载可能扩大可服务模型，但带来搬运成本。MLA 降低 attention 状态不会直接减少 MoE 权重，两个改进必须分别进入容量账本。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MIXTRAL：Mixtral of Experts](../sources/S-MIXTRAL.md)：Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。
  [完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。
  [完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
