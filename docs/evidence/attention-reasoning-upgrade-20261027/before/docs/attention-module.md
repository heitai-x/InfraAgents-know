# Attention 执行模块：本轮单模块交付

主入口是 [K-FW-006：Attention 执行接口](../knowledge/frameworks/attention-backends.md)。本次补齐 FlashAttention-2 与 FlashAttention-3。范围是精确 Attention 的数学语义、分块/分区算法、状态表示与执行成本，作为后续知识制作的具体范式。正文包含完整论证与伪代码，本页只给阅读定位。

| 用户确定的问题 | 正文中的具体内容 |
|---|---|
| 原因/动机、数学定义、输入输出和状态 | “先写出共同函数”：L、S、dₖ、dᵥ、可见集合与 head 映射；解释分数矩阵驻留与短 query 并行度问题 |
| 合法条件与实现选择 | “问题、不变量与设计轴”“非等长 causal 是一个具体反例”：可见性、归一化质量、布局与分派；区分模型语义和库的对齐约定 |
| 算法、数据流与状态变化 | “从零实现时的算法骨架”“分区 attention 合并”：初始化、逐块更新 m/ℓ/U、空行、输出与 LSE、分区合并 |
| 各项成本 | 分块重读字节、片上容量与并行度、转换与 GQA 物化、准备摊销、workspace 与并发、图缓冲生存期 |
| 何时保留、调整或放弃 | 短/长历史、batch、计划复用频率、布局转换与额外驻留改变时的收益反转；算例明确区分设定与实测 |
| 如何区分竞争解释 | “用同一逻辑状态比较实现”“诊断线索”：固定语义参考，分开检查准备、转换、执行、归约与资源完成 |

来源说明提供数学或实现逻辑的进一步展开，例如 [SDPA 的定义与分派](../knowledge/sources/S-SDPA.md)、[FlashAttention 的算法与 IO 结论](../knowledge/sources/S-FLASH.md)、[FlashInfer 论文的分区合并与划分](../knowledge/sources/S-FLASHINFERPAPER.md)。新增[FA2：算法与工作划分](../knowledge/sources/S-FLASHATTENTION2.md)和重写后的[FA3：异步流水与低精度](../knowledge/sources/S-FLASHATTENTION3.md)，正文分别有独立二级节。论文保留原始 PDF，源码逻辑以经过核对的文字与伪代码交付。

本模块的[独立检查](evidence/attention-fa2-fa3-20261027/attention-module.json)绑定当前正文与来源说明，检查阅读路径，并执行87项稠密参考/在线归约/分区合并的标量验证。覆盖非方形输入、QK/VO维度不同、mask、空行/空分区、尾块与极端分数；这些检查不代表某个 GPU kernel 已实现或达到性能目标。

[FA2/FA3专项检查](evidence/attention-fa2-fa3-20261027/algorithms.json)另含65项，检验sliced-Q行归属、两级流水的1/2/3块和尾块、缩放方向、缓冲读者完成条件与资源算术；[实际模块读取](evidence/attention-fa2-fa3-20261027/reader.json)覆盖当前正文和10份来源说明。
