# Attention 执行模块：内容、推导与验证范围

主入口是 [K-FW-006：Attention 执行](../knowledge/frameworks/attention-backends.md)，当前revision7。范围包括前向推理的数学语义、精确分块/分区、FA2/FA3数据流、显式近似的低精度示例及完整调用成本。关联章节承担平台与统计基础；本页给出阅读定位和内容作者的验收问题，不规定Agent的构建路线。

| 用户确定的问题 | 正文中的具体内容 |
|---|---|
| 原因/动机、数学定义、输入输出和状态 | “先写出共同函数”：L、S、dₖ、dᵥ、可见集合与 head 映射；解释分数矩阵驻留与短 query 并行度问题 |
| 合法条件与实现选择 | “问题、不变量与设计轴”“非等长 causal 是一个具体反例”：可见性、归一化质量、布局与分派；区分模型语义和库的对齐约定 |
| 算法、数据流与状态变化 | 在线归约、分区合并、FA2行归属、FA3流水；“异步缓冲”明确代次与最后读者；“FP8”给出量化器、Q/K/V/P尺度、归一化、缓存尾块及分层参考 |
| 各项成本 | “从形状推到工作、流量、驻留和时间”区分唯一KV、请求流量和物理事务，贯通CTA、FLOPs、部分输出、片上状态、准备与关键路径 |
| 何时保留、调整或放弃 | GQA decode算例比较C=1/4/8/16，展示并行饱和后的收益反转；组内共享、长prefill和精度改变时重新推导，不给固定最佳参数 |
| 如何区分竞争解释 | 三级流水诊断案例比较首尾、spill、驻留、编译重叠和转换准备；给出不同预测、可控干预和无法下结论的情形 |

来源说明提供进一步展开，例如[SDPA定义](../knowledge/sources/S-SDPA.md)、[FlashAttention算法与IO](../knowledge/sources/S-FLASH.md)、[FlashInfer分区与划分](../knowledge/sources/S-FLASHINFERPAPER.md)、[FA2工作划分](../knowledge/sources/S-FLASHATTENTION2.md)、[FA3异步与低精度](../knowledge/sources/S-FLASHATTENTION3.md)。11份来源说明与本地论文/文档组成可读链；FA3说明明确区分论文机制与正文独立构造的量化算法。源码不承担Agent补全关键知识的责任。

## 内容充分性如何检查

可以用以下问题检查阅读后的推导能力；它们是研究者可调整的题例，不是角色义务或固定实施顺序，也未声称已经运行Agent答题：

- 在不复制现有实现的前提下，从定义推导m/ℓ/U与分区合并，解释空行和非等长可见性。
- 对给定B/Hq/Hkv/L/S/d和数据流，分别计算任务数、唯一状态、请求流量与部分输出，说明哪些物理事务仍需测量。正文给出32 MiB唯一KV与128 MiB独立head请求的区别。
- 为低精度路径写出每个中间量的单位，解释为何旧码字不能只换scale，为什么单侧排列或更换旧K的正交身份会改变函数。
- 对1/2个环形槽、多个读者和跨代事件，给出允许覆盖的条件与错误反例，并说明有限模型不能证明实际设备内存语义。
- 面对“三级比两级慢”，给出竞争解释、各自预测与有区分力的实验；观测不支持时修正解释，差异落在噪声内时保留未知。

## 当前证据与尚未验证的效果

[本轮记录](evidence/attention-reasoning-upgrade-20261027/review.md)绑定当前内容。[模块检查](evidence/attention-reasoning-upgrade-20261027/attention-module.json)覆盖阅读路径与87项独立标量归约；[FA2/FA3检查](evidence/attention-reasoning-upgrade-20261027/algorithms.json)保留65项行归属/符号时序检查；[推导检查](evidence/attention-reasoning-upgrade-20261027/reasoning.json)覆盖明确选择的有限E4M3FN软件模型、分层参考、错误尺度/排列、有限缓冲协议与成本/诊断算术。[读取探针](evidence/attention-reasoning-upgrade-20261027/reader.json)通过真实冻结reader读取正文、11份说明和新增二级节。

这些证据分别证明局部模型与访问链，不能按检查数量推断内容完全充分。尚未执行目标GPU的FP8误差、同步、生成指令、物理重叠或性能实验，也未完成Agent独立构建/迁移/诊断的受控评估。知识对真实设计能力的帮助仍需这类使用证据；全库其他专题的深度复核状态不因本轮改变。
