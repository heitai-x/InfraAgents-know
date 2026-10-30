# Attention 执行模块：内容、推导与验证范围

主入口是 [K-FW-006：Attention 执行](../knowledge/frameworks/attention-backends.md)，准确修订与正文身份见 [catalog](../knowledge/catalog.json)，当前审查定位见[扩展状态](expansion-status.json)。范围包括前向推理的数学语义、精确分块/分区、FA2/FA3数据流、显式近似的低精度算法、硬件映射和组合状态。关联章节承担平台与统计基础；本页给出阅读定位和内容作者的验收问题，不规定Agent的构建路线。

| 用户确定的问题 | 正文中的具体内容 |
|---|---|
| 原因/动机、数学定义、输入输出和状态 | “问题、不变量与设计轴”“共同函数与可合并状态”：L、S、dₖ、dᵥ、可见集合与 head 映射；解释分数矩阵驻留与短 query 并行度问题 |
| 合法条件与实现选择 | 不变量、非等长causal、精确/近似语义；硬件实例区分公开矩阵契约与不透明fragment，组合实例区分逻辑所有者和在途读者 |
| 算法、数据流与状态变化 | 在线归约、分区合并、FA2/FA3与尺度完整的FP8；“一个完整的 warp 映射：16×16 分块与显式交接”贯通地址/搬运/矩阵输入/行归约/写回；“组合轨迹：共享前缀、量化尾块与在途图读者”贯通追加/封存/发布/图消费/取消/回收 |
| 各项成本 | 任务数、FLOPs、唯一KV、请求/物理事务、部分输出、片上状态、准备与关键路径；映射实例计算4096字节共享区与7680字节每轮逻辑片上交通；组合实例区分136/160字节页、1024字节预留与672字节相关对象存活 |
| 何时保留、调整或放弃 | GQA decode算例比较C=1/4/8/16，展示并行饱和后的收益反转；组内共享、长prefill和精度改变时重新推导，不给固定最佳参数 |
| 如何区分竞争解释 | 三级流水诊断案例比较首尾、spill、驻留、编译重叠和转换准备；给出不同预测、可控干预和无法下结论的情形 |

来源说明提供进一步展开，例如[SDPA定义](../knowledge/sources/S-SDPA.md)、[FlashAttention算法与IO](../knowledge/sources/S-FLASH.md)、[FlashInfer分区与划分](../knowledge/sources/S-FLASHINFERPAPER.md)、[FA2工作划分](../knowledge/sources/S-FLASHATTENTION2.md)、[FA3异步与低精度](../knowledge/sources/S-FLASHATTENTION3.md)、[CUDA矩阵契约](../knowledge/sources/S-CUDA-CAPABILITY.md)及[请求/bank模型](../knowledge/sources/S-CUDA-BEST.md)。来源说明与本地论文/文档组成可读链；FA3说明明确区分论文机制与正文独立构造的量化算法。源码不承担Agent补全关键知识的责任。

## 两个新增实例如何连接

硬件实例选一个warp、16×16×16矩阵操作，明确线程实际读取哪些字、怎样按定义布局交接、哪些lane共同维护一行、怎样保护尾块。它使用FP16概率乘V、高精度分母，这处近似与实际矩阵累加必须分层验证。内部fragment未公开保证，因此不会编造lane到内部寄存器的映射；更直接的指令布局可作为有依据的另一个设计。

组合实例选4-token页、可共享封存前缀、独立高精度尾块和两个固定地址图包。它展示取消不会自动结束已提交读取，封存不会允许只换scale，图地址稳定不会保证metadata永远有效。两个包并存也不绕过自回归token的上游依赖。消费时用FP8存储解码到FP16操作数，不能将容量收益误称为整条FP8计算吞吐收益。

## 内容充分性如何检查

可以用以下问题检查阅读后的推导能力；它们是研究者可调整的题例，不是角色义务或固定实施顺序，也未声称已经运行Agent答题：

- 在不复制现有实现的前提下，从定义推导m/ℓ/U与分区合并，解释空行和非等长可见性。
- 对给定B/Hq/Hkv/L/S/d和数据流，分别计算任务数、唯一状态、请求流量与部分输出，说明哪些物理事务仍需测量。正文给出32 MiB唯一KV与128 MiB独立head请求的区别。
- 为低精度路径写出每个中间量的单位，解释为何旧码字不能只换scale，为什么单侧排列或更换旧K的正交身份会改变函数。
- 对1/2个环形槽、多个读者和跨代事件，给出允许覆盖的条件与错误反例，并说明有限模型不能证明实际设备内存语义。
- 将硬件实例改为23行query、47个key及窗口可见性，重新确定尾块读取、行归约和数值参考，说明哪些资源账保持、哪些重算；不预先承诺目标GPU时间。
- 不改数学函数，改变lane拥有的元素或矩阵交接，证明覆盖与归约，预测请求/bank交通及驻留变化；仅有覆盖正确不足以证明访问高效。
- 对组合实例插入取消、发布失败或迟到事件，重建每个消费者实际看到的快照及仍不能回收的对象；改变页大小或共享尾块策略时重新推导。
- 将组合实例用于真正自回归decode，画出下一token的上游依赖，区分合法预先准备与尚不能开始的计算。
- 面对“三级比两级慢”，给出竞争解释、各自预测与有区分力的实验；观测不支持时修正解释，差异落在噪声内时保留未知。

上述题例是可选的评估问题，未因列入本页而成为已完成的独立Agent评估。实际使用评估应保存输入、知识可见范围、模型/预算、产物及失败原因；若比较知识版本，保持这些条件可比。以正确实现、迁移推导和诊断结果判断帮助，不以方案是否新奇或是否使用某章判断。

## 当前证据与尚未验证的效果

revision9 的[内容审查](evidence/knowledge-workflow-20261028-batch19/K-FW-006/round-2/review.json)与[实际检查](evidence/knowledge-workflow-20261028-batch19/K-FW-006/round-2/check-0.json)绑定该次正文和 `tools/check_k_fw_006_full.py`。检查覆盖确定性CPU标量在线状态、独立Decimal参考、分区、causal/GQA以及非单位尺度的特例与负对照；不执行真实warp映射、异步槽协议或GPU内存语义。登记与验证状态见[该批工作记录](evidence/knowledge-workflow-20261028-batch19/workflow.json)。后续正文修订应沿扩展状态读取其准确审查，不能自动沿用此处的旧版通过结论。

revision8 的[历史记录](evidence/attention-mapping-composition-20261027/review.md)、[映射/组合检查](evidence/attention-mapping-composition-20261027/mapping-composition.json)、[模块检查](evidence/attention-mapping-composition-20261027/attention-module.json)、[FA2/FA3检查](evidence/attention-mapping-composition-20261027/algorithms.json)、[既有推导](evidence/attention-mapping-composition-20261027/reasoning.json)、[算例](evidence/attention-mapping-composition-20261027/examples.json)与[冻结读取](evidence/attention-mapping-composition-20261027/reader.json)保留当时的输入和范围。它们不能直接作为revision9全文或未来修订的验证报告。

软件标量矩阵乘不模拟真实MMA的FP32归约树，逻辑bank模型不证明编译指令或物理事务，事件模型不证明GPU内存可见性。没有目标GPU的同步、低精度误差、真实重叠或性能结果，也没有受控Agent独立构建/迁移/诊断效果结论。内容是否足够最终需要这种使用证据；全库其他专题的深度复核状态不因本轮改变。
