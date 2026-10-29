# Astra 截止日前的 InfraAgents base 知识库

当前为 base-gpt-6-astra-2026-04-30-v2.3-draft5 构建草稿，含111篇、115项来源和235份活动历史。来源版本上限保持2026-04-30；编辑与独立推导使用真实日期。全库已登记问题、前提等导航字段，逐篇深度复核见[扩展状态](../docs/expansion-status.json)；当前深化分页、调度与GPU流水，见[本批模块](../docs/mechanism-expansion.md)。内容标准见[构建方法](../docs/knowledge-method.md)与[审查问题](../docs/knowledge-review.md)。

[主题索引](INDEX.md)、[设计入口](DESIGN.md)、[诊断入口](DIAGNOSIS.md) 和 [覆盖矩阵](COVERAGE.md) 提供可选阅读路径。[系统组合](system-semantics/design-synthesis.md) 展示条件化例子；它们不要求所有Agent采用同一组件划分、研发顺序或方案集合。

模型决定计算与状态；制品与加载建立可执行参数；算法决定更新、复用和调度；硬件与框架决定执行；协议与负载决定用户目标；工具验证机制预测。基础机制、公式、容量算例与未受影响章节保持连续。具体实现事实以本版本原文为准。

所有章节均为来源知识 seed，未声称在目标 Task 上实测有效。Evolution 仍可按既有规则提出章节增量，Diagnostician 独立核对，Runner 发布并保留历史。新实验使用真实日期，不能倒写成 base 事实。

[来源登记](SOURCES.md) · [时间规则](../docs/23-time-bounded-knowledge.md) · [版本与编辑记录](../VERSION.json)。

当前Attention修订的算法、硬件映射、组合状态和验证边界见[模块说明](../docs/attention-module.md)。
