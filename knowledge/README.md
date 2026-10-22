# Astra 截止日前的 InfraAgents base 知识库

当前为 base-gpt-6-astra-2026-04-30-v2.1 复审候选，来源版本上限 2026-04-30（包含该日）。保留原有 73 篇与八类组织，已接入 35 个新专题，共 108 篇、114 项固定来源。初版建设与验收见 [构建记录](../docs/BUILD-STATUS.md)，本次新审查和修订见 [审查记录](../docs/evidence/knowledge-review-loop-20261023/round-1/review.md)。

[系统组合](system-semantics/design-synthesis.md) → [设计入口](DESIGN.md) → [主题索引](INDEX.md)；[诊断入口](DIAGNOSIS.md) 与 [覆盖矩阵](COVERAGE.md) 保留反向阅读路径。

模型决定计算与状态；制品与加载建立可执行参数；算法决定更新、复用和调度；硬件与框架决定执行；协议与负载决定用户目标；工具验证机制预测。基础机制、公式、容量算例与未受影响章节保持连续。具体实现事实以本版本原文为准。

所有章节均为来源知识 seed，未声称在目标 Task 上实测有效。Evolution 仍可按既有规则提出章节增量，Diagnostician 独立核对，Runner 发布并保留历史。新实验使用真实日期，不能倒写成 base 事实。

[来源登记](SOURCES.md) · [时间规则](../docs/23-time-bounded-knowledge.md) · [版本与编辑记录](../VERSION.json)。
