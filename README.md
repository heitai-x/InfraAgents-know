# InfraAgents-know

InfraAgents 的独立知识仓库。不同分支承载不同内容版本，保持相同的目录、章节 ID 和现有读取接口。

当前分支：**gpt-6-astra-2026-04-30**。当前为 v2.1 独立复审候选：108 个 seed 专题、114 项固定来源和 12 类完整设计场景。v2 已交付；本次根据新一轮独立审查修正 MoE 算例和 HF 缓存来源归因，保留旧修订与原始归档，等待对实际新版本复审。资料版本上限仍为 2026-04-30；本轮发现及修订见 [独立审查](docs/evidence/knowledge-review-loop-20261023/round-1/review.md)，此前交付范围见 [交付记录](docs/BUILD-STATUS.md) 和 [覆盖说明](docs/inference-coverage.md)。

| 分支 | 内容 |
|---|---|
| [main](https://github.com/heitai-x/InfraAgents-know/tree/main) | 原始未限时 base，保留初始来源和章节历史；不作模型训练时间声明。 |
| [gpt-6-astra-2026-04-30](https://github.com/heitai-x/InfraAgents-know/tree/gpt-6-astra-2026-04-30) | 来源版本上限为 2026-04-30，在原 73 篇基础上完成 35 篇新增及原文修订，覆盖已声明的推理系统与依赖平台范围。 |

从 [设计入口](knowledge/DESIGN.md)、[完整专题索引](knowledge/INDEX.md) 和 [诊断入口](knowledge/DIAGNOSIS.md) 阅读；[来源登记](knowledge/SOURCES.md) 与 [完整原文](references/index.html) 提供本地依据。

知识以模型和状态、算法与调度、硬件与框架、系统语义、工作负载和观测关系组织。正文保留机制、公式、成本推导、适用条件和反例；来源归档与正文分开。

## 接入 InfraAgents

检出所需分支后，在 InfraAgents 的新任务中通过已有参数选择该目录：

```text
--knowledge-repository /path/to/InfraAgents-know
```

main 对应完整 base，Task 使用 `knowledge.view="full"`、`condition="seed"`。Astra 分支可使用该分支的 `knowledge/temporal-policy.json`，并将实际五角色模型与 profile 中的 `gpt-6-astra` 绑定。完整配置与验证入口见 [使用说明](docs/USAGE.md)。

后续 RSI 继续由 InfraAgents 的 Evolution 提案、Diagnostician 独立核对和既有发布器承担。新 Task 的实验、案例、trial 和运行资产保持真实来源；受控实验各自使用独立的学习与发布状态。原库中已经被 base 引用的历史微型案例只在 main 保留，其适用范围不扩大。

此仓库保存知识、来源和必要的版本证据，运行时仍由 [InfraAgents](https://github.com/heitai-x/InfraAgents) 提供。当前内容身份见 [VERSION.json](VERSION.json)；[原始迁移核查](docs/evidence/repository-migration.json) 仅证明最初迁移版本，后续扩充检查见构建记录。

当前未建立 DeepSeek 日期分支；没有确认其知识截止日期之前，不制作同日重复版本。
