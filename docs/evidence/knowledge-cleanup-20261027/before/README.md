# InfraAgents-know

InfraAgents 的独立知识仓库。不同分支承载不同内容版本，保持相同的目录、章节 ID 和现有读取接口。

目标内容分支为 **gpt-6-astra-2026-04-30**。当前本地修订版 **v2.2-r1** 保持 **108 篇专题、114 项固定来源**，相对 v2.1 深化 11 篇原专题。上一版六个新设章节已从活动目录撤回：有用的地址、API、缓冲和通信增量归回原章，重复机制与通用构建流程不再另立知识对象。来源上限仍为 2026-04-30。

研究目标保持为 Agent 从空工程构建并优化完整推理引擎。这个目标不预设组件划分、研发阶段或候选集合；[使用问题与探索边界](docs/knowledge-use-cases.md) 提供无序、非穷尽的问题示例。[本轮复审](docs/evidence/knowledge-open-exploration-20261024/review.md) 解释撤回与归并决定，[变更清单](docs/actionable-knowledge/change-manifest.json) 保存当前身份。旧v2.2的通过记录仅属于其当时的内容，不能证明新组织合理或Agent探索行为未受影响。

| 分支 | 内容 |
|---|---|
| [main](https://github.com/heitai-x/InfraAgents-know/tree/main) | 原始未限时 base，保留初始来源和章节历史；不作模型训练时间声明。 |
| [gpt-6-astra-2026-04-30](https://github.com/heitai-x/InfraAgents-know/tree/gpt-6-astra-2026-04-30) | 来源版本上限为 2026-04-30，当前修订保持108个原专题，深化11篇；问题入口不规定构建路线。内容身份以检出的VERSION.json为准。 |

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
