# InfraAgents-know

InfraAgents 的独立知识仓库。不同分支承载不同内容版本，保持相同的目录、章节 ID 和现有读取接口。

目标内容分支为 **gpt-6-astra-2026-04-30**。当前本地构建草稿为 **v2.3-draft19：111 篇、115 项来源**。来源上限保持 2026-04-30，原始归档字节不变；登记前正文、来源说明与元数据按准确身份保存。以 Attention 的深度样例为基础，当前继续深化分页 KV、连续批处理和 GPU 流水；算法、组合状态、成本与验证范围见[本批模块](docs/mechanism-expansion.md)。

制作标准见[知识库构建与修订方法](docs/knowledge-method.md)：说明问题的原因/动机、数学与状态定义、合法条件、具体算法和数据流、各项成本、反转条件及区分实验。允许用伪代码交付实现逻辑，论文与说明文档直接可读。[审查问题](docs/knowledge-review.md)复用多轮审阅发现的缺口；逐篇当前状态见[扩展状态](docs/expansion-status.json)，导航登记不等于深度验收。[构建 workflow](docs/knowledge-build-workflow.md)负责全库队列、修订、非作者复核、真实检查与当前版本登记。

研究目标是让 Agent 自主设计、构建和持续优化完整推理系统；阅读分类不规定组件、施工顺序或候选集合。[使用问题与探索边界](docs/knowledge-use-cases.md)保留无序示例，[变更清单](docs/actionable-knowledge/change-manifest.json)记录当前内容身份。此前 v2.2 系列的验证属于历史版本。

| 分支 | 内容 |
|---|---|
| [main](https://github.com/heitai-x/InfraAgents-know/tree/main) | 原始未限时 base，保留初始来源和章节历史；不作模型训练时间声明。 |
| [gpt-6-astra-2026-04-30](https://github.com/heitai-x/InfraAgents-know/tree/gpt-6-astra-2026-04-30) | 来源版本上限为 2026-04-30，当前保留108个原专题，构建草稿增加3篇；问题入口和参考方法不规定构建路线。内容身份以检出的VERSION.json为准。 |

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

当前Attention修订的算法、硬件映射、组合状态和验证边界见[模块说明](docs/attention-module.md)。
