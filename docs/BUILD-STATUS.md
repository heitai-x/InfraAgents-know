# 推理知识库 v2.1 交付记录

v2.1 已完成本轮独立审查与修订闭环：第一轮发现两项 P2 和一项 P3，修订后第二轮对实际内容提交 `69408e3fc62c8d34e576c2f1ce6d247488d568b9` 给出 **KB PASS**。详见 [第一轮](evidence/knowledge-review-loop-20261023/round-1/review.md)、[第二轮](evidence/knowledge-review-loop-20261023/round-2/review.md) 和 [历史及来源核对](evidence/knowledge-review-loop-20261023/round-2/repair-checks.json)。

修订包括合法的 MoE distinct top-2 分派算例、HF 缓存概览的来源归因边界和入口状态。13 篇当前章节各增加一次 revision；旧正文与 metadata 精确保留，其余 95 篇和全部原始归档未改变。当前仍为 108 篇、114 项固定来源，历史修订增为 96 份，来源版本上限仍为 2026-04-30。

当前验证绑定 GPU 兼容运行时 `0802c3f140e0b46e26bfdd686ae55662db39c042`：实际 `full + seed_and_learned`、五角色 Luna 和九项受限原文策略通过，完整读取 108 篇和 204 个当前/历史版本，27 个受限原文入口全部被拒绝，详见 [当前实际接口验证](evidence/knowledge-review-loop-20261023/round-2/runtime-probe.json)。独立审查两轮累计完整阅读 108 篇正文，原文阅读范围为 19 项（9 项完整文本、10 项相关范围），未宣称重读全部 114 项来源。

下列内容记录 v2 建设及其当时的验收。`final-linux-validation.json` 和 v2 发布/交付回执保持原始历史证据；它们不替代本轮 v2.1 的实际内容验证。知识库通过不等于新的真实 GPU Task 已运行或达成目标。

运行期发布另于 2026-10-23 完成真实接入补验：修复旧发布器对 CRLF 的隐式转换后，108 篇库与原批准反馈的隔离回放、幂等和历史恢复通过；实际 GPU run 已完成原阻断发布步骤并继续下一工作项。正文没有因修复改变。当前建议运行时为 `6d205853b6135adbc35b43525608ff919eb4dee6`，补验范围见 [独立发布审查](evidence/publication-runtime-fix-20261023/review.md)，使用方式见 [v2.1 发布接入](USAGE.md#v21-的运行期反馈与发布)。这不表示真实 Task 的系统质量或性能目标已经达成。

## v2 建设与验收历史

本轮已完成声明范围内的知识库修订：**108 篇专题、114 项固定来源、12 类完整场景**。原 73 篇全部处理，35 篇新增全部接入，C12 包含 embedding/重排、检索/RAG、CNN/检测、推荐和 GNN 五份独立契约。资料版本上限保持 **2026-04-30（含当天）**；编辑、复核与验收日期为 **2026-10-23**。

范围是完整推理系统及其集群、存储和部署依赖，不包括完整训练平台生命周期或全部设备产品百科。八类目录、非代码机制正文、来源登记、章节身份和精确历史保持原设计。完整范围见 [24 能力域与平台说明](inference-coverage.md)，逐篇状态见 [实施计划](expansion-plan.json)。

## 正文与交叉复核

三个 sub agents 分别承担服务/平台、非生成负载/编译量化、模型/多模态；主 agent 完成系统语义/观测/引擎导航并串行整合。所有 108 篇均有作者之外的实际审阅，范围与精确内容身份见 [整库语义复核](evidence/final-semantic-review-coverage.json)。初期 P0 发现的 4 篇、5 处计量/布局表达问题已修正并再次独立核对，见 [闭环复核](evidence/p0-final-resolution-independent-review.json)。

本轮新登记的 38 项固定来源已有完整技术正文/相应附录审阅记录，见 [来源复核汇总](evidence/final-new-source-review-coverage.json)。原有 76 项来源保留既有依据；本轮没有声称重新逐页阅读全部原始材料或递归阅读其所有参考文献。固定原文保持原始字节，公式/条件限制写入来源解释与独立推导，不修改原论文。

## Linux 实际读取验收

[最终验收](evidence/final-linux-validation.json)通过，使用真实 RepositoryKnowledge 和五个实际小写角色身份：planner、executor、experimenter、diagnostician、evolution。

- 108 篇正文、114 份来源全文和114份来源说明全部按真实接口分页读取，拼接字节与冻结身份一致。
- 结构、归档、时间资格与全部 83 份历史修订核对；原 73 篇 revision 1 与初始 Astra 提交逐字节/逐字段一致。
- 27 条场景与分类自然问题检索通过；生成的设计、诊断、目录及来源导航链接可解析。
- 五角色的实际读取与版本声明校验通过；错误版本、未读取声明和重复声明按合同拒绝。
- 第二快照限制全部114项原文，确认原文入口被拒绝、生成导航不暴露受限归档，来源说明仍可读。

这些检查证明本轮材料可按选定策略完整读取、追溯和限制访问。它们不证明模型理解了内容，也不代替真实模型数值、服务性能或知识收益实验；设定算例均保持假设身份，全部章节仍为 seed。

## 运行时接入

此前选定视图会把 DESIGN/DIAGNOSIS 替换为空入口。本轮修复根据冻结且已过滤的 catalog/source metadata 生成可用导航，既保留入口作用，也防止从未经登记的根文档带入被排除内容。修复在 [独立运行时分支](https://github.com/heitai-x/InfraAgents/tree/codex/knowledge-navigation-20261023)，提交 `aedff22c0885c4fc56db6058b9904e23d5e189b9`，未合并或切换正在运行的实验。

修复的28项针对测试、增强后的5项导航测试及独立7项检查均通过，见 [运行时验证](evidence/runtime-navigation-validation.json)。扩大知识测试为46项，其中45项通过；唯一失败是框架初始知识库引用的两份旧 learned 实验文件不在隔离检出中，已在未修改基线上复现。Astra seed 的最终审计无该告警。

Windows 早期检查的目录提升失败保留在历史记录；最终验收使用 Linux 原生快照提升，没有手工提升或吞掉错误。正在运行的 CPU/DeepSeek 实验目录没有切换。

## 版本与后续使用

v2 当时的内容版本为 `base-gpt-6-astra-2026-04-30-v2`，分支为 `gpt-6-astra-2026-04-30`。main 保留原始未限时 base，DeepSeek 日期分支未创建。知识库可通过现有参数选择，见 [使用说明](USAGE.md)。运行期 learned 继续由原有 Evolution、独立复核和发布流程管理。

[GitHub 发布核对](evidence/v2-publication.json)和[远端交付核对](evidence/v2-remote-delivery.json)均已通过。不可变内容标签为 `astra-2026-04-30-v2`，提交 `fdce576c5972c25b91ad81ba887c616d717ea3ca`，发布时全部 1645 个版本控制文件的 Git 树与本地一致。远端独立目录为 `/mnt/infra_agents/knowledge-worktrees/gpt-6-astra-2026-04-30-v2`，内容身份、来源资格和运行时文件身份再次核对通过；原 v1 目录和实际实验没有切换。

本分支后续交付回执提交只补充这两份记录与完成状态，标签仍固定在已验证的 v2 内容提交，未移动标签。正文、复核、验收和发布各自有独立证据，不以仓库推送成功代替内容与功能检查。
