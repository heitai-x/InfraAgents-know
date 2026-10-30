# 当前维护入口

当前内容身份和数量见 [VERSION.json](../VERSION.json)，逐篇正文与审查绑定见[扩展状态](expansion-status.json)。构建、恢复与状态查看使用[构建 workflow](knowledge-build-workflow.md)说明的项目级 `knowledge-workflow`；实际状态在所选配置的 `state_dir`，队列与登记记录在知识库 `docs/evidence/<run>/workflow.json`。旧 dated ops 执行器只保留历史证据。

2026-10-30 核查：全量构建已在[最终批次](evidence/knowledge-workflow-20261028-batch24/workflow.json)收口，后续[评测模块四项](evidence/evaluation-module-20261029/workflow.json)已登记验证。截至已登记的draft30，111篇均有对应正文的接受记录。此结论是内容与登记状态，不代表目标GPU或受控Agent构建收益已经得到证明。当前[定点优化](evidence/knowledge-optimization-20261030/workflow.json)处理入口、表达与使用验证；进行中的正文修改尚不继承旧版接受结论，以该记录的实际状态区分进行中和完成。

## 2026-10-28：全量构建启动时记录

2026-10-28，按 DS 作者、Sol 独立审查、实际隔离检查与分批登记推进全部剩余条目。当前已登记版本与数量以 [VERSION.json](../VERSION.json) 为准，逐篇建设范围见[扩展状态](expansion-status.json)，尚在编写或返修的候选不计为完成。执行入口与实际 PID、heartbeat、阶段、未完成数位于项目目录 `ops/knowledge-full-build-20261028/`；协调者已接通真实模型与失败保留机制，不能将启动循环写成全库已完成。

来源版本上限仍为2026-04-30；编辑、独立推导和运行观测使用真实日期。以下保留此前时点的具体交付记录。

# 此前状态：构建 workflow 首批验收 v2.3-draft6

2026-10-28，知识库构建过程已接入[固定 workflow](knowledge-build-workflow.md)及本地技能。全111章进入可恢复队列，首批分页KV、前缀缓存通过非作者内容审查、实际算法检查、结构与真实冻结读取，登记为revision5。当前111篇、115项来源、237份活动历史；其余109篇正文和登记不变，来源身份及归档字节不变。

分页补齐跨页warp输入映射及独占原地追加/共享COW对照；前缀补齐精确身份、共同恢复、保护/预留/发布/淘汰与收益反转。队列/登记工具分别通过25项和14项行为测试，分页新增28项与原55项模型、前缀37项模型、既有35项算例均通过。结构检查3288个本地链接、237份历史；冻结读取覆盖本批两章、全部23个二级节和9份来源说明，内容绑定当前修订。

首轮过程、实际发现与证据边界见[本轮记录](evidence/knowledge-workflow-20261028/review.md)。全库深度待复核由99篇降为98篇，完整队列并未全部验收。没有GPU实测、受控知识收益实验、远程发布或活动Task切换；工具不启动后台模型服务。以下保留此前时点记录。

# 此前状态：机制知识并行扩展 v2.3-draft5

2026-10-27开始，2026-10-28复核登记。以Attention深度为样例，三个sub-agents深化分页KV、连续批处理、GPU流水，主Agent完成六标准内容复核、边界修正与统一登记。三章由revision3升为4，Attention保持revision8；当前111篇、115项来源、235份活动历史，维护五份来源说明，原始归档字节与身份不变。

[模块入口](mechanism-expansion.md)定位新增算法/映射、组合状态与成本反转；[审查问题](knowledge-review.md)将多轮发现归回原六标准。[逐篇状态](expansion-status.json)仍有99篇待当前深度复核，不能由导航登记推断完成。作者模型、三次实际交叉阅读/迁移/反证练习、冻结读取及结构结果见[本轮记录](evidence/parallel-mechanisms-20261027/review.md)。使用练习共享上下文且无知识版本对照，不证明受控知识增益；没有运行目标GPU或完整引擎构建。本地未发布、未切换活动Task。以下为此前时点记录。

# 当前状态：Attention硬件映射与组合状态 v2.3-draft4

2026-10-27。K-FW-006由revision7升为8；当前111篇、115项来源、232份活动历史。新增从全局地址到warp矩阵/行归约/写回的完整实例，以及共享前缀、量化尾块、图快照、取消和在途读者的组合案例；补充两份已有CUDA来源说明。其他110篇及原始归档不变。

[模块入口](attention-module.md)定位正文；[本轮记录](evidence/attention-mapping-composition-20261027/review.md)与[映射/组合检查](evidence/attention-mapping-composition-20261027/mapping-composition.json)区分内容、软件模型和真实使用效果。读取及结构结果按本轮报告记录，不能推成GPU性能或Agent构建收益。全库逐篇深度复核未完成，本地未发布或切换Task。以下记录按原时点保留。

# 当前状态：Attention推导补强 v2.3-draft3

2026-10-27。K-FW-006由revision6升为7；当前111篇、115项来源、231份活动历史。新增尺度完整的低精度算法示例、异步缓冲代次协议、GQA完整成本推导与区分竞争解释的诊断案例，修正四份来源说明。原始资料版本与字节不变，其他110篇未修订。

[模块入口](attention-module.md)提供阅读位置与内容验收题例；[本轮记录](evidence/attention-reasoning-upgrade-20261027/review.md)区分内容、有限模型、读取与真实使用效果。检查通过不证明GPU性能或Agent构建收益；全库逐篇深度复核仍未完成。本地未发布或切换活动Task。以下是按发生时点保留的既有记录。

# 知识库构建草稿 v2.3-draft1

本轮登记111篇、114项来源，保存229份活动历史。Attention 与五篇引擎比较进行了具体算法、状态和成本修订；三篇继承草稿补充成本推导；其余章节的导航元数据已登记，不能据此声明完成全库深度复核。源码逻辑由来源说明承载，混合资料保留文档原文。

运行时增加目录分页、结构/按节读取、真实读取范围、问题检索及观测/workflow 的既有发布路径。新实现尚未部署到活动 Task。当前检查和遗留事项见[构建记录](evidence/knowledge-construction-20261027/review.md)。以下内容是此前版本的历史记录。

# 知识库 v2.2-r2：表达清理与验证接口更新

2026-10-27。本轮相对 v2.2-r1 修正4篇正文：模型接入、PyTorch 推理、可观测性与容量规划。将固定构建顺序、假设数量和分析先后改成按实际问题选用的方法；数学、状态和资源完成依赖保留。设计导航与范围说明同步去除固定候选数量和统一产物要求。

仍为108篇 seed、114项固定来源；新增4份准确旧修订，活动历史共121份，六篇撤回档案继续保留。固定原始资料未修改。来源定位修正、时间审计再生成、当前读取与采用绑定检查见 [本轮记录](evidence/knowledge-cleanup-20261027/review.md)；当前内容以 [VERSION.json](../VERSION.json) 为准。

本轮为本地修改与验证，不包含独立审查者结论、远端发布或活动 Task 切换。外层 v2.2-r1 bundle/patch 是历史交付，未包含本轮改动。以下旧报告保持其原有验证范围。

## v2.2-r1 历史记录：探索边界与专题归属修订

## 修订决定

原 v2.2 本地提案的六篇新增章节已撤回活动目录。来源与算例正确，不足以证明新建章节合理，也不证明它不会形成默认架构或构建顺序。当前保持原有108篇、114项固定来源；相对指定分支 f640148464929f803a53b55e27ee4cbb83dc8657 深化11篇。没有删除原分支的任何专题。

| 撤回的本地新章 | 修订处理 |
|---|---|
| K-SYS-020 从空工程构建 | 撤回通用阶段与固定组件/接口规划；重复KV与关键路径例不再另存；已有模型、系统组合与容量专题保持各自作用 |
| K-SYS-023 知识到设计决定 | 撤回统一产物与阶段式入口；布局转换成本留在SYS010，问题示例留在维护文档 |
| K-HW-015 CPU–GPU重叠 | 四阶段、槽位与共享资源量化归HW003；异构放置、stream和NUMA继续由原专题解释 |
| K-SYS-022 分布式执行 | count和标识反例归SYS005，ring时间归HW002；不重复TP/PP/EP/RDMA，不要求固定数量候选 |
| K-FW-013 张量接口 | 地址/别名增量归ALG016，具体缓存API归FW006；已有数值、预打包和生命周期不重复 |
| K-SYS-021 API契约 | ASGI语义归SYS008，推进/队列反例归SYS015；不要求固定三层架构 |

详情见 [复审](evidence/knowledge-open-exploration-20261024/review.md)、[专题撤回登记](actionable-knowledge/withdrawn-chapters.json) 和 [使用问题](knowledge-use-cases.md)。实际正文增量按 [变更清单](actionable-knowledge/change-manifest.json) 归属。

## 身份与验证

前次交付的修订号与正文保留，新修订再次增加revision，避免同一ID/revision对应不同内容。当前活动专题有117份历史；六篇撤回稿另存原v1正文与metadata，ID保留不复用，不能通过当前冻结知识视图读取。前次内容身份见 [旧VERSION](evidence/knowledge-open-exploration-20261024/previous-v2.2-VERSION.json)。

本轮分别提供 [结构检查](evidence/knowledge-open-exploration-20261024/structural.json)、[算例检查](evidence/knowledge-open-exploration-20261024/examples.json)、[真实读取探针](evidence/knowledge-open-exploration-20261024/runtime-probe.json) 和 [独立复审](evidence/knowledge-open-exploration-20261024/review.json)。复现入口见 [使用说明](USAGE.md#修订版的内容检查)。这些结果不证明Agent实际探索不受任何偏置，也不证明模型、GPU、服务性能或RSI收益。

运行时仍使用此前提供的源码归档并按模块SHA绑定；推荐分支获取时要求GitHub凭据，本轮未据此冒认推荐提交。没有改Controller或知识发布器，也没有运行Agent构建实验。工作位于本地 codex/knowledge-open-exploration-20261024，尚未推送远端。

## 既有正式基线记录

以下保留v2.1及更早版本在其当时内容与环境中的记录；原v2.2提案的结构/算术通过记录保存在 [旧复审](evidence/knowledge-actionability-20261024/review.md)，其组织方式已被本轮取代。

## v2.1交付记录（历史）

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
