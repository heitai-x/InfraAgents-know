# 前稿知识库探索边界审计

**内容的规范性越界成立；运行时并未把这套文字方法编译为固定执行顺序。** 本报告保留对修改前内容的独立判断。

root在报告落盘前开始改写工作区；前稿已从其提供的暂存tree 085646f6c2df3e479ee270f092131828f6f141df按原始字节恢复并绑定SHA，完整前稿另存scratch/knowledge_audit_previous。以下引文不引用修订后工作区。新稿另做独立审查。

## 保留的目标与边界

用户明确目标仍然是从零构建并优化完整的专用推理引擎。撤回一套编辑施工法不等于允许以局部kernel替代完整结果。实际Task约束和所选API/数学合法性必须满足；它们不能再被扩成未经要求的服务功能、组件划分、候选数量或统一交付格式。

## 八项具体发现

### E01：完整引擎被默认解释成在线服务的六类对象

开头与六类对象表把持久状态、请求/分支、调度、外部协议作为完整性的默认解释。“不要求六个类/进程”只放开物理分层，未放开概念分解；完整引擎不自动等于在线生成服务。

建议：保留从零完整引擎的研究目标，以实际Task输入输出/计算/必要状态/资源目标界定完整性；撤回服务适配、动态调度、KV等通用必需地位。在线服务作为有前提的一种实例。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第3行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：从零构建推理引擎，需要把模型函数、长期状态、一次执行、请求生命周期和外部服务连接成可独立验收的系统。本章讨论如何决定首先构建什么、哪些能力必须共同形成、何时可以转入优化，以及一次局部成功究竟推进了哪一部分系统。阶段划分与算例是依据固定资料作出的设计推导，不是某次 Agent 运行的观测结果。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第19行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：| 构建对象 | 必须明确的关系 | 可检查的交付物 |
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第25行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：| 服务适配 | 外部请求怎样进入引擎，输出怎样形成协议事件 | 输入验证、背压、错误、取消、版本及终止语义 |
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md` 第23行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：| 空工程启动 | 任务函数、持久状态、资源与对外服务之间的连接 | 必要能力、组件责任、接口与可交付阶段；未定问题另列 | 固定输入的真实执行与完整请求路径 |

### E02：能力依赖被包装成模型—状态—批处理—服务—优化的默认阶段

表头“先建立的能力 / 本阶段留下什么”及行序形成默认施工路径；docs贯穿案例用U01首先、U02、U03增加服务重复该路径。“不是固定的全局流程”没有撤销这些顺序信号。某个结果依赖什么条件，并不规定研究者必须先实现哪一层。

建议：撤回通用阶段表地位。只陈述某个具体设计的真实依赖和可验证问题；允许容量/布局/代价探索先于完整服务，也允许原型、接口与验证共同迭代。不要以另一条强制路径替换。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第34行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：| 当前缺口 | 先建立的能力 | 本阶段留下什么 | 尚不能据此声称什么 |
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第40行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：| 服务正确但目标未达到 | 端到端运行点、瓶颈解释与竞争设计 | 同条件下的候选比较、资源变化及失败边界 | 任意场景迁移或某项技术始终更快 |
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/knowledge-use-cases.md` 第65行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：U01 首先形成从输入到模型状态再到返回的闭合架构，并把缺失能力分成可实现阶段。U02 将位置、mask、KV 更新与后端布局连接成真实计算路径，检查局部数值和集成。U03 增加服务路径时，明确会话状态、请求代次、输出归属和取消回收，验证相互交错的请求。

### E03：实现前确定接口及每次执行的通用字段扩大了真实合法性约束

“在组件实现之前确定关键接口”和“一次执行至少需要知道”请求/分支/逻辑位置/持久状态等，把语义正确关系写成统一接口组织。静态编译、融合执行或调用关系可以隐式满足某些关系，并不一定需要独立请求对象、组件或运行时字段。

建议：删除实现前的时间性要求；只在具体共享、异步、重排等机制出现时说明必须成立的语义关系。显式/隐式表示、组件边界和接口何时定型保留为设计变量。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第46行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：## 在组件实现之前确定关键接口
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第48行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：内部接口不仅传递数据，也传递进度和所有权。一次执行至少需要知道每个输入属于哪个请求与分支、对应哪些逻辑位置、使用什么模型版本、允许读取哪些持久状态、输出应写到哪里，以及何时可以复用输入与结果缓冲。外部 API 再决定这些结果何时成为用户可见内容。具体边界见 [接口契约](api-contracts.md)。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第28行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：这些对象不要求分别对应六个进程或六个类。小模型可以把多个对象放在同一执行域，复杂模型可以拆分。需要稳定的是责任与接口；物理分层本身不证明可扩展性。Orca 的端点、请求池、调度器与执行引擎，以及 PagedAttention 的调度、状态管理与 worker 关系，提供了两种具体组织依据，但不要求新引擎照搬其文件布局或全部策略。[对应来源见文末](#来源与推导)。

### E04：每次阅读被附加通用输入、设计产物与备选方案要求

K-SYS-023要求可推进设计至少具备整组上下文，阅读后需要说明映射/证据/改动，有分支就保留替代方案；docs交接和模板末节再重复。这会把理解、排除、暂不采用或发现未知也变成一次完整设计交付。“不要求填满表格”只取消形式，未取消义务。

建议：撤回每次阅读后应交付的普遍要求。上下文只由当前推理需要确定，未知可以是研究对象；备选数量、记录格式和是否展开由任务与预算决定。不得反向强制Agent必须多样化。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md` 第13行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：一个可推进的设计问题至少需要任务函数与不可改变的质量约束、真实硬件/库版本、负载条件、已有能力、当前观察和待解释的不确定性。条件尚未获得时，应保留未知及其影响，不以章节中的设定数字填补实际环境。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md` 第35行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：阅读一个相关章节后，设计者需要说明当前对象与章节对象怎样对应、哪些前提已有证据、哪些仍未知，并给出选择造成的具体改变。例如“采用前缀复用”需要明确共享身份、可恢复位置、私有尾部与回收条件；它还应预测省掉哪些 prefill，而不是承诺全部 attention 流量同比下降。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md` 第37行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：对有实际分支的决定，保留一个满足约束的替代方案，并写出使选择反转的条件。无替代方案的数学必需关系不强行凑两个候选。选择来自事实约束、推导和待检验假设的组合，三者各自可追溯；源论文中的成功实例只能帮助提出候选。
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/knowledge-use-cases.md` 第57行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：交接应具体说明：采用了哪个机制版本；当前对象怎样满足其条件；尚未确认什么；由此改变了哪个接口、布局、调度或实验；什么观察会否定这个选择。选择尚未做出时，保留竞争方案和需要补充的事实。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/templates/entry.md` 第33行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：说明当前机制会在什么问题下被使用，需要哪些事实前提，以及使用后应形成什么设计、接口或实验产物。将 API 名称与真实版本、参数含义、shape/dtype/layout、状态、副作用、完成点、错误和回收条件连接起来；名字相同不证明接口语义相同。

### E05：不同证据支持范围被呈现为统一层级通关顺序

K-SYS-020标题写从局部到集成再到系统；K-SYS-023列四层并要求各层参考。局部与系统证据支持的结论确实不同，但不推出所有探索必须依次执行四层，也不推出单任务交付必须跨任务试用。审计时schema1已给U05 collective和U07跨任务试用加条件，但整体框架仍有通关含义。

建议：保留证据支持范围，撤回统一顺序。验证需求绑定具体主张和Task；跨任务支持只服务泛化/晋级主张。可以先观察完整系统再定位局部，也可复用未受影响证据。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第82行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：## 从局部到集成再到系统的验证
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md` 第57行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：## 四层验证回答四种不同问题
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md` 第66行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：各层使用独立程度适当的参考，并将数值、模型质量和协议语义分别处理。PyTorch 固定数值文档说明改变批量组织或归约路径可能改变舍入结果；因此变形测试应遵循任务原定的数值/输出标准，不能看到差异后临时放宽标准。若任务明确要求精确 greedy 输出，就须继续证明该要求，不能以一般浮点误差解释替代验收。

### E06：同一套未经任务检验的方法被正文、模板、导航反复赋予seed权威

K-SYS-020/023是scope全空的seed；docs把023登记为运行期共同入口，模板将其作完整示范。来源末尾虽然说明阶段/七类决定是编辑设计，但前文反复要求产物和流程。机制论文对某种组件关系的支持不等于从空工程方法的普适证明。

建议：撤回023作为共同默认方法入口；把布局转换成本等实质机制归并到相应旧章，把通用流程说明移出规范性知识。020保留用户主问题但移除统一施工法，或将其实际机制归并后按历史规则撤销；不能静默抹掉旧记录。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md` 第96行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：- [S-ORCA：Orca](../sources/S-ORCA.md)：原文 §3 中的 iteration-level scheduling 与 selective batching，尤其 Figure 4–5 及相邻完整论述，解释端点、请求池、调度器、执行引擎和 KV 管理如何共同组织请求推进。它支持组件关系与调度粒度分析，不提供本文的从空工程阶段方法或容量算例。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md` 第78行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：- [S-ORCA：Orca](../sources/S-ORCA.md) §3 的问题与迭代调度设计明确连接 endpoint、request pool、scheduler 与 execution engine，支持本章关于粒度改变需要跨组件契约的例子。[完整原文](../../references/S-ORCA/reader.html) · [全文文本](../../references/S-ORCA/fulltext.txt)。本章的七类决策时刻和使用产物是编辑设计，不归称为 Orca 的方法。
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/knowledge-use-cases.md` 第17行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：完整的运行期可读说明已登记为 [K-SYS-023：从知识到设计决定](../knowledge/system-semantics/design-decisions.md)。它属于有版本的章节，进入 catalog 和冻结快照。本页用于说明知识库维护与接入，不要求运行时把未登记的 docs 当成知识正文注入。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/templates/entry.md` 第37行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：上述内容是审阅问题，不规定所有专题采用相同标题、篇幅或短卡。运行中的项目事实、知识使用声明、原始证据和具有范围的认识分层保存；不以登记或审查通过代替真实目标结果。完整示范见 [从知识到设计决定](../system-semantics/design-decisions.md)。

### E07：use-cases.json制造类似任务合同的字段权威

schema1同时写not an implemented runtime dispatcher、runtime_entry、expected_outputs、required_validation；validate_knowledge要求这些字段非空。它是内容作者的结构门，不是工程执行门，但容易促使每类问题人为补固定产物与义务。

建议：撤回runtime_entry和通用expected_outputs/required_validation合同含义及相应非空校验；场景只保留非穷尽问题与导航例子，不换成另一组统一必填方法字段。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/docs/actionable-knowledge/use-cases.json` 第5行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：  "purpose": "Content routing and review contract; not an implemented runtime dispatcher.",
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/actionable-knowledge/use-cases.json` 第6行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：  "runtime_entry": "K-SYS-023",
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/actionable-knowledge/use-cases.json` 第29行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：      "expected_outputs": [
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/actionable-knowledge/use-cases.json` 第34行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：      "required_validation": [
- `/workspace/scratch/db4271884843/InfraAgents-know/tools/validate_knowledge.py` 第165行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：        for field in ("trigger", "inputs", "expected_outputs", "required_validation", "retrieval_queries"):

### E08：默认入口和检索强化方法框架，实际消费路径并非静态U表

静态DESIGN写先用023，并列逐阶段验收。真实选定快照由selected_navigation生成，仍有先固定再核对的通用指导及全部专题摘要；标题/tags/summary参与加权检索。这些可能塑造问题表述，但没有代码证据证明Agent实际只能选择一套方案。

建议：撤回必经方法入口和普遍施工摘要；保留从零完整引擎主题的可发现性。静态导航与框架生成导航分别处理；不添加强制任务路由器或流程状态机来修复文字过度规范。

前稿依据：

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/DESIGN.md` 第96行（tree `085646f6c2df3e479ee270f092131828f6f141df`）：从空工程到完整推理引擎时，先用 [知识到设计决定](system-semantics/design-decisions.md) 明确当前需要作出的决定、实际条件、工程产物和区分实验。U01–U07 表示工程时刻，与上面的 C01–C12 部署场景共同定位知识；完整说明见 [使用场景设计](../docs/knowledge-use-cases.md)。

## 真实运行时边界

### R01：没有按K-SYS-020/023或U01–U07表格执行的消费链路

- 全src检索 templates/entry、knowledge-use-cases、use-cases.json、K-SYS-020、K-SYS-023、expected_outputs、required_validation、runtime_entry 无命中
- infra_agent/rsi.py::prepare_snapshot 仅冻结catalog、来源和选定导航
- infra_agent/tools.py 76/77/84及745/747/764行：knowledge_index/search_knowledge/read_knowledge直接派发

### R02：Task合同与知识导航分别进入上下文，正文按需读取

- infra_agent/role_context.py 232行附近 global-task-contract
- infra_agent/role_context.py 255行附近 Catalogs remain navigation artifacts; bodies loaded only on request
- 选定snapshot不枚举docs/use-cases与entry模板，静态DESIGN也由生成导航替换

### R03：knowledge_used默认可空；非空校验身份，不校验施工方法

- infra_agent/role_contracts.py::KnowledgeUse只有entry_id/revision/purpose；PlannerReply等默认空列表
- infra_agent/role_runtime.py 340–343：uses非空才validate_uses
- infra_agent/knowledge.py::validate_uses校验ID/版本/非空purpose/读记录/冻结字节，没有步骤或组件合规检查；读范围完整性也未强制

### R04：真实硬门是知识资格/版本/路径/原文权限等，不是章节设计方法

- infra_agent/knowledge_selection.py::knowledge_policy/resolve_selection/selected_view
- infra_agent/knowledge.py::read精确ID@revision、snapshot text_sha256、字节SHA、读取字符范围

### R05：搜索没有自动证明Task条件适配

- tools公开参数仅query/mode/limit
- RepositoryKnowledge.search不传入Task scope
- knowledge_store.search有底层scope参数，但这里默认为空；Agent可见Task后自行组织query

### R06：typed workflow存在数据结构，没有查到步骤消费闭环

- knowledge_assets.py::validate_asset校验workflow trigger/preconditions/steps/gates
- controller.py只初始化KnowledgeAssetStore；全src引用主要为CLI与存储
- control_plane.py::_accumulate_work只累积knowledge_used记录，不以章节/字段生成执行序列

### R07：上轮API探针通过没有检验探索自由或知识收益

- validate_knowledge.py对use-case字段只检查非空
- probe_knowledge_runtime.py取固定retrieval_queries并记录knowledge_ids交集，未执行schema1的expected_outputs/required_validation

## 逐项修订/合并/撤回建议

- **K-SYS-020**：保留从零完整引擎目标和必要机制；撤回默认六对象、阶段/接口时间要求，或归并机制后撤销方法入口并保存历史。
- **K-SYS-023**：撤回共同默认方法入口；代价关系和证据边界可归并至机制/工具旧章，通用流程建议移出规范性知识。
- **docs**：保留研究目标，七项作为无序问题示例；去掉U01→U02→U03默认施工与统一交接义务。
- **use_cases_schema**：移除runtime_entry/expected_outputs/required_validation合同含义及作者非空门，保持导航例子。
- **entry_template**：保留机制/推导/来源/假设要求，撤回统一产物与唯一示范。
- **navigation**：区分静态文件与实际生成导航，不添加新的强制路由/工作流。

## 阅读记录及前稿字节身份

- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/engine-construction.md`；全文（修订前工具输出已逐段读取）；SHA-256 `d3254439c8dea9f29a2efc5d9242a3134a9a2a57dd2e87976634db838d43d285`。 恢复前稿：`/workspace/scratch/db4271884843/knowledge_audit_previous/knowledge/system-semantics/engine-construction.md`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-decisions.md`；全文（修订前工具输出已逐段读取）；SHA-256 `a9552b50f3ad86305a93a3e8f880f5ba6822363c4a2c5a56b9f66fd1b84bbdf2`。 恢复前稿：`/workspace/scratch/db4271884843/knowledge_audit_previous/knowledge/system-semantics/design-decisions.md`。
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/knowledge-use-cases.md`；全文（修订前工具输出已逐段读取）；SHA-256 `9e36e3ea2232454f4d4969a0293817683d8ac13b11208b7a3f3cbaae351ca914`。 恢复前稿：`/workspace/scratch/db4271884843/knowledge_audit_previous/docs/knowledge-use-cases.md`。
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/actionable-knowledge/use-cases.json`；全文（修订前工具输出已逐段读取）；SHA-256 `477d230ed963dbadbfafbea7e9128e164509b092cc0661ea911840e2d87d8ae3`。 恢复前稿：`/workspace/scratch/db4271884843/knowledge_audit_previous/docs/actionable-knowledge/use-cases.json`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/templates/entry.md`；全文（修订前工具输出已逐段读取）；SHA-256 `ed2add5dff67f6570058597a8ad3fccb0a11938d1ff3e4a4d210332796c31d1b`。 恢复前稿：`/workspace/scratch/db4271884843/knowledge_audit_previous/knowledge/templates/entry.md`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/DESIGN.md`；全文（修订前工具输出已逐段读取）；SHA-256 `2eb108ac94c0cd17937547c8d0b9b2c8f5eedf773a1f6cbf4c177561b08509a0`。 恢复前稿：`/workspace/scratch/db4271884843/knowledge_audit_previous/knowledge/DESIGN.md`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/DIAGNOSIS.md`；全文（修订前工具输出已逐段读取）；SHA-256 `51efd72c5832f95ed8660c451b17cf859b7bdce989672542789ef9a6b05930b5`。 恢复前稿：`/workspace/scratch/db4271884843/knowledge_audit_previous/knowledge/DIAGNOSIS.md`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/knowledge.py`；validate_uses/index/search/read全部相关方法；SHA-256 `aee95c8276b261b460863a129a449327752436a6c3e157b9716cb85a19faa6fe`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/knowledge_index.py`；全文；SHA-256 `94c15e8e5e700bbbf975b26699b2ff6493aba2d2f2e27acbf8e9196dad5c01a3`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/knowledge_selection.py`；全文；SHA-256 `8bca46f5e46b390c863975e9509df97d1da9d5afeff4777c94a25fd0e41921f2`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/knowledge_store.py`；158–232检索实现；SHA-256 `24b9c640ba096568f0bb01b11565c8b0d34760279bad7c96d8b2e4d7d6dc7df3`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/tools.py`；65–90与735–775 schema/派发；SHA-256 `acf71a076dc256c5439a972fcab53c289da66fa6b2e06e19ac75bedf94ac836a`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/role_contracts.py`；KnowledgeUse与角色输出字段；SHA-256 `c4aecf3596ad0d2ba408f3b3530382eee538b34f7836253078ac5235d99cedc4`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/role_runtime.py`；328–350使用声明验证；SHA-256 `90306d7ddd2928d945cb9dce7c1ba350a807c1ac1542bf144db18206a3f945df`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/role_context.py`；228–263 Task及知识导航；未把截断的整文件输出当全文；SHA-256 `7f4295b4cae944adc2567d387f950e0fdfd658905d745196325ce3239e6b062d`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/rsi.py`；13–130快照构建；SHA-256 `36a3b0136f2fc87e104b9b5f823a949878fd2745611917855e06423fc04a3900`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/knowledge_assets.py`；1–120工作流schema及全src引用搜索；SHA-256 `239fdb3015f1ce488ac31eb78f466b525305ecc3f2393fb3ed3f9e8c679360ea`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/controller.py`；70–100、815–838初始化与反馈引用；SHA-256 `abb598df3aaaa39d9e434e854445a325bc918c8fc31a765adbbebcdc85e877dc`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/control_plane.py`；1995–2017知识使用累积；SHA-256 `f7aacb16632368194435f4f3d17b35ac99a7bc2b389bec2d0a66de4fc1833e06`。
- `/workspace/scratch/db4271884843/review_src/InfraAgents-main/src/infra_agent/resources/protocols/knowledge.md`；全文；SHA-256 `ff26729bf49f177e567c154d6355b1b437d2d64377add1b5be6c893eec46fc56`。

## 限制

- 未运行Agent，不能声称已测出方案多样性下降或量化语义引导强度。
- 结论限于提供的本地运行时，不冒认为推荐私有/未获取分支。
- 没有重新审计整个ControlPlane；其既有Task、角色或roadmap约束与本次知识文本的顺序约束须区分。
- 没有修改任何仓库正文、模板、metadata或代码；只写scratch报告。
- 不要求Agent为探索而必须多样化，也不为证明开放而绕过真实任务与API合法性。
