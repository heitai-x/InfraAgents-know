# 探索边界修订稿独立审查

**修订后通过。** 新docs、schema2、模板和新增机制段解决了前稿的主要问题；K-SYS-010相邻旧文三处残留也已修正并独立复读关闭。保留从零构建完整引擎这一研究目标，没有把合法性扩展成固定组件/候选/步骤，也未反向要求Agent必须多样化。

## 已通过的内容

- **docs/knowledge-use-cases.md**：从零只描述Task起点，保留完整引擎目标；功能、质量与复用边界归Task；明确不增加组件/顺序/阶段门。
- **docs/knowledge-use-cases.md**：整体原型、编译先行、协议先行、局部成本实验与联合演化都以可以表述；没有把这些例子改成新的候选白名单。
- **docs/actionable-knowledge/use-cases.json**：schema2无roles/trigger/expected_outputs/required_validation/runtime_entry；ordering=none、mandatory_entry=null、must_use_knowledge=false，context_examples/possible_observations均受实际Task与所选机制约束。七问题不是执行阶段。
- **knowledge/templates/entry.md**：保留技术对象/版本/副作用及数学/API语义，明确不要求统一字段、交付物、候选数量或知识使用考核；因果依赖与研发顺序已区分。
- **knowledge/system-semantics/design-synthesis.md 新增两节**：新增部分本身允许改表示、融合边界、改变成本前提及放弃所有所列候选。1.0 / 0.6 / 0.5 ms例为条件化成本反例，R>1.25和R=2时1.7对2.0的算术成立，无实测声称。未把合法性扩成预设组件或候选限制。
- **knowledge/tools/correctness-oracles.md 增量**：检查由原Task/所选机制派生；负对照检验实际要求的辨别力；不要求每次探索复跑全部状态/协议场景。保持用户验收标准，不扩大为固定施工步骤。
- **knowledge/tools/statistical-experiments.md 增量**：对照设计已明确属于研究者可选择的评估，不是Agent构建流程。质量/成本结论与读取命中分离，未要求Agent必须给出不同架构或偏离示例。
- **反向多样化约束**：docs末节明确路径不同不等于收益、相同不等于失败，不向Agent下发必须多种架构/必须偏离示例；通过此项。

## 首轮意见与关闭记录

### N01（已关闭）

`/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-synthesis.md` 首轮第112行：一个内核或单实例方案满足资源上限之后，还需要 [集群放置](../workloads/cluster-resource-placement.md) 找到满足拓扑和故障余量的执行组，以 [存储服务](storage-services.md) 保证制品、缓存与提交对象的相应语义，并在 [部署环境](deployment-environment.md) 的实际配额、设备和版本下验证。集群空闲卡数、进程存活或模型文件存在，都不能独立证明服务容量已经可用。

仍把一个单实例方案通用地扩展为集群放置/存储服务要求，后面的开放声明不能撤销此处的还需要。

修正后：已限定为Task涉及多节点或选定方案使用共享制品/缓存/记录；明确单机/离线不会自动增加这些功能。

关闭时 SHA-256：`0480a8e7697850656cc1d8fe0534e97288505a9b08565d038ba2bc51958d7651`。

### N02（已关闭）

`/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-synthesis.md` 首轮第114行：每次组合都保留同一条说明链：任务条件与实际输入 → 状态及共同恢复点 → 计算/流量/容量/排队 → 合法候选 → 交互约束 → 独立参考和区分实验 → 条件改变后的重新选择。最终结论应能解释候选为何被保留或淘汰，而不是把每一种优化都叠到默认实现上。

仍将一种解释关系规定为每次组合的统一产物/顺序，与不设中间文档或施工路线的修订目标冲突。

修正后：改为这些示例的关系说明，允许其他表示/推导组织，明确不补造不存在的恢复/排队模块。

关闭时 SHA-256：`0480a8e7697850656cc1d8fe0534e97288505a9b08565d038ba2bc51958d7651`。

### N03（已关闭）

`/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-synthesis.md` 首轮第45行：先用不依赖精细拟合的容量和资源下界排除不可行组合，再选择少量能区分竞争解释的实验。记录一次性成本、稳态工作和失败恢复，避免候选只比较最顺利的一次执行。一个优化被拒绝时保留使选择反转的条件，例如带宽提高、复用距离缩短或质量门槛改变；这使探索可以回退到有依据的分支。

把下界筛选放在实验之前，仍是一项通用研发顺序要求；下界只能在其表示、资源和计算前提成立时排除对应候选。

修正后：下界排除只针对已明确的表示/资源假设；未知可实验取得，改变表示或执行组织可不受原模型约束。

关闭时 SHA-256：`0480a8e7697850656cc1d8fe0534e97288505a9b08565d038ba2bc51958d7651`。

## 最终读取身份

- `/workspace/scratch/db4271884843/InfraAgents-know/docs/knowledge-use-cases.md`，SHA-256 `b49e145bf67c0aa7df39242bb9d3225a001b883210f995409828a83b08c4785d`。
- `/workspace/scratch/db4271884843/InfraAgents-know/docs/actionable-knowledge/use-cases.json`，SHA-256 `1ca0d50cbbbac067db503063031aad4ee8a9be80c08bfd370a6e11be6058ba2a`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/templates/entry.md`，SHA-256 `a5cdbe653673d602e95ea8515fddfe7a4530cb7d2faac9115d68c6d3c9c33a47`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/system-semantics/design-synthesis.md`，SHA-256 `0480a8e7697850656cc1d8fe0534e97288505a9b08565d038ba2bc51958d7651`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/tools/correctness-oracles.md`，SHA-256 `c2f88bc9746ac25586fc6700aaf0f5a8d043a1b1f4aee99dfdee02e1ef468aab`。
- `/workspace/scratch/db4271884843/InfraAgents-know/knowledge/tools/statistical-experiments.md`，SHA-256 `ba32db33d1c75d61f1554689d8580dd9879707b4d00648cdd2e1aa8a202487b9`。

## 限制

- 当前结论是文字与代码边界审查，不能证明模型行为偏置已经消失。
- 本次没有执行新的引擎/Agent/GPU实验；算术只是既有设定值复算。
- 六章撤销、最终catalog/历史和schema2探针尚待root集成后核对，不能由文档宣告替代。
