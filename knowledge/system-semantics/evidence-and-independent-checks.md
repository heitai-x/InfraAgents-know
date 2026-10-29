# K-SYS-006：工程认识如何成为可迁移的专家知识

知识的增量是对系统机制与设计关系的新增解释；运行结果为这些解释提供事实、区分和修正。把一次现象提升为专家知识，需要连接对象、条件、干预、结果和竞争解释，使读者能够重建推导，在条件改变时判断哪些关系保留，并在反证出现后修正相应结论。

## 从现象到可检验命题

“改参数后更快”记录一次比较。“减少了哪些重复数据流、为何该运行点受到它限制、改变哪个条件后收益可能反转”才形成可迁移解释。例如开启前缀缓存后 TTFT 下降，可能来自少算 prompt，也可能来自排队、路由或预热差异；一个平均数不足以裁决。

将命题记为 `C = (id, kind, P, S, M, I, status, rev)`。`P` 是待检验的具体陈述，`S` 是可判定范围，`M` 是机制或推导，`I` 是失效条件集合，`status` 是当前支持状态，`rev` 固定修订身份。范围包含与判断相关的模型/权重/adapter、精度、布局、设备与后端、负载联合分布、到达方式、并发、缓存热状态、质量门槛及测量起止事件。范围不必罗列无关配置，但声明的条件必须能定位到实际对象。

证据记录为 `e = (source, version, condition, input, event, comparator, unit, dimension, outcome, dependency, cost)`。`unit` 区分独立运行、区组、请求与运行内样本；`dimension` 指明检查的是数值、索引、身份、终态还是性能；`dependency` 记录参考是否依赖被测路径的中间产物。原始输出、缺失、取消和失败均进入记录。来源支持一种机制的定义，不能自动支持目标部署上的速度或质量。

对于可重复的配对运行，定义第 `j` 个独立区组的差值 `d_j = Y_A(j) − Y_B(j)`，均值 `d̄ = Σ_j d_j/n`。当 `n ≥ 2`、差值独立、总体方差有限且为正，并满足相应 t 区间条件时，均值差区间为 `d̄ ± t_(1−α/2,n−1) s_d/√n`，其中 `s_d² = Σ_j(d_j−d̄)²/(n−1)`。正态差值和固定样本设计下可取得相应精确覆盖；采用近似条件时须另说明依据。`s_d` 和区间与指标 `Y` 同单位；若指标是延迟，负差值表示 A 更低。区间包含零表示尚不足以区分方向，不等于证明相等。同一运行中的相关 token 不能自动当作独立区组。S-STATS 不处理配对比较与固定效应；此式是本章在上述假设下采用的标准推导，不归称为该论文的配对方法。

## 合法性与支持状态

模型函数、输入、比较对象、版本、单位和终止事件必须对应同一命题。不同知识条件可以是实验的有意干预，但模型、任务、工具权限等应固定的条件不能随组别一起变化。设备错误可能在后续同步点才显现，最后一行报错不一定是首次偏离。完整终态要求每个计划单位都有契约允许的结果，不能把缺失、取消、超时或部分完成转换成成功，也不能为使测试运行而截短用户要求的历史。

定义、接口契约、代数推导、运行观测和构建效果承担不同结论。浮点归约不因实数结合律而获得逐位等价；稳定地址也不证明内容代次正确。端到端 SLO 支持一个完整路径的工作点，不能替代局部索引边界证明。

| 命题类型 | 本章的晋升条件 | 可取得的状态 | 未覆盖的结论 |
|---|---|---|---|
| `MECH`：机制、定义或推导 | M 已展开；固定来源或独立推导支持 M；S 落在该依据的适用范围 | `SOURCE_SUPPORTED` | 目标设备速度或目标负载收益 |
| `OBS`：同范围关系 | 完整同条件观察；原范围各声明维度的独立参考与负对照；M 解释观察关系；I 明确失效条件 | `OBSERVED`、`CHECKED`、`SUPPORTED` | 条件改变后的有效性 |
| `TRANSFER`：改变条件后的规则 | 原范围观察和检查；实际执行未参与选择的改变条件保留评价；结果满足 P 的预声明支持判据且无已消费的反驳证据；改变条件上的独立参考与负对照；M 解释保留的依赖 | `TRANSFERABLE`；未评价或未区分时保留 `CHECKED`，反驳时进入 `CONTESTED` | 未测试变化、其他设备或版本 |
| `BUILD_EFFECT`：知识条件对构建的作用 | 完整构建运行；配对或区组；预算匹配；固定且独立的评价器；保留构建任务；评价负对照；M 与使用路径分别成立 | `SUPPORTED_EFFECT` | 所有 Agent 均获益或最终吞吐必然提高 |

`OBSERVED` 表示合法观察已有，但独立参考或其负对照不足。`CHECKED` 表示原范围的观察和独立检查成立，但相应解释、迁移或构建效果门槛尚未满足；它不保证改变条件已检查，也不保证改变条件的收益方向已区分。`SUPPORTED` 还连接机制与失效条件。`CONTESTED` 表示同范围证据相互排斥，或改变条件上的保留结果反驳预测。

这里的冲突是对同一命题的互斥证据：违反确定性契约，或按预先规定的统计判定反驳同一个 P。两个带噪测量值不相等，不自动构成冲突。不同范围、不同指标或不同估计量应先分开。评价已经执行、结果支持、结果反驳和结果尚未区分是不同事实；“已执行且未反驳”不足以推出支持。

支持与反驳判据由具体 P 决定，并在观察保留结果前冻结。例如命题是延迟均值差小于零，可声明区间上端严格小于零才支持，下端严格大于零才反驳，其余结果未区分。若命题是至少降低某个幅度、满足等价范围、维持质量或遵守确定性契约，应使用各自的单位、边界及关系，不能把这个负方向区间规则泛化。无法合法构造区间或证据不足以应用规则时，保留缺口，不能补成支持。

构建效果中的 M 与使用路径不可互代。M 要给出“知识改变哪项设计或排错选择、经哪条工作/等待路径影响哪个指标”的可检验解释，并提出竞争解释的不同预测。使用路径则要把冻结知识条目、实际读取、具体设计决定、交付物及检查结果连起来，身份和先后顺序可追溯。只有读取日志，不能证明使用；只有事后写出的合理故事，也不能证明该机制进入实现。没有 M 时仍可报告受控实验的差值，但不取得本章解释性命题的 `SUPPORTED_EFFECT` 状态。该状态也不单独证明唯一中介机制或排除了所有未观测混杂。

## 独立核对与负对照

独立性应具体到被验证的维度。验证页表索引时，参考可以从每个请求的逻辑 token 全量重建，但不能调用被测页表 walk；验证量化执行时，应固定量化码、scale、zero point 和 bias 独立复算，再另查相对原模型的质量；验证协议时，预期事件来自服务契约，不能由同一个错误解析器同时产生实际值与参考值。

可以沿参考的依赖图，从输出向输入追踪。如果它读取被测组件产生的索引、归属或完成状态，该维度就没有获得独立核对。共享权重文件不一定破坏索引检查的独立性；换一个函数名、审查者或误差模型标签也不自动产生独立性。依赖图声明本身仍需作者核对，纯逻辑模型不能确认实际实现没有隐藏调用。

负对照主动改变一个应被拒绝的维度，并保持其他条件。例如改变一个被比较元素、复制其他请求的输出、删除终态、提前结束仍有在途工作的测量窗口，或把迟到完成放到槽位复用之后。检查器应拒绝该轨迹或按契约记录失败。某检查器接受已知错误时，使用它支持该维度的晋升被阻断；尚未运行负对照则是检查不足，两者不同。

负对照通过只支持已注入错误的辨别力。重复使用共享错误逻辑的检查不会因次数增加而成为独立证据；固定样例全部通过也不覆盖未触发的状态边界。测量参考独立和负对照有效，说明测量链在声明维度具备核对能力，不能补足结果尚未满足收益判据的问题。

## 证据归一化与完整晋升算法

先初始化 accepted、rejected 和按范围分组的索引。逐条读取 E，按版本、任务、单位、范围兼容性分区；不兼容记录进入 rejected 并保留理由，不参与该范围的晋升。再用计划单位集合检查缺失与唯一终态，以声明的维度集合查找参考、依赖与负对照，最后判断同范围冲突。删除不兼容记录不能替代取得有效观察。

下表给出算法和配套检查共同使用的全部独立谓词。它们是审查结果的抽象输入，不由晋升状态反推。缺少支持某个有利谓词的证据时，该谓词为假。记录“未发现接受错误”不能使 `base_control_rejected` 自动为真。

| 字段 | 对应条件 |
|---|---|
| `scope_declared` | S 非空，包含本命题必需的可定位条件 |
| `identity_match` | 归一化后必需的模型、任务、版本和比较身份成立 |
| `terminal_misreported` | 任一保留记录把缺失、超时、取消或部分完成记成成功 |
| `bad_control_accepted` | 用于本命题的任一检查器接受其声明维度的已知错误 |
| `same_scope_conflict` | 有效证据在同一范围按既定判定相互排斥 |
| `source_supported` | 固定来源或独立推导实际支持 M，而非仅存在引用 |
| `source_scope_match` | MECH 声明的 S 满足上述依据的适用前提 |
| `terminal_complete` | 每个计划单位有完整且唯一的合法终态 |
| `observation` | 存在有效观察；构建命题保留完整运行而非仅选中成功轨迹 |
| `same_condition` | 观察与所声称范围、指标和比较条件一致 |
| `base_reference` | 原范围非空的声明维度集合均有相应独立参考 |
| `base_control_rejected` | 原范围声明的负对照已执行且均被正确拒绝 |
| `mechanism` | M 已展开并连接本类型的观察关系、保留依赖或构建设计与指标 |
| `invalidation_stated` | OBS 的 I 给出会撤销或要求重测该关系的具体条件 |
| `heldout_changed` | 已在改变条件上执行未参与选择的保留评价，仅表示评价已发生 |
| `heldout_support` | 该评价的结果满足 P 预先声明的支持判据；未评价时不消费 |
| `heldout_conflict` | 该评价按预定规则反驳迁移预测；未评价时不消费 |
| `changed_reference` | 改变条件上的声明维度均有独立参考；不沿用原范围资格代替 |
| `changed_control_rejected` | 改变条件上的对应负对照已执行并被拒绝 |
| `paired_or_blocked` | 知识条件按完整任务/运行配对或采用声明的区组设计 |
| `budget_matched` | 各知识条件使用相同的预算口径与上限，并记录实际消耗 |
| `evaluator_fixed` | 各条件使用冻结且相同的任务标准、评价器和聚合规则 |
| `heldout_build` | 未参与选择的构建任务已评价，结果满足 P 预先声明的判定条件 |
| `documented_use_path` | 知识身份到读取、决定、交付物和检查的使用路径可追溯 |

`source_supported` 与 `source_scope_match` 只服务 MECH；`invalidation_stated` 是 OBS 的显式门槛。迁移与构建分支使用各自定义下的 M 和保留评价，不把无关字段加入门槛。`heldout_build` 同时要求实际评价及其结果符合所声称的 P，单有一个未运行的保留任务不成立。若差异区间不能满足预定收益判定，应报告未区分，不能将任务存在记为该字段成立。

迁移评价的三个布尔字段按下表消费。正常的单次评价应使支持与反驳互斥；归一化若收到矛盾摘要，不得通过丢弃反驳把它修成支持。

| `heldout_changed` | `heldout_support` | `heldout_conflict` | 迁移分支语义 |
|---|---|---|---|
| 假 | 任意 | 任意 | 未评价；两项结果字段不消费，保留异常或过期字段供核对，返回 `CHECKED` |
| 真 | 真 | 假 | 支持 P；其余迁移门槛全部成立才可晋升 |
| 真 | 假 | 真 | 反驳 P；返回 `CONTESTED` |
| 真 | 假 | 假 | 已评价但未区分，或缺少可应用判据的证据；返回 `CHECKED` 并记录下一检查 |
| 真 | 真 | 真 | 矛盾结果摘要或相互冲突的保留证据；按反驳优先进入 `CONTESTED`，核对记录身份、范围和判定规则 |

```pseudocode
ALGORITHM PROMOTE(C, E, R)
INPUT: C = (kind, P, S, M, I, rev)
       E = fixed evidence records and planned units
       R = claimed dimensions, reference dependencies and negative controls
OUTPUT: status, accepted, rejected, reason, next_check

PREPARE:
    accepted <- empty; rejected <- empty; groups <- empty
    FOR EACH record e IN E:
        IF version/task/unit/scope is incompatible:
            append (e, incompatibility_reason) to rejected
        ELSE:
            append e to accepted and its scope/dimension group
    Derive the 24 predicates in the field table from C, accepted, planned units and R
    Apply P's frozen decision rule to evaluated changed-condition results
    Preserve record identities, decision rule, unresolved reasons and contradictory summaries
    Preserve the supporting record identities for every true favorable predicate
    If a return below has missing prerequisites, next_check names those prerequisites
    If it has conflict, next_check distinguishes the competing predictions

P01 IF kind is not one of MECH, OBS, TRANSFER, BUILD_EFFECT:
        RETURN BLOCKED, unknown_kind
P02 IF NOT scope_declared OR NOT identity_match:
        RETURN BLOCKED, missing_scope_or_identity
P03 IF terminal_misreported:
        RETURN BLOCKED, terminal_misreported
P04 IF bad_control_accepted:
        RETURN BLOCKED, checker_not_discriminating
P05 IF same_scope_conflict:
        RETURN CONTESTED, same_scope_conflict
P06 IF kind = MECH:
        IF mechanism AND source_supported AND source_scope_match:
            RETURN SOURCE_SUPPORTED, scoped_source_or_derivation
        RETURN BLOCKED, mechanism_evidence_incomplete
P07 IF NOT terminal_complete:
        RETURN BLOCKED, terminal_outcomes_incomplete
P08 IF NOT observation OR NOT same_condition:
        RETURN BLOCKED, no_same_condition_observation
P09 IF NOT base_reference OR NOT base_control_rejected:
        RETURN OBSERVED, original_scope_check_incomplete
P10 IF kind = OBS:
        IF mechanism AND invalidation_stated:
            RETURN SUPPORTED, observation_check_mechanism_and_invalidation
        RETURN CHECKED, mechanism_or_invalidation_missing
P11 IF kind = TRANSFER:
        IF NOT heldout_changed:
            RETURN CHECKED, changed_condition_not_evaluated
P12     IF heldout_conflict:
            RETURN CONTESTED, transfer_prediction_failed_or_conflicting_results
P13     IF NOT heldout_support:
            RETURN CHECKED, transfer_prediction_unresolved,
                   next_check = resolve_missing_evidence_or_plan_valid_independent_repeats
P13b    IF NOT changed_reference OR NOT changed_control_rejected OR NOT mechanism:
            RETURN CHECKED, changed_condition_check_or_mechanism_missing
P14     RETURN TRANSFERABLE, scoped_transfer_rule
P15 IF NOT paired_or_blocked OR NOT budget_matched OR NOT evaluator_fixed:
        RETURN CHECKED, construction_comparison_not_controlled
P16 IF NOT heldout_build OR NOT documented_use_path OR NOT mechanism:
        RETURN CHECKED, construction_transfer_use_or_mechanism_missing
P17 RETURN SUPPORTED_EFFECT, scoped_construction_effect
```

P15–P17 只会由 BUILD_EFFECT 到达，因为 P01 排除了未知类型，其他类型均已返回。P16 分别检查 M 与使用路径，避免只记录路径就晋升。原范围检查通过也不能替代 P13b 的改变条件参考，更不能替代 P13 对预测结果的判断。

顺序具有语义：伪造成功或接受已知错误先阻断；同范围矛盾随后进入争议。MECH 不要求运行观察。改变条件的冲突只在原范围已检查且保留评价实际存在后消费；有反驳观察但尚缺改变条件参考、机制或支持资格时，P12 先保留争议。支持与反驳同时为真也不绕过它。原范围尚未完成检查时仍按 P09 返回 `OBSERVED`，同时保留已有反驳记录供后续诊断，不将其删除。

可将迁移部分的声明式证书写成逻辑条件。以下仅在已通过 P01–P09 且类型为 TRANSFER 的上下文中使用；全局阻断与同范围冲突仍保留原优先级。令 `E=heldout_changed`、`H=heldout_support`、`F=heldout_conflict`，`G=changed_reference ∧ changed_control_rejected ∧ mechanism`，则：`CONTESTED ⇔ E ∧ F`；`TRANSFERABLE ⇔ E ∧ ¬F ∧ H ∧ G`；`CHECKED ⇔ ¬E ∨ (E ∧ ¬F ∧ (¬H ∨ ¬G))`。三者互斥且完备。特别地，`E ∧ ¬F` 只说明已评价且没有被消费的反驳，不能省略 H。

一个最小区间反例是：P 声明改变条件后 A 的平均延迟低于 B，即 `μ_A−μ_B<0`，原范围已有充分支持，其余迁移检查均成立。改变条件上的 16 个独立配对区组给出 `d̄=0 ms`、`s_d=4 ms`，使用给定临界值 2.131，半宽为 `2.131×4/√16=2.131 ms`，区间为 `[-2.131,2.131] ms`。按上述预声明方向规则，结果既不支持负方向，也不反驳负方向，因此 `E=true,H=false,F=false`，P13 必须返回 `CHECKED`。独立参考和有效负对照不能把这个结果变成方向支持。

保持 n、标准差和临界值不变，将均值分别改为 −4 ms 和 +4 ms，区间分别为 `[-6.131,-1.869] ms` 与 `[1.869,6.131] ms`：前者在其他门槛满足时可取得 `TRANSFERABLE`，后者由 P12 取得 `CONTESTED`。区间仅接触零时也不满足这里的严格方向支持或反驳规则。上述数值是设定算例，不是运行测量，也不证明该区间方法在实际负载中的覆盖率。

未区分后的下一检查应针对原因：若独立单位不足，在冻结候选、任务与合法停止方案下安排新的独立区组；若顺序效应、漂移或 carryover 可疑，先随机或均衡 A/B 顺序并检查差值结构；若计时边界或参考不完整，先补足相应证据。不能不断查看同一保留结果直到刚好通过，也不能事后放宽收益门槛；保留集用于调参后，新评价需重新建立独立资格。新增重复也可能继续未区分，此时保留 `CHECKED` 与具体范围。

一个定向反例是：BUILD_EFFECT 的范围、完整观察、独立评价和负对照、配对、预算、固定评价器、保留任务及使用路径全部成立，唯独 M 缺失。P16 必须返回 `CHECKED`。另四个只撤掉一个条件的反例分别是 OBS 缺 I、TRANSFER 缺改变条件参考、BUILD_EFFECT 改变评价器、BUILD_EFFECT 缺使用路径；它们也必须停在 `CHECKED`。这些例子保持其他条件成立，能检出对应的漏查，而非依赖另一个无关错误阻断。

配套检查将顺序状态机与声明式状态证书比较。证书规定每个状态的必要且充分条件，独立收集所有可接受状态，要求结果集合恰有一个元素；它不调用顺序状态机产生参考值。按命题类型穷举相关字段，包含相互矛盾的输入组合以审查优先级，再测试只忽略一个谓词的故障版本。迁移算例从固定配对数据计算摘要和区间，再把支持、反驳及执行状态送入真正的晋升函数和独立证书；只遗漏支持门槛的故障版本必须错误晋升跨零反例，并由证书检出。字段表是人工核对正文与转写的接口；检查不解析自然语言，也不验证真实记录是否足以把某字段设为真。

支持状态不是不可回退的流水线。发现反例、依据版本变化或前提破坏后，应重新归一化并求状态，保留旧修订身份与回退理由。检查器修复后，定位受影响的规则和实验；原始输入及事件完整时用新检查重分析，缺信息时重跑相应路径。修复检查器不等于候选已通过新检查。

## 成本：时间、资源与驻留分别列账

先固定候选、输入、检查边界及重复层级。设 `n_b` 次构建，每次 `r` 次运行，每次 `m` 个测量样本，则 `n_run = n_b r`、`n_measure = n_b r m`。对全部阶段串行执行的设定，可写：

`T_serial = t_fixture + n_b t_build + n_run t_run_setup + n_measure(t_work + δ_observe) + n_xfer t_xfer + t_reference + t_sync_extra + t_review`。

所有 t 与 δ 的单位均为秒。`t_fixture` 是固定版本、准备输入/身份表/参考/负对照的实际耗时；`t_build` 是每次构建耗时；`t_run_setup` 仅含每次启动、加载及预热等准备；`t_work` 是一个样本的实际工作；`δ_observe` 仅含样本新增的记录、计时或采样开销；`t_xfer` 是未在其他项内计入的迁移、重排或序列化；`t_reference` 是独立参考的专属准备和执行；`t_sync_extra` 是其余项未包含的额外同步等待；`t_review` 是人工或 Agent 判断、复算和记录修订理由实际花费的时间。

运行准备项不包含测量样本本身。如果取得的是已经包含样本和观测开销的整段运行时间，则用该整段时间替换相应展开项，不能再加一次 `n_measure(t_work + δ_observe)`。传输、参考和同步同样按互斥事件归属计一次。准备可跨区组摊销时说明复用身份；每次重新加载或构建的工作不能当成一次性准备。

`T_serial` 是串行情形的经过时间，不是任意并行实验的墙钟时间。并行执行时构造含真实先后关系及资源串行化约束的事件 DAG，`T_wall` 由其关键路径得到；不能把重叠阶段的时间全相加。另保留 CPU 核秒、GPU 分配秒、人工/Agent 审查时间等资源向量，分别求和，不将这些累计秒数当作同一条墙钟。GPU 同步完成点与主机返回点也不能互换。

容量与驻留持续时间另列。内存峰值为 `M_peak = max_t Σ_o size_o(t)`，每个物理对象只计一次，共享对象不按读者数重复计量。证据存储的驻留量为 `Q_store = ∫ V_store(t) dt`，若 V 使用 GiB、t 使用秒，则 Q 单位为 GiB·s；恒定保存 V、持续 τ 时为 Vτ。容量峰值、保存多久和支付多少费用是不同量。

需要与决策损失比较时，把检查资源换成共同货币单位：`B_check = Σ_r price_r H_r + price_store Q_store + B_external`。`H_r` 的单位与费率对应，例如设备分配秒乘货币/分配秒，审查秒乘货币/秒，存储 GiB·s 乘货币/(GiB·s)。整实例账单与实例内 CPU/GPU 分项只能选择不重复的计费口径；内部访存或网络若已含在实例费用中，不再作为另一笔同类费用相加。访存量、传输字节和同步等待仍应作为诊断资源单列。

设 `p_ab = P(方案错误且被接受)`、`p_rg = P(方案正确且被拒绝)` 为同一决策总体中的联合概率，错误接受损失与错误拒绝损失分别为同一货币单位的 `ℓ_ab`、`ℓ_rg`，则 `L_money = ℓ_ab p_ab + ℓ_rg p_rg + B_check`。若使用条件错误率，必须再乘相应的错误/正确方案先验比例。不能把货币损失直接与秒相加。确需计入决策延迟的机会成本时，可另加 `price_delay T_wall`，并说明其未包含在上述损失中。本章没有估计这些概率。

一个可复算设定是：准备 60 s，构建 2 次各 120 s，每次构建运行 2 次、每次准备 10 s，每次运行测量 3 次、每样本工作 2 s 且观测额外 0.5 s；另有 4 次各 1 s 的传输、参考 30 s、额外同步 6 s、审查 180 s。串行总计为 `60+240+40+30+4+30+6+180 = 590 s`。若前 410 s 均占用一个按 0.01 货币单位/s 计费的机器资源，审查按 0.02 单位/s，证据 2 GiB 保存一天按 0.001 单位/(GiB·day)，则 `B_check = 4.10+3.60+0.002 = 7.702`。设联合错误概率为 0.01 和 0.02、损失为 1000 和 100，则总期望损失为 19.702 货币单位。这些都是设定算例，不是机器、检查器或构建实验的测量。

晋升本身通常远便宜于取得证据。用哈希索引处理 N 条记录、U 个计划单位、D 个检查维度、K 个负对照及 V 个节点、A 条边的参考依赖图时，元数据分区、集合核对和遍历的预期工作量为 `O(N+U+D+K+V+A)`；哈希冲突或采用排序会改变该界。原始输出和来源读取的字节量、参考复算、设备复制与长期归档另计，不能由谓词数量推得。保存全部原始对象的成本可能主导整个流程。

参考执行、profiling、复制与确定性同步会改变目标路径。诊断计数与正常路径性能分开报告。增加内层样本不能消除未重复的构建或环境层变异；S-STATS 的层级重复模型提醒我们先识别变异来源，再按实际重复成本分配预算。

## 条件变化后的保留、调整与放弃

模型函数、权重、位置规则与质量契约不变而机器速度改变时，可以保留机制关系，重新取得成本与性能证据。硬件拓扑、带宽域、后端和并发改变时，可保留适用的容量/流量下界，但排队、共享资源与关键路径须重算。负载长度、输出长度、复用间隔、取消率及其相关性改变时，收益可能反转；连续重复输入得到的理想命中率不能代表交错多租户。

前提被破坏时，应放弃原转移规则。例如共享前缀需要计算等价身份与兼容状态；量化 KV 迁移需要编码版本、scale/zero point、分片布局、完整范围和目标可见性共同成立。字节哈希一致不能证明目标按相同方式消费，单次文本一致也不能证明取消与重试安全。

前缀共享与异步取消的组合可沿完整轨迹核对。以下是设定的事件模型，不是 GPU 执行结果。共享页 P 的逻辑所有者为 A、B，私有页分别为 Q_A、Q_B；在途作业持有独立读取租约，租约包含请求、物理页、代次和作业身份。

| 事件 | 所有权、在途状态与允许动作 |
|---|---|
| A、B 提交计算 | P 有两个逻辑所有者及相应在途租约；Q_A、Q_B 各有私有所有者和在途租约 |
| 取消 A | A 取得唯一取消终态，撤销其后续提交权；逻辑所有权可解除，但 A 的在途租约仍保护 P 与 Q_A |
| C 请求槽位 | 若没有其他页，C 等待；不能仅因 A 已取消就把仍在途的 Q_A 交给 C。容量允许时可另分配 Q_C |
| A 的实际完成依赖被确认 | 按作业身份释放 A 的租约一次；Q_A 无所有者且无租约后才可回收。P 仍由 B 保护 |
| C 获得已回收的 Q_A | 分配代次递增，C 的内容、页表与输出归属绑定新代次；这一步发生在旧读取实际完成之后 |
| A 的迟到通知再次到达 | 用请求/作业/代次拒绝旧输出与重复释放，不能改写 C 或减少 C 的新租约，也不能把 A 的取消终态改为成功 |
| B、C 完成 | 分别核对逻辑前缀、输出身份和终态；最后所有者与租约解除后才回收对应对象，明确的缓存保留另计 |

若共享页大小为 p、私有页均为 q，则 A/B 在途时所列物理容量是 `p+2q`。取消后为 C 另分配一页时为 `p+3q`，额外容量 q 可减少 C 等待旧依赖完成的时间；池容量不足时只能等待或调整接纳。提前复用虽然看似节省 q，却违反所有权条件。参考从 B、C 各自逻辑 token 全量重建，另核对租约释放与终态，不调用被测页表。只比最终文本可能被偶然相同输出掩盖；只比地址又会漏掉代次混淆。真实设备的可见性、完成事件与内存排序仍需设备检查。

知识修订应保留仍成立的公式、对象模型和分支，把新认识放到它改变的位置。实现违反已写明的租约条件，首先支持实现偏离；若原机制没有保护在途读者，才需修正机制。一次成功也不能为所有被读条目同时记功。

## 区分实验、反例与竞争解释

区分实验改变一个使竞争解释产生不同预测的条件，并保持任务标准与比较边界。下表是实验设计，判定需要对应的实际证据。

| 观察 | 竞争解释 | 有区分力的干预与修正 |
|---|---|---|
| 前缀缓存后 TTFT 下降 | 少算 prompt；排队/预热变化；路由集中 | 固定到达率与热状态，记录命中 token、实际 prefill、队列和路由。若只在预热窗口下降，缩窄范围；若队列主导，不能仅凭命中率归因 |
| 自纠正后准确率上升 | 外部反馈；更多响应或选择预算；初始提示缺要求 | 去掉 oracle 标签，比较同响应预算基线，把任务要求放回初始提示。增益随反馈或预算消失时，修正为相应条件下的作用 |
| 知识扩充后交付率提高 | 设计改善；预算/模型/工具差异；评价遗漏失败 | 冻结比较条件，按完整构建运行配对，保留独立任务，检查使用路径，并注入删除失败的轨迹。只在筛选成功样本中存在的差异不能支持总体交付改善 |
| 更长上下文的成本未线性增加 | KV 复用；压缩/卸载；漏计排队和在途；长度与复用相关 | 同时记录复用、驻留、传输、完整完成窗口及联合负载。拆开节省与新增成本，再限定结论 |

研究知识贡献时，评价应区分发现方案、减少错误分支、到合格系统的成本和最终引擎表现。减少试验次数不一定减少成本，因为剩余试验可能更贵。固定预算未交付的运行保留为未成功；不能只平均成功运行的时间。选出候选后冻结版本，再在未参与选择的任务上评价；保留集一旦继续用于调参，就不再具有原先的独立资格。

一个身份反例是：预期 A=17、B=99，实际返回 A=17、B=17。只检查每个值是否来自合法值集合会接受；完整检查应核对身份集合、重复、缺失、额外结果、终态和每个身份的值。空结果也不能因“所有存在结果都正确”而成为完成。若契约允许取消，其预期终态来自契约，不能由实际返回值临时决定。

性能负对照可以删除慢请求、删除失败、提前结束仍有在途工作的窗口，或跳过真实模型路径。测量链应拒绝这些轨迹，或按预先定义的分母与失败规则记录。SLO 要求 deadline 内成功时，超时和质量失败不能被均值或样本筛选吞掉。

随章 `tools/check_k_sys_006_full.py` 是确定性、离线、只读的标准库模型检查。它核对晋升顺序与声明式证书、单条件遗漏、冲突优先级、固定区间算术、身份、合成依赖图和成本算例。迁移区间检查消费给定临界值，不求解 t 分位数。枚举仅覆盖各类型相关字段，无关字段另取全假和全真投影；不代表每种类型全部字段组合均穷举。它不验证原始证据抽取、真实参考隐藏依赖、统计覆盖率、保留集实际隔离、真实并发实现、设备可见性、性能或 Agent 构建收益。运行结果应由绑定当前脚本与输入的执行记录承担，形式化一致也不能替代对判据语义的审查。

## 运行记录、失败与知识修订

命令、路径、临时文件名和一次性环境故障通常属于运行证据。可进入正文的是更准确的成本关系、状态不变量、适用分支、可复算反例及其原因。来源版本、编辑日期与实验日期分别保存；今天的观测不能借引用旧论文变成截止前的事实。

失败分析先区分输入错误、实现偏离、资源不足、协议竞争和环境故障。最小化必须保留触发机制：跨页保留跨页，在途取消保留复用和迟到完成，数值问题保留敏感形状、位置与精度。删除并发后错误消失，并不表示已经解释原竞态。

反例应指出被推翻的量词。短请求卸载更慢，可以反驳“卸载总会更快”，不能反驳“容量不足时卸载提供可执行路径”。出现共同范围的矛盾时先区分函数、条件和证据层次，再拆分范围或提出新的区分实验；不靠多数票压制反例，也不把局部失败扩大成所有机制失效。

<!-- infraagents:provenance -->

## 来源与推导

来源事实受 2026-04-30 cutoff 约束，编辑日期不改变事实日期。本章的证据状态、晋升算法、成本换算、状态轨迹、方向判定算例及负对照是本库的解释与推导，设定算例不冒充实测。以下归档链接标识来源对象；来源说明中的历史阅读记录不自动成为其他作者的阅读资格。论文中的模型、任务和反馈条件随结论保留，不将其成绩直接作为本库或目标部署的效果证明。

- [S-ACE：Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](../sources/S-ACE.md)：固定 v1 的方法部分将上下文表示为有身份的条目，分开生成、反思和整理，局部 delta 经确定性合并，避免整体重写丢失细节。其限制部分说明反馈不可靠时上下文可能受污染。本章据此解释局部修订与反馈资格，不将论文报告的成绩转为本库构建收益。
  [完整原文](../../references/S-ACE/reader.html) · [全文文本](../../references/S-ACE/fulltext.txt)

- [S-GEPA：GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](../sources/S-GEPA.md)：固定 v2 说明以执行/评价轨迹反馈提出提示候选，使用 Pareto 候选选择和预算约束；用于选择的评价集与最终测试承担不同角色。本章据此解释搜索后保留评价及预算匹配，不把被反复用于选择的 Pareto 集直接当成未接触的最终保留集，也不将提示优化成绩作为 InfraAgents 收益。
  [完整原文](../../references/S-GEPA/reader.html) · [全文文本](../../references/S-GEPA/fulltext.txt)

- [S-SELF-CORRECTION：Large Language Models Cannot Self-Correct Reasoning Yet](../sources/S-SELF-CORRECTION.md)：该研究定义 intrinsic self-correction，并在其所考察推理设置中报告能力限制和退化现象；oracle 标签、额外响应预算和初始提示不完整会带来不同解释。本章据此区分外部反馈、匹配预算与提示条件；结论保留其模型、任务和实验设置范围，不外推到所有自纠正方法或当前模型。
  [完整原文](../../references/S-SELF-CORRECTION/reader.html) · [全文文本](../../references/S-SELF-CORRECTION/fulltext.txt)

- [S-REFLEXION：Reflexion: Language Agents with Verbal Reinforcement Learning](../sources/S-REFLEXION.md)：方法部分将任务反馈转为文字反思并保存在后续尝试使用的记忆中，改进载体不要求更新权重；自评价、测试生成及记忆限制会影响结果。本章借此解释反馈到记忆再到验证的链，不能由反思文本合理就推出其正确或可迁移。
  [完整原文](../../references/S-REFLEXION/reader.html) · [全文文本](../../references/S-REFLEXION/fulltext.txt)

- [S-STATS：Quantifying Performance Changes with Effect Size Confidence Intervals](../sources/S-STATS.md)：来源说明及相应论证连接平衡嵌套随机影响、重复层级、效应量区间及按成本分配预算，要求相应独立性、同方差与稳态假设；忽略高层变异时增加内层样本可能造成低覆盖。该文不处理配对比较与固定效应，本章不据它保证任意非平稳服务尾延迟区间；配对公式、货币换算及晋升规则是本章在明确假设下的推导。
  [完整原文](../../references/S-STATS/reader.html) · [全文文本](../../references/S-STATS/fulltext.txt)

## 延伸阅读

- [正确性 Oracle：函数参考、状态变形与服务契约](../tools/correctness-oracles.md)。
- [失败分析：最早偏离、最小复现与机制归因](../tools/failure-evidence.md)。
- [性能实验：独立重复、效应量与尾部证据](../tools/statistical-experiments.md)。
- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
- [系统设计推导：把模型、缓存、执行与目标组合起来](design-synthesis.md)。
- [分布式状态迁移：身份、传输与提交点](distributed-state-transfer.md)。
